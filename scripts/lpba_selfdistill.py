"""Self-distillation of DA3-SMALL's test-time ensemble: train the model to output, in one plain pass, the depth and
cameras that averaging 7 passes gives (knowledge/ideas/da3-tta.md), so PBA gets the ensemble's initialisation at one
forward pass. No labels: any clips work (here ARKitScenes windows, without their LiDAR / poses).

    # targets: 7 passes (plain, flip, reversed, reversed + flip, reference middle / saddle_balanced / saddle_sim_range)
    .venv/bin/python scripts/lpba_selfdistill.py targets <arkit root> <data dir> [max_clips=1000]
    # train the main depth branch (refinenet1-4, output_conv1/2) and the camera decoder on them
    .venv/bin/python scripts/lpba_selfdistill.py train <data dir> <run dir> [steps=3000 lr=5e-5 w_rot=1 w_trans=0.1]
    # predictions of the student in the evaluator's layout (its own depth, confidence and cameras)
    .venv/bin/python scripts/lpba_selfdistill.py export <src work dir> <run dir> <out work dir> [split=dev|held-out]

Targets per clip: log depth (each pass scaled to the plain pass's median, averaged, stored at H/4) and the cameras
relative to frame 0 (rotations: chordal mean; translations: mean, scaled with the depth). Loss: squared log-depth error
after one clip scale + w_rot |R_s - R_t|_F^2 + w_trans |t_s / mean|t_s| - t_t / mean|t_t||^2 over the frames.
"""

import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpba_tta import mean_poses, rel_poses  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASSES = ["plain", "flip", "rev", "revflip", "ref:middle", "ref:saddle_balanced", "ref:saddle_sim_range"]
TRAIN_HEAD = ("refinenet1.", "refinenet2.", "refinenet3.", "refinenet4.", "output_conv1.", "output_conv2.")
DEV = {"hiroom": 10, "7scenes": 2, "scannetpp": 3, "dtu64": 4, "eth3d": 3}


def load_api():
    from depth_anything_3.api import DepthAnything3

    return DepthAnything3.from_pretrained("depth-anything/DA3-SMALL").to("cuda").eval()


def raw_forward(api, ims, ref="first", grad=False):
    imgs_cpu, _, _ = api._preprocess_inputs(ims, None, None, 504, "upper_bound_resize")
    imgs, _, _ = api._prepare_model_inputs(imgs_cpu, None, None)
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
        out = api.model(imgs, None, None, [], False, False, ref)
    return out["depth"][0].float(), out["extrinsics"][0].float()  # [F, H, W], [F, 3, 4] world-to-camera


def ensemble(api, ims):
    base_d = None
    logs, rels = [], []
    for v in PASSES:
        flip, rev = "flip" in v, v.startswith("rev")
        ref = v[4:] if v.startswith("ref:") else "first"
        x = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in ims] if flip else list(ims)
        x = x[::-1] if rev else x
        d, E = raw_forward(api, x, ref)
        d, E = d.cpu().numpy(), E.cpu().double().numpy()
        if rev:
            d, E = d[::-1], E[::-1]
        if flip:
            d = d[:, :, ::-1]
            S = np.diag([-1.0, 1.0, 1.0])
            E = np.concatenate([S @ E[:, :, :3] @ S, S @ E[:, :, 3:]], 2)
        if base_d is None:
            base_d = d
        k = np.median(base_d) / np.median(d)
        E = E.copy()
        E[:, :, 3] *= k
        logs.append(np.log(np.maximum(d * k, 1e-6)))
        rels.append(rel_poses(E))
    return np.mean(logs, 0), mean_poses(rels)


def bench_clips(src, split, per_scene=3, frames=24, seed=0):
    """(name, files) frame subsets of the benchmark scenes of a split (no labels used)."""
    rng = random.Random(seed)
    for ds, n in DEV.items():
        d = os.path.join(src, "model_results", ds)
        scenes = sorted(os.path.dirname(p) for p, _, _ in os.walk(d) if os.path.basename(p) == "unposed")
        scenes = scenes[:n] if split == "dev" else scenes[n:]
        for sd in scenes:
            files = [str(f) for f in np.load(os.path.join(sd, "unposed", "exports", "gt_meta.npz"), allow_pickle=True)["image_files"]]
            name = f"{ds}__{os.path.relpath(sd, d).replace('/', '_')}"
            if len(files) <= frames:
                yield name + "__0", files
                continue
            for j in range(per_scene):
                idx = sorted(rng.sample(range(len(files)), frames))
                yield f"{name}__{j}", [files[i] for i in idx]


def cmd_targets(root, data, opts):
    os.makedirs(data, exist_ok=True)
    os.chdir(REPO)
    api = load_api()
    if "split" in opts:  # benchmark scenes (root = src work dir)
        clips = list(bench_clips(root, opts["split"]))
    else:
        clips = [(os.path.basename(d), sorted(glob.glob(os.path.join(d, "*.jpg"))))
                 for d in sorted(d for d in glob.glob(os.path.join(root, "*_*")) if os.path.isdir(d))[: int(opts.get("max_clips", 1000))]]
    for i, (name, files) in enumerate(clips):
        out = os.path.join(data, name + ".npz")
        if os.path.exists(out):
            continue
        if len(files) < 4:
            continue
        t0 = time.time()
        ims = [Image.open(f).convert("RGB") for f in files]
        ld, rel = ensemble(api, ims)
        H, W = ld.shape[1:]
        ldl = F.interpolate(torch.from_numpy(ld)[:, None], size=(H // 4, W // 4), mode="area")[:, 0].numpy()
        np.savez(out, files=np.array(files), logd=ldl.astype(np.float32), rel=rel.astype(np.float32))
        if i % 50 == 0:
            print(f"[targets] {i}/{len(clips)} {name}: {len(files)} frames, {time.time() - t0:.1f}s", flush=True)


def losses(api, z, cfg, grad=True):
    ims = [Image.open(f).convert("RGB") for f in z["files"]]
    d, E = raw_forward(api, ims, "first", grad)
    lt = torch.from_numpy(z["logd"]).cuda()
    ls = torch.log(F.interpolate(d[:, None], size=lt.shape[-2:], mode="area")[:, 0].clamp_min(1e-6))
    s = (lt - ls).mean()
    l_depth = ((ls + s - lt) ** 2).mean()
    E4 = torch.cat([E, torch.tensor([[[0, 0, 0, 1.0]]], device=E.device).expand(len(E), 1, 4)], 1)
    rel = E4 @ torch.linalg.inv(E4[0])
    rt = torch.from_numpy(z["rel"]).cuda()
    l_rot = ((rel[1:, :3, :3] - rt[1:, :3, :3]) ** 2).sum((-1, -2)).mean()
    ts, tt = rel[1:, :3, 3], rt[1:, :3, 3]
    ts = ts / ts.norm(dim=-1).mean().clamp_min(1e-6)
    tt = tt / tt.norm(dim=-1).mean().clamp_min(1e-6)
    l_trans = ((ts - tt) ** 2).sum(-1).mean()
    return l_depth, l_rot, l_trans


def cmd_train(data, run, opts):
    cfg = {"steps": 3000, "lr": 5e-5, "w_rot": 1.0, "w_trans": 0.1, "seed": 0, "eval_every": 500, "n_val": 40}
    for k, v in opts.items():
        cfg[k] = type(cfg[k])(v)
    os.makedirs(run, exist_ok=True)
    os.chdir(REPO)
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    api = load_api()
    params = []
    for n, p in api.model.named_parameters():
        train = (n.startswith("head.scratch.") and n[len("head.scratch."):].startswith(TRAIN_HEAD)) or n.startswith("cam_dec.")
        p.requires_grad_(train)
        if train:
            params.append(p)
    print(f"[sd] {sum(p.numel() for p in params) / 1e6:.2f}M trainable", flush=True)
    files = sorted(glob.glob(os.path.join(data, "*.npz")))
    random.Random(0).shuffle(files)
    val, train = files[: cfg["n_val"]], files[cfg["n_val"]:]
    cache = {}

    def get(p):
        if p not in cache:
            cache[p] = dict(np.load(p))
        return cache[p]

    def evaluate():
        r = []
        with torch.no_grad():
            for p in val:
                r.append([float(x) for x in losses(api, get(p), cfg, grad=False)])
        return np.mean(r, 0).tolist()

    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["steps"])
    log = {"cfg": cfg, "val": [[0, evaluate()]]}
    print(f"[sd] step 0 val depth / rot / trans {log['val'][-1][1]}", flush=True)
    t0, acc = time.time(), []
    for step in range(1, cfg["steps"] + 1):
        ld, lr_, lt = losses(api, get(random.choice(train)), cfg)
        loss = ld + cfg["w_rot"] * lr_ + cfg["w_trans"] * lt
        opt.zero_grad()
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        acc.append([float(ld), float(lr_), float(lt)])
        if step % 100 == 0:
            print(f"[sd] step {step} depth / rot / trans {np.mean(acc, 0).round(5).tolist()} ({time.time() - t0:.0f}s)", flush=True)
            acc = []
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            log["val"].append([step, evaluate()])
            print(f"[sd] step {step} val depth / rot / trans {log['val'][-1][1]}", flush=True)
            sd = {n: p.detach().cpu() for n, p in api.model.named_parameters() if p.requires_grad}
            torch.save(sd, os.path.join(run, "student.pt"))
            json.dump(log, open(os.path.join(run, "log.json"), "w"), indent=1)


def cmd_export(src, run, out_root, opts):
    split = opts.get("split", "dev")
    os.chdir(REPO)
    api = load_api()
    sd = torch.load(os.path.join(run, "student.pt"))
    missing = api.model.load_state_dict(sd, strict=False)
    assert not missing.unexpected_keys
    from depth_anything_3.utils.constants import EVAL_REF_VIEW_STRATEGY
    from lpba_depth_oracle import abs_rel, try_gt

    for ds in opts.get("datasets", ",".join(DEV)).split(","):
        d = os.path.join(src, "model_results", ds)
        scenes = sorted(os.path.dirname(p) for p, _, _ in os.walk(d) if os.path.basename(p) == "unposed")
        scenes = scenes[: DEV[ds]] if split == "dev" else scenes[DEV[ds]:]
        for sd_ in scenes:
            scene = os.path.relpath(sd_, d)
            ex = os.path.join(sd_, "unposed", "exports")
            meta = dict(np.load(os.path.join(ex, "gt_meta.npz"), allow_pickle=True))
            old = dict(np.load(os.path.join(ex, "mini_npz", "results.npz")))
            p = api.inference([str(f) for f in meta["image_files"]], ref_view_strategy=EVAL_REF_VIEW_STRATEGY)
            new = {"depth": np.asarray(p.depth, np.float32), "conf": np.asarray(p.conf, np.float32),
                   "extrinsics": np.asarray(p.extrinsics, np.float32)[:, :3], "intrinsics": np.asarray(p.intrinsics, np.float32)}
            o = os.path.join(out_root, "model_results", ds, scene, "unposed", "exports")
            os.makedirs(os.path.join(o, "mini_npz"), exist_ok=True)
            np.savez(os.path.join(o, "gt_meta.npz"), **meta)
            np.savez(os.path.join(o, "mini_npz", "results.npz"), **new)
            g = try_gt(meta["image_files"], *new["depth"].shape[1:])
            print(f"[export] {ds}/{scene}: abs.rel. {abs_rel(old['depth'], g):.4f} -> {abs_rel(new['depth'], g):.4f}", flush=True)


def main():
    cmd = sys.argv[1]
    pos = [a for a in sys.argv[2:] if "=" not in a]
    opts = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    {"targets": lambda: cmd_targets(pos[0], pos[1], opts), "train": lambda: cmd_train(pos[0], pos[1], opts),
     "export": lambda: cmd_export(pos[0], pos[1], pos[2], opts)}[cmd]()


if __name__ == "__main__":
    main()
