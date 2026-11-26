"""Fine-tune RoMa v2's refiners for geometry-guided use (scripts/gmr_ba.py): the warp they correct comes from a
feed-forward reconstruction (DA3, or DA3 + PBA), whose errors are larger than those of RoMa's own coarse matcher.

    .venv/bin/python scripts/gmr_train.py <out.pt> [key=value ...]

Data: benchmark scenes with GT depth and cameras (default: the held-out scenes of HiRoom and ScanNet++; the dev scenes
and the other datasets are never seen). Per step: one source frame and a few of its covisible neighbours; the initial
warp is the geometric warp of a saved state (`states`: e.g. DA3's prediction and its B′ refinement, chosen at random,
so the initial errors span 0-60 px); the target is the warp by the GT depth and cameras where the GT says the point is
visible. The schedule `stages` (as in gmr_ba.py, e.g. 4@2,4,4,2,1) is run once, every stage's output gets a robust L1
loss in px, and the final overlap logit a BCE loss towards "visible and now within 2 px". BatchNorm stays in eval mode.
knowledge/ideas/guided-match-refinement.md.
"""

import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gmr_ba as gb  # noqa: E402
import run_tto_pba as rp  # noqa: E402
import sba_fast as sf  # noqa: E402
from lpba_depth_oracle import try_gt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULTS = {
    "states": "workspace/evaluation/DA3-SMALL,workspace/tto/DA3-SMALL/sba_ho/pho25,workspace/tto/DA3-SMALL/sba_ref/pho25",
    "datasets": "hiroom,scannetpp",
    "subset": "held-out",  # held-out | dev | cv_a | cv_b (dev + every second held-out scene, starting at the first / second)
    "max_scenes": 0,
    "scene_stride": 1,  # keep every k-th benchmark training scene (data-size curve)
    "save_every": 1000,  # also keep <out>.<step>.pt every this many steps (learning curve)
    "arkit": "",  # directory of ARKitScenes windows (scripts/lpba_prep_arkit_pose.py): LiDAR depth + ARKit poses as GT;
    # DA3-SMALL is run on every window at load time (state 1) and B′ (iters=25) on the first arkit_pba windows (state 2)
    "arkit_n": 400,
    "arkit_pba": 150,
    "res": "384x512",
    "stages": "4@2,4,4,2,1",
    "train": "4",  # refiners that are trained (comma-separated ids); the others run frozen, without gradients (memory)
    "nbr": 10,
    "pairs": 4,  # neighbours per step
    "steps": 3000,
    "lr": 1e-4,
    "max_init": 64.0,  # px: pixels whose initial error is larger are not supervised
    "w_conf": 0.1,
    "gamma": 0.8,  # weight of an earlier stage's loss relative to the next
    "seed": 0,
    "log": 100,
}


def gt_warp(depth, R, t, K, i, js, h, w, occ=0.03):
    """As gmr_ba.geo_warp, for a GT depth with holes (0 = invalid): nearest sampling, validity of source and target."""
    N, H, W = depth.shape
    g = gb.norm_grid(h, w, depth.device)
    u = (g[..., 0] + 1) / 2 * W - 0.5
    v = (g[..., 1] + 1) / 2 * H - 0.5
    d = depth[i][v.round().long().clamp(0, H - 1), u.round().long().clamp(0, W - 1)]
    ray = torch.stack([(u - K[i, 0, 2]) / K[i, 0, 0], (v - K[i, 1, 2]) / K[i, 1, 1], torch.ones_like(u)], -1) * d[..., None]
    Xw = torch.einsum("ji,hwj->hwi", R[i], ray - t[i])
    Xc = torch.einsum("bij,hwj->bhwi", R[js], Xw) + t[js][:, None, None]
    z = Xc[..., 2]
    zs = z.clamp_min(1e-6)
    u2 = K[js, 0, 0][:, None, None] * Xc[..., 0] / zs + K[js, 0, 2][:, None, None]
    v2 = K[js, 1, 1][:, None, None] * Xc[..., 1] / zs + K[js, 1, 2][:, None, None]
    warp = torch.stack([(u2 + 0.5) / W * 2 - 1, (v2 + 0.5) / H * 2 - 1], -1)
    inside = (u2 >= 0) & (u2 <= W - 1) & (v2 >= 0) & (v2 <= H - 1) & (z > 1e-6)
    dj = depth[js[:, None, None], v2.round().long().clamp(0, H - 1), u2.round().long().clamp(0, W - 1)]
    known = (d > 0)[None].expand_as(inside)  # the source pixel has GT depth: its visibility is known
    vis = known & inside & (dj > 0) & ((z - dj).abs() < occ * dj)
    return warp, vis, known


def load_scenes(cfg, dev):
    Hl, Wl = (int(x) for x in cfg["res"].split("x"))
    states = cfg["states"].split(",")
    scenes = []
    for ds in cfg["datasets"].split(","):
        d = os.path.join(states[0], "model_results", ds)
        names = sorted(os.path.relpath(os.path.dirname(p), d) for p, _, _ in os.walk(d) if os.path.basename(p) == "unposed")
        n_dev = gb.DEV[ds]
        if cfg["subset"] == "dev":
            names = names[:n_dev]
        elif cfg["subset"] in ("cv_a", "cv_b"):  # two folds of the held-out scenes: train on dev + one half, test on the other
            names = names[:n_dev] + names[n_dev:][(0 if cfg["subset"] == "cv_a" else 1)::2]
        else:
            names = names[n_dev:]
        if cfg["max_scenes"]:
            names = names[: cfg["max_scenes"]]
        names = names[:: cfg["scene_stride"]]
        for scene in names:
            ex = os.path.join(d, scene, "unposed", "exports")
            meta = dict(np.load(os.path.join(ex, "gt_meta.npz"), allow_pickle=True))
            files = [str(f) for f in meta["image_files"]]
            st = []
            for s in states:
                p = os.path.join(s, "model_results", ds, scene, "unposed", "exports", "mini_npz", "results.npz")
                if not os.path.exists(p):
                    continue
                pr = np.load(p)
                dep = torch.from_numpy(pr["depth"].astype(np.float32))
                E = torch.from_numpy(pr["extrinsics"][:, :3].astype(np.float32))
                K = torch.from_numpy(pr["intrinsics"].astype(np.float32))
                nb = rp.select_neighbors(dep.to(dev), E[:, :, :3].to(dev), E[:, :, 3].to(dev), K.to(dev), cfg["nbr"], 0.05).cpu()
                st.append((dep, E[:, :, :3], E[:, :, 3], K, nb))
            N, H, W = st[0][0].shape
            gdep = try_gt(files, H, W)
            if gdep is None or not st:
                continue
            h0, w0 = cv2.imread(files[0]).shape[:2]
            gK = meta["intrinsics"].astype(np.float64).copy()
            gK[:, 0] *= W / w0
            gK[:, 1] *= H / h0
            gE = meta["extrinsics"].astype(np.float64)
            imgs = sf.load_images(files, H, W)
            lr = F.interpolate(imgs, size=(Hl, Wl), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
            scenes.append(dict(name=f"{ds}/{scene}", lr=(lr * 255).round().to(torch.uint8), states=st, gdep=torch.from_numpy(gdep).float(),
                               gR=torch.from_numpy(gE[:, :3, :3]).float(), gt=torch.from_numpy(gE[:, :3, 3]).float(), gK=torch.from_numpy(gK).float()))
            print(f"[train] loaded {ds}/{scene}: {N} frames, {len(st)} states", flush=True)
    return scenes


def load_arkit(cfg, dev):
    """ARKit windows as training scenes: predictions of DA3-SMALL made here, GT from LiDAR depth and the trajectory."""
    import glob

    from depth_anything_3.api import DepthAnything3
    from PIL import Image

    Hl, Wl = (int(x) for x in cfg["res"].split("x"))
    api = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL").to(dev).eval()
    pba_cfg = rp.parse_overrides(["iters=25"])
    scenes = []
    dirs = [d for d in sorted(glob.glob(os.path.join(cfg["arkit"], "*_*"))) if os.path.exists(os.path.join(d, "meta.npz"))]
    for wi, d in enumerate(dirs[: cfg["arkit_n"]]):
        try:
            mt = np.load(os.path.join(d, "meta.npz"))
            files = [os.path.join(d, f"{t}.jpg") for t in mt["ts"]]
            pil = [Image.open(f).convert("RGB") for f in files]
            w0, h0 = pil[0].size
            with torch.no_grad():
                p = api.inference(pil)
            pred = dict(depth=np.asarray(p.depth, np.float32), conf=np.asarray(p.conf, np.float32),
                        extrinsics=np.asarray(p.extrinsics, np.float32)[:, :3], intrinsics=np.asarray(p.intrinsics, np.float32))
            N, H, W = pred["depth"].shape
            preds = [pred]
            if wi < cfg["arkit_pba"]:
                o, _, _ = rp.refine_scene(files, pred, pba_cfg, torch.device(dev))
                preds.append(o)
            st = []
            for pr in preds:
                dep = torch.from_numpy(pr["depth"].astype(np.float32))
                E = torch.from_numpy(pr["extrinsics"][:, :3].astype(np.float32))
                K = torch.from_numpy(pr["intrinsics"].astype(np.float32))
                nb = rp.select_neighbors(dep.to(dev), E[:, :, :3].to(dev), E[:, :, 3].to(dev), K.to(dev), cfg["nbr"], 0.05).cpu()
                st.append((dep, E[:, :, :3], E[:, :, 3], K, nb))
            gdep = np.stack([cv2.resize(cv2.imread(os.path.join(d, f"{t}_depth.png"), -1).astype(np.float32) / 1000.0, (W, H), interpolation=cv2.INTER_NEAREST) for t in mt["ts"]])
            gE = np.stack([np.concatenate([cv2.Rodrigues(q[:3].astype(np.float64))[0], q[3:, None]], 1) for q in mt["traj"]])
            gK = np.tile(np.eye(3), (N, 1, 1))
            gK[:, 0, 0], gK[:, 1, 1] = mt["K"][:, 0] * W / w0, mt["K"][:, 1] * H / h0
            gK[:, 0, 2], gK[:, 1, 2] = mt["K"][:, 2] * W / w0, mt["K"][:, 3] * H / h0
            imgs = sf.load_images(files, H, W)
            lr = F.interpolate(imgs, size=(Hl, Wl), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
            scenes.append(dict(name=f"arkit/{os.path.basename(d)}", lr=(lr * 255).round().to(torch.uint8), states=st, gdep=torch.from_numpy(gdep).float(),
                               gR=torch.from_numpy(gE[:, :3, :3]).float(), gt=torch.from_numpy(gE[:, :3, 3]).float(), gK=torch.from_numpy(gK).float()))
        except Exception as e:  # noqa: BLE001
            print(f"[train] arkit {os.path.basename(d)} skipped: {e!r}", flush=True)
            continue
        if (wi + 1) % 25 == 0:
            print(f"[train] arkit: {len(scenes)} windows loaded", flush=True)
    del api
    torch.cuda.empty_cache()
    return scenes


def main():
    out = sys.argv[1]
    cfg = dict(DEFAULTS)
    for a in sys.argv[2:]:
        k, _, v = a.partition("=")
        cfg[k] = type(DEFAULTS[k])(v)
    os.chdir(REPO)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    dev = "cuda"
    Hl, Wl = (int(x) for x in cfg["res"].split("x"))
    scenes = load_scenes(cfg, dev) if cfg["datasets"] else []
    if cfg["arkit"]:
        scenes += load_arkit(cfg, dev)
    print(f"[train] {len(scenes)} scenes", flush=True)
    m = gb.refiner_model()
    m.eval()  # BatchNorm statistics stay as trained
    train_ids = {x for x in cfg["train"].split(",") if x}
    for p in m.refiners.parameters():
        p.requires_grad_(False)
    params = [p for k_, r in m.refiners.items() if k_ in train_ids for p in r.parameters()]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["steps"], pct_start=0.05)
    stages = [(int(x.split("@")[0]), int(x.split("@")[1]) if "@" in x else 1) for x in cfg["stages"].split(",") if x]
    ks = sorted({k for _, k in stages})
    px = torch.tensor([Wl / 2, Hl / 2], device=dev)  # normalised -> px at the refiner resolution
    rng = np.random.default_rng(cfg["seed"])
    logs = []
    t0 = time.time()
    for step in range(cfg["steps"]):
        sc = scenes[rng.integers(len(scenes))]
        dep, R, t, K, nb = sc["states"][rng.integers(len(sc["states"]))]
        N = dep.shape[0]
        i = int(rng.integers(N))
        js = nb[i]
        js = js[js != i]
        js = js[torch.from_numpy(rng.permutation(len(js))[: cfg["pairs"]])]
        fr = torch.cat([torch.tensor([i]), js])
        with torch.no_grad():
            lr_img = sc["lr"][fr].to(dev).float() / 255
            feats = {}
            for k in ks:
                lk = lr_img if k == 1 else F.interpolate(lr_img, size=(Hl // k, Wl // k), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
                feats[k] = m.refiner_features(lk)
            # local copies of the frames in this step: index 0 = source, 1.. = neighbours
            d_l, R_l, t_l, K_l = dep[fr].to(dev), R[fr].to(dev), t[fr].to(dev), K[fr].to(dev)
            g_l, gR, gt_, gK = sc["gdep"][fr].to(dev), sc["gR"][fr].to(dev), sc["gt"][fr].to(dev), sc["gK"][fr].to(dev)
            jl = torch.arange(1, len(fr), device=dev)
            B = len(jl)
            s0, k0 = stages[0]
            warp, vis = gb.geo_warp(d_l, R_l, t_l, K_l, 0, jl, Hl // (s0 * k0), Wl // (s0 * k0), 0.1)
            confd = torch.where(vis, 2.0, -2.0)[..., None].float()
        loss, n_st = 0.0, len(stages)
        rec = {}
        for si, (s, k) in enumerate(stages):
            hs, ws = Hl // (s * k), Wl // (s * k)
            if warp.shape[1] != hs:
                warp = F.interpolate(warp.permute(0, 3, 1, 2), size=(hs, ws), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
                confd = F.interpolate(confd.permute(0, 3, 1, 2), size=(hs, ws), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
            with torch.no_grad():
                wg, vg, known = gt_warp(g_l, gR, gt_, gK, 0, jl, hs, ws)
                e_in = ((warp.detach() - wg) * px).norm(dim=-1)
                if si == 0:
                    e_first = e_in
                    sup0 = None
                sup = vg & (e_in < cfg["max_init"])
            scale = torch.tensor((Wl / k / m.anchor_width, Hl / k / m.anchor_height), device=dev)
            fs = feats[k][s]
            with torch.set_grad_enabled(str(s) in train_ids):
                o = m.refiners[str(s)](f_A=fs[0:1].expand(B, -1, -1, -1), f_B=fs[1:], prev_warp=warp.detach().float(),
                                       prev_confidence=confd[..., :1].float() if k > 1 else confd.float(), scale_factor=scale)
            warp = o["warp"].float()
            last = not (si + 1 < n_st and stages[si + 1] == (s, k))
            if k == 1 and last:
                confd = o["confidence"].float()
            e_out = ((warp - wg) * px).norm(dim=-1)
            if sup.any():
                if str(s) in train_ids:
                    l = torch.sqrt(e_out[sup] ** 2 + 0.01).clamp(max=32.0).mean()  # robust L1 in px
                    loss = loss + cfg["gamma"] ** (n_st - 1 - si) * l
                    if k == 1 and known.any():  # this stage's overlap logit: visible and now within 3 px
                        tgt = (vg & (e_out.detach() < 3.0)).float()
                        lc = F.binary_cross_entropy_with_logits(o["confidence"][..., 0].float()[known], tgt[known])
                        loss = loss + cfg["w_conf"] * lc
                        rec[f"conf{si}"] = float(lc)
                rec[f"in{si}"] = float(e_in[sup].median())
                rec[f"out{si}"] = float(e_out[sup].detach().median())
        # overlap: visible by the GT and matched within 2 px, on pixels whose GT visibility is known
        with torch.no_grad():
            target = (vg & (e_out.detach() < 2.0)).float()
        if known.any() and confd.requires_grad:
            lc = F.binary_cross_entropy_with_logits(confd[..., 0][known], target[known])
            loss = loss + cfg["w_conf"] * lc
            rec["conf"] = float(lc)
        if not torch.is_tensor(loss):
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        rec["loss"] = float(loss)
        logs.append(rec)
        if (step + 1) % cfg["log"] == 0:
            keys = sorted({k_ for r in logs for k_ in r})
            print(f"[train] step {step + 1} {time.time() - t0:.0f}s " + " ".join(f"{k_} {np.mean([r[k_] for r in logs if k_ in r]):.3f}" for k_ in keys), flush=True)
            logs = []
        if (step + 1) % cfg["save_every"] == 0 or step + 1 == cfg["steps"]:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            ck = {"refiners": m.refiners.state_dict(), "cfg": cfg, "step": step + 1}
            torch.save(ck, out)
            torch.save(ck, f"{out[:-3] if out.endswith('.pt') else out}.{step + 1}.pt")


if __name__ == "__main__":
    main()
