"""Resolution distillation of DéjàView: the student runs at a lower input resolution (RoPE positions stretched to the
native grid, rope_scale.py) and K_s loop steps, the teacher is the frozen released model at 504 px and K_t = 16 on the
same frames. No ground truth (unlabeled ARKitScenes clips from prep_arkit.py, outside the DA3 benchmark).

Loss (all against the teacher):
  rays   L1 at the student resolution (teacher ray map area-downsampled)
  depth  L1 on log depth at 504 px (student depth bilinearly upsampled, as eval_student.py does at test time)
  conf   0.1 x L1 on log confidence at 504 px
  pose   --w_pose x L1 on the camera head's pose_enc (translation, quaternion, FoV), so that camera-head poses
         (DVLT_POSE=head) also survive the lower resolution

  .venv/bin/python scripts/distill_steps/train_res.py --name R1 --res 336,420 --ks 12,16 --hours 5
Outputs go to workspace/distill_runs/<name>/. If another process appears on the GPU (shared machine), the run
saves a resume checkpoint and exits with code 3; the wrapper waits for the GPU and restarts with --resume.
"""
import argparse
import copy
import glob
import json
import math
import os
import random
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "third_party", "dvlt", "src"))
import rope_scale  # noqa: E402
from train import load_dvlt  # noqa: E402

DATA = os.path.join(REPO, "workspace", "distill_data", "arkit_frames")
RUNS = os.path.join(REPO, "workspace", "distill_runs")


class Folders(torch.utils.data.Dataset):
    """Item i: a random video folder and a random sorted subset of frames, preprocessed at 504 px."""

    def __init__(self, dirs, length, min_views, max_views, seed, return_files=False, extra=(), p_extra=0.0):
        self.dirs, self.length, self.min_views, self.max_views, self.seed = dirs, length, min_views, max_views, seed
        self.return_files, self.extra, self.p_extra = return_files, list(extra), p_extra

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        from PIL import Image

        from dvlt.util.preprocess import preprocess_images

        rng = random.Random(self.seed * 1_000_003 + i)
        pool = self.extra if (self.extra and rng.random() < self.p_extra) else self.dirs
        d = pool[rng.randrange(len(pool))]
        files = sorted(glob.glob(os.path.join(d, "*.jpg")))
        n = min(rng.randint(self.min_views, self.max_views), len(files))
        # contiguous window of 2-4x n frames, so the views overlap like a benchmark clip
        span = min(len(files), n * rng.choice([2, 3, 4]))
        s0 = rng.randrange(len(files) - span + 1)
        idx = sorted(rng.sample(range(s0, s0 + span), n))
        imgs = [Image.open(files[j]).convert("RGB") for j in idx]
        out = preprocess_images(imgs, 504, 14, "cpu")["images"][0], os.path.basename(d)
        return (*out, [files[j] for j in idx]) if self.return_files else out


class TargetClips(torch.utils.data.Dataset):
    """Clips of gen_tto_targets.py: the same frames, preprocessed at 504 px, plus the TTO-refined cameras / depth."""

    def __init__(self, paths, length, seed):
        self.paths, self.length, self.seed = paths, length, seed

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        import numpy as np
        from PIL import Image

        from dvlt.util.preprocess import preprocess_images

        p = self.paths[random.Random(self.seed * 1_000_003 + i).randrange(len(self.paths))]
        d = np.load(p)
        imgs = [Image.open(os.path.join(DATA, str(f))).convert("RGB") for f in d["files"]]
        tgt = {k: torch.from_numpy(d[k]).float() for k in ("ext_teacher", "K_teacher", "ext_ref", "K_ref", "logdepth_ref")}
        return preprocess_images(imgs, 504, 14, "cpu")["images"][0], os.path.basename(p), tgt


def apply_targets(pt, tgt, dev):
    """Teacher predictions moved to the TTO result: rays += rays(refined cameras) - rays(teacher cameras) (so the
    teacher's non-pinhole ray detail is kept), depth *= exp(refined log-depth correction), pose_enc = refined cameras."""
    from dvlt.common.rays import compute_world_rays
    from dvlt.model_components.pose_encoding import extri_intri_to_pose_enc

    H, W = pt["depth"].shape[2:4]
    c2w = lambda e: torch.linalg.inv(torch.cat([e, torch.tensor([[[0, 0, 0, 1.0]]], device=dev).expand(len(e), 1, 4)], 1))
    ct, cr = c2w(tgt["ext_teacher"].to(dev)), c2w(tgt["ext_ref"].to(dev))
    rt = compute_world_rays(ct[None], tgt["K_teacher"].to(dev)[None], H, W)
    rr = compute_world_rays(cr[None], tgt["K_ref"].to(dev)[None], H, W)
    out = dict(pt)
    out["rays"] = pt["rays"].float() + (rr - rt)
    lr = F.interpolate(tgt["logdepth_ref"].to(dev)[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
    out["depth"] = pt["depth"].float() * lr.exp()[None, ..., None]
    pe = extri_intri_to_pose_enc(cr[:, :3], tgt["K_ref"].to(dev), (H, W))[None]
    sign = torch.sign((pe[..., 3:7] * pt["pose_enc"][..., 3:7].float()).sum(-1, keepdim=True)).clamp_min(0) * 2 - 1
    out["pose_enc"] = torch.cat([pe[..., :3], pe[..., 3:7] * sign, pe[..., 7:]], -1)
    return out


def student_size(H, W, res):
    s = res / max(H, W)
    return round(H * s / 14) * 14, round(W * s / 14) * 14


def run(model, images, K):
    from dvlt.model.dvlt.model import _slice_expand_flatten

    B, S, _, H, W = images.shape
    z0 = model._encode_images(images)
    reg = model.register_token.expand(B, S, -1, -1).reshape(B * S, model.num_register_tokens, -1)
    cam = _slice_expand_flatten(model.camera_token, B, S)
    x = torch.cat([cam, reg, z0], dim=1)
    pos = model._get_rope_positions(B * S, H, W, images.device)
    ts = torch.linspace(0.0, 1.0, K).tolist()
    for i in range(K):
        x = model._interval_step(x, ts[i], ts[i + 1] if i + 1 < K else 1.0, pos, B, S)
    return model._decode(x, H, W, B, S, pos)


def run_student(model, images, K, res):
    """images [1, S, 3, H, W] at 504 px -> predictions at the student resolution."""
    from torchvision.transforms.functional import resize

    H, W = images.shape[-2:]
    h, w = student_size(H, W, res)
    rope_scale.SCALE[:] = [(H // 14 - 1) / max(h // 14 - 1, 1), (W // 14 - 1) / max(w // 14 - 1, 1)]
    small = resize(images[0], [h, w], antialias=True)[None]
    try:
        return run(model, small, K)
    finally:
        rope_scale.SCALE[:] = [1.0, 1.0]


def losses(ps, pt):
    f = lambda t: t.float()
    _, S, h, w, _ = ps["rays"].shape
    H, W = pt["depth"].shape[2:4]
    rt = F.interpolate(f(pt["rays"])[0].permute(0, 3, 1, 2), size=(h, w), mode="area").permute(0, 2, 3, 1)
    l_ray = (f(ps["rays"])[0] - rt).abs().mean()
    up = lambda t: F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    ds = up(f(ps["depth"])[0].permute(0, 3, 1, 2))[:, 0]
    l_depth = (ds.clamp_min(1e-6).log() - f(pt["depth"])[0, ..., 0].clamp_min(1e-6).log()).abs().mean()
    cs = up(f(ps["depth_conf"])[0][:, None])[:, 0]
    l_conf = (cs.log() - f(pt["depth_conf"])[0].log()).abs().mean()
    l_pose = (f(ps["pose_enc"]) - f(pt["pose_enc"])).abs().mean()
    return dict(ray=l_ray, depth=l_depth, conf=l_conf, pose=l_pose)


def foreign_gpu_procs():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=30).stdout.split()
    except Exception:  # noqa: BLE001
        return []
    return [p for p in out if p.strip() and int(p) != os.getpid()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--res", default="336,420", help="student longest sides, sampled uniformly per step")
    ap.add_argument("--ks", default="12,16", help="student loop steps, sampled uniformly per step")
    ap.add_argument("--kt", type=int, default=16)
    ap.add_argument("--w_pose", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--enc_lr_mult", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--min_views", type=int, default=12)
    ap.add_argument("--max_views", type=int, default=32)
    ap.add_argument("--hours", type=float, default=5.0)
    ap.add_argument("--max_steps", type=int, default=100000)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default="", help="start the student from this checkpoint (teacher stays released)")
    ap.add_argument("--p_outdoor", type=float, default=0.0, help="share of clips from arkit_frames/outdoor (M360, T&T/DB)")
    ap.add_argument("--targets", default="", help="dir of gen_tto_targets.py clips: train towards teacher + TTO")
    ap.add_argument("--smoke", type=int, default=0, help="run N steps, print memory / speed, save nothing")
    args = ap.parse_args()
    res_choices = [int(r) for r in args.res.split(",")]
    k_choices = [int(k) for k in args.ks.split(",")]

    out = os.path.join(RUNS, args.name)
    os.makedirs(out, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    rope_scale.install()

    train_dirs = sorted(d for d in glob.glob(os.path.join(DATA, "train", "*")) if os.path.exists(f"{d}/DONE"))
    val_dirs = sorted(d for d in glob.glob(os.path.join(DATA, "val", "*")) if os.path.exists(f"{d}/DONE"))
    print(f"[data] {len(train_dirs)} train videos, {len(val_dirs)} val", flush=True)

    student = load_dvlt().to(dev)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    if args.init and not (args.resume and os.path.exists(os.path.join(out, "resume.pt"))):
        student.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["model"])
        print(f"[init] student from {args.init}", flush=True)
    student.train()
    student.enable_gradient_checkpointing()
    student.patch_embed_encoder.use_checkpoint = True
    student.patch_embed_encoder.use_reentrant = False
    for dec in (student.depth_decoder, student.ray_decoder):  # full-resolution conv activations per decode chunk
        fwd = dec.forward
        dec.forward = (lambda f: lambda *a, **k: torch.utils.checkpoint.checkpoint(f, *a, use_reentrant=False, **k))(fwd)
    enc =[p for n, p in student.named_parameters() if n.startswith("patch_embed_encoder.") and p.requires_grad]
    rest = [p for n, p in student.named_parameters() if not n.startswith("patch_embed_encoder.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr * args.enc_lr_mult}, {"params": rest, "lr": args.lr}],
                            weight_decay=0.0, betas=(0.9, 0.99))
    base_lrs = [g["lr"] for g in opt.param_groups]

    step, total, t_used = 0, args.max_steps, 0.0
    rpath = os.path.join(out, "resume.pt")
    if args.resume and os.path.exists(rpath):
        ck = torch.load(rpath, map_location="cpu", weights_only=False)
        student.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, total, t_used = ck["step"], ck["total"], ck["t_used"]
        print(f"[resume] step {step}/{total}, {t_used / 3600:.2f} h used", flush=True)
    elif not args.smoke:
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump({**vars(args), "train_videos": [os.path.basename(d) for d in train_dirs],
                       "val_videos": [os.path.basename(d) for d in val_dirs],
                       "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=1)

    if args.targets:
        tpaths = sorted(glob.glob(os.path.join(args.targets, "*.npz")))
        print(f"[data] {len(tpaths)} TTO target clips", flush=True)
        ds = TargetClips(tpaths, args.max_steps, args.seed)
    else:
        out_dirs = sorted(d for d in glob.glob(os.path.join(DATA, "outdoor", "*")) if os.path.exists(f"{d}/DONE"))
        if args.p_outdoor > 0:
            print(f"[data] {len(out_dirs)} outdoor scenes at p = {args.p_outdoor}", flush=True)
        ds = Folders(train_dirs, args.max_steps, args.min_views, args.max_views, args.seed, extra=out_dirs,
                     p_extra=args.p_outdoor)
    order = list(range(step, args.max_steps))
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, order), batch_size=None, shuffle=False,
                                     num_workers=args.workers, prefetch_factor=4 if args.workers else None)
    val = [Folders([d], 1, 24, 24, 1000)[0][0] for d in val_dirs]
    amp = dict(device_type="cuda", dtype=torch.bfloat16)
    log = open(os.path.join(out, "log.jsonl"), "a")
    t0 = time.time() - t_used
    rng = random.Random(args.seed + step)

    def validate():
        student.eval()
        rec = dict(step=step)
        with torch.no_grad(), torch.autocast(**amp):
            tp = [run(teacher, v[None].to(dev), args.kt) for v in val]
            for res, K in [(336, 16), (420, 12), (504, 12)]:
                acc = {}
                for v, pt in zip(val, tp):
                    ps = run_student(student, v[None].to(dev), K, res)
                    for k, x in losses(ps, pt).items():
                        acc[k] = acc.get(k, 0.0) + float(x) / len(val)
                rec.update({f"val{res}_K{K}_{k}": round(x, 5) for k, x in acc.items()})
        student.train()
        rec["t"] = time.time() - t0
        print(f"[val] {json.dumps(rec)}", flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    def save(path, full):
        d = {"step": step, "args": vars(args), "model": student.state_dict()}
        if full:
            d.update(opt=opt.state_dict(), total=total, t_used=time.time() - t0)
        torch.save(d, path + ".tmp")
        os.replace(path + ".tmp", path)

    if step == 0 and not args.smoke:
        validate()
    t_fit = None
    for item in dl:
        images, name, tgt = item if args.targets else (*item, None)
        if args.smoke and step >= args.smoke:
            break
        if step % 50 == 0 and foreign_gpu_procs() and not args.smoke:
            save(rpath, full=True)
            print(f"[pause] foreign GPU process {foreign_gpu_procs()} at step {step}; saved resume.pt", flush=True)
            sys.exit(3)
        images = images[None].to(dev, non_blocking=True)
        if total == args.max_steps and step >= 30 and t_fit is not None and not args.smoke:
            per = (time.time() - t_fit[1]) / (step - t_fit[0])
            total = min(args.max_steps, step + int((args.hours * 3600 - (time.time() - t0)) / per))
            print(f"[sched] {per:.2f} s/step -> {total} steps", flush=True)
        if t_fit is None and step >= 10 - (0 if not args.resume else 10):
            t_fit = (step, time.time())
        frac = min(1.0, (step + 1) / args.warmup)
        cos = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(total, 1))))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * frac * cos

        res, K = rng.choice(res_choices), rng.choice(k_choices)
        with torch.no_grad(), torch.autocast(**amp):
            pt = run(teacher, images, args.kt)
        with torch.autocast(**amp):
            ps = run_student(student, images, K, res)
        if tgt is not None:
            pt = apply_targets(pt, tgt, dev)
        L = losses(ps, pt)
        loss = L["ray"] + L["depth"] + 0.1 * L["conf"] + args.w_pose * L["pose"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        step += 1

        if step % 25 == 0 or step <= 5:
            rec = dict(step=step, loss=loss.item(), **{k: v.item() for k, v in L.items()}, gn=gn.item(),
                       lr=opt.param_groups[1]["lr"], views=images.shape[1], res=res, K=K, scene=name,
                       t=time.time() - t0, mem=torch.cuda.max_memory_allocated() / 2**30)
            print(json.dumps(rec), flush=True)
            if not args.smoke:
                log.write(json.dumps(rec) + "\n")
                log.flush()
        if args.smoke:
            continue
        if step % args.val_every == 0:
            validate()
        if step % 500 == 0:
            save(rpath, full=True)
        if step >= total:
            break
    if args.smoke:
        print(f"[smoke] {step} steps, {(time.time() - t0) / max(step, 1):.2f} s/step, peak "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        return
    validate()
    save(os.path.join(out, "final.pt"), full=False)
    if os.path.exists(rpath):
        os.remove(rpath)
    print(f"[done] {step} steps in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
