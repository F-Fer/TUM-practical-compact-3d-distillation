"""Depth-free pose refinement: dense patch matches + epipolar (Sampson) BA over poses and the shared focal.

    .venv/bin/python scripts/lpba_epi.py <state work dir> <dst work dir> [key=value ...]

Direction "photometric / feature-metric BA without depth" (knowledge/ideas/shared-depth-field.md): the depth is used only
to predict where to look, never as an unknown, so a frame can't absorb its pose error into its depth.

Per round (schedule `rounds` = sigma:radius:step,...):
  1. matching, poses fixed: `samples` gradient-weighted pixels per frame; a patch x patch window around each is warped
     into each of the `nbrs` spread neighbours with the state's depth and the current cameras (so the patch follows the
     surface); a 2D search of +-radius px (in steps of `step`) around the predicted location maximises NCC on images
     blurred by sigma; parabolic sub-pixel; kept if NCC > `min_ncc`, the best offset isn't at the window's border, and
     the best NCC outside a 2-step neighbourhood is at least `uniq` lower;
  2. poses (frame 0 fixed), focal: Adam (`steps`, cosine decay) on the Cauchy or Huber (`loss`, scale `huber` px) Sampson distance of all
     matches, plus `w_scale` * (log mean camera distance to frame 0 - its initial value)^2 (the epipolar cost has no
     scale).
Prints the pose AUC after every round (the iteration curve). Output in the evaluator's layout (the state's depth,
rescaled with the scene; the new cameras), metric_results/<dataset>_pba.json with auc*_r<k> per round, the number of
matches and the median Sampson distance before / after each round.
"""

import glob
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_tto_pba as rp  # noqa: E402
from depth_anything_3.bench.evaluator import Evaluator  # noqa: E402
from depth_anything_3.bench.utils import compute_pose  # noqa: E402
from depth_anything_3.utils.geometry import as_homogeneous  # noqa: E402

DEFAULTS = {
    "datasets": "",
    "subset": "dev",
    "rounds": "2:8:1,1:4:1,1:4:1,1:4:1",
    "samples": 1024,
    "patch": 7,
    "nbrs": 10,
    "min_ncc": 0.8,
    "uniq": 0.02,
    "steps": 300,
    "lr_pose": 1e-3,
    "lr_focal": 1e-3,
    "focal": True,
    "loss": "cauchy",  # huber | cauchy (scale `huber` px)
    "huber": 0.5,
    "w_scale": 1.0,
    "seed": 0,
}


def skew(t):
    z = torch.zeros_like(t[..., 0])
    return torch.stack([torch.stack([z, -t[..., 2], t[..., 1]], -1), torch.stack([t[..., 2], z, -t[..., 0]], -1),
                        torch.stack([-t[..., 1], t[..., 0], z], -1)], -2)


def sample(img, u, v):
    """img [H, W], u, v [...] pixel coordinates -> [...] (bilinear, zeros outside)."""
    H, W = img.shape
    g = torch.stack([u / (W - 1) * 2 - 1, v / (H - 1) * 2 - 1], -1).reshape(1, 1, -1, 2)
    return F.grid_sample(img[None, None], g, align_corners=True).view(u.shape)


@torch.no_grad()
def match(grey, depth, R, t, K, nbrs, pix, cfg, radius, step):
    """pix [N, M, 2] source pixels -> list of (i, j, x_i [m, 2], x_j [m, 2], ncc [m])."""
    N, H, W = grey.shape
    dev = grey.device
    r = cfg["patch"] // 2
    oy, ox = torch.meshgrid(torch.arange(-r, r + 1, device=dev).float(), torch.arange(-r, r + 1, device=dev).float(), indexing="ij")
    off = torch.stack([ox.reshape(-1), oy.reshape(-1)], -1) * max(1.0, step)  # [P, 2] patch offsets
    n = int(round(radius / step))
    sy, sx = torch.meshgrid(torch.arange(-n, n + 1, device=dev).float(), torch.arange(-n, n + 1, device=dev).float(), indexing="ij")
    srch = torch.stack([sx.reshape(-1), sy.reshape(-1)], -1) * step  # [S, 2]
    side = 2 * n + 1
    out = []
    for i in range(N):
        p = pix[i]  # [M, 2]
        pp = p[:, None] + off[None]  # [M, P, 2]
        a = sample(grey[i], pp[..., 0], pp[..., 1])  # [M, P]
        a = a - a.mean(1, keepdim=True)
        a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-6)
        d = sample(depth[i], pp[..., 0], pp[..., 1])
        ray = torch.stack([(pp[..., 0] - K[i, 0, 2]) / K[i, 0, 0], (pp[..., 1] - K[i, 1, 2]) / K[i, 1, 1], torch.ones_like(d)], -1)
        Xw = (ray * d[..., None] - t[i]) @ R[i]  # [M, P, 3]
        for j in nbrs[i].tolist():
            Xj = Xw @ R[j].T + t[j]
            z = Xj[..., 2]
            q = torch.stack([K[j, 0, 0] * Xj[..., 0] / z.clamp_min(1e-6) + K[j, 0, 2], K[j, 1, 1] * Xj[..., 1] / z.clamp_min(1e-6) + K[j, 1, 2]], -1)
            qs = q[:, None] + srch[None, :, None]  # [M, S, P, 2]
            b = sample(grey[j], qs[..., 0], qs[..., 1])
            b = b - b.mean(2, keepdim=True)
            b = b / b.norm(dim=2, keepdim=True).clamp_min(1e-6)
            ncc = (b * a[:, None]).sum(2)  # [M, S]
            inside = (z.min(1).values > 1e-6) & (qs[..., 0].amin((1, 2)) >= 0) & (qs[..., 0].amax((1, 2)) <= W - 1) \
                & (qs[..., 1].amin((1, 2)) >= 0) & (qs[..., 1].amax((1, 2)) <= H - 1)
            best, bi = ncc.max(1)
            by, bx = bi // side, bi % side
            g = ncc.view(-1, side, side)
            far = ((torch.arange(side, device=dev)[None, :, None] - by[:, None, None]).abs() > 2) | \
                  ((torch.arange(side, device=dev)[None, None, :] - bx[:, None, None]).abs() > 2)
            second = torch.where(far, g, torch.full_like(g, -2)).flatten(1).max(1).values
            ok = inside & (best > cfg["min_ncc"]) & (by > 0) & (by < side - 1) & (bx > 0) & (bx < side - 1) & (best - second > cfg["uniq"])
            if ok.sum() < 8:
                continue
            m = ok.nonzero()[:, 0]
            gy, gx, bym, bxm = g[m], g[m], by[m], bx[m]
            ar = torch.arange(len(m), device=dev)
            c = gx[ar, bym, bxm]
            l, rr = gx[ar, bym, bxm - 1], gx[ar, bym, bxm + 1]
            u_, dn = gy[ar, bym - 1, bxm], gy[ar, bym + 1, bxm]
            dx = torch.where((l - 2 * c + rr) < -1e-6, 0.5 * (l - rr) / (l - 2 * c + rr), torch.zeros_like(c)).clamp(-0.5, 0.5)
            dy = torch.where((u_ - 2 * c + dn) < -1e-6, 0.5 * (u_ - dn) / (u_ - 2 * c + dn), torch.zeros_like(c)).clamp(-0.5, 0.5)
            ctr = q[m, (cfg["patch"] ** 2) // 2]  # the predicted location of the centre pixel
            xj = ctr + srch[bi[m]] + torch.stack([dx, dy], -1) * step
            out.append((i, j, p[m], xj, best[m]))
    return out


def sampson(model, mt):
    """Sampson distance (px) of every match; mt = (I, J, xi [S, 2], xj [S, 2])."""
    I, J, xi, xj = mt
    R, t = model.poses()
    K = model.intrinsics()
    Rij = R[J] @ R[I].transpose(1, 2)
    tij = t[J] - (Rij @ t[I][..., None])[..., 0]
    Fm = torch.linalg.inv(K[J]).transpose(1, 2) @ skew(tij) @ Rij @ torch.linalg.inv(K[I])
    hi = torch.cat([xi, torch.ones_like(xi[:, :1])], -1)
    hj = torch.cat([xj, torch.ones_like(xj[:, :1])], -1)
    Fx = (Fm @ hi[..., None])[..., 0]
    Ftx = (Fm.transpose(1, 2) @ hj[..., None])[..., 0]
    num = (hj * Fx).sum(-1)
    den = Fx[:, 0] ** 2 + Fx[:, 1] ** 2 + Ftx[:, 0] ** 2 + Ftx[:, 1] ** 2
    return num.abs() / den.clamp_min(1e-12).sqrt()


def centre_spread(model):
    R, t = model.poses()
    c = -(R.transpose(1, 2) @ t[..., None])[..., 0]
    return torch.log((c - c[:1]).norm(dim=-1)[1:].mean().clamp_min(1e-9))


def refine(image_files, st, cfg, device, gt_ext):
    torch.manual_seed(cfg["seed"])
    tt = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)  # noqa: E731
    depth, conf, ext, K = tt(st["depth"]), tt(st["conf"]), tt(st["extrinsics"])[:, :3, :4], tt(st["intrinsics"])
    N, H, W = depth.shape
    images = rp.load_images(image_files, H, W).to(device)
    base = rp.parse_overrides(["iters=25", "depth_grid=0", "radial=false"])
    model = rp.PBA(images, depth, conf, ext, K, base).to(device)
    grey_full = images.mean(1, keepdim=True)
    with torch.no_grad():
        R, t = model.poses()
        nbrs = rp.select_neighbors(model.depth0, R, t, model.intrinsics(), cfg["nbrs"], 0.1, "spread", 0.2)
    probs = rp.sampling_probs(grey_full, model.valid)
    info, s0 = {}, centre_spread(model).detach()

    def auc():
        with torch.no_grad():
            R, t = model.poses()
            e = torch.cat([R, t[..., None] * model.scale], -1).cpu().numpy()
        return compute_pose(torch.from_numpy(as_homogeneous(e)), torch.from_numpy(as_homogeneous(gt_ext)))

    a0 = auc()
    params = [{"params": [model.rot, model.trans], "lr": cfg["lr_pose"]}] + ([{"params": [model.logf], "lr": cfg["lr_focal"]}] if cfg["focal"] else [])
    for k, spec in enumerate(cfg["rounds"].split(",")):
        sigma, radius, step = (float(x) for x in spec.split(":"))
        grey = rp.gaussian_blur(grey_full, sigma)[:, 0]
        idx = torch.multinomial(probs, cfg["samples"], replacement=False)
        pix = torch.stack([(idx % W).float(), (idx // W).float()], -1)
        with torch.no_grad():
            R, t = model.poses()
            ms = match(grey, model.depth0, R, t, model.intrinsics(), nbrs, pix, cfg, radius, step)
        if not ms:
            break
        I = torch.cat([torch.full((len(m[2]),), m[0], device=device) for m in ms])
        J = torch.cat([torch.full((len(m[2]),), m[1], device=device) for m in ms])
        mt = (I, J, torch.cat([m[2] for m in ms]), torch.cat([m[3] for m in ms]))
        with torch.no_grad():
            info[f"sampson_before_r{k}"] = float(sampson(model, mt).median())
        opt = torch.optim.Adam(params)
        lrs = [g["lr"] for g in opt.param_groups]
        for it in range(cfg["steps"]):
            f = 0.05 + 0.475 * (1 + math.cos(math.pi * it / cfg["steps"]))
            for g, lr in zip(opt.param_groups, lrs):
                g["lr"] = lr * f
            d = sampson(model, mt)
            c = cfg["huber"]
            loss = (rp.huber(d, c) if cfg["loss"] == "huber" else c * c * torch.log1p((d / c) ** 2)).mean() 
            loss = loss + cfg["w_scale"] * (centre_spread(model) - s0) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            info[f"sampson_after_r{k}"] = float(sampson(model, mt).median())
        info[f"matches_r{k}"] = len(I)
        a = auc()
        info[f"auc03_r{k}"], info[f"auc30_r{k}"] = a["auc03"], a["auc30"]
    with torch.no_grad():
        R, t = model.poses()
        # keep the scene's size: the depth is the state's, so rescale it with the camera spread (fixed by w_scale anyway)
        out = {"depth": (model.depth0 * model.scale).cpu().numpy().astype(np.float32),
               "extrinsics": torch.cat([R, t[..., None] * model.scale], -1).cpu().numpy().astype(np.float32),
               "intrinsics": model.intrinsics().cpu().numpy().astype(np.float32)}
    return out, info, a0


def main():
    src, dst = sys.argv[1], sys.argv[2]
    cfg = dict(DEFAULTS)
    for o in sys.argv[3:]:
        k, _, v = o.partition("=")
        if k not in cfg:
            sys.exit(f"unknown override {k}")
        cfg[k] = (v.lower() in ("1", "true")) if isinstance(DEFAULTS[k], bool) else type(DEFAULTS[k])(v)
    os.chdir(rp.REPO_ROOT)
    device = torch.device("cuda")
    os.makedirs(dst, exist_ok=True)
    json.dump(cfg, open(os.path.join(dst, "epi_config.json"), "w"), indent=2)
    datasets = [d for d in cfg["datasets"].split(",") if d] or list(rp.DEV)
    for ds in datasets:
        root = os.path.join(src, "model_results", ds)
        scenes = sorted(os.path.relpath(os.path.dirname(os.path.dirname(p)), root)
                        for p in glob.glob(os.path.join(root, "**", "unposed", "exports"), recursive=True))
        k = rp.DEV.get(ds, 0)
        scenes = scenes[:k] if cfg["subset"] == "dev" else scenes
        mpath = os.path.join(dst, "metric_results", f"{ds}_pba.json")
        os.makedirs(os.path.dirname(mpath), exist_ok=True)
        metrics = json.load(open(mpath)) if os.path.exists(mpath) else {}
        metrics.pop("mean", None)
        for name in scenes:
            if name in metrics:
                continue
            ex = os.path.join(root, name, "unposed", "exports")
            gtm = dict(np.load(os.path.join(ex, "gt_meta.npz"), allow_pickle=True))
            st = dict(np.load(os.path.join(ex, "mini_npz", "results.npz")))
            t0 = time.perf_counter()
            out, info, a0 = refine(gtm["image_files"], st, cfg, device, gtm["extrinsics"])
            info["epi_s"] = time.perf_counter() - t0
            oex = os.path.join(dst, "model_results", ds, name, "unposed", "exports")
            os.makedirs(os.path.join(oex, "mini_npz"), exist_ok=True)
            np.savez(os.path.join(oex, "mini_npz", "results.npz"), conf=st["conf"], **out)
            shutil.copy(os.path.join(ex, "gt_meta.npz"), os.path.join(oex, "gt_meta.npz"))
            p1 = compute_pose(torch.from_numpy(as_homogeneous(out["extrinsics"])), torch.from_numpy(as_homogeneous(gtm["extrinsics"])))
            metrics[name] = Evaluator._to_float_dict({**p1, **{f"{k_}_init": v for k_, v in a0.items()}, **info})
            Evaluator._dump_json(mpath, {**metrics, "mean": Evaluator._mean_of_dicts(metrics.values())})
            curve = " ".join(f"{info[f'auc03_r{r}']:.3f}" for r in range(len(cfg["rounds"].split(","))) if f"auc03_r{r}" in info)
            ms = " ".join(f"{info.get(f'matches_r{r}', 0)}:{info.get(f'sampson_before_r{r}', float('nan')):.2f}->{info.get(f'sampson_after_r{r}', float('nan')):.2f}"
                          for r in range(len(cfg["rounds"].split(","))))
            print(f"[epi] {ds} {name}: auc03 {a0['auc03']:.3f} -> {curve} | {ms} | t {info['epi_s']:.1f}s", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
