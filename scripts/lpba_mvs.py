"""Plane-sweep depth on fixed poses: can per-pixel multi-view evidence improve the depth that PBA leaves?

    .venv/bin/python scripts/lpba_mvs.py <state work dir> <dst work dir> [key=value ...]

Oracle-style test of option 1 of knowledge/ideas/shared-depth-field.md: the cameras (poses + focal) of a saved state
(e.g. B′ with lpba_depth_oracle.py, whose model_results are kept) are fixed, and each frame's depth is re-estimated by
a per-pixel sweep around the state's depth: hypotheses d = d_state * exp(delta), delta in [-range, range] (`hyps`
steps), i.e. surfaces parallel to the current depth rather than fronto-parallel planes. Cost per hypothesis and
neighbour: 1 - NCC over a `patch` x `patch` window (grey images, warped with every pixel's own hypothesis depth);
invisible (outside, behind, or occluded by more than `occ` w.r.t. the neighbour's current depth) neighbours are
dropped; the cost of a pixel is the mean of its `topk` best neighbours (`nbrs` spread neighbours as run_tto_pba.py).
Winner-take-all with a parabolic sub-step; a pixel is confident if its cost < `max_cost`, its reference patch has
std > `min_std`, its winner isn't at the border of the range and >= 2 neighbours see it.

`rounds` > 1 repeats the sweep with the range halved around the new estimate (and the new depths for the occlusion
test), so the effect of more iterations is visible. Output depth:

    out=pixel     the sweep's depth where confident, the state's elsewhere
    out=smooth    the confidence-weighted log correction, blurred with a Gaussian of `sigma` px (-> 0 without support)

`scale` > 1 runs the sweep on images upsampled by that factor (focal and principal point scaled, depth bilinearly
upsampled, the result area-downsampled back to the prediction's resolution).

Writes results.npz (the state's cameras and conf, the new depth) + gt_meta.npz in the evaluator's layout (score with
scripts/eval_tto.py; polish poses with scripts/lpba_stage.py) and metric_results/<dataset>_pba.json with the pose
metrics of the state, the depth error vs GT (abs. rel., one scale per scene) of the state and after every round, the
confident fraction, and the error on the confident pixels only (state vs sweep).
"""

import glob
import json
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
from lpba_depth_oracle import abs_rel, try_gt  # noqa: E402

DEFAULTS = {
    "datasets": "",
    "range": 0.15,
    "hyps": 33,
    "patch": 7,
    "nbrs": 10,
    "topk": 3,
    "occ": 0.1,
    "max_cost": 0.4,
    "min_std": 0.01,
    "rounds": 1,
    "out": "pixel",
    "sigma": 8.0,
    "scale": 1.0,
    "cams": "state",  # state | gt (diagnostic: GT cameras, the state's depth scaled to the GT's median)
}


def box(x, k):
    return F.avg_pool2d(x, k, stride=1, padding=k // 2, count_include_pad=False)


@torch.no_grad()
def sweep(grey, depth, R, t, K, nbrs, cfg, rng):
    """One sweep around `depth` [N, H, W]; returns the delta map [N, H, W] and the confidence mask [N, H, W]."""
    N, H, W = depth.shape
    k = cfg["patch"]
    dev = depth.device
    vs, us = torch.meshgrid(torch.arange(H, device=dev).float(), torch.arange(W, device=dev).float(), indexing="ij")
    pix = torch.stack([us, vs, torch.ones_like(us)], 0).reshape(3, -1)
    deltas = torch.linspace(-rng, rng, cfg["hyps"], device=dev)
    delta_out = torch.zeros(N, H, W, device=dev)
    conf_out = torch.zeros(N, H, W, dtype=torch.bool, device=dev)
    for i in range(N):
        J = nbrs[i]
        a = grey[i : i + 1, None]  # [1, 1, H, W]
        mu_a = box(a, k)
        var_a = (box(a * a, k) - mu_a**2).clamp_min(0)
        rays = torch.linalg.inv(K[i]) @ pix  # [3, HW]
        costs = torch.empty(cfg["hyps"], H * W, device=dev)
        nvis = torch.empty(cfg["hyps"], H * W, device=dev)
        for h, dl in enumerate(deltas):
            Xc = rays * (depth[i].reshape(1, -1) * torch.exp(dl))
            Xw = R[i].T @ (Xc - t[i][:, None])
            Xj = R[J] @ Xw[None] + t[J][:, :, None]  # [nb, 3, HW]
            z = Xj[:, 2]
            uj = K[J, 0, 0, None] * Xj[:, 0] / z.clamp_min(1e-6) + K[J, 0, 2, None]
            vj = K[J, 1, 1, None] * Xj[:, 1] / z.clamp_min(1e-6) + K[J, 1, 2, None]
            g = torch.stack([uj / (W - 1) * 2 - 1, vj / (H - 1) * 2 - 1], -1).view(len(J), H, W, 2)
            b = F.grid_sample(grey[J, None], g, align_corners=True)  # [nb, 1, H, W]
            dj = F.grid_sample(depth[J, None], g, align_corners=True).view(len(J), -1)
            vis = (z > 1e-6) & (uj >= 0) & (uj <= W - 1) & (vj >= 0) & (vj <= H - 1) & (z < dj * (1 + cfg["occ"]))
            mu_b = box(b, k)
            cov = box(a * b, k) - mu_a * mu_b
            var_b = (box(b * b, k) - mu_b**2).clamp_min(0)
            ncc = (cov / torch.sqrt(var_a * var_b + 1e-8)).view(len(J), -1)
            c = torch.where(vis, 1 - ncc, torch.full_like(ncc, 3.0))
            top = c.topk(min(cfg["topk"], len(J)), dim=0, largest=False).values
            ok = top < 3
            costs[h] = (top * ok).sum(0) / ok.sum(0).clamp_min(1) + 3.0 * (ok.sum(0) == 0)
            nvis[h] = vis.sum(0).float()
        best = costs.argmin(0)
        cmin = costs.gather(0, best[None])[0]
        lo = costs.gather(0, (best - 1).clamp_min(0)[None])[0]
        hi = costs.gather(0, (best + 1).clamp_max(cfg["hyps"] - 1)[None])[0]
        den = lo - 2 * cmin + hi
        sub = torch.where(den > 1e-6, 0.5 * (lo - hi) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
        step = (deltas[1] - deltas[0]) if cfg["hyps"] > 1 else torch.tensor(0.0, device=dev)
        delta = deltas[best] + sub * step
        conf = (cmin < cfg["max_cost"]) & (best > 0) & (best < cfg["hyps"] - 1)
        conf &= nvis.gather(0, best[None])[0] >= 2
        conf &= var_a.view(-1).sqrt() > cfg["min_std"]
        delta_out[i] = delta.view(H, W)
        conf_out[i] = conf.view(H, W)
    return delta_out, conf_out


def blur_correction(delta, conf, sigma):
    w = conf.float()[:, None]
    num = rp.gaussian_blur(delta[:, None] * w, sigma)
    den = rp.gaussian_blur(w, sigma)
    return (num / (den + 1e-2))[:, 0]


def refine(image_files, st, cfg, device, gt, gtm=None):
    t_ = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)  # noqa: E731
    if cfg["cams"] == "gt":
        v = gt > 0
        Kg = gtm["intrinsics"].astype(np.float32).copy()
        h0, w0 = 2 * Kg[0, 1, 2], 2 * Kg[0, 0, 2]  # principal point at the centre
        Kg[:, 0] *= st["depth"].shape[2] / w0
        Kg[:, 1] *= st["depth"].shape[1] / h0
        st = {**st, "depth": st["depth"] * np.median(gt[v]) / np.median(st["depth"][v]), "extrinsics": gtm["extrinsics"][:, :3], "intrinsics": Kg}
    depth0 = t_(st["depth"])
    N, H, W = depth0.shape
    ext = t_(st["extrinsics"])[:, :3, :4]
    K = t_(st["intrinsics"]).clone()
    R, t = ext[:, :, :3], ext[:, :, 3]
    nbrs = rp.select_neighbors(depth0, R, t, K, cfg["nbrs"], 0.1, "spread", 0.2)
    s = cfg["scale"]
    Hs, Ws = int(round(H * s)), int(round(W * s))
    grey = rp.load_images(image_files, Hs, Ws).to(device).mean(1)
    Ks = K.clone()
    Ks[:, 0] *= Ws / W
    Ks[:, 1] *= Hs / H
    cur = F.interpolate(depth0[:, None], (Hs, Ws), mode="bilinear", align_corners=False)[:, 0] if s != 1 else depth0
    base = cur.clone()
    info = {"depth_err_in": abs_rel(st["depth"], gt)}
    conf_any = torch.zeros_like(cur, dtype=torch.bool)
    for r in range(cfg["rounds"]):
        delta, conf = sweep(grey, cur, R, t, Ks, nbrs, cfg, cfg["range"] / 2**r)
        conf_any |= conf
        if cfg["out"] == "smooth":
            cur = cur * torch.exp(blur_correction(delta, conf, cfg["sigma"] * s))
        else:
            cur = torch.where(conf, cur * torch.exp(delta), cur)
        out = F.interpolate(cur[:, None], (H, W), mode="area")[:, 0] if s != 1 else cur
        info[f"depth_err_r{r}"] = abs_rel(out.cpu().numpy(), gt)
    cf = (F.interpolate(conf_any[:, None].float(), (H, W), mode="area")[:, 0] > 0.5) if s != 1 else conf_any
    info["conf_frac"] = float(cf.float().mean())
    out_np = out.cpu().numpy().astype(np.float32)
    if gt is not None:
        g = np.where(cf.cpu().numpy(), gt, 0)
        info["depth_err_conf_in"] = abs_rel(st["depth"], g)
        info["depth_err_conf_out"] = abs_rel(out_np, g)
    info["depth_err_out"] = info[f"depth_err_r{cfg['rounds'] - 1}"]
    del base
    return out_np, info, st


def main():
    src, dst = sys.argv[1], sys.argv[2]
    cfg = dict(DEFAULTS)
    for o in sys.argv[3:]:
        k, _, v = o.partition("=")
        if k not in cfg:
            sys.exit(f"unknown override {k}")
        cfg[k] = type(DEFAULTS[k])(v)
    os.chdir(rp.REPO_ROOT)
    device = torch.device("cuda")
    os.makedirs(dst, exist_ok=True)
    json.dump(cfg, open(os.path.join(dst, "mvs_config.json"), "w"), indent=2)
    datasets = [d for d in cfg["datasets"].split(",") if d] or sorted(os.listdir(os.path.join(src, "model_results")))
    for ds in datasets:
        root = os.path.join(src, "model_results", ds)
        scenes = sorted(os.path.relpath(os.path.dirname(os.path.dirname(p)), root)
                        for p in glob.glob(os.path.join(root, "**", "unposed", "exports"), recursive=True))
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
            N, H, W = st["depth"].shape
            gt = try_gt(gtm["image_files"], H, W)
            t0 = time.perf_counter()
            depth, info, st = refine(gtm["image_files"], st, cfg, device, gt, gtm)
            info["mvs_s"] = time.perf_counter() - t0
            oex = os.path.join(dst, "model_results", ds, name, "unposed", "exports")
            os.makedirs(os.path.join(oex, "mini_npz"), exist_ok=True)
            np.savez(os.path.join(oex, "mini_npz", "results.npz"), **{**st, "depth": depth})
            shutil.copy(os.path.join(ex, "gt_meta.npz"), os.path.join(oex, "gt_meta.npz"))
            pose = compute_pose(torch.from_numpy(as_homogeneous(st["extrinsics"][:, :3])), torch.from_numpy(as_homogeneous(gtm["extrinsics"])))
            metrics[name] = Evaluator._to_float_dict({**pose, **info})
            Evaluator._dump_json(mpath, {**metrics, "mean": Evaluator._mean_of_dicts(metrics.values())})
            rr = " ".join(f"{info[f'depth_err_r{r}']:.4f}" for r in range(cfg["rounds"]))
            print(f"[mvs] {ds} {name}: depth {info['depth_err_in']:.4f} -> {rr} | conf {info['conf_frac']:.2f} "
                  f"conf-px {info.get('depth_err_conf_in', float('nan')):.4f} -> {info.get('depth_err_conf_out', float('nan')):.4f} "
                  f"t {info['mvs_s']:.1f}s", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
