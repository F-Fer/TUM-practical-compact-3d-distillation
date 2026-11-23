"""Prototype of a scene-level depth correction shared by all frames (knowledge/ideas/shared-depth-field.md, step 3).

    .venv/bin/python scripts/lpba_field.py <src work dir> <dst work dir> [key=value ...]

    # the shared field (16 cells per axis) vs the per-frame 8-cell grid, same autograd optimiser
    ... depth=field field_cells=16
    ... depth=grid

Per scene: poses (frame 0 fixed), one shared focal, per-frame relative brightness and a depth correction are optimised
with Adam on the blur levels of run_tto_pba.py (sigma 8, 4, 2, 1, `iters` each, cosine decay, 1,024 gradient-weighted
pixels per frame and iteration, 10 spread neighbours, grey Huber photometric + Huber log-depth geometric term), with
autograd through PBA.residuals (pinhole, no radial term). The depth correction is either

    grid    run_tto_pba.py's per-frame log-depth grid (depth_grid cells along the long side), the control;
    field   one log-depth field in world coordinates: a trilinear grid of field_cells^3 cells over the robust bounding
            box (2-98 % per axis) of the initial back-projected points, looked up at each pixel's initial world
            position (from DA3's depth and cameras, fixed), plus one log scale per frame. A frame can't bend a region
            without bending it for every frame that sees it.

Priors: w_prior * mean(correction^2) on the grid / field cells, and on the per-frame scales. Output in the evaluator's
layout (results.npz with the refined depth, extrinsics, intrinsics, conf; gt_meta.npz), so scripts/lpba_stage.py can
polish the poses with the depth fixed and scripts/eval_tto.py can score it; metric_results/<dataset>_pba.json with
pose AUC before / after, time and the depth error vs GT (lpba_depth_oracle.gt_depth; nan without GT).
"""

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
from lpba_depth_oracle import abs_rel, try_gt  # noqa: E402

DEFAULTS = {
    "subset": "dev",
    "datasets": "",
    "depth": "field",  # field | grid
    "field_cells": 16,
    "depth_grid": 8,
    "iters": 25,
    "levels": "8,4,2,1",
    "samples": 1024,
    "neighbors": 10,
    "lr_pose": 3e-3,
    "lr_focal": 3e-3,
    "lr_depth": 5e-3,
    "lr_bright": 1e-2,
    "w_geo": 1.0,
    "w_prior": 1.0,
    "huber_photo": 0.05,
    "huber_geo": 0.05,
    "seed": 0,
}


class FieldPBA(rp.PBA):
    """PBA with the per-frame grid replaced by a shared world-space log-depth field + per-frame log scales."""

    def __init__(self, images, depth, conf, ext, K, cfg, cells):
        super().__init__(images, depth, conf, ext, K, {**cfg, "depth_grid": 1})
        N, H, W = self.N, self.H, self.W
        # initial world points of every pixel (DA3's depth and cameras, normalised scene), fixed
        vs, us = torch.meshgrid(torch.arange(H, device=depth.device).float(), torch.arange(W, device=depth.device).float(), indexing="ij")
        Kinv = torch.linalg.inv(self.K0)
        pix = torch.stack([us, vs, torch.ones_like(us)], -1).reshape(-1, 3)  # [HW, 3]
        rays = torch.einsum("nij,pj->npi", Kinv, pix)  # [N, HW, 3]
        Xc = rays * self.depth0.reshape(N, -1, 1)
        Xw = torch.einsum("nji,npj->npi", self.R0, Xc - self.t0[:, None])  # R^T (X - t)
        flat = Xw.reshape(-1, 3)
        sub = flat[:: max(1, len(flat) // 200000)]
        lo, hi = torch.quantile(sub, 0.02, dim=0), torch.quantile(sub, 0.98, dim=0)
        self.register_buffer("box_lo", lo)
        self.register_buffer("box_size", (hi - lo).clamp_min(1e-6))
        self.register_buffer("Xw", Xw.permute(2, 0, 1).reshape(3, -1))  # [3, N*H*W] for bilinear_gather
        self.field = torch.nn.Parameter(torch.zeros(1, 1, cells, cells, cells, device=depth.device))
        self.fscale = torch.nn.Parameter(torch.zeros(N, device=depth.device))

    def field_at(self, X):
        """log correction of the field at world points X [S, 3] (clamped to the box)."""
        g = ((X - self.box_lo) / self.box_size).clamp(0, 1) * 2 - 1  # grid_sample: x, y, z = last dims W, H, D
        return F.grid_sample(self.field, g.view(1, 1, 1, -1, 3), mode="bilinear", padding_mode="border", align_corners=True).view(-1)

    def grid_at(self, idx, u, v):
        X = rp.bilinear_gather(self.Xw, idx, u, v, self.H, self.W)  # [S, 3]
        return self.fscale[idx] + self.field_at(X)

    def dense_depth(self):
        corr = self.field_at(self.Xw.T).view(self.N, self.H, self.W) + self.fscale[:, None, None]
        return self.depth0 * torch.exp(corr)

    def prior(self):
        return (self.field**2).mean() + (self.fscale**2).mean()


def refine(image_files, pred, cfg, device):
    torch.manual_seed(cfg["seed"])
    start = time.perf_counter()
    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)  # noqa: E731
    depth, conf, ext, K = t(pred["depth"]), t(pred["conf"]), t(pred["extrinsics"]), t(pred["intrinsics"])
    N, H, W = depth.shape
    images = rp.load_images(image_files, H, W).to(device)
    base = rp.parse_overrides(["iters=25", f"depth_grid={cfg['depth_grid']}", "radial=false"])
    if cfg["depth"] == "field":
        model = FieldPBA(images, depth, conf, ext, K, base, cfg["field_cells"]).to(device)
        depth_params = [model.field, model.fscale]
    else:
        model = rp.PBA(images, depth, conf, ext, K, base).to(device)
        depth_params = [model.grid]
    with torch.no_grad():
        R, tt = model.poses()
        nbrs = rp.select_neighbors(model.depth0, R, tt, model.intrinsics(), cfg["neighbors"], base["occ_thresh"], "spread", base["min_covis"])
    opt = torch.optim.Adam([{"params": [model.rot, model.trans], "lr": cfg["lr_pose"]}, {"params": [model.logf], "lr": cfg["lr_focal"]},
                            {"params": depth_params, "lr": cfg["lr_depth"]}, {"params": [model.bright], "lr": cfg["lr_bright"]}])
    lrs = [g["lr"] for g in opt.param_groups]
    grey_full = images.mean(1, keepdim=True)
    M = cfg["samples"]
    for sigma in [float(x) for x in cfg["levels"].split(",")]:
        img = rp.gaussian_blur(grey_full, sigma)
        img_flat = img.permute(1, 0, 2, 3).reshape(1, -1)
        probs = rp.sampling_probs(img, model.valid)
        pool = torch.multinomial(probs, 16 * M, replacement=True)
        for st in opt.state.values():  # fresh Adam state per level, as run_tto_pba.py
            st.clear()
        T = cfg["iters"]
        for it in range(T):
            f = 0.1 + 0.45 * (1 + math.cos(math.pi * it / T))
            for g, lr in zip(opt.param_groups, lrs):
                g["lr"] = lr * f
            pix = pool.gather(1, torch.randint(0, pool.shape[1], (N, M), device=device))
            u, v = (pix % W).float(), (pix // W).float()
            i_tgt, i_src, geo, vis = model.residuals(u, v, nbrs, img_flat)
            m = vis.float()
            n = m.sum().clamp_min(1)
            lp = (rp.huber(i_tgt - i_src, cfg["huber_photo"]).mean(-1) * m).sum() / n
            lg = (rp.huber(geo, cfg["huber_geo"]) * m).sum() / n
            prior = model.prior() if cfg["depth"] == "field" else (model.grid**2).mean()
            loss = lp + cfg["w_geo"] * lg + cfg["w_prior"] * prior
            opt.zero_grad()
            loss.backward()
            opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    opt_s = time.perf_counter() - start
    with torch.no_grad():
        R, tt = model.poses()
        out = {"depth": (model.dense_depth() * model.scale).cpu().numpy().astype(np.float32),
               "extrinsics": torch.cat([R, tt[..., None] * model.scale], -1).cpu().numpy().astype(np.float32),
               "intrinsics": model.intrinsics().cpu().numpy().astype(np.float32)}
    return out, {"opt_s": opt_s, "num_frames": N}


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
    json.dump(cfg, open(os.path.join(dst, "field_config.json"), "w"), indent=2)
    datasets = [d for d in cfg["datasets"].split(",") if d] or list(rp.DEV)
    for ds in datasets:
        scenes = sorted(os.path.dirname(p) for p, _, _ in os.walk(os.path.join(src, "model_results", ds)) if os.path.basename(p) == "unposed")
        k = rp.DEV.get(ds, 0)
        scenes = scenes[:k] if cfg["subset"] == "dev" else scenes[k:] if cfg["subset"] == "held-out" else scenes
        mpath = os.path.join(dst, "metric_results", f"{ds}_pba.json")
        os.makedirs(os.path.dirname(mpath), exist_ok=True)
        metrics = json.load(open(mpath)) if os.path.exists(mpath) else {}
        metrics.pop("mean", None)
        for sd in scenes:
            name = os.path.relpath(sd, os.path.join(src, "model_results", ds))
            if name in metrics:
                continue
            ex = os.path.join(sd, "unposed", "exports")
            gt = dict(np.load(os.path.join(ex, "gt_meta.npz"), allow_pickle=True))
            pred = dict(np.load(os.path.join(ex, "mini_npz", "results.npz")))
            out, info = refine(gt["image_files"], pred, cfg, device)
            oex = os.path.join(dst, "model_results", ds, name, "unposed", "exports")
            os.makedirs(os.path.join(oex, "mini_npz"), exist_ok=True)
            np.savez(os.path.join(oex, "mini_npz", "results.npz"), conf=pred["conf"], **out)
            shutil.copy(os.path.join(ex, "gt_meta.npz"), os.path.join(oex, "gt_meta.npz"))
            g = torch.from_numpy(as_homogeneous(gt["extrinsics"]))
            p0 = compute_pose(torch.from_numpy(as_homogeneous(pred["extrinsics"])), g)
            p1 = compute_pose(torch.from_numpy(as_homogeneous(out["extrinsics"])), g)
            gd = try_gt(gt["image_files"], *pred["depth"].shape[1:])
            rec = {**p1, **{f"{k_}_init": v for k_, v in p0.items()}, **info,
                   "depth_err_da3": abs_rel(pred["depth"], gd) if gd is not None else float("nan"),
                   "depth_err_out": abs_rel(out["depth"], gd) if gd is not None else float("nan")}
            metrics[name] = Evaluator._to_float_dict(rec)
            Evaluator._dump_json(mpath, {**metrics, "mean": Evaluator._mean_of_dicts(metrics.values())})
            print(f"[field] {ds} {name}: auc03 {p0['auc03']:.3f}->{p1['auc03']:.3f} depth {rec['depth_err_da3']:.4f}->{rec['depth_err_out']:.4f} "
                  f"t {info['opt_s']:.1f}s", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
