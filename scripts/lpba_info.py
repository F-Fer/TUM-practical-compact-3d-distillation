"""Checks 1 and 2 after G0 (reports/Learned refinement step between DA3 and DeepBA.md, §8-9).

    .venv/bin/python scripts/lpba_info.py <PBA run dir with saved predictions> --out <dir> [--configs grey:spread10,grey:all,...]
        [--samples 8192] [--src workspace/evaluation/DA3-SMALL]

Per scene, at the run's final state (its poses, focal and depth; grid zero, no radial term), for every configuration
<features>:<pairs>[:fixed] (features grey | vgg, pairs spread10 = the 10 spread neighbours of PBA, all = every other frame,
pairs without overlap add nothing): the Gauss-Newton matrix of the LM level (sigma 1, Huber), its Schur complement S onto
the poses of frames 1..N-1 (focal and grid eliminated), with the same source pixels for every configuration (":fixed": focal and grid held fixed instead of eliminated, i.e. the
pose block alone; ":fixgrid": only the grid held fixed). Reported:

    cond         condition number of S
    share10      share of the remaining error d (run -> GT, PBA parameters) in the weakest 10 % of S's directions
    weak10_auc   AUC@3 after removing d in those directions (G0b's oracle, per configuration)
    info_d       total information along d: n_vis d^T S d / |d|^2 (n_vis = visible warps, i.e. a sum over warps)
    info_w       mean total information along the weakest 10 % of grey:spread10's directions, and info_s along its
                 strongest 50 %; their ratios to grey:spread10 say whether the extra pairs or features stiffen the
                 weak directions more than the rest (= better conditioned, not just less noisy)

The first configuration also writes <out>/<dataset>__<scene>.npz for the probe of check 2 (scripts/lpba_probe.py):
d, the run's update from DA3, poses, the weak basis and per-frame features.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_tto_pba as rp  # noqa: E402
from depth_anything_3.bench.utils import compute_pose  # noqa: E402
from depth_anything_3.utils.geometry import as_homogeneous  # noqa: E402
from lpba_weak_oracle import apply  # noqa: E402

SETS = ("hiroom", "7scenes", "scannetpp", "dtu64", "eth3d")


def schur(kernel, feat, quad, nbrs, N, fixed=""):
    chunk = max(1, int(3e6 / (nbrs.shape[1] * feat.shape[1] * feat.shape[-1])))
    Hpp, Hgp, Hgg, gp, gg, Hx, _ = kernel.normal_eq(feat, quad, nbrs, chunk)
    n, nc = len(gp), kernel.nc
    Hf = torch.zeros(n + N * nc, n + N * nc, dtype=torch.float64, device=gp.device)
    Hf[:n, :n] = Hpp
    Hf[n:, :n] = Hgp.reshape(N * nc, n)
    Hf[:n, n:] = Hgp.reshape(N * nc, n).T
    Hf[n:, n:] = Hx + torch.block_diag(*Hgg)
    g = torch.cat([gp, gg.reshape(-1)])
    m = 6 * (N - 1)
    A, B, C = Hf[:m, :m], Hf[:m, m:], Hf[m:, m:]
    CiBt = torch.linalg.solve(C, torch.cat([B.T, g[m:, None]], 1))
    S = A - B @ CiBt[:, :-1]
    gs = g[:m] - B @ CiBt[:, -1]  # reduced gradient of the poses
    if fixed == "fixed":  # focal and depth grid held at their values instead of eliminated: the pose block alone
        S, gs = A, g[:m]
    elif fixed == "fixgrid":  # depth grid held fixed, focal eliminated
        b_, c_ = Hf[:m, m], Hf[m, m]
        S, gs = A - torch.outer(b_, b_) / c_, g[:m] - b_ * g[m] / c_
    return (0.5 * (S + S.T)).cpu().numpy(), gs.cpu().numpy(), kernel.last_n_vis


def rel(Ra, ta, Rb, tb):
    """Pose of b relative to a (world-to-camera): R_ba = R_b R_a^T, t_ba = t_b - R_ba t_a -> rotvec, t [6]."""
    R = Rb @ Ra.T
    return np.concatenate([Rotation.from_matrix(R).as_rotvec(), tb - R @ ta])


def scene(run, src, dataset, name, configs, samples, device, out_dir):
    ex = os.path.join(src, "model_results", dataset, name, "unposed/exports")
    pred = dict(np.load(os.path.join(ex, "mini_npz/results.npz")))
    gt = dict(np.load(os.path.join(ex, "gt_meta.npz"), allow_pickle=True))
    out = dict(np.load(os.path.join(run, "model_results", dataset, name, "unposed/exports/mini_npz/results.npz")))
    ext = out["extrinsics"][:, :3].astype(np.float64)
    depth = torch.from_numpy(out["depth"]).float().to(device)
    N, H, W = depth.shape
    images = rp.load_images(gt["image_files"], H, W).to(device)
    t_ = lambda a: torch.from_numpy(np.asarray(a)).float().to(device)  # noqa: E731
    cfg = rp.parse_overrides(["focal_init=pred", "radial=false"])
    model = rp.PBA(images, depth, t_(pred["conf"]), t_(ext), t_(out["intrinsics"]), cfg).to(device)
    gen = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        R, t = model.poses()
        nb = {"spread10": rp.select_neighbors(model.depth0, R, t, model.intrinsics(), 10, cfg["occ_thresh"], "spread", cfg["min_covis"]),
              "all": rp.select_neighbors(model.depth0, R, t, model.intrinsics(), N - 1, cfg["occ_thresh"], "top")}
        grey = rp.gaussian_blur(images.mean(1, keepdim=True), 1.0)
        pix = torch.multinomial(rp.sampling_probs(grey, model.valid), samples, replacement=True, generator=gen)
        maps = {"grey": grey}
        if any(c.startswith("vgg") for c in configs):
            maps["vgg"] = rp.gaussian_blur(rp.vgg_features(images, 8), 1.0)
    d = rp.gt_pose_error(gt["extrinsics"][:, :3], ext, model.scale)
    dv = d[1:].reshape(-1)
    gt_ext = torch.from_numpy(as_homogeneous(gt["extrinsics"]))
    score = lambda e: compute_pose(torch.from_numpy(as_homogeneous(e)), gt_ext)["auc03"]  # noqa: E731
    m = 6 * (N - 1)
    nw = max(1, round(0.1 * m))
    res = {"frames": N, "run_auc03": float(score(ext))}
    ref = None
    for ci, c in enumerate(configs):
        feats, pairs, *rest = c.split(":")
        kernel = rp.Kernel(model, {**cfg, "features": feats})
        with torch.no_grad():
            kernel.set_level(maps[feats], feats == "grey", 1.0)
            feat, quad = kernel.make_pool(pix)
            S, gs, n_vis = schur(kernel, feat, quad, nb[pairs], N, fixed=rest[0] if rest else "")
        ev, V = np.linalg.eigh(S)
        St = S * n_vis
        if ref is None:
            ref = V
        c2 = (V.T @ dv) ** 2
        y = V[:, :nw] @ (V[:, :nw].T @ dv)
        full = np.zeros_like(d)
        full[1:] = y.reshape(N - 1, 6)
        res[c] = {
            "cond": float(ev[-1] / max(ev[0], 1e-30)),
            "share10": float(c2[:nw].sum() / max(c2.sum(), 1e-30)),
            "weak10_auc": float(score(apply(ext, full, model.scale))),
            "info_d": float(dv @ St @ dv / max(dv @ dv, 1e-30)),
            "info_w": float(np.mean(np.einsum("ij,ik,kj->j", ref[:, :nw], St, ref[:, :nw]))),
            "info_s": float(np.mean(np.einsum("ij,ik,kj->j", ref[:, m // 2:], St, ref[:, m // 2:]))),
            "n_vis": n_vis,
        }
        if ci == 0 and out_dir:
            # per-frame data for the probe (check 2); poses relative to frame index neighbours, the run's update from
            # DA3, the reduced gradient, the per-frame diagonal block of S, DA3's confidence
            Rr, tr = ext[:, :, :3], ext[:, :, 3] / model.scale
            upd = rp.gt_pose_error(ext, pred["extrinsics"][:, :3].astype(np.float64), model.scale)  # DA3 -> run
            prev = np.stack([rel(Rr[max(i - 1, 0)], tr[max(i - 1, 0)], Rr[i], tr[i]) for i in range(N)])
            nxt = np.stack([rel(Rr[min(i + 1, N - 1)], tr[min(i + 1, N - 1)], Rr[i], tr[i]) for i in range(N)])
            blocks = np.zeros((N, 6))
            for i in range(1, N):
                blocks[i] = np.log(np.maximum(np.linalg.eigvalsh(S[6 * (i - 1):6 * i, 6 * (i - 1):6 * i]), 1e-30))
            proj = np.zeros(N)
            Pw = V[:, :nw]
            proj[1:] = (Pw**2).reshape(N - 1, 6, nw).sum((1, 2))
            conf = pred["conf"].reshape(N, -1)
            grad = np.zeros((N, 6))
            grad[1:] = gs.reshape(N - 1, 6) / max(np.sqrt((gs**2).mean()), 1e-30)
            np.savez(os.path.join(out_dir, f"{dataset}__{name.replace('/', '_')}.npz"), d=d, upd=upd, prev=prev, nxt=nxt, blocks=blocks,
                     proj=proj, grad=grad, conf_mean=np.log(conf.mean(1) / conf.mean()), conf_med=np.log(np.median(conf, 1) / np.median(conf)),
                     Vw=Pw, ext=ext, gt_ext=gt["extrinsics"][:, :3], scale=model.scale, dataset=dataset)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--src", default="workspace/evaluation/DA3-SMALL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", default="grey:spread10,grey:all,vgg:spread10,vgg:all")
    ap.add_argument("--samples", type=int, default=8192)
    a = ap.parse_args()
    os.chdir(rp.REPO_ROOT)
    os.makedirs(a.out, exist_ok=True)
    configs = a.configs.split(",")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_path = os.path.join(a.out, "info.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for ds in SETS:
        root = os.path.join(a.run, "model_results", ds)
        names = sorted(os.path.relpath(p, root).split("/unposed/")[0] for p in glob.glob(os.path.join(root, "**/unposed/exports/mini_npz/results.npz"), recursive=True))
        for s in names:
            if s in res.get(ds, {}):
                continue
            r = scene(a.run, a.src, ds, s, configs, a.samples, device, a.out)
            res.setdefault(ds, {})[s] = r
            json.dump(res, open(res_path, "w"), indent=1)
            print(f"{ds} {s}: N={r['frames']} auc {r['run_auc03']:.3f} | " + " | ".join(
                f"{c} cond {r[c]['cond']:.0e} share10 {r[c]['share10']:.2f} weak10 {r[c]['weak10_auc']:.3f} info_d {r[c]['info_d']:.2e} w {r[c]['info_w']:.2e} s {r[c]['info_s']:.2e}"
                for c in configs), flush=True)
            torch.cuda.empty_cache()
    print("config | median cond | share10 | weak10 AUC@3 (mean of datasets) | median ratio to first config: info_d, info_w, info_s")
    c0 = configs[0]
    for c in configs:
        allv = [v for sc in res.values() for v in sc.values()]
        ds_mean = lambda k: np.mean([np.mean([v[c][k] for v in sc.values()]) for sc in res.values()])  # noqa: E731
        ratio = lambda k: np.median([v[c][k] / max(v[c0][k], 1e-30) for v in allv])  # noqa: E731
        print(f"{c:14s} | {np.median([v[c]['cond'] for v in allv]):.1e} | {ds_mean('share10'):.2f} | {100 * ds_mean('weak10_auc'):.1f} | "
              f"{ratio('info_d'):.2f} {ratio('info_w'):.2f} {ratio('info_s'):.2f}")


if __name__ == "__main__":
    main()
