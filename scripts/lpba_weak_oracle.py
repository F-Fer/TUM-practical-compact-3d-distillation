"""G0b of reports/Learned refinement step between DA3 and DeepBA.md: how much pose AUC is left in the weak directions?

    .venv/bin/python scripts/lpba_weak_oracle.py <PBA run dir with saved predictions> [--out x.json] [--samples 16384] [overrides]

Per scene, at the run's final state (its poses, focal and depth, grid zero, no radial term): the Gauss-Newton matrix of
the LM level (sharp grey images with sigma 1, Huber, geometric term, 8-cell grid, shared focal), its Schur complement
onto the poses of frames 1..N-1 (focal and grid eliminated) and its eigenvectors. Poses are PBA's parameters (rotation
vector in rad, translation in median depths), as in weak_modes.py (commit 1c1a921).

The remaining error d is the correction that takes the run's poses to the GT (gt_pose_error, in the run's gauge). The
oracle removes the part of d in a set of eigen-directions and scores the pose AUC of the result:
    weakXX      the weakest XX % of the directions (an upper bound of what a step that only fixes the flat directions gains)
    strong50    the strongest half (control)
    all         all of d (sanity check: AUC = 1)
    weak10_ok   the weakest 10 %, but frames with a gross error (> 15 deg or > 0.25 median depths) left as they are
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

SETS = ("hiroom", "7scenes", "scannetpp", "dtu64", "eth3d")


def apply(ext, d, scale):
    """Poses ext [N, 3, 4] (metric) updated by PBA parameters d [N, 6]: R = exp(w) R, t = exp(w) t + tau scale."""
    dR = Rotation.from_rotvec(d[:, :3]).as_matrix()
    R = dR @ ext[:, :, :3]
    t = np.einsum("nij,nj->ni", dR, ext[:, :, 3]) + d[:, 3:] * scale
    return np.concatenate([R, t[..., None]], 2)


def scene(run, src, dataset, name, cfg, device, gen):
    ex = os.path.join(src, "model_results", dataset, name, "unposed/exports")
    pred = dict(np.load(os.path.join(ex, "mini_npz/results.npz")))
    gt = dict(np.load(os.path.join(ex, "gt_meta.npz"), allow_pickle=True))
    out = dict(np.load(os.path.join(run, "model_results", dataset, name, "unposed/exports/mini_npz/results.npz")))
    ext = out["extrinsics"][:, :3].astype(np.float64)
    depth = torch.from_numpy(out["depth"]).float().to(device)
    N, H, W = depth.shape
    images = rp.load_images(gt["image_files"], H, W).to(device)
    t_ = lambda a: torch.from_numpy(np.asarray(a)).float().to(device)  # noqa: E731
    model = rp.PBA(images, depth, t_(pred["conf"]), t_(ext), t_(out["intrinsics"]), cfg).to(device)
    with torch.no_grad():
        R, t = model.poses()
        nbrs = rp.select_neighbors(model.depth0, R, t, model.intrinsics(), cfg["neighbors"], cfg["occ_thresh"], cfg["nbr_mode"], cfg["min_covis"])
        kernel = rp.Kernel(model)
        img = rp.gaussian_blur(images.mean(1, keepdim=True), 1.0)
        kernel.set_level(img, True, 1.0)
        probs = rp.sampling_probs(img, model.valid)
        pix = torch.multinomial(probs, cfg["lm_samples"], replacement=True, generator=gen)
        feat, quad = kernel.make_pool(pix)
        chunk = max(1, int(3e6 / (nbrs.shape[1] * cfg["lm_samples"] * feat.shape[-1])))
        Hpp, Hgp, Hgg, gp, gg, Hx, _ = kernel.normal_eq(feat, quad, nbrs, chunk)
    n, nc = len(gp), kernel.nc
    Hf = torch.zeros(n + N * nc, n + N * nc, dtype=torch.float64, device=device)
    Hf[:n, :n] = Hpp
    Hf[n:, :n] = Hgp.reshape(N * nc, n)
    Hf[:n, n:] = Hgp.reshape(N * nc, n).T
    Hf[n:, n:] = Hx + torch.block_diag(*Hgg)
    npose = 6 * (N - 1)
    A, B, C = Hf[:npose, :npose], Hf[:npose, npose:], Hf[npose:, npose:]
    S = A - B @ torch.linalg.solve(C, B.T)
    S = 0.5 * (S + S.T)
    evals, V = torch.linalg.eigh(S)
    V = V.cpu().numpy()

    d = rp.gt_pose_error(gt["extrinsics"][:, :3], ext, model.scale)  # [N, 6], frame 0 = 0
    gross = (np.linalg.norm(d[:, :3], axis=1) > np.radians(15)) | (np.linalg.norm(d[:, 3:], axis=1) > 0.25)
    gt_ext = torch.from_numpy(as_homogeneous(gt["extrinsics"]))
    score = lambda e: compute_pose(torch.from_numpy(as_homogeneous(e)), gt_ext)  # noqa: E731

    def fix(cols, x):
        y = V[:, cols] @ (V[:, cols].T @ x[1:].reshape(-1))
        full = np.zeros_like(x)
        full[1:] = y.reshape(N - 1, 6)
        return apply(ext, full, model.scale)

    m = npose
    res = {"frames": N, "gross": float(gross[1:].mean()), "cond": float(evals[-1] / max(float(evals[0]), 1e-30)), "run": score(ext)}
    for q in (10, 25, 50):
        res[f"weak{q}"] = score(fix(slice(0, max(1, round(q / 100 * m))), d))
    res["strong50"] = score(fix(slice(round(0.5 * m), m), d))
    res["all"] = score(fix(slice(0, m), d))
    d_ok = d.copy()
    d_ok[gross] = 0
    res["weak10_ok"] = score(fix(slice(0, max(1, round(0.1 * m))), d_ok))
    c = (V.T @ d[1:].reshape(-1)) ** 2
    res["share_weak10"] = float(c[: round(0.1 * m)].sum() / max(c.sum(), 1e-30))
    return {k: ({kk: float(vv) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--src", default="workspace/evaluation/DA3-SMALL")
    ap.add_argument("--out")
    ap.add_argument("--samples", type=int, default=16384)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    os.chdir(rp.REPO_ROOT)
    cfg = rp.parse_overrides(["focal_init=pred", "radial=false", f"lm_samples={a.samples}"] + a.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(0)
    res = {}
    for ds in SETS:
        root = os.path.join(a.run, "model_results", ds)
        names = sorted(os.path.relpath(p, root).split("/unposed/")[0] for p in glob.glob(os.path.join(root, "**/unposed/exports/mini_npz/results.npz"), recursive=True))
        for s in names:
            r = scene(a.run, a.src, ds, s, cfg, device, gen)
            res.setdefault(ds, {})[s] = r
            print(f"{ds} {s}: N={r['frames']} gross={r['gross']:.2f} cond={r['cond']:.1e} share_weak10={r['share_weak10']:.2f} auc03 run "
                  f"{r['run']['auc03']:.3f} weak10 {r['weak10']['auc03']:.3f} weak10_ok {r['weak10_ok']['auc03']:.3f} weak25 {r['weak25']['auc03']:.3f} "
                  f"weak50 {r['weak50']['auc03']:.3f} strong50 {r['strong50']['auc03']:.3f} all {r['all']['auc03']:.3f}", flush=True)
            torch.cuda.empty_cache()
    keys = ("run", "weak10", "weak10_ok", "weak25", "weak50", "strong50", "all")
    summary = {}
    for metric in ("auc03", "auc30"):
        per = {k: {ds: float(np.mean([v[k][metric] for v in sc.values()])) for ds, sc in res.items()} for k in keys}
        summary[metric] = {k: {**per[k], "mean": float(np.mean(list(per[k].values())))} for k in keys}
    print("AUC@3 x100 | " + " | ".join(SETS) + " | mean")
    for k in keys:
        row = summary["auc03"][k]
        print(f"{k:10s} | " + " | ".join(f"{100 * row.get(ds, float('nan')):.1f}" for ds in SETS) + f" | {100 * row['mean']:.1f}")
    if a.out:
        json.dump({"summary": summary, "scenes": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
