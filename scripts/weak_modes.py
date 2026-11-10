"""Does DA3's pose error lie in the weakly constrained directions of the photometric BA? (session 5, experiment E4)

    uv run scripts/weak_modes.py <run dir with saved predictions, e.g. workspace/tto/DA3-SMALL/y_b50_s0> [--out x.json] [overrides]

Per scene, at DA3's initial state: the Gauss-Newton matrix of the LM level of the B configuration (sharp images, 16384 fresh
samples per frame, Huber, geometric term, 8-cell grid, shared focal; no radial term), its Schur complement onto the poses of
frames 1..N-1 (the grid cells and the focal eliminated), and the eigenvectors of that 6 (N - 1) matrix. Poses are PBA's
parameters (rotation vector in rad, translation in units of the median depth, so a unit of either moves a point at median
depth by about one unit).

The error is the correction that would take DA3's poses to the ground truth, written in the same parameters: GT in DA3's
gauge (frame 0 at DA3's pose, translations scaled to DA3's camera spread as init_gt does), w = log(R_gt R0^T),
tau = t_gt - exp(w) t0. Reported per scene: the share of its squared norm in the weakest 10 % of the directions, in the
10-50 % band and in the strongest half (a random direction has 10 / 40 / 50 %), the same for the error that is left after
the run (GT - run, in the coordinates of the initial state) and for the run's own update, with and without the grossly
wrong frames (> 15 deg or > 25 % of the median depth, set to zero).
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
from pba_analytic import Kernel  # noqa: E402

SETS = ("hiroom", "7scenes", "scannetpp", "dtu64", "eth3d")
B = ("nbr_mode=spread neighbors=10 lr_rot=3e-3 lr_trans=3e-3 lr_focal=3e-3 levels=8,4,2,1,1 lm_levels=4 samples=1024 impl=analytic "
     "lm_impl=analytic lm_solver=dense32 lm_iters=15 lm_samples=4096 lm_stop=2e-3 bright_rel=true").split()


def params_of(ext, R0, t0, scale):
    """PBA's (rot, trans) [N, 6] that turn the initial poses (R0, t0 normalised) into ext [N, 3, 4] (metric)."""
    R, t = ext[:, :, :3], ext[:, :, 3] / scale
    dR = R @ R0.transpose(0, 2, 1)
    w = Rotation.from_matrix(dR).as_rotvec()
    tau = t - np.einsum("nij,nj->ni", dR, t0)
    return np.concatenate([w, tau], 1)


def gt_in_pred_gauge(gt_ext, pred_ext):
    """GT world-to-camera poses moved into DA3's gauge: frame 0 at DA3's frame 0, camera spread as DA3's."""
    def h(e):
        T = np.tile(np.eye(4), (len(e), 1, 1))
        T[:, :3] = e[:, :3]
        return T
    G, P = h(gt_ext.astype(np.float64)), h(pred_ext.astype(np.float64))
    spread = lambda e: np.linalg.norm((c := -np.einsum("nji,nj->ni", e[:, :3, :3], e[:, :3, 3])) - c.mean(0), axis=1).mean()  # noqa: E731
    G[:, :3, 3] *= spread(P) / max(spread(G), 1e-12)
    out = G @ np.linalg.inv(G[0]) @ P[0]
    return out[:, :3]


def band_shares(vecs, x):
    """Shares of |x|^2 in the weakest 10 %, 10-50 % and strongest 50 % of the eigenvectors vecs (ascending eigenvalues)."""
    c = (vecs.T @ x) ** 2
    n = len(c)
    tot = max(c.sum(), 1e-30)
    a, b = int(round(0.1 * n)), int(round(0.5 * n))
    return [float(c[:a].sum() / tot), float(c[a:b].sum() / tot), float(c[b:].sum() / tot)]


def scene(run, dataset, scene_name, cfg, device, gen):
    ex = os.path.join("workspace/evaluation/DA3-SMALL/model_results", dataset, scene_name, "unposed/exports")
    pred = dict(np.load(os.path.join(ex, "mini_npz/results.npz")))
    gt = dict(np.load(os.path.join(ex, "gt_meta.npz"), allow_pickle=True))
    out = np.load(os.path.join(run, "model_results", dataset, scene_name, "unposed/exports/mini_npz/results.npz"))
    depth = torch.from_numpy(pred["depth"]).float().to(device)
    N, H, W = depth.shape
    images = rp.load_images(gt["image_files"], H, W).to(device)
    model = rp.PBA(images, depth, torch.from_numpy(pred["conf"]).float().to(device), torch.from_numpy(pred["extrinsics"]).float().to(device),
                   torch.from_numpy(pred["intrinsics"]).float().to(device), cfg).to(device)
    with torch.no_grad():
        R, t = model.poses()
        nbrs, _ = rp.select_neighbors(model.depth0, R, t, model.intrinsics(), cfg["neighbors"], cfg["occ_thresh"], cfg["nbr_mode"], cfg["min_covis"])
        kernel = Kernel(model)
        img = rp.gaussian_blur(images.mean(1, keepdim=True), 1.0)
        kernel.set_level(img, True, 1.0)
        probs = rp.sampling_probs(img, model.valid, cfg["grad_weight"])
        pix = torch.multinomial(probs, cfg["lm_samples"], replacement=True, generator=gen)
        feat, quad = kernel.make_pool(pix)
        k = nbrs.shape[1]
        chunk = max(1, int(1.6e7 / (k * cfg["lm_samples"] * feat.shape[-1])))
        Hpp, Hgp, Hgg, gp, gg, Hx, _ = kernel.normal_eq(feat, quad, nbrs, chunk, True)
    n, nc = len(gp), kernel.nc
    Hf = torch.zeros(n + N * nc, n + N * nc, dtype=torch.float64, device=device)
    Hf[:n, :n] = Hpp
    Hf[n:, :n] = Hgp.reshape(N * nc, n)
    Hf[:n, n:] = Hgp.reshape(N * nc, n).T
    Hf[n:, n:] = Hx + torch.block_diag(*Hgg)
    npose = 6 * (N - 1)
    A, Bm, C = Hf[:npose, :npose], Hf[:npose, npose:], Hf[npose:, npose:]
    S = A - Bm @ torch.linalg.solve(C, Bm.T)  # poses, with focal and grid eliminated
    S = 0.5 * (S + S.T)
    evals, evecs = torch.linalg.eigh(S)
    evecs = evecs.cpu().numpy()
    R0, t0, sc = model.R0.cpu().numpy().astype(np.float64), model.t0.cpu().numpy().astype(np.float64), model.scale
    d_gt = params_of(gt_in_pred_gauge(gt["extrinsics"][:, :3], pred["extrinsics"]), R0, t0, sc)[1:]
    d_run = params_of(out["extrinsics"][:, :3].astype(np.float64), R0, t0, sc)[1:]
    gross = (np.linalg.norm(d_gt[:, :3], axis=1) > np.radians(15)) | (np.linalg.norm(d_gt[:, 3:], axis=1) > 0.25)
    res = {"frames": N, "gross": float(gross.mean()), "cond": float(evals[-1] / max(float(evals[0]), 1e-30)),
           "rayleigh_gt_rel": float((d_gt.reshape(-1) @ S.cpu().numpy() @ d_gt.reshape(-1)) / max(d_gt.reshape(-1) @ d_gt.reshape(-1), 1e-30) / float(evals.mean()))}
    for name, x in (("gt", d_gt), ("left", d_gt - d_run), ("update", d_run)):
        res[name] = band_shares(evecs, x.reshape(-1))
        y = x.copy()
        y[gross] = 0
        res[name + "_ok"] = band_shares(evecs, y.reshape(-1))
        res[name + "_norm"] = float(np.linalg.norm(x))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out")
    ap.add_argument("--samples", type=int, default=16384)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    os.chdir(rp.REPO_ROOT)
    cfg = rp.parse_overrides(B + [f"lm_samples={a.samples}"] + a.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(0)
    res = {}
    for d in SETS:
        root = os.path.join(a.run, "model_results", d)
        names = sorted(os.path.relpath(p, root).split("/unposed/")[0] for p in glob.glob(os.path.join(root, "**/unposed/exports/mini_npz/results.npz"), recursive=True))
        for s in names:
            r = scene(a.run, d, s, cfg, device, gen)
            res.setdefault(d, {})[s] = r
            print(f"{d} {s}: N={r['frames']} gross={r['gross']:.2f} cond={r['cond']:.1e} R/mean={r['rayleigh_gt_rel']:.3f} "
                  f"gt {np.round(r['gt'], 2)} gt_ok {np.round(r['gt_ok'], 2)} left {np.round(r['left'], 2)} update {np.round(r['update'], 2)}", flush=True)
            torch.cuda.empty_cache()
    print("dataset | gt weak/mid/strong | gt w/o gross | left after run | left w/o gross | run update | R(gt)/mean eig")
    for d, sc in res.items():
        m = lambda k: np.mean([v[k] for v in sc.values()], 0)  # noqa: E731
        print(f"{d:9s} | {np.round(m('gt'), 2)} | {np.round(m('gt_ok'), 2)} | {np.round(m('left'), 2)} | {np.round(m('left_ok'), 2)} | {np.round(m('update'), 2)} | {np.median([v['rayleigh_gt_rel'] for v in sc.values()]):.3f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
