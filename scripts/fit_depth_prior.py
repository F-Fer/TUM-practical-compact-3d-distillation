"""Fit the depth-grid prior of the photometric BA to DA3's own depth error (session 5, experiment 3b).

    uv run scripts/fit_depth_prior.py DA3-SMALL [--cells 8] [--dev]

Per frame, DA3's log-depth error (prediction minus GT, minus the scene's median) is projected onto PBA's bilinear grid
(least squares over the valid pixels). The per-frame mean of the cells is the frame's offset, the rest its shape. A Gaussian
prior with precision alpha (I - J / nc) + beta L on the shape (L the 4-neighbour Laplacian of the cells, J the all-ones matrix)
and 1 / var on the offset is fitted by maximum likelihood. In PBA's units (prior = w_prior mean(shape^2) + w_scale ... +
w_smooth ..., see run_tto_pba.py) the ratios are w_smooth / w_prior = beta / alpha and w_scale / w_prior =
1 / (nc alpha var(offset)). --dev uses the scenes of the dev subset only (HiRoom 10, 7-Scenes 2, ScanNet++ 3, ETH3D 3, DTU-49 4).
"""

import argparse
import glob
import os

import numpy as np
from scipy.optimize import minimize

import analyse_failure_modes as afm

DEV = {"hiroom": 10, "7scenes": 2, "scannetpp": 3, "eth3d": 3, "dtu": 4}


def grid_shape(H, W, g):
    return (g, max(2, round(g * W / H))) if H > W else (max(2, round(g * H / W)), g)


def bilinear_rows(u, v, H, W, gh, gw):
    """[P, gh*gw] bilinear weights of pixels (u, v) on a grid with corners at the image corners."""
    gu, gv = u * (gw - 1) / (W - 1), v * (gh - 1) / (H - 1)
    u0, v0 = np.clip(np.floor(gu), 0, gw - 2).astype(int), np.clip(np.floor(gv), 0, gh - 2).astype(int)
    du, dv = gu - u0, gv - v0
    A = np.zeros((len(u), gh * gw))
    for (a, b, wgt) in ((0, 0, (1 - du) * (1 - dv)), (1, 0, du * (1 - dv)), (0, 1, (1 - du) * dv), (1, 1, du * dv)):
        np.add.at(A, (np.arange(len(u)), (v0 + b) * gw + u0 + a), wgt)
    return A


def laplacian(gh, gw):
    n = gh * gw
    L = np.zeros((n, n))
    for y in range(gh):
        for x in range(gw):
            i = y * gw + x
            for yy, xx in ((y + 1, x), (y, x + 1)):
                if yy < gh and xx < gw:
                    j = yy * gw + xx
                    L[i, i] += 1; L[j, j] += 1; L[i, j] -= 1; L[j, i] -= 1
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--cells", type=int, default=8)
    ap.add_argument("--dev", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    shapes, offsets, per_ds = [], [], {}
    gshape = None
    for ds, n_dev in DEV.items():
        root = os.path.join(afm.REPO_ROOT, "workspace/evaluation", a.model, "model_results", ds)
        scenes = sorted(os.path.relpath(p, root).split("/unposed/")[0] for p in glob.glob(os.path.join(root, "**/unposed/exports/mini_npz/results.npz"), recursive=True))
        for sc in scenes[: n_dev if a.dev else None]:
            s = afm.load_scene(a.model, ds, sc)
            pd, gd, valid = s["pred_depth"], s["gt_depth"], s["valid"] & (s["gt_depth"] > 0) & (s["pred_depth"] > 0)
            N, H, W = pd.shape
            gh, gw = grid_shape(H, W, a.cells)
            gshape = (gh, gw)
            e_all = np.log(pd[valid]) - np.log(gd[valid])
            med = np.median(e_all)
            for i in range(N):
                vv, uu = np.nonzero(valid[i])
                if len(uu) < 500:
                    continue
                pick = rng.choice(len(uu), min(4000, len(uu)), replace=False)
                u, v = uu[pick].astype(float), vv[pick].astype(float)
                e = np.log(pd[i][vv[pick], uu[pick]]) - np.log(gd[i][vv[pick], uu[pick]]) - med
                e = np.clip(e, -0.5, 0.5)
                A = bilinear_rows(u, v, H, W, gh, gw)
                g = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ e)
                offsets.append(g.mean())
                shapes.append(g - g.mean())
                per_ds.setdefault(ds, []).append(g)
    groups = {}  # grid shape -> frames' shape vectors (the datasets' aspect ratios differ)
    for g in shapes:
        groups.setdefault(len(g), []).append(g)
    gshapes = {gh * gw: (gh, gw) for gh, gw in {grid_shape(378, 504, a.cells), grid_shape(504, 504, a.cells), grid_shape(336, 504, a.cells), grid_shape(392, 504, a.cells)}}
    data = []
    for nc, gs in groups.items():
        S = np.array(gs)
        L = laplacian(*gshapes[nc])
        ev, U = np.linalg.eigh(np.eye(nc) - np.ones((nc, nc)) / nc)
        U = U[:, ev > 0.5]
        data.append((U.T @ L @ U, S @ U, nc))
    n_frames = sum(len(Z) for _, Z, _ in data)

    def nll(x):
        al, be = np.exp(x)
        out = 0.0
        for Ls, Z, _ in data:
            Q = al * np.eye(len(Ls)) + be * Ls
            out -= 0.5 * len(Z) * np.linalg.slogdet(Q)[1] - 0.5 * np.einsum("fi,ij,fj->", Z, Q, Z)
        return out / n_frames

    r = minimize(nll, np.log([100.0, 100.0]), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
    al, be = np.exp(r.x)
    r0 = minimize(lambda x: nll([x[0], -30.0]), [np.log(100.0)], method="Nelder-Mead")
    var_off = float(np.var(offsets))
    nc = max(groups, key=lambda k: len(groups[k]))
    S = np.array(groups[nc])
    gshape = gshapes[nc]
    print(f"grid {gshape} ({nc} cells), {len(S)} frames from {len(per_ds)} datasets; RMS shape {np.sqrt((S**2).mean()):.4f}, RMS offset {np.sqrt(np.mean(np.square(offsets))):.4f}")
    print(f"ML: alpha {al:.1f} beta {be:.1f} (nll {r.fun:.3f}; alpha only: {np.exp(r0.x[0]):.1f}, nll {r0.fun:.3f})")
    print(f"-> w_smooth / w_prior = {be / al:.3f}, w_scale / w_prior = {1 / (nc * al * var_off):.3f}")
    for ds, gs in per_ds.items():
        gs = np.array(gs)
        print(f"  {ds:9s} frames {len(gs):4d} RMS shape {np.sqrt(((gs - gs.mean(1, keepdims=True))**2).mean()):.4f} RMS offset {np.sqrt((gs.mean(1)**2).mean()):.4f}")


if __name__ == "__main__":
    main()
