"""Levenberg-Marquardt bundle adjustment with a Schur complement, started from a state dumped by run_tto_ba_iterations.py.

Same objective as the pycolmap BA of run_tto_ba.py: reprojection error of SIMPLE_PINHOLE / SIMPLE_RADIAL cameras under a
Cauchy loss, poses of all images but the first, the focal length (and radial parameter) of every camera and all points
refined, the principal point fixed. Unlike pycolmap it exposes every iteration, so the cost and the poses can be followed.

    state = load(npz)                      # the problem before the BA
    history = solve(state, max_iters=100)  # list of dicts: iteration, cost, rms, step accepted, poses, focal, time
"""
import os
import time

import numpy as np
import torch
from torch.func import jacfwd, vmap

DT = torch.float64
DEV = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("BA_DEVICE", "cuda") != "cpu" else "cpu")


def ar(*a):
    return torch.arange(*a, device=DEV)


def skew(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack([z, -w[..., 2], w[..., 1], w[..., 2], z, -w[..., 0], -w[..., 1], w[..., 0], z], -1).reshape(*w.shape[:-1], 3, 3)


def rodrigues(w):
    """Rotation matrices [N, 3, 3] of rotation vectors [N, 3]."""
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    K = skew(w / th)
    th = th[..., None]
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def load(path, prefix="pre_"):
    d = path if isinstance(path, dict) else np.load(path, allow_pickle=True)  # a dumped file, or the state dict itself
    t = lambda k: torch.from_numpy(np.asarray(d[prefix + k])).to(DEV, DT)  # noqa: E731
    obs = t("obs")
    params = t("cam_params")
    radial = ["RADIAL" in str(m) for m in d[prefix + "cam_model"]]
    cam_ids = d[prefix + "cam_ids"]
    cam_index = {int(c): i for i, c in enumerate(cam_ids)}
    s = dict(
        img=obs[:, 0].long(), pt=obs[:, 1].long(), uv=obs[:, 2:4],
        R=t("poses")[:, :, :3].clone(), t=t("poses")[:, :, 3].clone(), X=t("xyz").clone(),
        cam=torch.tensor([cam_index[int(c)] for c in d[prefix + "cam_of_img"]], device=DEV),
        f=params[:, 0].clone(), cx=params[:, 1], cy=params[:, 2],
        k=params[:, 3].clone() if params.shape[1] > 3 else torch.zeros(len(params), dtype=DT, device=DEV),
        refine_k=torch.tensor(radial, device=DEV),
        loss_scale=float(d["loss"][1]),
    )
    s["R0"], s["t0"] = s["R"].clone(), s["t"].clone()
    c = -torch.einsum("nji,nj->ni", s["R"], s["t"])  # camera centres
    s["tscale"] = float((c - c.mean(0)).norm(dim=1).mean().clamp_min(1e-9))  # the spread of the cameras: the unit of the translation prior
    return s


def project(R, t, f, k, cx, cy, X):
    Xc = torch.einsum("...ij,...j->...i", R, X) + t
    x, y = Xc[..., 0] / Xc[..., 2], Xc[..., 1] / Xc[..., 2]
    rad = 1 + k * (x * x + y * y)
    return torch.stack([f * x * rad + cx, f * y * rad + cy], -1), Xc[..., 2]


def residuals(s, R=None, t=None, f=None, k=None, X=None):
    R, t, f, k, X = (s[n] if v is None else v for n, v in (("R", R), ("t", t), ("f", f), ("k", k), ("X", X)))
    c = s["cam"][s["img"]]
    uv, z = project(R[s["img"]], t[s["img"]], f[c], k[c], s["cx"][c], s["cy"][c], X[s["pt"]])
    return uv - s["uv"], z


def cauchy(r, a):
    """Cost 0.5 * sum rho(|r|^2) with rho(s) = a^2 log(1 + s / a^2), and the weights rho'(s) of the IRLS step."""
    sq = (r * r).sum(-1)
    return 0.5 * (a * a * torch.log1p(sq / (a * a))).sum(), 1.0 / (1.0 + sq / (a * a))


def rotvec(R):
    """Rotation vectors [N, 3] of rotation matrices [N, 3, 3] (logarithm map)."""
    cos = ((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2).clamp(-1 + 1e-12, 1 - 1e-12)
    th = torch.acos(cos)
    v = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    f = torch.where(th < 1e-6, torch.full_like(th, 0.5), th / (2 * torch.sin(th).clamp_min(1e-12)))
    return v * f[:, None]


def prior_cost(s, R=None, t=None):
    """0.5 * (w_rot^2 |log(R R0^T)|^2 + (w_trans / spread)^2 |t - t0|^2), summed over the images but the first (anchor: the initial poses)."""
    w_rot, w_trans = s.get("prior", (0.0, 0.0))
    if w_rot <= 0 and w_trans <= 0:
        return torch.zeros((), dtype=DT, device=DEV)
    R, t = s["R"] if R is None else R, s["t"] if t is None else t
    rr = rotvec(R[1:] @ s["R0"][1:].transpose(1, 2))
    d = t[1:] - s["t0"][1:]
    return 0.5 * (w_rot ** 2 * (rr * rr).sum() + (w_trans / s["tscale"]) ** 2 * (d * d).sum())


def jac_fn(R0, t0, f0, k0, cx, cy, X0, uv):
    """Residual [2] as a function of v = (rotation increment 3, translation increment 3, focal increment, radial increment, point increment 3)."""
    def r(v):
        Xc0 = R0 @ (X0 + v[8:11])
        Xc = Xc0 + torch.linalg.cross(v[:3], Xc0) + t0 + v[3:6]
        x, y = Xc[0] / Xc[2], Xc[1] / Xc[2]
        rad = 1 + (k0 + v[7]) * (x * x + y * y)
        return torch.stack([(f0 + v[6]) * x * rad + cx, (f0 + v[6]) * y * rad + cy]) - uv
    return jacfwd(r)(torch.zeros(11, dtype=R0.dtype, device=R0.device))


@torch.no_grad()
def step(s, lam, chunk=400000):
    """One LM step at the current state: returns the proposed increments and the predicted cost reduction."""
    N, P, C = len(s["R"]), len(s["X"]), len(s["f"])
    img, pt = s["img"], s["pt"]
    M = len(img)
    c = s["cam"][img]
    r, _ = residuals(s)
    cost, w = cauchy(r, s["loss_scale"])
    cost = cost + prior_cost(s)
    J = torch.cat([vmap(jac_fn)(s["R"][img[i:i + chunk]], s["t"][img[i:i + chunk]], s["f"][c[i:i + chunk]], s["k"][c[i:i + chunk]],
                                s["cx"][c[i:i + chunk]], s["cy"][c[i:i + chunk]], s["X"][pt[i:i + chunk]], s["uv"][i:i + chunk]) for i in range(0, M, chunk)])
    Jpose, Jintr, Jx = J[:, :, :6], J[:, :, 6:8].clone(), J[:, :, 8:11]
    Jintr[:, :, 1] = Jintr[:, :, 1] * s["refine_k"][c].to(DT)[:, None]  # the radial parameter of a pinhole camera stays fixed
    Jpose = Jpose * (img != 0).to(DT)[:, None, None]  # the first image keeps its pose
    D = 6 * (N - 1) + 2 * C
    Jcam = torch.cat([Jpose, Jintr], -1)  # [M, 2, 8]
    idx = torch.cat([6 * (img[:, None] - 1) + ar(6)[None], (6 * (N - 1) + 2 * c)[:, None] + ar(2)[None]], 1)  # [M, 8]
    idx = torch.where(torch.cat([(img != 0)[:, None].expand(-1, 6), torch.ones(M, 2, dtype=torch.bool, device=DEV)], 1), idx, torch.full_like(idx, D))
    wj = w[:, None, None]
    Hcc = torch.zeros(D + 1, D + 1, dtype=DT, device=DEV)
    Hcc.index_put_((idx[:, :, None].expand(-1, -1, 8), idx[:, None, :].expand(-1, 8, -1)), (Jcam.transpose(1, 2) @ (wj * Jcam)), accumulate=True)
    gc = torch.zeros(D + 1, dtype=DT, device=DEV)
    gc.index_add_(0, idx.reshape(-1), (Jcam.transpose(1, 2) @ (w[:, None] * r)[:, :, None]).reshape(-1))
    w_rot, w_trans = s.get("prior", (0.0, 0.0))
    if (w_rot > 0 or w_trans > 0) and N > 1:
        a2 = (w_trans / s["tscale"]) ** 2
        rr = rotvec(s["R"][1:] @ s["R0"][1:].transpose(1, 2))
        d = s["t"][1:] - s["t0"][1:]
        rows_rot = (6 * ar(N - 1))[:, None] + ar(3)[None]
        rows_t = rows_rot + 3
        Hcc[rows_rot.reshape(-1), rows_rot.reshape(-1)] += w_rot ** 2
        Hcc[rows_t.reshape(-1), rows_t.reshape(-1)] += a2
        gc[rows_rot.reshape(-1)] += w_rot ** 2 * rr.reshape(-1)
        gc[rows_t.reshape(-1)] += a2 * d.reshape(-1)
    V = torch.zeros(P, 3, 3, dtype=DT, device=DEV).index_add_(0, pt, Jx.transpose(1, 2) @ (wj * Jx))
    gp = torch.zeros(P, 3, dtype=DT, device=DEV).index_add_(0, pt, (Jx.transpose(1, 2) @ (w[:, None] * r)[:, :, None])[:, :, 0])
    W = Jcam.transpose(1, 2) @ (wj * Jx)  # [M, 8, 3]

    eps = 1e-9
    Vd = V + lam * torch.diag_embed(torch.diagonal(V, dim1=1, dim2=2).clamp_min(eps)) + eps * torch.eye(3, dtype=DT, device=DEV)
    Vinv = torch.linalg.inv(Vd)
    U = W @ Vinv[pt]  # [M, 8, 3]
    # Schur complement: S = Hcc - sum over points of W_p Vinv W_p^T, over all pairs of observations of the same point
    order = torch.argsort(pt)
    pt_s, counts = pt[order], torch.bincount(pt, minlength=P)
    start = torch.cumsum(counts, 0) - counts
    S = Hcc.clone()
    Ls = counts[pt_s]
    total_pairs = int(Ls.sum())
    pos = torch.cumsum(Ls, 0) - Ls
    for a0 in range(0, M, 60000):
        a1 = min(M, a0 + 60000)
        a_rep = torch.repeat_interleave(ar(a0, a1), Ls[a0:a1])
        within = ar(len(a_rep)) - (pos[a0:a1] - pos[a0])[a_rep - a0]
        b = order[start[pt_s[a_rep]] + within]
        a = order[a_rep]
        for j in range(0, len(a), 500000):
            aa, bb = a[j:j + 500000], b[j:j + 500000]
            S.index_put_((idx[aa][:, :, None].expand(-1, -1, 8), idx[bb][:, None, :].expand(-1, 8, -1)), -(U[aa] @ W[bb].transpose(1, 2)), accumulate=True)
    Sd = S[:D, :D] + lam * torch.diag(torch.diagonal(Hcc[:D, :D]).clamp_min(eps))
    # right-hand side: -gc + sum_m U_m gp[p_m] (accumulated into the camera-side rows)
    corr = torch.zeros(D + 1, dtype=DT, device=DEV).index_add_(0, idx.reshape(-1), (U @ gp[pt][:, :, None])[:, :, 0].reshape(-1))
    rhs = -gc[:D] + corr[:D]
    dc = torch.linalg.solve(Sd, rhs)
    dc_full = torch.cat([dc, torch.zeros(1, dtype=DT, device=DEV)])
    dcam = dc_full[idx]  # [M, 8]
    dp = -torch.einsum("pij,pj->pi", Vinv, gp + torch.zeros(P, 3, dtype=DT, device=DEV).index_add_(0, pt, (W.transpose(1, 2) @ dcam[:, :, None])[:, :, 0]))
    # predicted reduction of the IRLS surrogate
    lin = r + (Jcam @ dcam[:, :, None])[:, :, 0] + (Jx @ dp[pt][:, :, None])[:, :, 0]
    pred = 0.5 * (w * ((r * r).sum(-1) - (lin * lin).sum(-1))).sum()
    if (w_rot > 0 or w_trans > 0) and N > 1:
        dw, dt = dc[:6 * (N - 1)].reshape(N - 1, 6)[:, :3], dc[:6 * (N - 1)].reshape(N - 1, 6)[:, 3:]
        pred = pred + 0.5 * (w_rot ** 2 * ((rr * rr).sum() - ((rr + dw) ** 2).sum()) + a2 * ((d * d).sum() - ((d + dt) ** 2).sum()))
    pose_d = torch.zeros(N, 6, dtype=DT, device=DEV)
    pose_d[1:] = dc[:6 * (N - 1)].reshape(N - 1, 6)
    intr_d = dc[6 * (N - 1):].reshape(C, 2)
    gnorm = float(max(gc[:D].abs().max(), gp.abs().max()))
    return cost, pose_d, intr_d, dp, pred, gnorm


def apply(s, pose_d, intr_d, dp):
    n = dict(s)
    n["R"] = rodrigues(pose_d[:, :3]) @ s["R"]
    n["t"] = s["t"] + pose_d[:, 3:]
    n["f"] = s["f"] + intr_d[:, 0]
    n["k"] = s["k"] + intr_d[:, 1] * s["refine_k"].to(DT)
    n["X"] = s["X"] + dp
    return n


def rms(cost, n_res):
    return float(np.sqrt(float(cost) / n_res))  # as the bundle adjustment report of COLMAP: sqrt(cost / residuals)


def solve(s, max_iters=100, lam=1e-4, verbose=False, stop=None, prior=(0.0, 0.0)):
    """Run LM; every iteration (accepted or not) is recorded. stop(history) -> True ends early."""
    s = dict(s, prior=tuple(prior))  # (rotation weight per rad, translation weight per camera spread); 0 = no prior
    hist = []
    n_res = 2 * len(s["img"])
    t0 = time.perf_counter()
    cost0, _ = cauchy(residuals(s)[0], s["loss_scale"])
    cost0 = cost0 + prior_cost(s)
    hist.append(dict(it=0, cost=float(cost0), rms=rms(cost0, n_res), accepted=True, lam=lam, R=s["R"].cpu(), t=s["t"].cpu(), f=s["f"].cpu(), time=0.0))
    nu = 2.0
    for it in range(1, max_iters + 1):
        cost, pose_d, intr_d, dp, pred, gnorm = step(s, lam)
        new = apply(s, pose_d, intr_d, dp)
        cost_new, _ = cauchy(residuals(new)[0], s["loss_scale"])
        cost_new = cost_new + prior_cost(new)
        ratio = float(cost - cost_new) / max(float(pred), 1e-30)
        ok = float(cost_new) < float(cost) and ratio > 0
        if ok:
            s = new
            lam = max(lam * max(1 / 3, 1 - (2 * ratio - 1) ** 3), 1e-12)
            nu = 2.0
        else:
            lam *= nu
            nu *= 2
        cur = cost_new if ok else cost
        hist.append(dict(it=it, cost=float(cur), rms=rms(cur, n_res), accepted=ok, lam=lam, gnorm=gnorm, R=s["R"].cpu(), t=s["t"].cpu(), f=s["f"].cpu(), time=time.perf_counter() - t0))
        if verbose and (it <= 10 or it % 10 == 0):
            print(f"it {it:3d} cost {float(cur):.6f} rms {rms(cur, n_res):.6f} {'ok ' if ok else 'rej'} lam {lam:.1e} |g| {gnorm:.2e}")
        if stop and stop(hist):
            break
    return hist, s
