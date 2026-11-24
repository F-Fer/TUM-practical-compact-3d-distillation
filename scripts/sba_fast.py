"""Fast sparse (track-based) bundle adjustment on DA3 predictions, everything on the GPU: how fast can the standard
pipeline be made? (knowledge/ideas/fast-sparse-ba.md)

    .venv/bin/python scripts/sba_fast.py <src work dir> <dst work dir> [key=value ...]

Pipeline per scene (all timed except image decoding, like run_tto_pba.py's opt_s):
  1. pairs: each frame's `nbr` neighbours by DA3's covisibility (run_tto_pba.select_neighbors, mode `nbr_mode`)
  2. features: SuperPoint / ALIKED / DISK (`feat`, `max_kp`) on every frame, at the prediction's resolution
     (`feat=roma`: RoMa v2 query-frame tracks from scripts/tracks_roma.py instead of steps 2-4)
  3. matching: LightGlue on all pairs, batched (`lg_batch` pairs per call)
  4. tracks: greedy union-find over the matches (`track_mode`, see build_tracks)
  5. points: unprojected with DA3's depth and cameras, mean over the track's observations
  6. BA: own Levenberg-Marquardt on the GPU (analytic Jacobians, Schur complement, Cauchy loss), poses (frame 0
     fixed), one shared focal length, one shared radial coefficient (`radial`), all points; `iters` iterations
  7. depth: DA3's depth scaled per frame to the BA points (`depth_fit=frame`), or by the camera spread (`scene`)

Writes <dst>/model_results/... (results.npz as the benchmark reads it) and <dst>/metric_results/<ds>_pba.json (runs before 2026-11-24 17:30: _ba.json; pose AUC
before / after, timings per stage: feat_s, match_s, track_s (= 1-5), ba_s, opt_s, sizes).
"""

import json
import os
import shutil
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_anything_3.bench.evaluator import Evaluator  # noqa: E402
from depth_anything_3.bench.utils import compute_pose  # noqa: E402
from depth_anything_3.utils.geometry import as_homogeneous  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = {"hiroom": 10, "7scenes": 2, "scannetpp": 3, "dtu64": 4, "eth3d": 3}

DEFAULTS = {
    "subset": "dev",  # dev | held-out | all
    "datasets": "",
    "scenes": "",
    "feat": "superpoint",  # superpoint | aliked | disk | roma
    "max_kp": 2048,
    "det_thresh": 0.0005,  # SuperPoint detection threshold
    "nbr": 10,  # neighbours per frame (pairs = union over frames)
    "nbr_mode": "spread",  # spread | top
    "lg_batch": 32,  # LightGlue pairs per batched call
    "lg_filter": 0.1,  # LightGlue match score threshold
    "lg_depth_conf": 0.95,  # LightGlue early stopping (-1 = off)
    "lg_mp": 0,  # LightGlue in mixed precision (fp16 autocast)
    "min_len": 2,  # minimum track length
    "max_len": 0,  # tracks longer than this are dropped (0 = no limit)
    "track_mode": "greedy",  # greedy | pairs | cc (see build_tracks)
    "iters": 25,  # LM iterations
    "lam": 1e-4,
    "loss_scale": 0.5,  # Cauchy scale in px (prediction resolution)
    "radial": 1,  # 1 = one shared radial coefficient
    "refine_focal": 1,
    "prior": 0.0,  # weight of a prior on DA3's poses (rotation per rad, translation per camera spread), 0 = off
    "depth_fit": "frame",  # frame | scene
    "roma_setting": "turbo",
    "query_frames": 8,
    "max_query_pts": 4096,
    "vis_thresh": 0.5,
    "extra_queries": 4,
    "min_visible": 256,
    "pairs": 0,
    "query_offset": 0,
    "cache": "",  # feat=roma: directory with cached RoMa tracks per scene (written if missing; the time of the run that made them is reused)
    "dtype": "float32",  # BA Jacobians / Schur products (the reduced camera system is always float64)
    "seed": 0,
    "save": 1,
}


def parse(argv):
    cfg = dict(DEFAULTS)
    for a in argv:
        k, _, v = a.partition("=")
        if k not in DEFAULTS:
            sys.exit(f"unknown option {k}, expected one of {list(DEFAULTS)}")
        cfg[k] = type(DEFAULTS[k])(v)
    return cfg


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ------------------------------------------------------------------ front-end

_EXTRACTORS, _MATCHERS = {}, {}


def extractor(cfg):
    key = (cfg["feat"], cfg["max_kp"], cfg["det_thresh"])
    if key not in _EXTRACTORS:
        import lightglue

        if cfg["feat"] == "superpoint":
            m = lightglue.SuperPoint(max_num_keypoints=cfg["max_kp"], detection_threshold=cfg["det_thresh"])
        elif cfg["feat"] == "aliked":
            m = lightglue.ALIKED(max_num_keypoints=cfg["max_kp"])
        elif cfg["feat"] == "disk":
            m = lightglue.DISK(max_num_keypoints=cfg["max_kp"])
        else:
            raise ValueError(cfg["feat"])
        _EXTRACTORS[key] = m.eval().cuda()
    return _EXTRACTORS[key]


def matcher(cfg):
    key = (cfg["feat"], cfg["lg_filter"], cfg["lg_depth_conf"], cfg["lg_mp"])
    if key not in _MATCHERS:
        import lightglue

        _MATCHERS[key] = lightglue.LightGlue(features=cfg["feat"], filter_threshold=cfg["lg_filter"],
                                             depth_confidence=cfg["lg_depth_conf"], width_confidence=-1,
                                             mp=bool(cfg["lg_mp"])).eval().cuda()
    return _MATCHERS[key]


@torch.no_grad()
def features(imgs, cfg):
    """imgs [N, 3, H, W] float on the GPU -> keypoints [N, K, 2], descriptors [N, K, D], valid counts [N]."""
    ex = extractor(cfg)
    kps, descs, n = [], [], []
    for i in range(len(imgs)):
        f = ex.extract(imgs[i], resize=None)
        kps.append(f["keypoints"][0])
        descs.append(f["descriptors"][0])
        n.append(len(f["keypoints"][0]))
    K = max(n)
    kp = torch.full((len(imgs), K, 2), -1e4, device="cuda")  # padding far outside the image
    de = torch.zeros(len(imgs), K, descs[0].shape[1], device="cuda")
    for i in range(len(imgs)):
        kp[i, : n[i]] = kps[i]
        de[i, : n[i]] = descs[i]
    return kp, de, torch.tensor(n, device="cuda")


@torch.no_grad()
def match_pairs(kp, de, n, pairs, size, cfg):
    """pairs [E, 2] -> list of (i, j, idx_i [m], idx_j [m]) on the CPU."""
    lg = matcher(cfg)
    H, W = size
    out = []
    B = cfg["lg_batch"]
    img_size = torch.tensor([[W, H]], device="cuda", dtype=torch.float32)
    for s in range(0, len(pairs), B):
        p = pairs[s: s + B]
        i, j = torch.from_numpy(p[:, 0]).cuda(), torch.from_numpy(p[:, 1]).cuda()
        data = {"image0": {"keypoints": kp[i], "descriptors": de[i], "image_size": img_size.expand(len(p), -1)},
                "image1": {"keypoints": kp[j], "descriptors": de[j], "image_size": img_size.expand(len(p), -1)}}
        r = lg(data)
        m0 = r["matches0"]  # [b, K], -1 = none
        valid = (m0 > -1) & (torch.arange(m0.shape[1], device="cuda")[None] < n[i][:, None])
        valid &= torch.gather(torch.arange(kp.shape[1], device="cuda")[None].expand(len(p), -1) < n[j][:, None], 1, m0.clamp_min(0))
        b, a = torch.nonzero(valid, as_tuple=True)
        bb = b.cpu().numpy()
        aa = a.cpu().numpy()
        cc = m0[b, a].cpu().numpy()
        for k in range(len(p)):
            sel = bb == k
            out.append((int(p[k, 0]), int(p[k, 1]), aa[sel], cc[sel]))
    return out


def build_tracks(matches, n, N, min_len, max_len, mode="greedy"):
    """Observations (track, frame, keypoint) from pairwise matches.

    greedy: union-find that skips a union when the two tracks share a frame (no track sees a frame twice; COLMAP-like)
    pairs: every match is its own two-view point (no track building)
    cc: connected components, components that see a frame twice are dropped entirely (loses most matches)"""
    offs = np.concatenate([[0], np.cumsum(n)])
    if mode == "pairs":
        fr, kp = [], []
        for i, j, a, b in matches:
            fr.append(np.stack([np.full(len(a), i), np.full(len(b), j)], 1))
            kp.append(np.stack([a, b], 1))
        if not fr:
            return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
        fr, kp = np.concatenate(fr), np.concatenate(kp)
        tr = np.repeat(np.arange(len(fr)), 2)
        return tr, fr.reshape(-1), kp.reshape(-1)
    if mode == "greedy":
        T = int(offs[-1])
        parent = list(range(T))
        frames = [1 << int(f) for f in np.repeat(np.arange(N), n)]

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j, a, b in matches:
            for u, v in zip((offs[i] + a).tolist(), (offs[j] + b).tolist()):
                ru, rv = find(u), find(v)
                if ru == rv or frames[ru] & frames[rv]:
                    continue
                parent[rv] = ru
                frames[ru] |= frames[rv]
        roots = np.array([find(x) for x in range(T)])
        used = np.zeros(T, bool)
        for i, j, a, b in matches:
            used[offs[i] + a] = True
            used[offs[j] + b] = True
        node = np.nonzero(used)[0]
        frame = np.searchsorted(offs, node, side="right") - 1
        kpi = node - offs[frame]
        _, tr = np.unique(roots[node], return_inverse=True)
        length = np.bincount(tr)
        keep = length >= min_len
        if max_len > 0:
            keep &= length <= max_len
        m = keep[tr]
        _, tr = np.unique(tr[m], return_inverse=True)
        return tr, frame[m], kpi[m]
    offs = np.concatenate([[0], np.cumsum(n)])
    rows, cols = [], []
    for i, j, a, b in matches:
        rows.append(offs[i] + a)
        cols.append(offs[j] + b)
    if not rows:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    T = int(offs[-1])
    g = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(T, T))
    _, lab = connected_components(g, directed=False)
    node = np.unique(np.concatenate([rows, cols]))
    frame = np.searchsorted(offs, node, side="right") - 1
    kpi = node - offs[frame]
    tr = lab[node]
    # drop tracks that contain two keypoints of the same frame, and short / long ones
    key = tr * N + frame
    uk, cnt = np.unique(key, return_counts=True)
    bad = np.unique(uk[cnt > 1] // N)
    length = np.bincount(tr, minlength=lab.max() + 1)
    keep = (length >= min_len) & ~np.isin(np.arange(len(length)), bad)
    if max_len > 0:
        keep &= length <= max_len
    m = keep[tr]
    tr, frame, kpi = tr[m], frame[m], kpi[m]
    _, tr = np.unique(tr, return_inverse=True)
    return tr, frame, kpi


def bilinear(depth, f, uv):
    """depth [N, H, W] tensor, frame idx [M], uv [M, 2] -> [M]."""
    N, H, W = depth.shape
    u = uv[:, 0].clamp(0, W - 1)
    v = uv[:, 1].clamp(0, H - 1)
    u0, v0 = u.floor().long().clamp(max=W - 2), v.floor().long().clamp(max=H - 2)
    du, dv = u - u0, v - v0
    d = depth
    return ((1 - du) * (1 - dv) * d[f, v0, u0] + du * (1 - dv) * d[f, v0, u0 + 1] + (1 - du) * dv * d[f, v0 + 1, u0]
            + du * dv * d[f, v0 + 1, u0 + 1])


# ------------------------------------------------------------------ BA

def skew(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack([z, -w[..., 2], w[..., 1], w[..., 2], z, -w[..., 0], -w[..., 1], w[..., 0], z], -1).reshape(*w.shape[:-1], 3, 3)


def rodrigues(w):
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    K = skew(w / th)
    th = th[..., None]
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def rotvec(R):
    cos = ((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    th = torch.acos(cos)
    v = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    f = torch.where(th < 1e-4, torch.full_like(th, 0.5), th / (2 * torch.sin(th).clamp_min(1e-12)))
    return v * f[:, None]


def sbmm(A, B):
    """Batched product of tiny matrices [M, a, k] @ [M, k, b] as k broadcast multiply-adds (cuBLAS batched GEMM is slow
    for 2x3 / 8x2 blocks: ~27 ms per call at 600k observations on the 3080, this ~1 ms)."""
    out = A[:, :, 0, None] * B[:, None, 0, :]
    for i in range(1, A.shape[2]):
        out = out + A[:, :, i, None] * B[:, None, i, :]
    return out


def mm_split(A, B):
    """A @ B.T for float64 A, B with three float32 products (hi/lo split): ~float64 accuracy at float32 speed."""
    Ah, Bh = A.float(), B.float()
    Al, Bl = (A - Ah.double()).float(), (B - Bh.double()).float()
    return (Ah @ Bh.T).double() + ((Ah @ Bl.T) + (Al @ Bh.T)).double()


class BA:
    """Reprojection BA: poses R, t [N] (frame 0 fixed), shared focal f and radial k, points X [P]; obs (img, pt, uv)."""

    def __init__(self, R, t, f, k, cx, cy, X, img, pt, uv, cfg):
        dt = getattr(torch, cfg["dtype"])
        self.dt = dt
        self.R, self.t, self.X = R.to(dt), t.to(dt), X.to(dt)
        self.f, self.k = torch.tensor(float(f), dtype=dt, device="cuda"), torch.tensor(float(k), dtype=dt, device="cuda")
        self.cx, self.cy = float(cx), float(cy)
        self.img, self.pt, self.uv = img, pt, uv.to(dt)
        self.a = cfg["loss_scale"]
        self.radial, self.refine_focal = bool(cfg["radial"]), bool(cfg["refine_focal"])
        self.N, self.P = len(R), len(X)
        self.prior = cfg["prior"]
        self.R0, self.t0 = self.R.clone(), self.t.clone()
        c = -torch.einsum("nji,nj->ni", self.R, self.t)
        self.tscale = float((c - c.mean(0)).norm(dim=1).mean().clamp_min(1e-9))
        self.order = torch.argsort(pt, stable=True)
        self.counts = torch.bincount(pt, minlength=self.P)
        self.img_order = torch.argsort(img, stable=True)
        self.img_counts = torch.bincount(img, minlength=self.N)

    def residuals(self, R=None, t=None, f=None, k=None, X=None):
        R = self.R if R is None else R
        t = self.t if t is None else t
        f = self.f if f is None else f
        k = self.k if k is None else k
        X = self.X if X is None else X
        Xc = (R[self.img] * X[self.pt][:, None, :]).sum(-1) + t[self.img]
        z = Xc[:, 2]
        z = torch.where(z.abs() < 1e-6, torch.full_like(z, 1e-6), z)  # no clamp to positive: points behind a camera keep finite Jacobians (as ba_lm.py)
        x, y = Xc[:, 0] / z, Xc[:, 1] / z
        rad = 1 + k * (x * x + y * y)
        return torch.stack([f * x * rad + self.cx, f * y * rad + self.cy], -1) - self.uv, Xc, x, y, rad, z

    def cost(self, r):
        sq = (r * r).sum(-1)
        a2 = self.a * self.a
        return 0.5 * (a2 * torch.log1p(sq / a2)).sum(dtype=torch.float64), 1.0 / (1.0 + sq / a2)

    def prior_cost(self, R=None, t=None):
        if self.prior <= 0:
            return 0.0
        R = self.R if R is None else R
        t = self.t if t is None else t
        rr = rotvec((R[1:] @ self.R0[1:].transpose(1, 2)).double())
        d = (t[1:] - self.t0[1:]).double()
        return 0.5 * self.prior ** 2 * ((rr * rr).sum() + (d * d).sum() / self.tscale ** 2)

    def total(self, **kw):
        r = self.residuals(**kw)[0]
        c, _ = self.cost(r)
        return c + self.prior_cost(kw.get("R"), kw.get("t"))

    @torch.no_grad()
    def step(self, lam):
        """One LM step. Per-observation Jacobians in self.dt (float32 by default); the normal equations are assembled in
        float64 from per-image and per-point block sums (no scatter of millions of entries into a few addresses), the
        camera-by-point block W is dense per chunk of points and S = Hcc - W V^-1 W^T is one matrix product."""
        N, P = self.N, self.P
        img, pt = self.img, self.pt
        M = len(img)
        dd = torch.float64
        r, Xc, x, y, rad, z = self.residuals()
        cost, w = self.cost(r)
        cost = cost + self.prior_cost()
        f, k = self.f, self.k
        r2 = x * x + y * y
        dxr_dx = rad + 2 * k * x * x
        dxr_dy = 2 * k * x * y
        dyr_dy = rad + 2 * k * y * y
        iz = 1 / z
        A = torch.stack([f * dxr_dx, f * dxr_dy, f * dxr_dy, f * dyr_dy], -1).view(M, 2, 2)
        Bm = torch.zeros(M, 2, 3, dtype=self.dt, device="cuda")
        Bm[:, 0, 0], Bm[:, 0, 2], Bm[:, 1, 1], Bm[:, 1, 2] = iz, -x * iz, iz, -y * iz
        Jxc = sbmm(A, Bm)  # [M, 2, 3]
        RX = Xc - self.t[img]
        Jpose = torch.cat([-sbmm(Jxc, skew(RX)), Jxc], -1)  # rotation (left increment), translation
        Jf = torch.stack([x * rad, y * rad], -1) * float(self.refine_focal)
        Jk = torch.stack([f * x * r2, f * y * r2], -1) * float(self.radial)
        Jc = torch.cat([Jpose, Jf[:, :, None], Jk[:, :, None]], -1)  # [M, 2, 8], per-observation products in self.dt,
        Jx = sbmm(Jxc, self.R[img])  # [M, 2, 3]                                sums in float64
        rd, wd = r, w
        JcW = Jc.transpose(1, 2) * wd[:, None, None]  # [M, 8, 2]
        # camera blocks, summed per image
        io, ic = self.img_order, self.img_counts
        Himg = torch.segment_reduce(sbmm(JcW, Jc)[io].reshape(M, 64).to(dd), "sum", lengths=ic).view(N, 8, 8)
        gimg = torch.segment_reduce(sbmm(JcW, rd[:, :, None])[io].reshape(M, 8).to(dd), "sum", lengths=ic)
        D = 6 * (N - 1) + 2
        Hcc = torch.zeros(D, D, dtype=dd, device="cuda")
        r6 = (6 * torch.arange(N - 1, device="cuda"))[:, None] + torch.arange(6, device="cuda")[None]  # rows of images 1..N-1
        ri = torch.tensor([D - 2, D - 1], device="cuda")
        Hcc[r6[:, :, None], r6[:, None, :]] = Himg[1:, :6, :6]
        Hcc[r6[:, :, None], ri[None, None, :]] = Himg[1:, :6, 6:]
        Hcc[ri[None, :, None], r6[:, None, :]] = Himg[1:, 6:, :6]
        Hcc[ri[:, None], ri[None, :]] = Himg[:, 6:, 6:].sum(0)
        gc = torch.zeros(D, dtype=dd, device="cuda")
        gc[r6.reshape(-1)] = gimg[1:, :6].reshape(-1)
        gc[ri] = gimg[:, 6:].sum(0)
        if self.prior > 0 and N > 1:
            a2 = 1.0 / self.tscale ** 2
            rr = rotvec((self.R[1:] @ self.R0[1:].transpose(1, 2)).double())
            d = (self.t[1:] - self.t0[1:]).double()
            p2 = self.prior ** 2
            rot, tr = r6[:, :3].reshape(-1), r6[:, 3:].reshape(-1)
            Hcc[rot, rot] += p2
            Hcc[tr, tr] += p2 * a2
            gc[rot] += p2 * rr.reshape(-1)
            gc[tr] += p2 * a2 * d.reshape(-1)
        # point blocks, summed per point
        po, pc = self.order, self.counts
        JxW = Jx.transpose(1, 2) * wd[:, None, None]
        V = torch.segment_reduce(sbmm(JxW, Jx)[po].reshape(M, 9).to(dd), "sum", lengths=pc).view(P, 3, 3)
        gp = torch.segment_reduce(sbmm(JxW, rd[:, :, None])[po].reshape(M, 3).to(dd), "sum", lengths=pc)
        Wb = sbmm(JcW, Jx).to(dd)  # [M, 8, 3]
        eps = 1e-12
        Vd = V + lam * torch.diag_embed(torch.diagonal(V, dim1=1, dim2=2).clamp_min(eps)) + 1e-9 * torch.eye(3, dtype=dd, device="cuda")
        Vinv = torch.linalg.inv(Vd)
        Wi_all = torch.segment_reduce(Wb[po][:, 6:, :].reshape(M, 6), "sum", lengths=pc).view(P, 2, 3)  # intrinsics rows per point
        nz = img != 0
        Pc = max(1, int(6.0e7 // (3 * D)))  # points per dense chunk (~0.5 GB per float64 block)

        def chunk_W(p0, p1):
            m = nz & (pt >= p0) & (pt < p1)
            Wd6 = torch.zeros(N - 1, 6, p1 - p0, 3, dtype=dd, device="cuda")
            Wd6[img[m] - 1, :, pt[m] - p0, :] = Wb[m, :6, :]
            return torch.cat([Wd6.reshape(6 * (N - 1), p1 - p0, 3), Wi_all[p0:p1].permute(1, 0, 2)], 0)  # [D, p, 3]

        S = Hcc.clone()
        rhs = -gc.clone()
        for p0 in range(0, P, Pc):
            p1 = min(P, p0 + Pc)
            Wd = chunk_W(p0, p1)
            Vi = Vinv[p0:p1]
            Ud = Wd[:, :, 0:1] * Vi[None, :, 0] + Wd[:, :, 1:2] * Vi[None, :, 1] + Wd[:, :, 2:3] * Vi[None, :, 2]  # elementwise (fp64 bmm is slow)
            S -= mm_split(Ud.reshape(D, -1), Wd.reshape(D, -1))
            rhs += (Ud * gp[None, p0:p1]).sum((1, 2))
            del Wd, Ud
        Sd = S + lam * torch.diag(torch.diagonal(Hcc).clamp_min(eps))
        try:
            dc = torch.linalg.solve(Sd, rhs)
        except RuntimeError:
            dc = torch.linalg.lstsq(Sd, rhs[:, None]).solution[:, 0]
        WTdc = torch.zeros(P, 3, dtype=dd, device="cuda")
        for p0 in range(0, P, Pc):
            p1 = min(P, p0 + Pc)
            WTdc[p0:p1] = (chunk_W(p0, p1) * dc[:, None, None]).sum(0)
        dp = -torch.einsum("pij,pj->pi", Vinv, gp + WTdc)
        dpose = torch.zeros(N, 6, dtype=dd, device="cuda")
        dpose[1:] = dc[: 6 * (N - 1)].view(N - 1, 6)
        dcam = torch.cat([dpose[img], dc[D - 2:].expand(M, 2)], 1).to(self.dt)  # [M, 8]
        lin = rd + (Jc * dcam[:, None, :]).sum(-1) + (Jx * dp[pt].to(self.dt)[:, None, :]).sum(-1)
        pred = 0.5 * (wd * ((rd * rd).sum(-1) - (lin * lin).sum(-1))).sum(dtype=dd)
        return cost, dpose.to(self.dt), dc[D - 2].to(self.dt), dc[D - 1].to(self.dt), dp.to(self.dt), pred

    def solve(self, iters, lam):
        hist = []
        nu = 2.0
        for it in range(iters):
            cost, pose_d, df, dk, dp, pred = self.step(lam)
            # the rotation increment is applied about the camera centre frame: R <- exp(w) R, t <- t + dt (as ba_lm.py)
            new = dict(R=rodrigues(pose_d[:, :3]) @ self.R, t=self.t + pose_d[:, 3:], f=self.f + df, k=self.k + dk, X=self.X + dp)
            cn = self.total(**new)
            ratio = float(cost - cn) / max(float(pred), 1e-30)
            ok = bool(cn < cost) and ratio > 0
            if ok:
                self.R, self.t, self.f, self.k, self.X = new["R"], new["t"], new["f"], new["k"], new["X"]
                lam = max(lam * max(1 / 3, 1 - (2 * ratio - 1) ** 3), 1e-12)
                nu = 2.0
            else:
                lam *= nu
                nu *= 2
            hist.append((float(cn if ok else cost), ok))
        return hist


# ------------------------------------------------------------------ scene

def load_images(files, H, W):
    imgs = np.stack([cv2.resize(cv2.imread(str(f))[..., ::-1], (W, H), interpolation=cv2.INTER_AREA) for f in files])
    return torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div(255)


_WARM = set()


def warm_up(imgs, cfg):
    """Load the networks and run them once before the first timed scene (model loading and CUDA/cuDNN set-up are not
    TTO time; run_tto_pba.py's first scene is replaced by the median in the plots for the same reason)."""
    key = (cfg["feat"], cfg["max_kp"], cfg["roma_setting"])
    if key in _WARM:
        return
    g = imgs[:2].cuda()
    with torch.no_grad():
        if cfg["feat"] == "roma":
            from tracks_roma import _roma

            _roma(cfg["roma_setting"]).match(g[:1], g[1:2])
        else:
            kp, de, n = features(g, cfg)
            match_pairs(kp, de, n, np.array([[0, 1]]), imgs.shape[2:], cfg)
    sync()
    _WARM.add(key)


def refine_scene(files, pred, cfg):
    import run_tto_pba as rp

    depth, ext, K = pred["depth"].astype(np.float32), pred["extrinsics"][:, :3].astype(np.float64), pred["intrinsics"].astype(np.float64)
    N, H, W = depth.shape
    imgs = load_images(files, H, W)  # decoding not timed
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    warm_up(imgs, cfg)
    info = {"num_frames": N}
    sync()
    t0 = time.perf_counter()
    dev = "cuda"
    dep = torch.from_numpy(depth).to(dev)
    Rt = torch.from_numpy(ext).float().to(dev)
    Kt = torch.from_numpy(K).float().to(dev)
    if cfg["feat"] == "roma":
        from tracks_roma import track_roma

        rcfg = {k: cfg[k] for k in ("roma_setting", "query_frames", "max_query_pts", "vis_thresh", "extra_queries", "min_visible", "pairs", "query_offset")}
        cf = os.path.join(cfg["cache"], cfg["_scene_key"] + ".npz") if cfg["cache"] else ""
        if cf and os.path.exists(cf):
            c = np.load(cf)
            tracks, vis = c["tracks"], c["vis"]
            t0 -= float(c["match_s"])
        else:
            tracks, vis, _, _, _ = track_roma(files, pred, rcfg, torch.device(dev))
        sync()
        info["feat_s"] = 0.0
        info["match_s"] = time.perf_counter() - t0
        if cf and not os.path.exists(cf):
            os.makedirs(os.path.dirname(cf), exist_ok=True)
            np.savez(cf, tracks=tracks, vis=vis, match_s=info["match_s"])
        ok = vis > cfg["vis_thresh"]
        inside = (tracks[..., 0] >= 0) & (tracks[..., 0] <= W - 1) & (tracks[..., 1] >= 0) & (tracks[..., 1] <= H - 1)
        ok &= inside
        ok[:, ok.sum(0) < cfg["min_len"]] = False
        f_idx, p_idx = np.nonzero(ok)
        keep = np.unique(p_idx)
        remap = -np.ones(tracks.shape[1], int)
        remap[keep] = np.arange(len(keep))
        tr, frame = remap[p_idx], f_idx
        uv = torch.from_numpy(tracks[f_idx, p_idx]).float().to(dev)
        info["num_pairs"] = 0
    else:
        nb = rp.select_neighbors(dep, Rt[:, :, :3], Rt[:, :, 3], Kt, cfg["nbr"], 0.05, mode=cfg["nbr_mode"]).cpu().numpy()
        pairs = np.unique(np.sort(np.stack([np.repeat(np.arange(N), nb.shape[1]), nb.reshape(-1)], 1), 1), axis=0)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        g = imgs.to(dev)
        kp, de, n = features(g, cfg)
        sync()
        info["feat_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        matches = match_pairs(kp, de, n, pairs, (H, W), cfg)
        sync()
        info["match_s"] = time.perf_counter() - t1
        tr, frame, kpi = build_tracks(matches, n.cpu().numpy(), N, cfg["min_len"], cfg["max_len"], cfg["track_mode"])
        uv = kp[torch.from_numpy(frame).to(dev), torch.from_numpy(kpi).to(dev)]
        info["num_pairs"] = int(len(pairs))
        info["num_matches"] = int(sum(len(m[2]) for m in matches))
    P = int(tr.max()) + 1 if len(tr) else 0
    info.update(num_tracks=P, num_obs=int(len(tr)))
    if P < 16:
        info.update(track_s=time.perf_counter() - t0, ba_s=0.0, opt_s=time.perf_counter() - t0, ba=0)
        return pred["extrinsics"], pred["intrinsics"], depth, info
    fr = torch.from_numpy(frame).long().to(dev)
    ptt = torch.from_numpy(tr).long().to(dev)
    # points from DA3: unproject every observation, mean per track
    d = bilinear(dep, fr, uv)
    ray = torch.stack([(uv[:, 0] - Kt[fr, 0, 2]) / Kt[fr, 0, 0], (uv[:, 1] - Kt[fr, 1, 2]) / Kt[fr, 1, 1], torch.ones_like(d)], -1) * d[:, None]
    Xw = torch.einsum("mji,mj->mi", Rt[fr, :, :3], ray - Rt[fr, :, 3])
    X = torch.zeros(P, 3, device=dev).index_add_(0, ptt, Xw) / torch.bincount(ptt, minlength=P)[:, None].float()
    sync()
    info["track_s"] = time.perf_counter() - t0
    t2 = time.perf_counter()
    f0 = float(np.median(K[:, 0, 0]))
    ba = BA(Rt[:, :, :3].double(), Rt[:, :, 3].double(), f0, 0.0, float(np.median(K[:, 0, 2])), float(np.median(K[:, 1, 2])), X.double(),
            fr, ptt, uv, cfg)
    c0 = float(ba.total())
    hist = ba.solve(cfg["iters"], cfg["lam"])
    sync()
    info["ba_s"] = time.perf_counter() - t2
    info["opt_s"] = time.perf_counter() - t0
    info.update(ba=1, cost0=c0, cost=hist[-1][0] if hist else c0, focal=float(ba.f), radial=float(ba.k), focal0=f0,
                n_accept=sum(h[1] for h in hist))
    R, t = ba.R.double().cpu().numpy(), ba.t.double().cpu().numpy()
    ext_new = np.concatenate([R, t[:, :, None]], 2).astype(np.float32)
    Kn = K.copy()
    Kn[:, 0, 0] = Kn[:, 1, 1] = float(ba.f)
    Kn[:, 0, 2], Kn[:, 1, 2] = float(ba.cx), float(ba.cy)
    # depth: DA3's, scaled per frame to the BA points (their depth in the refined cameras over DA3's depth there)
    zc = (torch.einsum("mij,mj->mi", ba.R[fr].double(), ba.X[ptt].double()) + ba.t[fr].double())[:, 2]
    ratio = (zc / d.double().clamp_min(1e-6)).cpu().numpy()
    fr_np = frame
    scene_s = float(np.median(ratio[np.isfinite(ratio) & (ratio > 0)])) if np.any(ratio > 0) else 1.0
    scales = np.full(N, scene_s)
    if cfg["depth_fit"] == "frame":
        for i in range(N):
            ri = ratio[(fr_np == i) & (ratio > 0)]
            if len(ri) >= 20:
                scales[i] = np.median(ri)
    depth_new = depth * scales[:, None, None].astype(np.float32)
    return ext_new, Kn.astype(np.float32), depth_new, info


def main():
    src, dst, cfg = sys.argv[1], sys.argv[2], parse(sys.argv[3:])
    sys.stdout.reconfigure(line_buffering=True)
    os.chdir(REPO)
    os.makedirs(dst, exist_ok=True)
    json.dump(cfg, open(os.path.join(dst, "sba_config.json"), "w"), indent=2)
    datasets = [d for d in cfg["datasets"].split(",") if d] or list(DEV)
    for ds in datasets:
        data_dir = os.path.join(src, "model_results", ds)
        scenes = sorted(os.path.dirname(p) for p, _, _ in os.walk(data_dir) if os.path.basename(p) == "unposed")
        if cfg["subset"] == "dev":
            scenes = scenes[: DEV[ds]]
        elif cfg["subset"] == "held-out":
            scenes = scenes[DEV[ds]:]
        mp = os.path.join(dst, "metric_results", f"{ds}_pba.json")  # the name scripts/index_runs.py reads (Antonin's metric_file=pba)
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        metrics = json.load(open(mp)) if os.path.exists(mp) else {}
        metrics.pop("mean", None)
        for sd in scenes:
            scene = os.path.relpath(sd, data_dir)
            if (cfg["scenes"] and scene not in cfg["scenes"].split(",")) or scene in metrics:
                continue
            ex = os.path.join(sd, "unposed", "exports")
            gt = dict(np.load(os.path.join(ex, "gt_meta.npz"), allow_pickle=True))
            pred = dict(np.load(os.path.join(ex, "mini_npz", "results.npz")))
            try:
                ext, K, depth, info = refine_scene([str(f) for f in gt["image_files"]], pred, {**cfg, "_scene_key": f"{ds}/{scene}"})
            except torch.cuda.OutOfMemoryError as e:
                print(f"[sba] {ds} {scene} FAILED {e!r}")
                torch.cuda.empty_cache()
                continue
            if cfg["save"]:
                o = os.path.join(dst, "model_results", ds, scene, "unposed", "exports")
                os.makedirs(os.path.join(o, "mini_npz"), exist_ok=True)
                np.savez(os.path.join(o, "mini_npz", "results.npz"), depth=depth, conf=pred["conf"], extrinsics=ext, intrinsics=K)
                shutil.copy(os.path.join(ex, "gt_meta.npz"), os.path.join(o, "gt_meta.npz"))
            gte = torch.from_numpy(as_homogeneous(gt["extrinsics"]))
            pose = compute_pose(torch.from_numpy(as_homogeneous(np.asarray(ext)[:, :3])), gte)
            pose0 = compute_pose(torch.from_numpy(as_homogeneous(pred["extrinsics"][:, :3])), gte)
            metrics[scene] = Evaluator._to_float_dict({**pose, **{f"{k}_init": v for k, v in pose0.items()}, **info})
            Evaluator._dump_json(mp, {**metrics, "mean": Evaluator._mean_of_dicts(metrics.values())})
            m = metrics[scene]
            print(f"[sba] {ds} {scene}: auc03 {m['auc03_init']:.3f}->{m['auc03']:.3f} tracks {m['num_tracks']} obs {m['num_obs']} "
                  f"feat {m.get('feat_s', 0):.2f} match {m.get('match_s', 0):.2f} track {m['track_s']:.2f} ba {m['ba_s']:.2f} "
                  f"opt {m['opt_s']:.2f}s f {m.get('focal0', 0):.0f}->{m.get('focal', 0):.0f} k {m.get('radial', 0):.3f}")


if __name__ == "__main__":
    main()
