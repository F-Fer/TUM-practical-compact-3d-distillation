"""Geometry-guided match refinement + BA: the feed-forward geometry replaces the coarse stage of a dense matcher.

    .venv/bin/python scripts/gmr_ba.py <src work dir> <dst work dir> [key=value ...]

RoMa v2 = DINOv3 ViT-L descriptor + transformer matcher (coarse warp at 1/16) + three small convolutional refiners
(1/4, 1/2, 1/1 on VGG19-BN features) that correct a given warp by a local correlation. Here the coarse warp of every
pair comes from the current geometry instead (depth + cameras of <src>, e.g. DA3 or a PBA run): source pixels are
back-projected and projected into each covisible neighbour, the refiners correct that warp and predict an overlap
confidence, and a reprojection BA (scripts/sba_fast.py's GPU LM: poses, shared focal, radial, free points) runs on the
corrected correspondences. No descriptor, no global matcher, no track building: every source pixel is a point seen in
its frame and in up to `nbr` neighbours. `rounds` > 1 repeats refine -> BA from the new geometry.
knowledge/ideas/guided-match-refinement.md.
"""

import json
import os
import shutil
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sba_fast as sf  # noqa: E402
from lpba_depth_oracle import abs_rel, try_gt  # noqa: E402
from depth_anything_3.bench.evaluator import Evaluator  # noqa: E402
from depth_anything_3.bench.utils import compute_pose  # noqa: E402
from depth_anything_3.utils.geometry import as_homogeneous  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = sf.DEV

DEFAULTS = {
    "subset": "dev",  # dev | held-out | all (held-out = all but the first DEV[ds] scenes of <src>: for a source that
    # holds only the held-out scenes, e.g. sba_ho/*, use all)
    "datasets": "",
    "scenes": "",
    "nbr": 10,  # covisible neighbours per source frame
    "nbr_mode": "spread",
    "res": "384x512",  # H x W the refiners run at (multiples of 4); images are resized to it
    "stages": "4,2,1",  # refiner stages: "<s>" = refiner s (4 = 1/4 resolution with local correlation radius 3, 2, 1) at
    # `res`, "<s>@<k>" = on the images downscaled by k (wider basin); ";" separates the stage lists of the rounds
    "pts": 2048,  # source pixels per frame that become BA points
    "vis_thresh": 0.5,  # overlap confidence above which a neighbour observation is used
    "occ": 0.1,  # relative depth tolerance of the geometric visibility that initialises the overlap logit
    "logit0": 2.0,  # initial overlap logit: +logit0 where the geometry says visible, -logit0 elsewhere
    "max_shift": 0.0,  # > 0: drop observations the refiners moved by more than this many px (prediction resolution)
    "rounds": 1,  # refine -> BA rounds
    "iters": 25,  # LM iterations per round
    "lam": 1e-4,
    "loss_scale": 0.5,  # Cauchy scale, px at the prediction resolution
    "loss_rounds": "",  # comma-separated Cauchy scales per round (the last repeats), overrides loss_scale: with several
    # rounds, a large scale first lets matches far from the current geometry move it, the re-match then starts closer
    "gnc": "",  # graduated robust scale: comma-separated Cauchy scales (px) run for gnc_iters LM iterations each before
    # the `iters` iterations at loss_scale (matches far from the current geometry keep their pull at first)
    "gnc_iters": 8,
    "radial": 1,
    "refine_focal": 1,
    "prior": 0.0,
    "refiner_ckpt": "",  # refiners fine-tuned by scripts/gmr_train.py (empty: RoMa v2's)
    "eval_matches": 0,  # 1: with GT depth + cameras (HiRoom, ScanNet++, ETH3D), the error of the geometric and of the
    # refined matches against the true correspondence, binned by the geometric (initial) error: the refiners' basin
    "ba_init": "",  # gt: diagnostic, the BA starts from the GT cameras (matches still from <src>'s geometry): do the matches
    # prefer the truth, or the geometry they were predicted from?
    "depth_fit": "frame",  # frame | scene | points (per-frame smooth log-depth grid fitted to the BA points)
    "dtype": "float32",
    "amp": 1,
    "seed": 0,
    "save": 1,
}

_MODEL = {}


def refiner_model(ckpt=""):
    if ckpt and _MODEL.get("ckpt") != ckpt:
        m = refiner_model()
        m.refiners.load_state_dict(torch.load(ckpt, map_location="cuda")["refiners"])
        m.eval()
        _MODEL["ckpt"] = ckpt
    if "m" not in _MODEL:
        from romav2 import RoMaV2

        m = RoMaV2()
        m.f, m.matcher = None, None  # the descriptor and the global matcher are not used
        torch.cuda.empty_cache()
        _MODEL["m"] = m
    return _MODEL["m"]


def norm_grid(h, w, dev):
    ys = torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=dev)
    xs = torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=dev)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], -1)


def geo_warp(depth, R, t, K, i, js, h, w, occ):
    """Warp of frame i's h x w pixel grid into frames js by the geometry: normalised target coordinates [B, h, w, 2]
    and visibility [B, h, w] (in front, inside, not occluded by more than occ)."""
    N, H, W = depth.shape
    g = norm_grid(h, w, depth.device)
    u = (g[..., 0] + 1) / 2 * W - 0.5
    v = (g[..., 1] + 1) / 2 * H - 0.5
    d = F.grid_sample(depth[i][None, None], g[None], mode="bilinear", align_corners=False, padding_mode="border")[0, 0]
    ray = torch.stack([(u - K[i, 0, 2]) / K[i, 0, 0], (v - K[i, 1, 2]) / K[i, 1, 1], torch.ones_like(u)], -1) * d[..., None]
    Xw = torch.einsum("ji,hwj->hwi", R[i], ray - t[i])
    Xc = torch.einsum("bij,hwj->bhwi", R[js], Xw) + t[js][:, None, None]
    z = Xc[..., 2]
    zs = z.clamp_min(1e-6)
    u2 = K[js, 0, 0][:, None, None] * Xc[..., 0] / zs + K[js, 0, 2][:, None, None]
    v2 = K[js, 1, 1][:, None, None] * Xc[..., 1] / zs + K[js, 1, 2][:, None, None]
    warp = torch.stack([(u2 + 0.5) / W * 2 - 1, (v2 + 0.5) / H * 2 - 1], -1)
    inside = (warp.abs().amax(-1) < 1) & (z > 1e-6)
    dj = F.grid_sample(depth[js][:, None], warp.clamp(-1, 1), mode="bilinear", align_corners=False, padding_mode="border")[:, 0]
    vis = inside & (z < dj * (1 + occ))
    return warp.clamp(-2, 2), vis


@torch.inference_mode()
def refine_pairs(m, feats, depth, R, t, K, nb, cfg, Hl, Wl):
    """Per source frame: geometric warp into its neighbours, corrected by the refiners. Returns, for the sampled
    source pixels, the observations (frame, point, uv) and the 3D points, plus diagnostics."""
    dev = depth.device
    N, H, W = depth.shape
    # stage tokens: "<s>" = refiner s at the full refiner resolution, "<s>@<k>" = refiner s on the images downscaled by k
    # (a wider basin: the 1/4 refiner's correlation window of +-3 cells is then +-12 k px); a coarse stage (k > 1) only
    # updates the warp, the overlap logit stays the geometric one
    stages = [(int(x.split("@")[0]), int(x.split("@")[1]) if "@" in x else 1) for x in cfg["stages"].split(",") if x]
    fr_l, pt_l, uv_l, X_l, shift_l = [], [], [], [], []
    n_pts = 0
    conf = cfg["_conf"]
    for i in range(N):
        js = nb[i]
        js = js[js != i]
        B = len(js)
        s0, k0 = stages[0]
        warp, vis = geo_warp(depth, R, t, K, i, js, Hl // (s0 * k0), Wl // (s0 * k0), cfg["occ"])
        warp0 = warp
        confd = torch.where(vis, cfg["logit0"], -cfg["logit0"])[..., None].float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(cfg["amp"])):
            for si, (s, k) in enumerate(stages):
                hs, ws = Hl // (s * k), Wl // (s * k)
                if warp.shape[1] != hs:
                    warp = F.interpolate(warp.permute(0, 3, 1, 2), size=(hs, ws), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
                    confd = F.interpolate(confd.permute(0, 3, 1, 2), size=(hs, ws), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
                scale = torch.tensor((Wl / k / m.anchor_width, Hl / k / m.anchor_height), device=dev)
                fs = feats[k][s]
                out = m.refiners[str(s)](f_A=fs[i : i + 1].expand(B, -1, -1, -1), f_B=fs[js], prev_warp=warp.float(),
                                         prev_confidence=confd[..., :1].float() if k > 1 else confd.float(), scale_factor=scale)
                warp = out["warp"].float()
                # a stage that is repeated (iterated like a recurrent update: each pass re-centres the correlation
                # window) only moves the warp; the confidence is taken from its last pass
                if k == 1 and not (si + 1 < len(stages) and stages[si + 1] == (s, k)):
                    confd = out["confidence"].float()
        hs, ws = warp.shape[1:3]
        # source pixels: random among the confident ones, at the refiners' output resolution
        c = F.interpolate(conf[i][None, None], size=(hs, ws), mode="bilinear", align_corners=False)[0, 0]
        ok = (c > cfg["_conf_thr"]).flatten()
        idx = torch.nonzero(ok)[:, 0]
        idx = idx[torch.randperm(len(idx), device=dev)[: cfg["pts"]]]
        g = norm_grid(hs, ws, dev).reshape(-1, 2)[idx]
        su = torch.stack([(g[:, 0] + 1) / 2 * W - 0.5, (g[:, 1] + 1) / 2 * H - 0.5], -1)  # [P, 2] source pixels
        tw = warp.reshape(B, -1, 2)[:, idx]
        tu = torch.stack([(tw[..., 0] + 1) / 2 * W - 0.5, (tw[..., 1] + 1) / 2 * H - 0.5], -1)  # [B, P, 2]
        ov = confd[..., 0].sigmoid().reshape(B, -1)[:, idx]
        inside = (tu[..., 0] >= 0) & (tu[..., 0] <= W - 1) & (tu[..., 1] >= 0) & (tu[..., 1] <= H - 1)
        keep = (ov > cfg["vis_thresh"]) & inside
        w0 = F.interpolate(warp0.permute(0, 3, 1, 2), size=(hs, ws), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).reshape(B, -1, 2)[:, idx]
        shift = ((tw - w0) * torch.tensor([W / 2, H / 2], device=dev)).norm(dim=-1)  # px the refiners moved the match
        if cfg["max_shift"] > 0:
            keep &= shift < cfg["max_shift"]
        if cfg.get("_gt") is not None:  # true correspondences from the GT depth and cameras
            gd, gR, gt_, gK = cfg["_gt"]
            ii = torch.full((len(idx),), i, device=dev, dtype=torch.long)
            dg = gd[i][su[:, 1].round().long().clamp(0, H - 1), su[:, 0].round().long().clamp(0, W - 1)]
            ray = torch.stack([(su[:, 0] - gK[i, 0, 2]) / gK[i, 0, 0], (su[:, 1] - gK[i, 1, 2]) / gK[i, 1, 1], torch.ones_like(dg)], -1) * dg[:, None]
            Xw = torch.einsum("ji,pj->pi", gR[i], ray - gt_[i])
            Xc = torch.einsum("bij,pj->bpi", gR[js], Xw) + gt_[js][:, None]
            zz = Xc[..., 2].clamp_min(1e-6)
            tg = torch.stack([gK[js, 0, 0][:, None] * Xc[..., 0] / zz + gK[js, 0, 2][:, None], gK[js, 1, 1][:, None] * Xc[..., 1] / zz + gK[js, 1, 2][:, None]], -1)
            ok = (dg > 0)[None] & (Xc[..., 2] > 1e-6) & (tg[..., 0] >= 0) & (tg[..., 0] <= W - 1) & (tg[..., 1] >= 0) & (tg[..., 1] <= H - 1)
            dj = gd[js[:, None].expand(-1, len(idx)), tg[..., 1].round().long().clamp(0, H - 1), tg[..., 0].round().long().clamp(0, W - 1)]
            ok &= (dj > 0) & ((Xc[..., 2] - dj).abs() < 0.03 * dj)  # visible in the neighbour by the GT depth
            t0 = torch.stack([(w0[..., 0] + 1) / 2 * W - 0.5, (w0[..., 1] + 1) / 2 * H - 0.5], -1)
            e0 = (t0 - tg).norm(dim=-1)[ok]
            e1 = (tu - tg).norm(dim=-1)[ok]
            cfg["_match_eval"].append(torch.stack([e0, e1, ov[ok], keep[ok].float()], 1).cpu())
        has = keep.any(0)
        P = int(has.sum())
        if P == 0:
            continue
        pid = torch.full((len(idx),), -1, device=dev, dtype=torch.long)
        pid[has] = torch.arange(P, device=dev) + n_pts
        # observations: the source pixel itself, then the kept neighbour matches
        fr_l.append(torch.full((P,), i, device=dev, dtype=torch.long))
        pt_l.append(pid[has])
        uv_l.append(su[has])
        bb, pp = torch.nonzero(keep, as_tuple=True)
        fr_l.append(js[bb])
        pt_l.append(pid[pp])
        uv_l.append(tu[bb, pp])
        shift_l.append(shift[bb, pp])
        d = sf.bilinear(depth, torch.full((P,), i, device=dev, dtype=torch.long), su[has])
        ray = torch.stack([(su[has][:, 0] - K[i, 0, 2]) / K[i, 0, 0], (su[has][:, 1] - K[i, 1, 2]) / K[i, 1, 1], torch.ones_like(d)], -1) * d[:, None]
        X_l.append(torch.einsum("ji,pj->pi", R[i], ray - t[i]))
        n_pts += P
    if not n_pts:
        return None
    return torch.cat(fr_l), torch.cat(pt_l), torch.cat(uv_l), torch.cat(X_l), torch.cat(shift_l)


def fit_depth(depth, fr, uv, zc, d, N, mode, w=None):
    """Bring DA3's depth to the BA's result. frame / scene: one scale (median ratio of the BA points' depth to the
    predicted depth at their pixels) per frame / scene. grid:G[:lam]: per frame a bilinear log-depth correction grid
    with G cells along the long side (nodes at the image corners, as PBA's grid) fitted to log(z_BA / d) of the frame's
    observations by robust ridge least squares (Huber 0.05; lam x a prior that pulls every node to the frame's median
    and neighbouring nodes together): the triangulated points as a sparse depth measurement, DA3 as the interpolant."""
    ratio = (zc / d.double().clamp_min(1e-6))
    good = torch.isfinite(ratio) & (ratio > 0.2) & (ratio < 5)
    if w is not None:
        good &= w > 0.5  # BA inliers (Cauchy weight)
    rn, frn, gn = ratio.cpu().numpy(), fr.cpu().numpy(), good.cpu().numpy()
    scene_s = float(np.median(rn[gn])) if gn.any() else 1.0
    scales = np.full(N, scene_s)
    if mode != "scene":
        for i in range(N):
            ri = rn[(frn == i) & gn]
            if len(ri) >= 20:
                scales[i] = np.median(ri)
    if not mode.startswith("grid"):
        return depth * scales[:, None, None].astype(np.float32)
    parts = mode.split(":")
    G, lam = int(parts[1]), float(parts[2]) if len(parts) > 2 else 1.0
    H, W = depth.shape[1:]
    gh, gw = (G + 1, max(2, round(G * W / H)) + 1) if H > W else (max(2, round(G * H / W)) + 1, G + 1)
    dev = fr.device
    sc = torch.from_numpy(scales).to(dev)
    y = torch.where(good, torch.log(ratio.clamp_min(1e-6)) - torch.log(sc[fr]), torch.zeros_like(ratio))  # log residual after the frame's scale
    gu, gv = uv[:, 0].double() * (gw - 1) / (W - 1), uv[:, 1].double() * (gh - 1) / (H - 1)
    u0, v0 = gu.floor().clamp(0, gw - 2).long(), gv.floor().clamp(0, gh - 2).long()
    du, dv = (gu - u0).clamp(0, 1), (gv - v0).clamp(0, 1)
    Kn = gh * gw
    cols = torch.stack([v0 * gw + u0, v0 * gw + u0 + 1, (v0 + 1) * gw + u0, (v0 + 1) * gw + u0 + 1], 1)  # [M, 4]
    wts = torch.stack([(1 - du) * (1 - dv), du * (1 - dv), (1 - du) * dv, du * dv], 1)
    # prior: nodes near zero (ridge) and neighbouring nodes equal (first differences)
    L = torch.zeros(Kn, Kn, dtype=torch.float64, device=dev)
    idx = torch.arange(Kn, device=dev).view(gh, gw)
    for a_, b_ in ((idx[:, :-1], idx[:, 1:]), (idx[:-1], idx[1:])):
        a_, b_ = a_.reshape(-1), b_.reshape(-1)
        L[a_, a_] += 1
        L[b_, b_] += 1
        L[a_, b_] -= 1
        L[b_, a_] -= 1
    cnt = torch.bincount(fr[good], minlength=N).double().clamp_min(1)
    c = torch.zeros(N, Kn, dtype=torch.float64, device=dev)
    wt = good.double()
    for _ in range(3):  # IRLS
        A = torch.zeros(N, Kn, Kn, dtype=torch.float64, device=dev)
        bvec = torch.zeros(N, Kn, dtype=torch.float64, device=dev)
        ww = wts * wt[:, None]
        flat = (fr[:, None, None] * Kn * Kn + cols[:, :, None] * Kn + cols[:, None, :]).reshape(-1)
        A.view(-1).index_add_(0, flat, (ww[:, :, None] * wts[:, None, :]).reshape(-1))
        bvec.view(-1).index_add_(0, (fr[:, None] * Kn + cols).reshape(-1), (ww * y[:, None]).reshape(-1))
        A = A / cnt[:, None, None] + lam * (0.01 * torch.eye(Kn, dtype=torch.float64, device=dev) + 0.1 * L)[None]
        c = torch.linalg.solve(A, bvec / cnt[:, None])
        res = y - (c[fr[:, None], cols] * wts).sum(1)
        wt = good.double() * (0.05 / res.abs().clamp_min(0.05))
    corr = F.interpolate(c.view(N, 1, gh, gw).float(), size=(H, W), mode="bilinear", align_corners=True)[:, 0]
    return depth * scales[:, None, None].astype(np.float32) * torch.exp(corr).cpu().numpy()


def refine_scene(files, pred, cfg):
    import run_tto_pba as rp

    depth, ext, K = pred["depth"].astype(np.float32), pred["extrinsics"][:, :3].astype(np.float64), pred["intrinsics"].astype(np.float64)
    N, H, W = depth.shape
    Hl, Wl = (int(x) for x in cfg["res"].split("x"))
    imgs = sf.load_images(files, H, W)  # decoding not timed
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    m = refiner_model(cfg["refiner_ckpt"])
    dev = "cuda"
    if "warm" not in _MODEL:  # cuDNN set-up is not TTO time
        with torch.inference_mode():
            m.refiner_features(F.interpolate(imgs[:2].to(dev), size=(Hl, Wl), mode="bicubic", align_corners=False, antialias=True))
        _MODEL["warm"] = 1
    info = {"num_frames": N}
    sf.sync()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    dep = torch.from_numpy(depth).to(dev)
    Rt = torch.from_numpy(ext).float().to(dev)
    Kt = torch.from_numpy(K).float().to(dev)
    conf = torch.from_numpy(pred["conf"].astype(np.float32)).to(dev)
    cfg = {**cfg, "_conf": conf, "_conf_thr": float(torch.quantile(conf.flatten()[:: max(1, conf.numel() // 100000)], 0.1))}
    with torch.inference_mode():
        lr = F.interpolate(imgs.to(dev), size=(Hl, Wl), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        feats = {}
        toks = [x for x in cfg["stages"].replace(";", ",").split(",") if x]
        for k in sorted({int(x.split("@")[1]) if "@" in x else 1 for x in toks}):
            lk = lr if k == 1 else F.interpolate(lr, size=(Hl // k, Wl // k), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
            need = {int(x.split("@")[0]) for x in toks if (int(x.split("@")[1]) if "@" in x else 1) == k}
            fk = {}
            for a in range(0, N, 8):  # preallocated per scale: a list + cat would hold every map twice
                for s_, x in m.refiner_features(lk[a : a + 8]).items():
                    if s_ not in need:
                        continue
                    if s_ not in fk:
                        fk[s_] = torch.empty((N, *x.shape[1:]), dtype=x.dtype, device=dev)
                    fk[s_][a : a + 8] = x
            feats[k] = fk
            del lk
        del lr
    sf.sync()
    info["feat_s"] = time.perf_counter() - t0
    cfg["_gt"], cfg["_match_eval"] = None, []
    if cfg["eval_matches"]:
        import cv2

        gdep = try_gt(files, H, W)
        if gdep is not None:
            h0, w0_ = cv2.imread(str(files[0])).shape[:2]
            gK = cfg["_gt_K"].astype(np.float64).copy()
            gK[:, 0] *= W / w0_
            gK[:, 1] *= H / h0
            gE = cfg["_gt_ext"]
            cfg["_gt"] = (torch.from_numpy(gdep).float().to(dev), torch.from_numpy(gE[:, :3, :3]).float().to(dev),
                          torch.from_numpy(gE[:, :3, 3]).float().to(dev), torch.from_numpy(gK).float().to(dev))
    R, t = Rt[:, :, :3], Rt[:, :, 3]
    f0 = float(np.median(K[:, 0, 0]))
    cx, cy = float(np.median(K[:, 0, 2])), float(np.median(K[:, 1, 2]))
    k0 = 0.0
    info.update(match_s=0.0, ba_s=0.0, focal0=f0)
    depth_cur = depth
    for rnd in range(cfg["rounds"]):
        t1 = time.perf_counter()
        nb = rp.select_neighbors(dep, R, t, Kt, cfg["nbr"], 0.05, mode=cfg["nbr_mode"])
        rs = cfg["stages"].split(";")  # per round, the last one repeats
        obs = refine_pairs(m, feats, dep, R, t, Kt, nb, {**cfg, "stages": rs[min(rnd, len(rs) - 1)]}, Hl, Wl)
        sf.sync()
        info["match_s"] += time.perf_counter() - t1
        if obs is None:
            break
        fr, ptt, uv, X, shift = obs
        info.update(num_tracks=int(X.shape[0]), num_obs=int(len(fr)), shift_med=float(shift.median()), shift_p90=float(shift.quantile(0.9)) if len(shift) < 1e7 else -1.0)
        t2 = time.perf_counter()
        Rb, tb = R.double(), t.double()
        if cfg["ba_init"] == "gt":
            G = np.tile(np.eye(4), (N, 1, 1))
            G[:, :3] = cfg["_gt_ext"][:, :3]
            E = np.tile(np.eye(4), (N, 1, 1))
            E[:, :3] = torch.cat([R, t[:, :, None]], 2).double().cpu().numpy()
            G[:, :3, 3] *= rp.camera_spread(E[:, :3]) / max(rp.camera_spread(G[:, :3]), 1e-12)
            G = G @ np.linalg.inv(G[0]) @ E[0]
            Gt = torch.from_numpy(G).to(dev)
            # the points move with their source camera: X = R_gt^T (R X + t - t_gt), per point from its first observation
            first = torch.zeros(X.shape[0], dtype=torch.long, device=dev).index_reduce_(0, ptt, torch.arange(len(ptt), device=dev), "amin", include_self=False)
            src = fr[first]
            Xc = torch.einsum("pij,pj->pi", Rb[src], X.double()) + tb[src]
            X = torch.einsum("pji,pj->pi", Gt[src, :3, :3], Xc - Gt[src, :3, 3]).float()
            Rb, tb = Gt[:, :3, :3].clone(), Gt[:, :3, 3].clone()
        ba = sf.BA(Rb, tb, f0, k0, cx, cy, X.double(), fr, ptt, uv, cfg)
        c0 = float(ba.total())
        for a_ in [float(x) for x in cfg["gnc"].split(",") if x]:
            ba.a = a_
            ba.solve(cfg["gnc_iters"], cfg["lam"])
        lr_ = [float(x) for x in cfg["loss_rounds"].split(",") if x]
        ba.a = lr_[min(rnd, len(lr_) - 1)] if lr_ else cfg["loss_scale"]
        hist = ba.solve(cfg["iters"], cfg["lam"])
        sf.sync()
        info["ba_s"] += time.perf_counter() - t2
        info.update(cost0=c0, cost=hist[-1][0] if hist else c0, n_accept=sum(h[1] for h in hist))
        # new geometry: cameras from the BA, DA3's depth scaled per frame to the BA points
        d = sf.bilinear(dep, fr, uv)
        zc = (torch.einsum("mij,mj->mi", ba.R[fr].double(), ba.X[ptt].double()) + ba.t[fr].double())[:, 2]
        wr = ba.cost(ba.residuals()[0])[1]
        wpt = torch.zeros(ba.P, dtype=wr.dtype, device=dev).index_reduce_(0, ptt, wr, "amin", include_self=False)  # a point's worst observation
        depth_cur = fit_depth(depth_cur, fr, uv, zc, d, N, cfg["depth_fit"], wpt[ptt])
        dep = torch.from_numpy(depth_cur).to(dev)
        R, t = ba.R.float(), ba.t.float()
        f0, k0 = float(ba.f), float(ba.k)
        Kt = Kt.clone()
        Kt[:, 0, 0] = Kt[:, 1, 1] = f0
        Kt[:, 0, 2], Kt[:, 1, 2] = cx, cy
    sf.sync()
    info["opt_s"] = time.perf_counter() - t0
    info.update(focal=f0, radial=k0, peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30)
    gt = try_gt(files, H, W)
    info.update(depth_err_in=abs_rel(depth, gt), depth_err_out=abs_rel(depth_cur, gt))
    if cfg["_match_eval"]:
        ev = torch.cat(cfg["_match_eval"]).numpy()
        edges = [0, 1, 2, 4, 8, 16, 32, 64, 1e9]
        for a_, b_ in zip(edges[:-1], edges[1:]):
            mk = (ev[:, 0] >= a_) & (ev[:, 0] < b_)
            tag = f"me_{a_}"
            kept = mk & (ev[:, 3] > 0)
            info.update({tag + "_n": float(mk.sum()), tag + "_med": float(np.median(ev[mk, 1])) if mk.any() else float("nan"),
                         tag + "_lt1": float((ev[mk, 1] < 1).mean()) if mk.any() else float("nan"),
                         tag + "_lt2": float((ev[mk, 1] < 2).mean()) if mk.any() else float("nan"),
                         tag + "_keep": float(ev[mk, 3].mean()) if mk.any() else float("nan"),
                         tag + "_kept_med": float(np.median(ev[kept, 1])) if kept.any() else float("nan"),
                         tag + "_kept_lt2": float((ev[kept, 1] < 2).mean()) if kept.any() else float("nan")})
    ext_new = torch.cat([R, t[:, :, None]], 2).cpu().numpy().astype(np.float32)
    return ext_new, Kt.cpu().numpy().astype(np.float32), depth_cur, info


def main():
    src, dst = sys.argv[1], sys.argv[2]
    cfg = dict(DEFAULTS)
    for a in sys.argv[3:]:
        k, _, v = a.partition("=")
        if k not in DEFAULTS:
            sys.exit(f"unknown option {k}, expected one of {list(DEFAULTS)}")
        cfg[k] = type(DEFAULTS[k])(v)
    sys.stdout.reconfigure(line_buffering=True)
    os.chdir(REPO)
    os.makedirs(dst, exist_ok=True)
    json.dump(cfg, open(os.path.join(dst, "gmr_config.json"), "w"), indent=2)
    datasets = [d for d in cfg["datasets"].split(",") if d] or list(DEV)
    for ds in datasets:
        data_dir = os.path.join(src, "model_results", ds)
        scenes = sorted(os.path.dirname(p) for p, _, _ in os.walk(data_dir) if os.path.basename(p) == "unposed")
        if cfg["subset"] == "dev":
            scenes = scenes[: DEV[ds]]
        elif cfg["subset"] == "held-out":
            scenes = scenes[DEV[ds]:]
        mp = os.path.join(dst, "metric_results", f"{ds}_pba.json")
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
                ext, K, depth, info = refine_scene([str(f) for f in gt["image_files"]], pred, {**cfg, "_gt_ext": gt["extrinsics"].astype(np.float64), "_gt_K": gt["intrinsics"]})
            except torch.cuda.OutOfMemoryError as e:
                print(f"[gmr] {ds} {scene} FAILED {e!r}")
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
            mm = metrics[scene]
            print(f"[gmr] {ds} {scene}: auc03 {mm['auc03_init']:.3f}->{mm['auc03']:.3f} auc30 {mm['auc30_init']:.3f}->{mm['auc30']:.3f} "
                  f"pts {mm.get('num_tracks', 0):.0f} obs {mm.get('num_obs', 0):.0f} cost {mm.get('cost0', 0):.0f}->{mm.get('cost', 0):.0f} acc {mm.get('n_accept', 0):.0f} shift {mm.get('shift_med', 0):.2f}/{mm.get('shift_p90', 0):.2f}px "
                  f"feat {mm['feat_s']:.2f} match {mm['match_s']:.2f} ba {mm['ba_s']:.2f} opt {mm['opt_s']:.2f}s "
                  f"f {mm.get('focal0', 0):.0f}->{mm.get('focal', 0):.0f} k {mm.get('radial', 0):.3f} mem {mm['peak_mem_gib']:.1f}G "
                  f"depth {100 * mm.get('depth_err_in', float('nan')):.2f}->{100 * mm.get('depth_err_out', float('nan')):.2f}%")


if __name__ == "__main__":
    main()
