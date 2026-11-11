"""Refine saved DA3 predictions with photometric or feature-metric bundle adjustment (dense, direct).

Usage (from anywhere; paths are resolved relative to the repo root):
    .venv/bin/python scripts/run_tto_pba.py <src work dir> <dst work dir> [key=value ...]

    # photometric BA on the dev subset (the defaults: the photometric line of results/pba/figures5)
    .venv/bin/python scripts/run_tto_pba.py workspace/evaluation/DA3-SMALL workspace/tto/DA3-SMALL/pho subset=dev
    # feature-metric: VGG16 features + Huber (the feature-metric line of the plot)
    ... features=vgg
    # photometric Adam, then 40 LM iterations on NCC of 3x3 patches (best pose on SMALL/BASE, ~5 s per scene at T = 800)
    ... iters=800 lm_photo_loss=ncc lm_iters=40 lm_stop=0
    # feature-metric VGG + NCC in all levels (3x the time of VGG + Huber)
    ... features=vgg photo_loss=ncc samples=4096

Per scene, the camera poses (frame 0 fixed), one shared focal length, a coarse log-depth correction grid per frame,
a relative affine brightness per frame (grey Huber only) and one radial lens coefficient are optimised so that pixels of
each frame warped into its covisible neighbours match in intensity or features (photometric term) and depth (geometric
term). Adam on 4 blur levels (sigma 8, 4, 2, 1 px, `iters` each), then Levenberg-Marquardt on the sharp images, with
hand-derived gradients and Gauss-Newton blocks (class Kernel). What each choice is based on, and what was tried and
dropped: knowledge/ideas/pba-method.md (the code of the dropped alternatives is at commit 1c1a921).

Scenes: all of <src>/model_results, or `subset=dev` (the first 10 / 2 / 3 / 4 / 3 scenes of HiRoom / 7-Scenes /
ScanNet++ / DTU-64 / ETH3D, the set of the compute plots), `subset=held-out` (the other scenes of those datasets), or
datasets / max_scenes / first_scene / scenes. Finished scenes are skipped, so an interrupted run can be resumed.

Outputs go to <dst work dir>, mirroring <src work dir>:
    pba_config.json   the configuration
    model_results/    refined depth, extrinsics and intrinsics per scene (unposed/exports/mini_npz/results.npz) plus a
                      copy of gt_meta.npz, so that scripts/eval_tto.py (or run_bench.py eval.eval_only=true) scores them
    metric_results/   <dataset>_pba.json: pose AUC before and after, focal error, time (opt_s) and other per-scene
                      stats; <dataset>_pba_hist.jsonl: the loss curve of every scene
"""

import json
import math
import os
import shutil
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from depth_anything_3.bench.evaluator import Evaluator
from depth_anything_3.bench.utils import compute_pose
from depth_anything_3.utils.geometry import as_homogeneous

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = {
    "subset": "",  # dev | held-out | "" (then datasets, max_scenes, first_scene, scenes)
    "datasets": "",  # comma-separated, default: all in <src>/model_results (with subset: the five pose datasets)
    "max_scenes": 0,  # per dataset, 0 = all
    "first_scene": 0,  # per dataset: skip this many scenes first
    "scenes": "",  # comma-separated scene names, default: all
    "features": "grey",  # grey (photometric, the mean of the RGB channels) | vgg (VGG16 relu1_2 + relu2_2, PCA-reduced)
    "feat_dim": 8,  # vgg: PCA channels
    "photo_loss": "huber",  # huber (per pixel) | ncc (1 - NCC of 3x3 patches; samples then count pixels, 9 per patch)
    "lm_photo_loss": "",  # photo_loss of the LM levels, empty: photo_loss
    "neighbors": 10,  # covisible target frames per source frame
    "nbr_mode": "spread",  # spread (evenly over the covisibility ranking) | top (the most covisible)
    "min_covis": 0.2,
    "samples": 1024,  # source pixels per frame and Adam iteration
    "levels": "8,4,2,1,1",  # blur sigma (px) of each level; Adam levels run `iters` iterations each
    "iters": 50,  # T: Adam iterations per level (T = 25 is as good on dev; held-out tests used 50)
    "lm_levels": "4",  # comma-separated level indices run with Levenberg-Marquardt instead of Adam
    "lm_iters": 15,  # LM iterations of those levels (at most, see lm_stop)
    "lm_samples": 4096,  # source pixels per frame and LM iteration, 0 = samples
    "lm_lambda": 1e-3,  # initial damping, relative to diag(J^T J)
    "lm_lambda_min": 1e-7,  # lower bound of the damping
    "lm_stop": 2e-3,  # stop an LM level when the cost on fixed held-out pixels falls by less than this fraction in an iteration, 0 = off
    "lm_stop_samples": 2048,  # held-out source pixels per frame of lm_stop
    "lr_rot": 3e-3,  # rad
    "lr_trans": 3e-3,  # in units of the median depth
    "lr_focal": 3e-3,  # log focal
    "lr_depth": 5e-3,  # log depth grid
    "lr_bright": 1e-2,
    "lr_radial": 1e-3,
    "poses": True,  # optimise the poses
    "focal": "shared",  # shared (one scale for all frames) | none (fixed)
    "focal_init": "median",  # median (replace DA3's per-frame focals with their median first) | pred
    "depth_grid": 8,  # cells along the longer side of the log-depth correction grid, 0 = depth fixed
    "radial": True,  # one shared radial coefficient k of the division model x_u = x_d / (1 + k r_d^2) (lens distortion, ETH3D)
    "radial_max": 0.0,  # upper bound of k; 0 allows only barrel distortion
    "radial_gate": 0.01,  # at the first LM level: if |k| is below this, set it to 0 and leave it out (no distortion)
    "w_radial": 1e-3,  # keeps k near zero where the data do not determine it
    "w_photo": 1.0,
    "w_geo": 1.0,  # multi-view depth consistency
    "w_prior": 1.0,  # keeps the log-depth correction near zero (also fixes the scale gauge)
    "w_pose": 0.0,  # keeps the pose updates (rotation in rad, translation in median depths) near zero (= DA3's poses)
    "huber_photo": 0.05,  # scale of the photometric Huber kernel (intensities in [0, 1])
    "huber_geo": 0.05,  # log depth
    "occ_thresh": 0.05,  # relative depth; a sample is occluded in the target if it lies further away
    "conf_quantile": 0.1,  # source pixels below this DA3 confidence quantile (per scene) are not sampled
    "strat_cells": 0,  # source pixels: equal sampling mass per cell of a grid with this many cells along the long side, 0 = off
    "init_from": "",  # work dir whose extrinsics/intrinsics replace DA3's (e.g. a track-BA run), depth stays DA3's
    "init_gt": "",  # start from the ground truth: pose | pose_focal, to test where the objective's minimum lies
    "seed": 0,
    "save": True,  # write results.npz (False: only metrics)
}

# the dev subset: the first scenes of each pose dataset (sorted by name), as in results/pba and the compute plots
DEV = {"hiroom": 10, "7scenes": 2, "scannetpp": 3, "dtu64": 4, "eth3d": 3}
POOL_FACTOR = 16  # Adam: candidate pixels per frame drawn once per level = POOL_FACTOR x samples


def parse_overrides(overrides):
    """key=value pairs, converted to the type of the default."""
    cfg = dict(DEFAULTS)
    for o in overrides:
        key, _, value = o.partition("=")
        if key not in DEFAULTS:
            sys.exit(f"unknown override {key!r}, expected one of {list(DEFAULTS)}")
        default = DEFAULTS[key]
        cfg[key] = value.lower() == "true" if isinstance(default, bool) else type(default)(value)
    for key, allowed in (("features", ("grey", "vgg")), ("photo_loss", ("huber", "ncc")), ("lm_photo_loss", ("", "huber", "ncc")),
                         ("nbr_mode", ("spread", "top")), ("focal", ("shared", "none")), ("focal_init", ("median", "pred")),
                         ("subset", ("", "dev", "held-out")), ("init_gt", ("", "pose", "pose_focal"))):
        if cfg[key] not in allowed:
            sys.exit(f"{key}={cfg[key]!r}, expected one of {allowed}")
    return cfg


# ---------------------------------------------------------------- geometry and images


def so3_exp(w):
    """Rodrigues: axis-angle [..., 3] -> rotation [..., 3, 3], smooth at 0."""
    theta2 = (w * w).sum(-1, keepdim=True)[..., None]
    theta = torch.sqrt(theta2 + 1e-12)
    a = torch.sin(theta) / theta
    b = 2 * torch.sin(theta / 2) ** 2 / (theta2 + 1e-12)
    b = torch.where(theta2 < 1e-8, 0.5 - theta2 / 24, b)
    x, y, z = w.unbind(-1)
    o = torch.zeros_like(x)
    k = torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(*w.shape[:-1], 3, 3)
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(k)
    return eye + a * k + b * (k @ k)


def so3_exp_jl(w):
    """Rodrigues and the left Jacobian J_l(w) = I + b [w]x + c [w]x^2. w [..., 3].
    No branches for small angles (they cost tensor operations in every iteration): in float32, 1 - cos and 1 - sin(t)/t
    lose their relative precision below ~1e-3 rad, but they are then multiplied by [w]x^2 ~ t^2."""
    th2 = (w * w).sum(-1, keepdim=True)[..., None] + 1e-12
    th = torch.sqrt(th2)
    a = torch.sin(th) / th
    b = (1 - torch.cos(th)) / th2
    c = (1 - a) / th2
    x, y, z = w.unbind(-1)
    o = torch.zeros_like(x)
    k = torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(*w.shape[:-1], 3, 3)
    k2 = k @ k
    eye = torch.eye(3, device=w.device, dtype=w.dtype)
    return eye + a * k + b * k2, eye + b * k + c * k2


def bilinear_gather(src, idx, u, v, H, W):
    """Bilinearly sample src [C, N*H*W] (N frames, flattened) at frame idx [S], pixel (u, v) [S] -> [S, C].
    (u, v) must be inside [0, W-1] x [0, H-1]."""
    u0 = u.detach().floor().clamp(0, W - 2)
    v0 = v.detach().floor().clamp(0, H - 2)
    du, dv = (u - u0)[:, None], (v - v0)[:, None]
    base = idx * (H * W) + v0.long() * W + u0.long()
    g = lambda o: src[:, base + o].T  # noqa: E731
    top = g(0) * (1 - du) + g(1) * du
    bot = g(W) * (1 - du) + g(W + 1) * du
    return top * (1 - dv) + bot * dv


def gaussian_blur(x, sigma):
    """x [N, C, H, W]"""
    if sigma <= 0:
        return x
    r = int(math.ceil(3 * sigma))
    k = torch.exp(-((torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2) / 2)
    k = k / k.sum()
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return x


def huber(r, c):
    a = r.abs()
    return torch.where(a < c, 0.5 * r * r / c, a - 0.5 * c)


def huber_grad(r, c):
    """huber'(r)"""
    return (r / c).clamp(-1, 1)


def huber_weight(r, c):
    """IRLS weight huber'(r) / r (the LM levels minimise sum 0.5 w r^2 with these weights)."""
    return torch.where(r.abs() < c, 1 / c, 1 / r.abs().clamp_min(1e-12))


def load_images(image_files, H, W):
    """RGB in [0, 1], resized to the prediction's resolution (the aspect ratios match exactly)."""
    imgs = [cv2.resize(cv2.imread(str(f))[..., ::-1], (W, H), interpolation=cv2.INTER_AREA) for f in image_files]
    return torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255


_VGG_NET = {}  # the network of this process, by device (kept between scenes)


@torch.no_grad()
def vgg_features(images, dim, scale=0.1, chunk=8):
    """VGG16 relu1_2 and relu2_2 (upsampled) of images [N, 3, H, W], PCA-reduced to dim channels whose per-channel RMS
    is scale on average (comparable to intensities, so the same Huber threshold applies). PCA from the eigenvectors of
    the covariance of a random subset of pixels of every frame."""
    import torchvision

    key = str(images.device)
    if key not in _VGG_NET:
        _VGG_NET[key] = torchvision.models.vgg16(weights="IMAGENET1K_V1").features[:9].to(images.device).eval()
    vgg = _VGG_NET[key]
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device)[:, None, None]
    H, W = images.shape[-2:]

    def feats(x):
        f1 = vgg[:4]((x - mean) / std)
        f2 = F.interpolate(vgg[4:9](f1), size=(H, W), mode="bilinear", align_corners=False)
        return torch.cat([f1 / f1.std(), f2 / f2.std()], 1)  # both layers equally weighted

    gen = torch.Generator(device=images.device).manual_seed(0)
    samples = []
    for i in range(0, len(images), chunk):
        f = feats(images[i : i + chunk]).flatten(2)  # [n, C, HW]
        idx = torch.randint(0, f.shape[2], (2048,), device=f.device, generator=gen)
        samples.append(f[:, :, idx].permute(0, 2, 1).reshape(-1, f.shape[1]))
    X = torch.cat(samples)
    mu = X.mean(0)
    Xc = X - mu
    ev, U = torch.linalg.eigh(Xc.T @ Xc)
    P = U[:, -dim:].flip(1)  # [C, dim], the largest eigenvalues first
    norm = scale / (ev[-dim:] / len(X)).mean().sqrt()
    out = torch.empty(len(images), dim, H, W, device=images.device)
    for i in range(0, len(images), chunk):
        f = feats(images[i : i + chunk])
        out[i : i + chunk] = torch.einsum("nchw,cd->ndhw", f - mu[:, None, None], P) * norm
    return out


@torch.no_grad()
def select_neighbors(depth, R, t, K, k, occ_thresh, mode="spread", min_covis=0.2, n_grid=24):
    """Neighbours [N, k] of each frame by covisibility (fraction of its grid points seen unoccluded).

    mode top: the k most covisible frames (short baselines). mode spread: k frames evenly spaced over the
    covisibility ranking of the frames with covisibility >= min_covis, from most covisible to the widest baseline
    that still overlaps (wide baselines constrain rotation vs. translation, cf. the bas-relief ambiguity).
    """
    N, H, W = depth.shape
    k = min(k, N - 1)
    vs, us = torch.meshgrid(
        torch.linspace(0, H - 1, n_grid, device=depth.device), torch.linspace(0, W - 1, n_grid, device=depth.device), indexing="ij"
    )
    u, v = us.reshape(-1), vs.reshape(-1)
    d = depth[:, v.long(), u.long()]  # [N, P]
    pix = torch.stack([u, v, torch.ones_like(u)], 0)  # [3, P]
    Xc = torch.linalg.inv(K) @ pix[None] * d[:, None]  # [N, 3, P]
    Xw = R.transpose(1, 2) @ (Xc - t[:, :, None])  # [N, 3, P]
    score = torch.zeros(N, N, device=depth.device)
    flat = depth.reshape(1, -1)
    for j in range(N):
        Xj = R[j] @ Xw + t[j][None, :, None]  # [N, 3, P]
        z = Xj[:, 2]
        uj = K[j, 0, 0] * Xj[:, 0] / z.clamp_min(1e-6) + K[j, 0, 2]
        vj = K[j, 1, 1] * Xj[:, 1] / z.clamp_min(1e-6) + K[j, 1, 2]
        inside = (z > 1e-6) & (uj >= 0) & (uj <= W - 1) & (vj >= 0) & (vj <= H - 1)
        S = inside.numel()
        dj = bilinear_gather(flat, torch.full((S,), j, device=depth.device), uj.reshape(-1).clamp(0, W - 1), vj.reshape(-1).clamp(0, H - 1), H, W)
        visible = inside & (z < dj.view_as(z) * (1 + occ_thresh)) & (z > dj.view_as(z) * (1 - occ_thresh))
        score[:, j] = visible.float().mean(1)
    score.fill_diagonal_(-1)
    order = score.argsort(dim=1, descending=True)[:, : N - 1]
    if mode == "top":
        return order[:, :k]
    n_ok = (score >= min_covis).sum(1).clamp_min(k)  # at least k candidates per frame
    pos = torch.linspace(0, 1, k, device=depth.device)[None] * (n_ok[:, None] - 1).float()
    return order.gather(1, pos.round().long())


def sampling_probs(grey, valid, strat_cells=0):
    """[N, H*W] sampling probability of source pixels: valid and ~|grad I| of the (blurred) grey image. With strat_cells > 0
    every cell of a grid with this many cells along the long side gets the same mass, so that texture can't concentrate the
    samples (helps long LM runs)."""
    gx = F.pad(grey[..., :, 2:] - grey[..., :, :-2], (1, 1, 0, 0))
    gy = F.pad(grey[..., 2:, :] - grey[..., :-2, :], (0, 0, 1, 1))
    g = torch.sqrt(gx**2 + gy**2)[:, 0]
    p = (g + 1e-3) * valid
    if strat_cells > 0:
        H, W = p.shape[-2:]
        s = max(1, math.ceil(max(H, W) / strat_cells))
        mass = F.avg_pool2d(p[:, None], s, ceil_mode=True)
        p = p / F.interpolate(mass, scale_factor=s, mode="nearest")[:, 0, :H, :W].clamp_min(1e-12)
    return p.reshape(len(p), -1) + 1e-12 * valid.reshape(len(p), -1)


# ---------------------------------------------------------------- the scene's parameters


class PBA(torch.nn.Module):
    """Parameters of one scene: pose updates (rot, trans; frame 0 fixes the gauge), log focal scale, log-depth grid,
    brightness (log gain, bias) per frame and the radial coefficient kr. The scene is normalised to median depth 1."""

    def __init__(self, images, depth, conf, ext, K, cfg):
        super().__init__()
        N, _, H, W = images.shape
        dev = images.device
        self.cfg, self.N, self.H, self.W = cfg, N, H, W
        self.scale = float(depth.median())
        self.register_buffer("depth0", depth / self.scale)
        self.register_buffer("R0", ext[:, :, :3].clone())
        self.register_buffer("t0", ext[:, :, 3] / self.scale)
        K = K.clone()
        if cfg["focal_init"] == "median":  # every benchmark scene has one camera, DA3's per-frame focals differ by 0.5-1 %
            K[:, 0, 0] = K[:, 0, 0].median()
            K[:, 1, 1] = K[:, 1, 1].median()
        self.register_buffer("K0", K)
        self.register_buffer("images", images)
        self.register_buffer("free", (torch.arange(N, device=dev) > 0).float()[:, None])

        self.rot = torch.nn.Parameter(torch.zeros(N, 3, device=dev))
        self.trans = torch.nn.Parameter(torch.zeros(N, 3, device=dev))
        self.logf = torch.nn.Parameter(torch.zeros(1, device=dev))
        g = max(cfg["depth_grid"], 1)
        gh, gw = (g, max(2, round(g * W / H))) if H > W else (max(2, round(g * H / W)), g)
        self.grid = torch.nn.Parameter(torch.zeros(N, gh, gw, device=dev))
        self.bright = torch.nn.Parameter(torch.zeros(N, 2, device=dev))
        self.kr = torch.nn.Parameter(torch.zeros(1, device=dev))

        # sampling mask: confident, not at the border
        thr = torch.quantile(conf.reshape(-1)[:: conf.numel() // 2**22 + 1], cfg["conf_quantile"]) if cfg["conf_quantile"] > 0 else -1
        valid = conf > thr
        valid[(valid.flatten(1).float().mean(1) < 0.01)] = True  # frames with (almost) no confident pixels: use all
        valid[:, :2], valid[:, -2:], valid[:, :, :2], valid[:, :, -2:] = False, False, False, False
        self.register_buffer("valid", valid)

    def poses(self):
        dR = so3_exp(self.rot * self.free) if self.cfg["poses"] else torch.eye(3, device=self.R0.device).expand(self.N, 3, 3)
        R = dR @ self.R0
        t = (dR @ self.t0[..., None])[..., 0] + (self.trans * self.free if self.cfg["poses"] else 0)
        return R, t

    def intrinsics(self):
        f = torch.exp(self.logf if self.cfg["focal"] != "none" else self.logf.detach() * 0).expand(self.N)
        scale = 1 + torch.diag_embed(torch.stack([f - 1, f - 1, torch.zeros_like(f)], -1))  # only fx, fy
        return self.K0 * scale

    def grid_at(self, idx, u, v):
        """log-depth correction at pixel (u, v) of frame idx (bilinear, grid corners at the image corners)."""
        out = torch.zeros_like(u)
        if self.cfg["depth_grid"] > 0:
            N, gh, gw = self.grid.shape
            gu, gv = u * (gw - 1) / (self.W - 1), v * (gh - 1) / (self.H - 1)
            out = out + bilinear_gather(self.grid.reshape(1, -1), idx, gu, gv, gh, gw)[:, 0]
        return out

    def dense_depth(self):
        corr = torch.zeros_like(self.depth0)
        if self.cfg["depth_grid"] > 0:
            corr = corr + F.interpolate(self.grid[:, None], size=(self.H, self.W), mode="bilinear", align_corners=True)[:, 0]
        return self.depth0 * torch.exp(corr)

    def residuals(self, u, v, nbrs, img_flat):
        """Source pixels (u, v) [N, M] of every frame warped into its neighbours nbrs [N, k] (pinhole, without the radial
        coefficient; for eval_photo): target and source intensities after brightness [N, k, M, C] and [N, 1, M, C],
        geometric (log depth) residuals and the visibility mask [N, k, M]."""
        H, W, N = self.H, self.W, self.N
        R, t = self.poses()
        K = self.intrinsics()
        idx = torch.arange(N, device=u.device)[:, None].expand_as(u)
        base = self.depth0
        d_src = base[idx, v.long(), u.long()] * torch.exp(self.grid_at(idx.reshape(-1), u.reshape(-1), v.reshape(-1)).view_as(u))
        x, y = (u - K[:, None, 0, 2]) / K[:, None, 0, 0] * d_src, (v - K[:, None, 1, 2]) / K[:, None, 1, 1] * d_src
        Xc = torch.stack([x, y, d_src], -1)  # [N, M, 3]
        Xw = (Xc - t[:, None]) @ R  # R^T (X - t), row vectors
        Rt, tt, Kt = R[nbrs], t[nbrs], K[nbrs]  # [N, k, ...]
        Xt = Xw[:, None] @ Rt.transpose(-1, -2) + tt[:, :, None]  # [N, k, M, 3]
        z = Xt[..., 2]
        zc = z.clamp_min(1e-4)
        ut, vt = Kt[..., None, 0, 0] * Xt[..., 0] / zc + Kt[..., None, 0, 2], Kt[..., None, 1, 1] * Xt[..., 1] / zc + Kt[..., None, 1, 2]
        inside = (z > 1e-4) & (ut >= 0) & (ut <= W - 1) & (vt >= 0) & (vt <= H - 1)
        ut, vt = ut.clamp(0, W - 1), vt.clamp(0, H - 1)
        tgt = nbrs[:, :, None].expand_as(ut).reshape(-1)
        utf, vtf = ut.reshape(-1), vt.reshape(-1)
        d_tgt = bilinear_gather(base.reshape(1, -1), tgt, utf, vtf, H, W)[:, 0] * torch.exp(self.grid_at(tgt, utf, vtf))
        d_tgt = d_tgt.view_as(z)
        visible = inside & (z.detach() < d_tgt.detach() * (1 + self.cfg["occ_thresh"]))
        geo = torch.log(zc) - torch.log(d_tgt.clamp_min(1e-4))
        on = self.free[:, 0] * (self.cfg["features"] == "grey" and self.cfg["photo_loss"] != "ncc")
        a, b = self.bright[:, 0] * on, self.bright[:, 1] * on
        C = img_flat.shape[0]
        i_src = bilinear_gather(img_flat, idx.reshape(-1), u.reshape(-1), v.reshape(-1), H, W).view(N, 1, -1, C)
        i_tgt = bilinear_gather(img_flat, tgt, utf, vtf, H, W).view(*z.shape, C)
        i_tgt = i_tgt * torch.exp(a[nbrs] - a[:, None])[..., None, None] + (b[nbrs] - b[:, None])[..., None, None]
        return i_tgt, i_src, geo, visible


@torch.no_grad()
def eval_photo(model, nbrs, img_flat, n=8192, seed=1):
    """Fixed-sample photometric (RGB, sharp images) and geometric error, before and after, for the per-scene stats."""
    gen = torch.Generator(device=img_flat.device).manual_seed(seed)
    N, H, W = model.N, model.H, model.W
    probs = model.valid.reshape(N, -1).float()
    pix = torch.multinomial(probs, n, replacement=True, generator=gen)
    u, v = (pix % W).float(), (pix // W).float()
    i_tgt, i_src, geo, mask = model.residuals(u, v, nbrs, img_flat)
    photo = i_tgt - i_src
    m = mask.float()
    out = {
        "photo_l1": float((photo.abs().mean(-1) * m).sum() / m.sum()),
        "geo_l1": float((geo.abs() * m).sum() / m.sum()),
        "visible": float(m.mean()),
    }
    # distribution of the photometric residuals of visible (sample, neighbour) pairs: quantiles of |r|, robust sigma
    # (1.4826 MAD of the signed per-channel residuals) and the fraction of pairs beyond 0.05 / 0.1 / 0.2
    a = photo.abs().mean(-1)[mask]
    if a.numel():
        srt = a.sort().values
        for q in (50, 75, 90, 95, 99):
            out[f"photo_q{q}"] = float(srt[min(int(q / 100 * len(srt)), len(srt) - 1)])
        r = photo[mask].reshape(-1)
        r = r[:: max(1, r.numel() // 4_000_000)]
        out["photo_sigma"] = float(1.4826 * (r - r.median()).abs().median())
        for t in (5, 10, 20):
            out[f"photo_out{t:02d}"] = float((a > t / 100).float().mean())
    return out


# ---------------------------------------------------------------- the objective: gradients and Gauss-Newton blocks


class FlatAdam:
    """Adam on one vector holding all parameters (the parameters become views of it), with a learning rate per entry.
    Same update as torch.optim.Adam with one group per parameter, in 6 instead of ~50 tensor operations."""

    def __init__(self, params, lrs, betas=(0.9, 0.999), eps=1e-8):
        self.params, self.b1, self.b2, self.eps = params, betas[0], betas[1], eps
        self.x = torch.cat([p.data.reshape(-1) for p in params])
        self.g = torch.zeros_like(self.x)
        self.grads, o = [], 0
        for p in params:
            n = p.numel()
            p.data = self.x[o : o + n].view(p.shape)
            self.grads.append(self.g[o : o + n].view(p.shape))
            o += n
        self.lr = torch.cat([torch.full((p.numel(),), float(lr), device=self.x.device, dtype=self.x.dtype) for p, lr in zip(params, lrs)])
        self.reset()

    def reset(self):
        self.m, self.v, self.t = torch.zeros_like(self.x), torch.zeros_like(self.x), 0

    def step(self, scale=1.0):
        self.t += 1
        self.m.lerp_(self.g, 1 - self.b1)
        self.v.mul_(self.b2).addcmul_(self.g, self.g, value=1 - self.b2)
        denom = (self.v.sqrt() / math.sqrt(1 - self.b2**self.t)).add_(self.eps)
        self.x.sub_(self.lr * (self.m / denom) * (scale / (1 - self.b1**self.t)))


class Warps:
    """Forward quantities of a batch of warps, see Kernel.warp."""

    __slots__ = ("rad", "A", "E", "Xc", "Xt", "ds", "s", "Rji", "tji", "src", "nbrs", "iz", "ux", "vy", "I", "Iu", "Iv", "lu", "lv", "geo", "vis", "base_g", "wt", "fxt", "fyt")


class Kernel:
    """The objective at one pyramid level, its gradient (Adam) and its Gauss-Newton blocks (LM). All in torch.no_grad.

    A warp of source pixel p of frame i into neighbour j, in the camera of the source:

        X_c = (x, y, d) with d = d0(p) exp(g_i(p)), x = (u - cx) / f * d, y = (v - cy) / f * d
        X_t = R_ji X_c + t_ji,  R_ji = R_j R_i^T,  t_ji = t_j - R_ji t_i

    and the derivatives follow from the 3-vector dL/dX_t of each warp. With the pose parametrisation of PBA.poses
    (R = exp(w) R0, t = exp(w) t0 + tau) and phi = J_l(w) dw, the left tangent:

        target frame:  dX_t / dphi_j . a = (X_t - tau_j) x a      dX_t / dtau_j . a = a
        source frame:  dX_t / dphi_i . a = (X_c - tau_i) x (-R_ij a)     dX_t / dtau_i . a = -R_ij a

    so the pose gradient of a frame pair needs only sum(dL/dX_t) and sum(X_t x dL/dX_t). Photometric term: Huber on the
    grey image with relative brightness r = I_j exp(a_j - a_i) + b_j - b_i - I_i, Huber on features, or 1 - NCC of 3x3
    patches; geometric term: Huber on log(z_warped / depth_target); prior w_prior mean(grid^2). The code was checked against
    autograd, forward-mode AD and central differences in float64 (scripts/bench_pba.py at commit 1c1a921).
    """

    def __init__(self, model, cfg=None):
        cfg = cfg or model.cfg
        self.m, self.cfg = model, cfg
        self.N, self.H, self.W = model.N, model.H, model.W
        self.gh, self.gw = model.grid.shape[1:]
        self.nc = self.gh * self.gw
        self.sx, self.sy = (self.gw - 1) / (self.W - 1), (self.gh - 1) / (self.H - 1)
        self.has_grid, self.has_focal, self.has_pose = cfg["depth_grid"] > 0, cfg["focal"] == "shared", cfg["poses"]
        self.dp, self.dg = cfg["huber_photo"], cfg["huber_geo"]
        self.ncc = cfg["photo_loss"] == "ncc"
        self.radial = cfg["radial"]  # x_undistorted = x_distorted / (1 + k r_distorted^2)
        self.quad = torch.tensor([0, 1, self.gw, self.gw + 1], device=model.R0.device)
        self.frames = torch.arange(self.N, device=model.R0.device)
        self.fx0, self.fy0, self.cx, self.cy = model.K0[:, 0, 0], model.K0[:, 1, 1], model.K0[:, 0, 2], model.K0[:, 1, 2]
        dev, dt = model.R0.device, model.R0.dtype
        self.f0 = torch.stack([self.fx0, self.fy0], -1)
        self.c0 = torch.stack([self.cx, self.cy], -1)
        self.uv_lo = torch.zeros(2, device=dev, dtype=dt)
        self.uv_hi = torch.tensor([self.W - 1.0, self.H - 1.0], device=dev, dtype=dt)  # last pixel
        self.uv_base = torch.tensor([self.W - 2.0, self.H - 2.0], device=dev, dtype=dt)  # last pixel a bilinear cell can start at
        self.tex_quad = torch.tensor([0, 1, self.W, self.W + 1], device=dev)
        self.grid_scale = torch.tensor([self.sx, self.sy], device=dev, dtype=dt)
        self.grid_base = torch.tensor([self.gw - 2.0, self.gh - 2.0], device=dev, dtype=dt)
        self.use_bright = False

    def prep(self, nbrs):
        """What warp needs of the neighbours nbrs [n, k] and does not change during a level."""
        return self.f0[nbrs][:, :, None], self.c0[nbrs][:, :, None], nbrs[:, :, None] * (self.H * self.W), nbrs[:, :, None] * self.nc

    # ---------------------------------------------------------------- data of one level
    def set_level(self, img, use_bright, stride=1.0):
        """img [N, C, H, W]. The texture holds the channels and the depth, so that one lookup fetches both. use_bright:
        per-frame brightness (grey images with the Huber loss)."""
        self.C, self.stride = img.shape[1], stride
        self.tex = torch.cat([img.permute(0, 2, 3, 1).reshape(-1, self.C), self.m.depth0.reshape(-1, 1)], 1)
        self.use_bright = bool(use_bright)
        self.bright_on = self.m.free[:, 0] * float(self.use_bright)

    def make_pool(self, pix):
        """Candidate pixels pix [N, P] (long) -> per-sample constants feat [N, P, D] (image coordinates at the initial focal
        length x, y, depth, 4 bilinear grid weights, C channels) and grid quads [N, P], where D = 7 + C.
        With NCC: of the 3x3 patches around them, feat [N, P, 9, D + 1] and quads [N, P, 9], the source patch zero-mean,
        its squared norm in the last channel."""
        W, H = self.W, self.H
        u, v = (pix % W).float(), (pix // W).float()
        if self.ncc:
            off = torch.tensor([-1.0, 0.0, 1.0], device=u.device) * self.stride
            du_, dv_ = off[None, :].expand(3, 3).reshape(-1), off[:, None].expand(3, 3).reshape(-1)
            u, v = (u[..., None] + du_).clamp(0, W - 1), (v[..., None] + dv_).clamp(0, H - 1)
        frame = self.frames[:, None, None] if self.ncc else self.frames[:, None]
        u0, v0 = u.long(), v.long()
        flat = self.tex[frame * (H * W) + v0 * W + u0]
        if self.ncc and self.stride != int(self.stride):  # not on the pixel grid: bilinear, as bilinear_gather
            u1, v1 = u.floor().clamp_max(W - 2), v.floor().clamp_max(H - 2)
            a, b = (u - u1)[..., None], (v - v1)[..., None]
            base = frame * (H * W) + (v1 * W + u1).long()
            top = self.tex[base] * (1 - a) + self.tex[base + 1] * a
            bot = self.tex[base + W] * (1 - a) + self.tex[base + W + 1] * a
            ch = (top * (1 - b) + bot * b)[..., :-1]
            flat = torch.cat([ch, flat[..., -1:]], -1)
        gu, gv = u * self.sx, v * self.sy
        gu0, gv0 = gu.floor().clamp(0, self.gw - 2), gv.floor().clamp(0, self.gh - 2)
        du, dv = gu - gu0, gv - gv0
        sh = (self.N, 1, 1) if self.ncc else (self.N, 1)  # image coordinates at the initial focal length
        xb, yb = (u - self.cx.view(sh)) / self.fx0.view(sh), (v - self.cy.view(sh)) / self.fy0.view(sh)
        head = torch.stack([xb, yb, flat[..., -1], (1 - du) * (1 - dv), du * (1 - dv), (1 - du) * dv, du * dv], -1)
        ch = flat[..., :-1]
        if self.ncc:
            ch = ch - ch.mean((-2, -1), keepdim=True)
            aa = (ch * ch).sum((-2, -1), keepdim=True).expand(*ch.shape[:-1], 1)
            ch = torch.cat([ch, aa], -1)
        return torch.cat([head, ch], -1), gv0.long() * self.gw + gu0.long()

    def take(self, feat_pool, quad_pool, idx):
        """The candidates idx [N, P] of a pool as samples [N, M, D] (with NCC, M = 9 P)."""
        rows = self.frames[:, None]
        f, q = feat_pool[rows, idx], quad_pool[rows, idx]
        return (f.flatten(1, 2), q.flatten(1, 2)) if self.ncc else (f, q)

    # ---------------------------------------------------------------- state
    def state(self):
        m = self.m
        fr = m.free
        if self.has_pose:
            dR, Jl = so3_exp_jl(m.rot.data * fr)
            tau = m.trans.data * fr
            R = dR @ m.R0
            t = (dR @ m.t0[..., None])[..., 0] + tau
        else:
            R, Jl, tau, t = m.R0, None, torch.zeros_like(m.t0), m.t0
        s = torch.exp(m.logf.data) if self.has_focal else 1.0
        return R, t, Jl, tau, s, m.bright.data * self.bright_on[:, None] if self.use_bright else None

    # ---------------------------------------------------------------- forward
    def warp(self, st, src, feat, quad, nbrs, const):
        """Source frames src [n] (None: all frames) with samples feat [n, M, D], quad [n, M] into their neighbours nbrs
        [n, k], const = prep(nbrs). u and v are handled as one vector and the 4 corners of a bilinear lookup as one index:
        every tensor operation costs ~4 us on a GPU, whatever its size."""
        R, t, _, _, s_f, _ = st
        W = self.W
        w = Warps()
        f_t, c_t, off, goff = const
        fxy = f_t * s_f
        ws = feat[..., 3:7]
        grid = self.m.grid.data.view(self.N, -1)
        if self.has_grid:
            rows = self.frames[:, None, None] if src is None else src[:, None, None]
            ds = feat[..., 2] * torch.exp((grid[rows, quad[..., None] + self.quad] * ws).sum(-1))
        else:
            ds = feat[..., 2]
        Rs, ts = (R, t) if src is None else (R[src], t[src])
        xy = feat[..., :2] / s_f
        if self.radial:
            k_r = self.m.kr.data
            rd2 = (xy * xy).sum(-1)
            D = 1 + k_r * rd2
            xy = xy / D[..., None]
            w.A, w.E = (1 - k_r * rd2) / D, rd2 / D  # d log x_u / d(-log f) and d log x_u / d(-k), source side
        ds1 = ds[..., None]
        Xc = torch.cat([xy * ds1, ds1], -1)
        Rji = R[nbrs] @ Rs[:, None].transpose(-1, -2)
        tji = t[nbrs] - (Rji @ ts[:, None, :, None])[..., 0]
        Xt = Xc[:, None] @ Rji.transpose(-1, -2) + tji[:, :, None]
        z = Xt[..., 2]
        zc = z.clamp_min(1e-4)
        iz = 1 / zc
        Xn = Xt[..., :2] * iz[..., None]
        if self.radial:  # r_d = s r_u with s = 2 / (1 + sqrt(1 - 4 k r_u^2))
            xn, yn = Xn.unbind(-1)
            r2 = (Xn * Xn).sum(-1)
            q = torch.sqrt((1 - 4 * k_r * r2).clamp_min(1e-4))
            sc = 2 / (1 + q)
            w.rad = (xn, yn, r2, sc, k_r * sc * sc / q, q)  # the 5th is d s / d r_u^2
            uvx = fxy * sc[..., None] * Xn
        else:
            uvx = fxy * Xn
        uv = uvx + c_t
        uvc = uv.clamp(self.uv_lo, self.uv_hi)
        inside = (uv == uvc).all(-1) & (z > 1e-4)
        # texture: channels + depth, bilinear, with its derivatives in u and v
        uv0 = torch.minimum(uvc.floor(), self.uv_base)
        duv = uvc - uv0
        du, dv = duv[..., :1], duv[..., 1:]
        base = off + (uv0[..., 1] * W + uv0[..., 0]).long()
        a, b, c, d = self.tex[base[..., None] + self.tex_quad].unbind(-2)
        ab, cd = b - a, d - c
        top, bot = a + ab * du, c + cd * du
        Tv = bot - top
        val, Tu = top + Tv * dv, ab + (cd - ab) * dv
        # depth correction of the target at the warped pixel, with its derivatives
        if self.has_grid:
            guv = uvc * self.grid_scale
            guv0 = torch.minimum(guv.floor(), self.grid_base)
            gdu, gdv = (guv - guv0).unbind(-1)
            base_g = goff + (guv0[..., 1] * self.gw + guv0[..., 0]).long()
            a, b, c, d = grid.reshape(-1)[base_g[..., None] + self.quad].unbind(-1)
            ab, cd = b - a, d - c
            top, bot = a + ab * gdu, c + cd * gdu
            gv_ = bot - top
            g_t, g_u = top + gv_ * gdv, (ab + (cd - ab) * gdv) * self.sx
            g_v = gv_ * self.sy
            e_g = torch.exp(g_t)
            w.base_g, w.wt = base_g, (gdu, gdv)
        else:
            e_g, g_u, g_v = 1.0, 0.0, 0.0
            w.base_g, w.wt = None, None
        dt = val[..., -1] * e_g
        w.vis = inside & (z < dt * (1 + self.cfg["occ_thresh"]))
        idt = 1 / dt.clamp_min(1e-4)
        w.geo = torch.log(zc * idt)
        idD = e_g * idt
        w.lu, w.lv = Tu[..., -1] * idD + g_u, Tv[..., -1] * idD + g_v  # d log(depth of the target) / d(u, v)
        w.I, w.Iu, w.Iv = val[..., :-1], Tu[..., :-1], Tv[..., :-1]
        w.ux, w.vy = uvx.unbind(-1)
        w.fxt, w.fyt = fxy.unbind(-1)
        w.Xc, w.Xt, w.ds, w.s, w.Rji, w.tji, w.src, w.nbrs, w.iz = Xc, Xt, ds, feat[..., 7:], Rji, tji, src, nbrs, iz
        return w

    def photo_res(self, st, w):
        """Photometric residual [n, k, M, C] after the relative brightness r = I_j exp(a_j - a_i) + b_j - b_i - I_i (only
        the source's contrast counts, so the residual can't vanish by shrinking all frames' contrast), and exp(a_j - a_i)
        [n, k, 1, 1] (1.0 without brightness)."""
        br = st[-1]
        if br is None:
            return w.I - w.s[:, None], 1.0
        a, b = br.unbind(-1)
        a_s, b_s = (a, b) if w.src is None else (a[w.src], b[w.src])
        eat = torch.exp(a[w.nbrs] - a_s[:, None])[:, :, None, None]
        return w.I * eat + (b[w.nbrs] - b_s[:, None])[:, :, None, None] - w.s[:, None], eat

    def ncc_terms(self, w, m):
        """Per-patch NCC of the warped patches: zero-mean target and source patch [n, k, P, D], D = 9 C, squared norm of
        the source patch [n, 1, P], of the target [n, k, P], NCC [n, k, P] and the mask of fully visible patches."""
        n, k, M, C = *w.I.shape[:3], w.I.shape[3]
        P = M // 9
        b = w.I.reshape(n, k, P, 9 * C)
        b = b - b.mean(-1, keepdim=True)
        a = w.s[:, None, ..., :-1].reshape(n, 1, P, 9 * C)
        aa = w.s[:, None, ::9, -1]
        bb = (b * b).sum(-1)
        D = torch.sqrt(aa * bb + 1e-8)
        return b, a, aa, bb, D, (a * b).sum(-1) / D, m.reshape(n, k, P, 9).amin(-1)

    def priors(self):
        """Value of the non-data terms."""
        m, cfg = self.m, self.cfg
        v = 0.0
        if self.has_grid:
            v = v + cfg["w_prior"] * float((m.grid.data**2).mean())
        if cfg["w_pose"] > 0:
            v = v + cfg["w_pose"] * float(((m.rot.data * m.free) ** 2 + (m.trans.data * m.free) ** 2).mean())
        if self.radial:
            v = v + cfg["w_radial"] * float((m.kr.data**2).sum())
        return v

    def cost(self, feat, quad, nbrs, chunk):
        """The objective for samples feat [N, M, D]."""
        cfg = self.cfg
        st = self.state()
        sp, sg, n, np_ = 0.0, 0.0, 0.0, 0.0
        for i in range(0, self.N, chunk):
            src = self.frames[i : i + chunk]
            nb = nbrs[src]
            w = self.warp(st, src, feat[src], quad[src], nb, self.prep(nb))
            m = w.vis.float()
            if self.ncc:
                *_, ncc, mp = self.ncc_terms(w, m)
                sp, np_ = sp + ((1 - ncc) * mp).sum(), np_ + mp.sum()
            else:
                r, _ = self.photo_res(st, w)
                sp = sp + (huber(r, self.dp).mean(-1) * m).sum()
            sg, n = sg + (huber(w.geo, self.dg) * m).sum(), n + m.sum()
        n = max(float(n), 1.0)
        return cfg["w_photo"] * float(sp) / (max(float(np_), 1.0) if self.ncc else n) + cfg["w_geo"] * float(sg) / n + self.priors()

    # ---------------------------------------------------------------- gradient for Adam
    def gradient(self, feat, quad, nbrs, const, grads, want_loss=False):
        """Fill grads (rot, trans, logf, grid, bright, kr) with the gradient of the objective. Returns (total, parts) if
        want_loss. No float() in here: it would make every iteration wait for the GPU."""
        cfg, m_, N = self.cfg, self.m, self.N
        st = self.state()
        _, _, Jl, tau, _, br = st
        w = self.warp(st, None, feat, quad, nbrs, const)
        m = w.vis.float()
        den = m.sum().clamp_min(1)
        c = self.C
        dLg = m * huber_grad(w.geo, self.dg) * (cfg["w_geo"] / den)  # [N, k, M]
        if self.ncc:  # dL / d(warped value), L = (1 - NCC) over the fully visible patches
            b_, a_, aa, bb, D, ncc, mp = self.ncc_terms(w, m)
            den_p = mp.sum().clamp_min(1)
            dLt = ((a_ - (ncc * aa / D)[..., None] * b_) * (-cfg["w_photo"] * mp / (D * den_p))[..., None]).reshape(*w.I.shape)
            r = dLr = None
        else:
            r, eat = self.photo_res(st, w)
            dLr = m[..., None] * huber_grad(r, self.dp) * (cfg["w_photo"] / (c * den))  # [N, k, M, C]
            dLt = dLr * eat if br is not None else dLr
        Gu = (dLt * w.Iu).sum(-1) - dLg * w.lu
        Gv = (dLt * w.Iv).sum(-1) - dLg * w.lv
        T = Gu * w.ux + Gv * w.vy
        if self.radial:
            xn, yn, r2, sc, dsc, q = w.rad
            Pu, Pv = Gu * w.fxt, Gv * w.fyt  # dL / d(u, v) in normalised image coordinates
            cross_ = 2 * dsc * (Pu * xn + Pv * yn)  # from d s / d r^2: d L / d(x_n, y_n) = (Pu, Pv) s + cross_ (x_n, y_n)
            gX = torch.stack([(Pu * sc + cross_ * xn) * w.iz, (Pv * sc + cross_ * yn) * w.iz, (dLg - T - cross_ * r2) * w.iz], -1)
        else:
            gX = torch.stack([Gu * w.fxt * w.iz, Gv * w.fyt * w.iz, (dLg - T) * w.iz], -1)  # dL / dX_t
        rot, trans, logf, grid, bright = grads[:5]
        S1 = gX.sum(2)
        Xr = w.Xt - w.tji[:, :, None]  # R_ji X_c
        if self.has_pose:
            S2 = torch.linalg.cross(w.Xt, gX, dim=-1).sum(2)
            Rij = w.Rji.transpose(-1, -2)
            S1s = (Rij @ S1[..., None])[..., 0]  # R_ij S1, in the source camera
            S2s = (Rij @ (S2 - torch.linalg.cross(w.tji, S1, dim=-1))[..., None])[..., 0]
            S1ss = S1s.sum(1)
            trans.copy_(-S1ss)
            rot.copy_(torch.linalg.cross(tau, S1ss, dim=-1) - S2s.sum(1))
            j = nbrs.reshape(-1)
            trans.index_add_(0, j, S1.reshape(-1, 3))
            rot.index_add_(0, j, (S2 - torch.linalg.cross(tau[nbrs], S1, dim=-1)).reshape(-1, 3))
            rot.copy_((Jl.transpose(-1, -2) @ rot[..., None])[..., 0] * m_.free)
            trans.mul_(m_.free)
            if cfg["w_pose"] > 0:
                f = 2 * cfg["w_pose"] / (3 * N)
                rot.add_(m_.rot.data * m_.free * f)
                trans.add_(m_.trans.data * m_.free * f)
        else:
            rot.zero_(), trans.zero_()
        dLs = (gX * Xr).sum(-1)  # dL / dlog(source depth) per warp
        if self.has_grid:
            ws, gflat = feat[..., 3:7], grid.view(N, -1)
            gflat.zero_()
            gflat.scatter_add_(1, (quad[..., None] + self.quad).reshape(N, -1), (dLs.sum(1)[..., None] * ws).reshape(N, -1))
            gdu, gdv = w.wt
            a, b = dLg * (gdu - 1), -dLg * gdu  # -dLg (1 - gdu), -dLg gdu
            ga, gb, gw_ = a * gdv, b * gdv, self.gw
            base = w.base_g
            grid.reshape(-1).index_add_(0, torch.cat([base, base + 1, base + gw_, base + gw_ + 1], -1).reshape(-1),
                                        torch.cat([a - ga, b - gb, ga, gb], -1).reshape(-1))
            grid.add_(m_.grid.data, alpha=2 * cfg["w_prior"] / m_.grid.numel())
        else:
            grid.zero_()
        if self.has_focal or self.radial:
            r3 = w.Rji[..., 2][:, :, None]
            src_part = -dLs + w.ds[:, None] * (gX * r3).sum(-1)  # d L / d(-log x_u) of the source's (x, y)
        if self.has_focal:
            logf.fill_((T + (src_part * w.A[:, None] if self.radial else src_part)).sum())
        else:
            logf.zero_()
        if self.radial:
            grads[5].fill_((T * r2 * sc / q + src_part * w.E[:, None]).sum())
            grads[5].add_(m_.kr.data, alpha=2 * cfg["w_radial"])
        if br is not None:  # r = I_j exp(a_j - a_i) + b_j - b_i - I_i
            s_all = dLr.sum(-1).sum(2)  # [N, k]
            ga_, gb_ = bright[:, 0], bright[:, 1]
            ja = nbrs.reshape(-1)
            e_ = (dLr * w.I).sum((-1, -2)) * eat[:, :, 0, 0]
            ga_.copy_(-e_.sum(1))
            ga_.index_add_(0, ja, e_.reshape(-1))
            gb_.copy_(-s_all.sum(1))
            gb_.index_add_(0, ja, s_all.reshape(-1))
            bright.mul_(self.bright_on[:, None])
        else:
            bright.zero_()
        if not want_loss:
            return None
        lp = float(((1 - ncc) * mp).sum() / den_p) if self.ncc else float((huber(r, self.dp).mean(-1) * m).sum() / den)
        lg = float((huber(w.geo, self.dg) * m).sum() / den)
        total = cfg["w_photo"] * lp + cfg["w_geo"] * lg + self.priors()
        lprior = float((m_.grid.data**2).mean()) if self.has_grid else 0.0
        return total, {"photo": lp, "geo": lg, "prior": lprior, "visible": float(m.mean())}

    # ---------------------------------------------------------------- Gauss-Newton
    def normal_eq(self, feat, quad, nbrs, chunk):
        """Gauss-Newton blocks of the IRLS-weighted objective at the current parameters and samples feat [N, M, D].

        Unknowns: p = poses of frames 1..N-1 (rot, trans), the log focal, (the radial coefficient); g = the grid cells.
        Returns the blocks Hpp [np, np], Hgp [N, nc, np], Hgg [N, nc, nc] (the coupling of a frame's own cells), gp [np],
        gg [N, nc], the coupling of the cells of different frames Hx [N nc, N nc] and the loss.
        """
        cfg, N, nc = self.cfg, self.N, self.nc
        dev, st = self.frames.device, self.state()
        Jl, tau = st[2], st[3]
        n_pose = 6 * (N - 1) if self.has_pose else 0
        nq = 14 if self.radial else 13  # columns per warp: 2 x 6 poses, log focal, (radial coefficient)
        np_ = n_pose + 1 + int(self.radial)
        dummy = np_  # index of everything that is fixed (frame 0, or the poses or the focal when they are not optimised)
        pose_idx = torch.full((N, 6), dummy, dtype=torch.long, device=dev)
        if self.has_pose:
            pose_idx[1:] = torch.arange(n_pose, device=dev).view(N - 1, 6)
        focal_idx = n_pose if self.has_focal else dummy
        k_idx = n_pose + 1
        Hpp = torch.zeros((np_ + 1) ** 2, device=dev)
        gp = torch.zeros(np_ + 1, device=dev)
        has_grid = self.has_grid
        Hgp = torch.zeros(N * nc * (np_ + 1), device=dev) if has_grid else None
        Hgg = torch.zeros(N, nc, nc, device=dev) if has_grid else None
        gg = torch.zeros(N, nc, device=dev) if has_grid else None
        Hx = torch.zeros(N * nc, N * nc, device=dev) if has_grid else None
        sp, sg, n_vis, np_sum = 0.0, 0.0, 0.0, 0.0
        ncc_scale = 1.0
        if self.ncc:  # the rows are normalised by the visible samples at the end, NCC's by the visible patches
            nv, npatch = 0.0, 0.0
            for i in range(0, N, chunk):
                src = self.frames[i : i + chunk]
                nb = nbrs[src]
                w = self.warp(st, src, feat[src], quad[src], nb, self.prep(nb))
                m = w.vis.float()
                nv, npatch = nv + float(m.sum()), npatch + float(m.reshape(*m.shape[:2], -1, 9).amin(-1).sum())
            ncc_scale = max(nv, 1.0) / max(npatch, 1.0)
        for i in range(0, N, chunk):
            src = self.frames[i : i + chunk]
            nb = nbrs[src]
            n, k = len(src), nb.shape[1]
            w = self.warp(st, src, feat[src], quad[src], nb, self.prep(nb))
            M, C = w.I.shape[2], w.I.shape[3]
            m = w.vis.float()
            sg += float((huber(w.geo, self.dg) * m).sum())
            n_vis += float(m.sum())
            iz = w.iz
            if self.radial:
                xn, yn, r2, sc, dsc, q = w.rad
                fi, gi = w.fxt * iz, w.fyt * iz
                a_, xy = sc + 2 * r2 * dsc, 2 * xn * yn * dsc
                P1 = torch.stack([fi * (sc + 2 * xn * xn * dsc), fi * xy, -fi * xn * a_], -1)
                P2 = torch.stack([gi * xy, gi * (sc + 2 * yn * yn * dsc), -gi * yn * a_], -1)
                k_t = r2 * sc / q  # d u / dk = u_x k_t (and v)
            else:
                P1 = torch.stack([w.fxt * iz, torch.zeros_like(iz), -w.ux * iz], -1)  # d u / dX_t
                P2 = torch.stack([torch.zeros_like(iz), w.fyt * iz, -w.vy * iz], -1)  # d v / dX_t
            Xr = w.Xt - w.tji[:, :, None]
            Rji = w.Rji
            r3 = Rji[..., 2][:, :, None]
            Xs = (w.Xc - tau[src][:, None])[:, None]
            Xs_t = w.Xt - tau[nb][:, :, None]
            ds_ = w.ds[:, None]

            def rows(gX, T):
                """Columns rot_s, trans_s, rot_t, trans_t (in tangent coordinates), logf (and k) and dlog(source depth) of
                warps with X-space gradient gX [n, k, M, 3]."""
                ts = -(gX @ Rji)
                dds = (gX * Xr).sum(-1)
                sp_ = -dds + ds_ * (gX * r3).sum(-1)
                cols = [torch.linalg.cross(Xs, ts, dim=-1), ts, torch.linalg.cross(Xs_t, gX, dim=-1), gX]
                if self.radial:
                    cols += [(T + sp_ * w.A[:, None])[..., None], (T * k_t + sp_ * w.E[:, None])[..., None]]
                else:
                    cols += [(T + sp_)[..., None]]
                return torch.cat(cols, -1), dds

            # geometric row
            gXg = -w.lu[..., None] * P1 - w.lv[..., None] * P2
            gXg[..., 2] += iz
            Jg, dg_ = rows(gXg, -(w.lu * w.ux + w.lv * w.vy))
            wg = m * huber_weight(w.geo, self.dg) * cfg["w_geo"]
            if self.ncc:
                # photometric rows: difference of the normalised patches, half its squared norm is 1 - NCC
                b_, a_, aa, bb, D_, ncc, mp = self.ncc_terms(w, m)
                sp += float(((1 - ncc) * mp).sum())
                np_sum += float(mp.sum())
                Ju, du_ = rows(P1, w.ux)
                Jv, dv_ = rows(P2, w.vy)
                Jb = w.Iu[..., None] * torch.cat([Ju, du_[..., None]], -1)[..., None, :] + w.Iv[..., None] * torch.cat([Jv, dv_[..., None]], -1)[..., None, :]
                P_ = M // 9
                D = 9 * C
                Jb = Jb.reshape(n, k, P_, D, nq + 1)
                Jb = Jb - Jb.mean(-2, keepdim=True)
                nb_ = torch.sqrt(bb + 1e-8)
                bh = b_ / nb_[..., None]
                Jb = (Jb - bh[..., None] * (bh[..., None, :] @ Jb)) / nb_[..., None, None]
                r = bh - a_ / torch.sqrt(aa + 1e-8)[..., None]  # [n, k, P, D]
                Jp, dp_ = Jb[..., :nq], Jb[..., nq]
                wp = (mp * cfg["w_photo"] * ncc_scale)[..., None].expand_as(r)
            else:
                r, eat = self.photo_res(st, w)
                sp += float((huber(r, self.dp).mean(-1) * m).sum())
                # X-space gradient of the photometric rows [n, k, M, C, 3]: per channel
                E = eat[..., 0] if torch.is_tensor(eat) else eat
                Tp = (w.Iu * w.ux[..., None] + w.Iv * w.vy[..., None])
                if C == 1:
                    Jp, dp_ = rows((w.Iu[..., 0, None] * P1 + w.Iv[..., 0, None] * P2) * (E[..., None] if torch.is_tensor(E) else E), Tp[..., 0] * E)
                    Jp, dp_ = Jp[..., None, :], dp_[..., None]
                else:
                    Ju, du_ = rows(P1, w.ux)
                    Jv, dv_ = rows(P2, w.vy)
                    Ea = E[..., None, None] if torch.is_tensor(E) else E
                    Jp = (w.Iu[..., None] * Ju[..., None, :] + w.Iv[..., None] * Jv[..., None, :]) * Ea
                    dp_ = (w.Iu * du_[..., None] + w.Iv * dv_[..., None]) * (E[..., None] if torch.is_tensor(E) else E)
                wp = m[..., None] * huber_weight(r, self.dp) * (cfg["w_photo"] / C)
            if not self.has_focal:
                Jg[..., 12] = 0
                Jp[..., 12] = 0
            # pose and focal: one 13 x 13 block per frame pair
            Ap = Jp.reshape(n, k, -1, nq)
            wa = wp.reshape(n, k, -1)
            ra = r.reshape(n, k, -1)
            H13 = (Ap * wa[..., None]).transpose(-1, -2) @ Ap + (Jg * wg[..., None]).transpose(-1, -2) @ Jg
            g13 = (Ap * (wa * ra)[..., None]).sum(2) + (Jg * (wg * w.geo)[..., None]).sum(2)
            # to the coordinates of the parameters (dphi = J_l(rot) drot)
            T13 = torch.eye(nq, device=dev, dtype=H13.dtype).repeat(n, k, 1, 1)
            if self.has_pose:
                T13[..., 0:3, 0:3] = Jl[src][:, None].transpose(-1, -2).expand(n, k, 3, 3)
                T13[..., 6:9, 6:9] = Jl[nb].transpose(-1, -2)
            H13 = T13 @ H13 @ T13.transpose(-1, -2)
            g13 = (T13 @ g13[..., None])[..., 0]
            ids = [pose_idx[src][:, None].expand(n, k, 6), pose_idx[nb], torch.full((n, k, 1), focal_idx, device=dev)]
            if self.radial:
                ids.append(torch.full((n, k, 1), k_idx, device=dev))
            ids = torch.cat(ids, -1)
            Hpp.index_add_(0, (ids[..., None] * (np_ + 1) + ids[..., None, :]).reshape(-1), H13.reshape(-1))
            gp.index_add_(0, ids.reshape(-1), g13.reshape(-1))
            if not has_grid:
                continue
            # depth grid: dense bilinear weights [.., M, nc] of the source and the warped pixels
            ws = feat[src][..., 3:7]
            Ws = torch.zeros(n, M, nc, device=dev).scatter_(2, quad[src][..., None] + self.quad, ws)
            gdu, gdv = w.wt
            ct = w.base_g[..., None] - nb[:, :, None, None] * nc + self.quad  # cell within the target's grid
            wt = torch.stack([(1 - gdu) * (1 - gdv), gdu * (1 - gdv), (1 - gdu) * gdv, gdu * gdv], -1)
            Wt = torch.zeros(n, k, M, nc, device=dev).scatter_(3, ct, wt)
            # source role: the row's derivative in log depth is dp_ / dg_. The photometric rows of NCC belong to a patch
            # and use the mean weights of its samples
            dsp = (dp_ * wp)  # [n, k, P, Dp]
            Vp = (dsp[..., None] * Jp).sum(3)  # [n, k, P, 13]
            Vg = (dg_ * wg)[..., None] * Jg  # [n, k, M, 13]
            gsp = (dsp * r).sum(-1).sum(1)  # [n, P]
            gsg = (dg_ * wg * w.geo).sum(1)  # [n, M]
            h2p = (dsp * dp_).sum(-1).sum(1)
            h2g = (dg_ * dg_ * wg).sum(1)
            if self.ncc:
                Wp = Ws.reshape(n, -1, 9, nc).mean(2)  # [n, P, nc]
                Hs = Wp[:, None].transpose(-1, -2) @ Vp + Ws[:, None].transpose(-1, -2) @ Vg
                gg[src] += (Wp.transpose(-1, -2) @ gsp[..., None])[..., 0] + (Ws.transpose(-1, -2) @ gsg[..., None])[..., 0]
                Hgg[src] += (Wp * h2p[..., None]).transpose(-1, -2) @ Wp + (Ws * h2g[..., None]).transpose(-1, -2) @ Ws
            else:
                Hs = Ws[:, None].transpose(-1, -2) @ (Vp + Vg)  # [n, k, nc, 13]
                gg[src] += (Ws.transpose(-1, -2) @ (gsp + gsg)[..., None])[..., 0]
                Hgg[src] += (Ws * (h2p + h2g)[..., None]).transpose(-1, -2) @ Ws
            # target role: only the geometric row depends on the target's cells, with derivative -1
            Ht = Wt.transpose(-1, -2) @ (-wg[..., None] * Jg)  # [n, k, nc, 13]
            gt = Wt.transpose(-1, -2) @ (-wg * w.geo)[..., None]  # [n, k, nc, 1]
            gg.index_add_(0, nb.reshape(-1), gt[..., 0].reshape(-1, nc))
            # 16 entries per warp, by frame of the target (a matrix product would touch nc^2 per warp)
            fr = nb[:, :, None, None]
            flat = (fr * nc + ct)[..., None] * nc + ct[..., None, :]  # frame nc^2 + cell nc + cell
            Hgg.view(-1).index_add_(0, flat.reshape(-1), (wt[..., None] * wt[..., None, :] * wg[..., None, None]).reshape(-1))
            Tt = T13.transpose(-1, -2)
            Hs, Ht = Hs @ Tt, Ht @ Tt
            size = nc * (np_ + 1)
            cell = torch.arange(nc, device=dev)[None, None, :, None] * (np_ + 1)
            Hgp.index_add_(0, (src[:, None, None, None] * size + cell + ids[:, :, None]).reshape(-1), Hs.reshape(-1))
            Hgp.index_add_(0, (nb[:, :, None, None] * size + cell + ids[:, :, None]).reshape(-1), Ht.reshape(-1))
            # 16 entries per warp: cell of the source frame x cell of the target frame
            cs_ = src[:, None, None, None] * nc + (quad[src][..., None] + self.quad)[:, None]  # [n, 1, M, 4]
            ct_ = fr * nc + ct  # [n, k, M, 4]
            flat = cs_[..., None] * (N * nc) + ct_[..., None, :]
            vals = ws[:, None, :, :, None] * wt[..., None, :] * (-dg_ * wg)[..., None, None]
            Hx.view(-1).index_add_(0, flat.reshape(-1), vals.reshape(-1))
        n_vis = max(n_vis, 1.0)
        Hpp = Hpp.view(np_ + 1, np_ + 1)[:np_, :np_].double() / n_vis
        gp = gp[:np_].double() / n_vis
        if self.has_pose and cfg["w_pose"] > 0:
            f = 2 * cfg["w_pose"] / (3 * N)
            Hpp[range(n_pose), range(n_pose)] += f
            gp[:n_pose] += f * torch.cat([self.m.rot.data[1:], self.m.trans.data[1:]], 1).reshape(-1).double()
        if self.radial:  # keeps the coefficient near zero where the data do not determine it
            Hpp[k_idx, k_idx] += 2 * cfg["w_radial"]
            gp[k_idx] += 2 * cfg["w_radial"] * float(self.m.kr.data[0])
        if has_grid:
            Hgp = Hgp.view(N, nc, np_ + 1)[..., :np_].double() / n_vis
            Hgg = Hgg.double() / n_vis
            gg = gg.double() / n_vis
            f = 2 * cfg["w_prior"] / self.m.grid.numel()
            Hgg[:, range(nc), range(nc)] += f
            gg += f * self.m.grid.data.view(N, -1).double()
            Hx = (Hx + Hx.T).double() / n_vis
        loss = cfg["w_photo"] * sp / (max(np_sum, 1.0) if self.ncc else n_vis) + cfg["w_geo"] * sg / n_vis + self.priors()
        return Hpp, Hgp, Hgg, gp, gg, Hx, loss

    def apply(self, delta_p, delta_g, sign=1.0):
        m = self.m
        d = delta_p.float() * sign
        if self.has_pose:
            n = 6 * (self.N - 1)
            q = d[:n].view(self.N - 1, 6)
            m.rot.data[1:] += q[:, :3]
            m.trans.data[1:] += q[:, 3:]
            if self.has_focal:
                m.logf.data += d[n]
            if self.radial:
                m.kr.data += d[n + 1]
        else:
            if self.has_focal:
                m.logf.data += d[0]
            if self.radial:
                m.kr.data += d[1]
        if self.radial:
            m.kr.data.clamp_(max=self.cfg["radial_max"])
        if delta_g is not None:
            m.grid.data += delta_g.float().view_as(m.grid.data) * sign

    def lm_step(self, feat, quad, nbrs, chunk, lam, max_tries=6):
        """One Levenberg-Marquardt iteration on fixed samples. A step that lowers the cost is tried twice as long (the
        Gauss-Newton model overestimates the curvature of this large-residual objective). Returns the new damping, the cost
        before and after and whether a step was accepted."""
        cfg = self.cfg
        Hpp, Hgp, Hgg, gp, gg, Hx, cost0 = self.normal_eq(feat, quad, nbrs, chunk)
        if self.has_grid:
            dmax = max(float(Hpp.diagonal().max()), float(Hgg.diagonal(dim1=1, dim2=2).max()))
        else:
            dmax = float(Hpp.diagonal().max())
        floor = 1e-12 * dmax + 1e-30
        Dp = Hpp.diagonal().clamp_min(floor)
        Dg = Hgg.diagonal(dim1=1, dim2=2).clamp_min(floor) if self.has_grid else None
        params = (self.m.rot, self.m.trans, self.m.logf, self.m.grid, self.m.kr)
        state = [x.data.clone() for x in params]
        for _ in range(max_tries):
            step = self.solve(Hpp, Hgp, Hgg, gp, gg, Hx, Dp, Dg, lam)
            if step is not None:
                dp_, dg_ = step
                self.apply(dp_, dg_)
                cost1 = self.cost(feat, quad, nbrs, chunk)
                if cost1 < cost0:
                    self.apply(dp_, dg_)
                    cost2 = self.cost(feat, quad, nbrs, chunk)
                    lam_min = cfg["lm_lambda_min"]
                    if cost2 < cost1:
                        return max(lam / 3, lam_min), cost0, cost2, True
                    self.apply(dp_, dg_, -1.0)
                    return max(lam / 3, lam_min), cost0, cost1, True
                for x, s_ in zip(params, state):
                    x.data.copy_(s_)
            lam *= 4
        return lam, cost0, cost0, False

    def solve(self, Hpp, Hgp, Hgg, gp, gg, Hx, Dp, Dg, lam):
        """(H + lam diag(H)) delta = -g by a dense Cholesky of all unknowns, in single precision after Jacobi scaling
        (about half the time of float64 on a GPU, same result). None if not positive definite."""
        S = Hpp + torch.diag_embed(lam * Dp)
        if not self.has_grid:
            L, info = torch.linalg.cholesky_ex(S)
            return (-torch.cholesky_solve(gp[:, None], L)[:, 0], None) if int(info) == 0 else None
        A = Hgg + torch.diag_embed(lam * Dg)
        N, nc = self.N, self.nc
        n = len(gp)
        T = n + N * nc
        H = torch.zeros(T, T, device=gp.device, dtype=gp.dtype)
        H[:n, :n] = S
        Hgpf = Hgp.reshape(N * nc, n)
        H[n:, :n] = Hgpf
        H[:n, n:] = Hgpf.T
        H[n:, n:] = Hx
        H[n:, n:] += torch.block_diag(*A)
        rhs = torch.cat([gp, gg.reshape(-1)])
        sc = H.diagonal().clamp_min(1e-30).sqrt()
        L, info = torch.linalg.cholesky_ex((H / sc[:, None] / sc[None, :]).float())
        if int(info) != 0:
            return None
        d = -torch.cholesky_solve((rhs / sc).float()[:, None], L)[:, 0].double() / sc
        return d[:n], d[n:].view(N, nc)


# ---------------------------------------------------------------- one scene


def refine_scene(image_files, pred, cfg, device):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()

    depth = torch.from_numpy(pred["depth"]).float().to(device)
    N, H, W = depth.shape
    conf = torch.from_numpy(pred["conf"]).float().to(device)
    ext = torch.from_numpy(pred["extrinsics"]).float().to(device)
    K = torch.from_numpy(pred["intrinsics"]).float().to(device)
    images = load_images(image_files, H, W).to(device)
    model = PBA(images, depth, conf, ext, K, cfg).to(device)

    decode_s = time.perf_counter() - start
    with torch.no_grad():
        R, t = model.poses()
        nbrs = select_neighbors(model.depth0, R, t, model.intrinsics(), cfg["neighbors"], cfg["occ_thresh"], cfg["nbr_mode"], cfg["min_covis"])
    if device.type == "cuda":
        torch.cuda.synchronize()
    covis_s = time.perf_counter() - start - decode_s
    k = nbrs.shape[1]
    img_sharp = images.permute(1, 0, 2, 3).reshape(3, -1)
    info = {"num_frames": N, "neighbors": k, "decode_s": decode_s, "covis_s": covis_s, "feat_s": 0.0}
    before = eval_photo(model, nbrs[:, :10], img_sharp)  # (at most 10 neighbours: memory)
    load_s = time.perf_counter() - start

    levels = [float(s) for s in cfg["levels"].split(",") if s]  # empty: no optimisation (a reference run)
    lm_levels = {int(s) for s in cfg["lm_levels"].split(",") if s}
    M, iters = cfg["samples"], cfg["iters"]
    hist = []
    start = time.perf_counter()
    kernel = Kernel(model)
    fadam = FlatAdam([model.rot, model.trans, model.logf, model.grid, model.bright, model.kr],
                     [cfg["lr_rot"], cfg["lr_trans"], cfg["lr_focal"], cfg["lr_depth"], cfg["lr_bright"], cfg["lr_radial"]])
    lam = max(cfg["lm_lambda"], cfg["lm_lambda_min"])
    grey_huber = cfg["features"] == "grey" and cfg["photo_loss"] != "ncc"  # the brightness parameters are only used then
    maps = {"grey": images.mean(1, keepdim=True), "_grey": images.mean(1, keepdim=True)}  # _grey: the sampling distribution
    blurred = {}  # per (map, sigma): levels of the same blur share their images

    def blur(name, sig):
        if (name, sig) not in blurred:
            blurred[(name, sig)] = gaussian_blur(maps[name], sig)
        return blurred[(name, sig)]

    for level, sigma in enumerate(levels):
        if cfg["features"] not in maps:
            t_feat = time.perf_counter()
            maps["vgg"] = vgg_features(images, cfg["feat_dim"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            info["feat_s"] = time.perf_counter() - t_feat
        img = blur(cfg["features"], sigma)
        stride = max(1.0, sigma)  # NCC patch spacing grows with the blur
        kernel.set_level(img, grey_huber, stride)
        # a pool of candidate pixels per frame, drawn once per level (multinomial over all pixels is slow)
        probs = sampling_probs(img if cfg["features"] == "grey" else blur("_grey", sigma), model.valid, cfg["strat_cells"])
        pool = torch.multinomial(probs, POOL_FACTOR * M, replacement=True)
        n_pix = M if cfg["photo_loss"] != "ncc" else M // 9  # with NCC, M samples = M/9 patches of 9 pixels

        if level in lm_levels:
            if cfg["radial"] and cfg["radial_gate"] > 0 and kernel.radial:  # gate: keep k only if Adam moved it
                info["radial_k_gate"] = float(model.kr)
                if abs(info["radial_k_gate"]) < cfg["radial_gate"]:
                    with torch.no_grad():
                        model.kr.zero_()
                    kernel.radial = False
                    info["radial_gated"] = 1
            lm_loss = cfg["lm_photo_loss"] or cfg["photo_loss"]
            K_lm = kernel
            if lm_loss != cfg["photo_loss"]:
                K_lm = Kernel(model, {**cfg, "photo_loss": lm_loss})
                K_lm.radial = kernel.radial
                K_lm.set_level(img, cfg["features"] == "grey" and lm_loss != "ncc", stride)
            S = 9 if lm_loss == "ncc" else 1
            D = S * img.shape[1] + S
            if cfg["lm_samples"]:
                n_pix = cfg["lm_samples"] // S
                pool = torch.multinomial(probs, 4 * cfg["lm_samples"], replacement=True)
            chunk = max(1, int((1.6e7 if device.type == "cuda" else 4e6) / (k * n_pix * D)))  # source frames per block
            held_prev = None
            if cfg["lm_stop"] > 0:  # fixed held-out pixels with their own random stream
                gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 77)
                held = K_lm.make_pool(torch.multinomial(probs, cfg["lm_stop_samples"] // S, replacement=True, generator=gen))
                held = (held[0].flatten(1, 2), held[1].flatten(1, 2)) if S > 1 else held
                held_prev = K_lm.cost(*held, nbrs, chunk)
            for it in range(cfg["lm_iters"]):
                pix = pool.gather(1, torch.randint(0, pool.shape[1], (N, n_pix), device=device))
                feat, quad = K_lm.make_pool(pix)
                if S > 1:
                    feat, quad = feat.flatten(1, 2), quad.flatten(1, 2)
                lam, c0, c1, ok = K_lm.lm_step(feat, quad, nbrs, chunk, lam)
                hist.append({"level": level, "it": it, "loss": c0, "loss_after": c1, "lambda": lam, "accepted": ok, "t": time.perf_counter() - start})
                if held_prev is not None:
                    held_cur = K_lm.cost(*held, nbrs, chunk)
                    hist[-1]["heldout"] = held_cur
                    if (held_prev - held_cur) / max(held_prev, 1e-12) < cfg["lm_stop"]:
                        info["lm_stopped_at"] = it + 1
                        break
                    held_prev = held_cur
            continue

        # Adam (fresh state per level, cosine decay of the learning rate to 0.1 x); with NCC the pool of 16 M centres is
        # cut to 16 M / 9, each with its 9 samples
        fadam.reset()
        feat_pool, quad_pool = kernel.make_pool(pool[:, : POOL_FACTOR * n_pix])
        const = kernel.prep(nbrs)
        for it in range(iters):
            idx = torch.randint(0, feat_pool.shape[1], (N, n_pix), device=device)
            log = it % 20 == 0 or it == iters - 1
            res = kernel.gradient(*kernel.take(feat_pool, quad_pool, idx), nbrs, const, fadam.grads, want_loss=log)
            fadam.step(0.1 + 0.45 * (1 + math.cos(math.pi * it / iters)))
            if cfg["radial"]:
                model.kr.data.clamp_(max=cfg["radial_max"])
            if log:
                hist.append({"level": level, "it": it, "loss": res[0], **res[1], "t": time.perf_counter() - start})
    if device.type == "cuda":
        torch.cuda.synchronize()
    opt_s = time.perf_counter() - start
    after = eval_photo(model, nbrs[:, :10], img_sharp)

    with torch.no_grad():
        R, t = model.poses()
        ext_out = torch.cat([R, t[..., None] * model.scale], -1).cpu().numpy()
        K_out = model.intrinsics().cpu().numpy()
        depth_out = (model.dense_depth() * model.scale).cpu().numpy()
        f_scale = torch.exp(model.logf).cpu().numpy()
    for key, value in (("lm_stopped_at", 0), ("radial_k_gate", 0.0), ("radial_gated", 0)):  # the same keys in every scene:
        info.setdefault(key, value)  # the evaluator's mean takes them from the first one
    info.update(
        load_s=load_s,
        opt_s=opt_s,
        peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
        photo_before=before["photo_l1"],
        photo_after=after["photo_l1"],
        **{f"{k}_before": v for k, v in before.items() if k.startswith(("photo_q", "photo_sigma", "photo_out"))},
        **{f"{k}_after": v for k, v in after.items() if k.startswith(("photo_q", "photo_sigma", "photo_out"))},
        geo_before=before["geo_l1"],
        geo_after=after["geo_l1"],
        visible_before=before["visible"],
        visible_after=after["visible"],
        focal_scale=float(np.mean(f_scale)),
        radial_k=float(model.kr.detach()[0]),
        depth_corr_abs=float(np.abs(model.grid.detach().cpu().numpy()).mean()),
        bright_gain_abs=float(model.bright.detach()[:, 0].abs().mean()),
        bright_bias_abs=float(model.bright.detach()[:, 1].abs().mean()),
        bright_gain_mean=float(model.bright.detach()[1:, 0].mean()),
    )
    return {"depth": depth_out.astype(np.float32), "extrinsics": ext_out.astype(np.float32), "intrinsics": K_out.astype(np.float32)}, info, hist


def camera_spread(ext):
    """Mean distance of the camera centres from their mean, of world-to-camera extrinsics [N, 3, 4]."""
    c = -np.einsum("nji,nj->ni", ext[:, :, :3], ext[:, :, 3])
    return np.linalg.norm(c - c.mean(0), axis=1).mean()


def main():
    if len(sys.argv) < 3 or sys.argv[1].startswith("-"):
        sys.exit(__doc__)
    src, dst, cfg = sys.argv[1], sys.argv[2], parse_overrides(sys.argv[3:])
    sys.stdout.reconfigure(line_buffering=True)
    os.chdir(REPO_ROOT)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "pba_config.json"), "w") as f:
        json.dump({**cfg, "code_version": 2}, f, indent=2)  # 2: the single-file version of 2026-11-10 (scripts/index_runs.py)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = os.path.join(src, "model_results")
    datasets = [d for d in cfg["datasets"].split(",") if d] or (list(DEV) if cfg["subset"] else sorted(os.listdir(results_dir)))

    for dataset in datasets:
        first, count = cfg["first_scene"], cfg["max_scenes"]
        if cfg["subset"]:
            if dataset not in DEV:
                print(f"[run_tto_pba] {dataset}: not a dataset of subset={cfg['subset']}, skipped")
                continue
            first, count = (0, DEV[dataset]) if cfg["subset"] == "dev" else (DEV[dataset], 0)
        data_dir = os.path.join(results_dir, dataset)
        metric_path = os.path.join(dst, "metric_results", f"{dataset}_pba.json")
        hist_path = os.path.join(dst, "metric_results", f"{dataset}_pba_hist.jsonl")
        os.makedirs(os.path.dirname(metric_path), exist_ok=True)
        metrics = json.load(open(metric_path)) if os.path.exists(metric_path) else {}
        metrics.pop("mean", None)  # recomputed after every scene, so an interrupted run can be resumed

        scenes = sorted(os.path.dirname(p) for p, _, _ in os.walk(data_dir) if os.path.basename(p) == "unposed")
        for i, scene_dir in enumerate(scenes[first : first + count if count else None], first):
            scene = os.path.relpath(scene_dir, data_dir)
            if (cfg["scenes"] and scene not in cfg["scenes"].split(",")) or scene in metrics:
                continue
            exports = os.path.join(scene_dir, "unposed", "exports")
            gt = dict(np.load(os.path.join(exports, "gt_meta.npz"), allow_pickle=True))
            pred = dict(np.load(os.path.join(exports, "mini_npz", "results.npz")))
            if cfg["init_from"]:
                other = os.path.join(cfg["init_from"], "model_results", dataset, scene, "unposed", "exports", "mini_npz", "results.npz")
                if not os.path.exists(other):
                    print(f"[run_tto_pba] {dataset} {scene}: no {other}, skipped")
                    continue
                other = np.load(other)
                ext = other["extrinsics"][:, :3].astype(np.float32)
                # the other run's scale is arbitrary (COLMAP BA leaves it free): match DA3's camera-centre spread,
                # otherwise its translations are inconsistent with DA3's depth maps
                ext[:, :, 3] *= camera_spread(pred["extrinsics"]) / max(camera_spread(ext), 1e-12)
                pred.update(extrinsics=ext, intrinsics=other["intrinsics"].astype(np.float32))
            if cfg["init_gt"]:  # GT poses in DA3's scale (and GT focal at the prediction's resolution)
                ext = gt["extrinsics"][:, :3].astype(np.float32).copy()
                ext[:, :, 3] *= camera_spread(pred["extrinsics"]) / max(camera_spread(ext), 1e-12)
                pred.update(extrinsics=ext)
                if cfg["init_gt"] == "pose_focal":
                    w_orig = cv2.imread(str(gt["image_files"][0])).shape[1]
                    k_ = pred["intrinsics"].astype(np.float32).copy()
                    k_[:, 0, 0] = gt["intrinsics"][:, 0, 0] * pred["depth"].shape[2] / w_orig
                    k_[:, 1, 1] = gt["intrinsics"][:, 1, 1] * pred["depth"].shape[2] / w_orig
                    pred.update(intrinsics=k_)
            try:
                out, info, hist = refine_scene(gt["image_files"], pred, cfg, device)
            except torch.cuda.OutOfMemoryError as e:  # not stored, so a rerun retries it
                print(f"[run_tto_pba] {dataset} {i + 1}/{len(scenes)} {scene} FAILED: {e!r}")
                torch.cuda.empty_cache()
                continue

            if cfg["save"]:
                out_exports = os.path.join(dst, "model_results", dataset, scene, "unposed", "exports")
                os.makedirs(os.path.join(out_exports, "mini_npz"), exist_ok=True)
                np.savez(os.path.join(out_exports, "mini_npz", "results.npz"), conf=pred["conf"], **out)
                shutil.copy(os.path.join(exports, "gt_meta.npz"), os.path.join(out_exports, "gt_meta.npz"))

            gt_ext = torch.from_numpy(as_homogeneous(gt["extrinsics"]))
            pose0 = compute_pose(torch.from_numpy(as_homogeneous(pred["extrinsics"])), gt_ext)
            pose = compute_pose(torch.from_numpy(as_homogeneous(out["extrinsics"])), gt_ext)
            gt_f = gt["intrinsics"][:, 0, 0] * pred["depth"].shape[2] / cv2.imread(str(gt["image_files"][0])).shape[1]
            focal_err0 = np.abs(pred["intrinsics"][:, 0, 0] / gt_f - 1).mean()
            focal_err = np.abs(out["intrinsics"][:, 0, 0] / gt_f - 1).mean()
            rec = {**pose, **{f"{k}_init": v for k, v in pose0.items()}, "focal_err": focal_err, "focal_err_init": focal_err0, **info}
            metrics[scene] = Evaluator._to_float_dict(rec)
            Evaluator._dump_json(metric_path, {**metrics, "mean": Evaluator._mean_of_dicts(metrics.values())})
            with open(hist_path, "a") as f:
                f.write(json.dumps({"scene": scene, "hist": hist}) + "\n")
            m = metrics[scene]
            print(
                f"[run_tto_pba] {dataset} {i + 1}/{len(scenes)} {scene}: auc03 {m['auc03_init']:.3f}->{m['auc03']:.3f} "
                f"auc30 {m['auc30_init']:.3f}->{m['auc30']:.3f} focal_err {focal_err0:.3f}->{focal_err:.3f} "
                f"photo {m['photo_before']:.4f}->{m['photo_after']:.4f} geo {m['geo_before']:.4f}->{m['geo_after']:.4f} "
                f"t {m['opt_s']:.1f}s mem {m['peak_mem_gib']:.2f}G"
            )
        if metrics:
            mean = Evaluator._mean_of_dicts(metrics.values())
            print(f"[run_tto_pba] {dataset} mean: " + " ".join(f"{k} {v:.4f}" for k, v in mean.items()))


if __name__ == "__main__":
    main()
