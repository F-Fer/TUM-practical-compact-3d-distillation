"""Refine saved DA3 predictions with photometric bundle adjustment (dense, direct).

Usage (from anywhere; paths are resolved relative to the repo root):
    .venv/bin/python scripts/run_tto_pba.py <src work dir> <dst work dir> [key=value ...]

    .venv/bin/python scripts/run_tto_pba.py workspace/evaluation/DA3-SMALL workspace/tto/DA3-SMALL-pba datasets=hiroom

Optimises, per scene, the camera poses (frame 0 fixed), focal lengths, a low-resolution log-depth
correction grid per frame and a per-frame affine brightness, so that pixels of each frame reprojected
into its covisible neighbours match in colour (photometric term) and depth (geometric term).
Coarse-to-fine over blurred images, Adam, random gradient-weighted pixel samples per iteration.

Outputs go to <dst work dir>, mirroring <src work dir>:
    model_results/    refined depth, extrinsics and intrinsics per scene (unposed/exports/mini_npz/results.npz)
                      plus a copy of gt_meta.npz, so that `run_bench.py ... eval.eval_only=true` scores them
    metric_results/   <dataset>_pose.json as written by the DA3 evaluator, with per-scene PBA stats added
                      (renamed to <dataset>_pba.json, so that the evaluator's eval_only run can write its own)
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
    "datasets": "",  # comma-separated, default: all in <src>/model_results
    "max_scenes": 0,  # per dataset, 0 = all
    "scenes": "",  # comma-separated scene names, default: all
    "neighbors": 6,  # covisible target frames per source frame
    "nbr_mode": "top",  # top (most covisible) | spread (evenly over the covisibility ranking, see select_neighbors)
    "min_covis": 0.2,
    "samples": 4096,  # source pixels per frame and iteration
    "levels": "8,4,2,1",  # blur sigma (px) of the coarse-to-fine levels, 0 = sharp
    "iters": 100,  # per level (L-BFGS: iterations, each with one or more evaluations in the line search)
    "optimizer": "adam",  # adam (fresh random samples per iteration) | lbfgs (fixed samples per level, strong Wolfe)
    "lbfgs_samples": 4,  # L-BFGS: fixed samples per frame = lbfgs_samples * samples
    "lr_rot": 1e-3,  # rad
    "lr_trans": 1e-3,  # in units of the median depth
    "lr_focal": 1e-3,  # log focal
    "lr_depth": 5e-3,  # log depth grid
    "lr_bright": 1e-2,
    "focal": "shared",  # none | shared (one scale for all frames) | per_frame
    "focal_init": "pred",  # pred | median (replace per-frame focals with their median first)
    "depth_grid": 8,  # cells along the longer side of the log-depth correction grid, 0 = depth fixed
    "poses": True,  # optimise poses
    "w_photo": 1.0,
    "w_geo": 1.0,  # multi-view depth consistency
    "w_prior": 1.0,  # keeps the log-depth correction near zero (also fixes the scale gauge)
    "w_focal": 0.0,  # keeps the log focal scale near zero (DA3's focal); focal and depth trade off on narrow-FOV scenes
    "huber_photo": 0.05,  # intensities in [0, 1]
    "huber_geo": 0.05,  # log depth
    "occ_thresh": 0.05,  # relative depth; a sample is occluded in the target if it lies further away
    "conf_quantile": 0.1,  # source pixels below this DA3 confidence quantile (per scene) are not sampled
    "photo_loss": "huber",  # huber (per pixel, after affine brightness) | ncc (1 - NCC of 3x3 patches)
    "patch_stride": 1.0,  # px between patch samples, multiplied by max(1, blur sigma)
    "grad_weight": 1.0,  # sampling probability ~ |grad I|^grad_weight (0 = uniform)
    "features": "rgb",  # rgb (photometric) | vgg (feature-metric: VGG16 relu1_2 + relu2_2, PCA-reduced)
    "feat_dim": 16,
    "init_from": "",  # work dir whose extrinsics/intrinsics replace DA3's (e.g. the sparse-BA output), depth stays DA3's
    "seed": 0,
    "save": True,  # write results.npz (False: only metrics)
}


def parse_overrides(overrides):
    """key=value pairs, converted to the type of the default."""
    cfg = dict(DEFAULTS)
    for o in overrides:
        key, _, value = o.partition("=")
        if key not in DEFAULTS:
            sys.exit(f"unknown override {key!r}, expected one of {list(DEFAULTS)}")
        default = DEFAULTS[key]
        cfg[key] = value.lower() == "true" if isinstance(default, bool) else type(default)(value)
    return cfg


# ---------------------------------------------------------------- geometry


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


def bilinear_gather(src, idx, u, v, H, W):
    """Bilinearly sample src [C, N*H*W] (N frames, flattened) at frame idx [S], pixel (u, v) [S] -> [S, C].

    Differentiable w.r.t. src and (u, v); (u, v) must be inside [0, W-1] x [0, H-1].
    """
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


def huber(r, delta):
    a = r.abs()
    return torch.where(a < delta, 0.5 * r * r / delta, a - 0.5 * delta)


# ---------------------------------------------------------------- scene


def load_images(image_files, H, W):
    """RGB in [0, 1], resized to the prediction's resolution (the aspect ratios match exactly)."""
    imgs = [cv2.resize(cv2.imread(str(f))[..., ::-1], (W, H), interpolation=cv2.INTER_AREA) for f in image_files]
    return torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255


@torch.no_grad()
def vgg_features(images, dim, chunk=2, scale=0.1):
    """VGG16 relu1_2 and relu2_2 (upsampled) of images [N, 3, H, W], PCA-reduced to dim channels whose
    per-channel RMS is scale on average (comparable to intensities, so the same Huber threshold applies)."""
    import torchvision

    vgg = torchvision.models.vgg16(weights="IMAGENET1K_V1").features[:9].to(images.device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device)[:, None, None]
    H, W = images.shape[-2:]

    def feats(x):
        f1 = vgg[:4]((x - mean) / std)
        f2 = F.interpolate(vgg[4:9](f1), size=(H, W), mode="bilinear", align_corners=False)
        return torch.cat([f1 / f1.std(), f2 / f2.std()], 1)  # both layers equally weighted

    # PCA basis from a random subset of pixels of every frame
    gen = torch.Generator(device=images.device).manual_seed(0)
    samples = []
    for i in range(0, len(images), chunk):
        f = feats(images[i : i + chunk]).flatten(2)  # [n, C, HW]
        idx = torch.randint(0, f.shape[2], (2048,), device=f.device, generator=gen)
        samples.append(f[:, :, idx].permute(0, 2, 1).reshape(-1, f.shape[1]))
    X = torch.cat(samples)
    mu = X.mean(0)
    _, S, V = torch.linalg.svd(X - mu, full_matrices=False)
    P = V[:dim].T  # [C, dim]
    norm = scale / (S[:dim] ** 2 / len(X)).mean().sqrt()
    out = torch.empty(len(images), dim, H, W, device=images.device)
    for i in range(0, len(images), chunk):
        f = feats(images[i : i + chunk])
        out[i : i + chunk] = torch.einsum("nchw,cd->ndhw", f - mu[:, None, None], P) * norm
    del vgg
    return out


@torch.no_grad()
def select_neighbors(depth, R, t, K, k, occ_thresh, mode="top", min_covis=0.2, n_grid=24):
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
        return order[:, :k], score
    n_ok = (score >= min_covis).sum(1).clamp_min(k)  # at least k candidates per frame
    pos = torch.linspace(0, 1, k, device=depth.device)[None] * (n_ok[:, None] - 1).float()
    return order.gather(1, pos.round().long()), score


class PBA(torch.nn.Module):
    def __init__(self, images, depth, conf, ext, K, cfg):
        super().__init__()
        N, _, H, W = images.shape
        dev = images.device
        self.cfg, self.N, self.H, self.W = cfg, N, H, W
        self.scale = float(depth.median())  # scene normalised to median depth 1
        self.register_buffer("depth0", depth / self.scale)
        self.register_buffer("R0", ext[:, :, :3].clone())
        self.register_buffer("t0", ext[:, :, 3] / self.scale)
        K = K.clone()
        if cfg["focal_init"] == "median":
            K[:, 0, 0] = K[:, 0, 0].median()
            K[:, 1, 1] = K[:, 1, 1].median()
        self.register_buffer("K0", K)
        self.register_buffer("images", images)
        self.register_buffer("free", (torch.arange(N, device=dev) > 0).float()[:, None])  # frame 0 fixes the gauge

        self.rot = torch.nn.Parameter(torch.zeros(N, 3, device=dev))
        self.trans = torch.nn.Parameter(torch.zeros(N, 3, device=dev))
        self.logf = torch.nn.Parameter(torch.zeros(N if cfg["focal"] == "per_frame" else 1, device=dev))
        g = max(cfg["depth_grid"], 1)
        gh, gw = (g, max(2, round(g * W / H))) if H > W else (max(2, round(g * H / W)), g)
        self.grid = torch.nn.Parameter(torch.zeros(N, gh, gw, device=dev))
        self.bright = torch.nn.Parameter(torch.zeros(N, 2, device=dev))  # log gain, bias

        # sampling mask: confident, not at the border
        thr = torch.quantile(conf.reshape(-1)[:: conf.numel() // 2**22 + 1], cfg["conf_quantile"]) if cfg["conf_quantile"] > 0 else -1
        valid = conf > thr
        valid[(valid.flatten(1).float().mean(1) < 0.01)] = True  # frames with (almost) no confident pixels: use all
        valid[:, :2], valid[:, -2:], valid[:, :, :2], valid[:, :, -2:] = False, False, False, False
        self.register_buffer("valid", valid)

    # --- current state
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
        if self.cfg["depth_grid"] <= 0:
            return torch.zeros_like(u)
        N, gh, gw = self.grid.shape
        gu, gv = u * (gw - 1) / (self.W - 1), v * (gh - 1) / (self.H - 1)
        return bilinear_gather(self.grid.reshape(1, -1), idx, gu, gv, gh, gw)[:, 0]

    def dense_depth(self):
        if self.cfg["depth_grid"] <= 0:
            return self.depth0
        corr = F.interpolate(self.grid[:, None], size=(self.H, self.W), mode="bilinear", align_corners=True)[:, 0]
        return self.depth0 * torch.exp(corr)

    # --- residuals
    def residuals(self, u, v, nbrs, img_flat):
        """Reproject source pixels (u, v) [N, M] of every frame into its neighbours nbrs [N, k].

        Returns target and source intensities (after brightness) [N, k, M, C] and [N, 1, M, C], geometric (log depth) [N, k, M] residuals and the visibility mask [N, k, M].
        Per-frame quantities are broadcast rather than gathered per sample (much faster backward).
        """
        H, W, N = self.H, self.W, self.N
        R, t = self.poses()
        K = self.intrinsics()
        idx = torch.arange(N, device=u.device)[:, None].expand_as(u)
        d_src = self.depth0[idx, v.long(), u.long()] * torch.exp(self.grid_at(idx.reshape(-1), u.reshape(-1), v.reshape(-1)).view_as(u))
        x = (u - K[:, None, 0, 2]) / K[:, None, 0, 0] * d_src
        y = (v - K[:, None, 1, 2]) / K[:, None, 1, 1] * d_src
        Xc = torch.stack([x, y, d_src], -1)  # [N, M, 3]
        Xw = (Xc - t[:, None]) @ R  # R^T (X - t), row vectors
        Rt, tt, Kt = R[nbrs], t[nbrs], K[nbrs]  # [N, k, ...]
        Xt = Xw[:, None] @ Rt.transpose(-1, -2) + tt[:, :, None]  # [N, k, M, 3]
        z = Xt[..., 2]
        zc = z.clamp_min(1e-4)
        ut = Kt[..., None, 0, 0] * Xt[..., 0] / zc + Kt[..., None, 0, 2]
        vt = Kt[..., None, 1, 1] * Xt[..., 1] / zc + Kt[..., None, 1, 2]
        inside = (z > 1e-4) & (ut >= 0) & (ut <= W - 1) & (vt >= 0) & (vt <= H - 1)
        ut, vt = ut.clamp(0, W - 1), vt.clamp(0, H - 1)

        tgt = nbrs[:, :, None].expand_as(ut).reshape(-1)
        utf, vtf = ut.reshape(-1), vt.reshape(-1)
        d_tgt = bilinear_gather(self.depth0.reshape(1, -1), tgt, utf, vtf, H, W)[:, 0] * torch.exp(self.grid_at(tgt, utf, vtf))
        d_tgt = d_tgt.view_as(z)
        visible = inside & (z.detach() < d_tgt.detach() * (1 + self.cfg["occ_thresh"]))
        geo = torch.log(zc) - torch.log(d_tgt.clamp_min(1e-4))

        # affine brightness only for intensities with the Huber loss (NCC and features are invariant to it, and
        # Adam would turn its ~0 gradient into a random walk)
        on = self.free[:, 0] * (self.cfg["features"] == "rgb" and self.cfg["photo_loss"] == "huber")
        a, b = self.bright[:, 0] * on, self.bright[:, 1] * on
        C = img_flat.shape[0]
        i_src = bilinear_gather(img_flat, idx.reshape(-1), u.reshape(-1), v.reshape(-1), H, W).view(N, 1, -1, C)
        i_src = i_src * torch.exp(a)[:, None, None, None] + b[:, None, None, None]
        i_tgt = bilinear_gather(img_flat, tgt, utf, vtf, H, W).view(*z.shape, C)
        i_tgt = i_tgt * torch.exp(a[nbrs])[..., None, None] + b[nbrs][..., None, None]
        return i_tgt, i_src, geo, visible

    def loss(self, u, v, nbrs, img_flat, stride=1.0):
        cfg = self.cfg
        if cfg["photo_loss"] == "ncc":
            # u, v are patch centres [N, P]; expand to 3x3 patches [N, 9P]
            off = torch.tensor([-1.0, 0.0, 1.0], device=u.device) * stride
            du, dv = off[None, :].expand(3, 3).reshape(-1), off[:, None].expand(3, 3).reshape(-1)
            N, P = u.shape
            u = (u[..., None] + du).clamp(0, self.W - 1).reshape(N, -1)
            v = (v[..., None] + dv).clamp(0, self.H - 1).reshape(N, -1)
        i_tgt, i_src, geo, mask = self.residuals(u, v, nbrs, img_flat)
        m = mask.float()
        denom = m.sum().clamp_min(1)
        if cfg["photo_loss"] == "ncc":
            shp = (*i_tgt.shape[:2], P, 9, i_tgt.shape[-1])
            a, b = i_src.expand_as(i_tgt).reshape(shp), i_tgt.reshape(shp)
            a = a - a.mean((-2, -1), keepdim=True)
            b = b - b.mean((-2, -1), keepdim=True)
            ncc = (a * b).sum((-2, -1)) / ((a * a).sum((-2, -1)) * (b * b).sum((-2, -1)) + 1e-8).sqrt()
            mp = m.reshape(*shp[:-1]).amin(-1)  # patch visible in all its samples
            l_photo = ((1 - ncc) * mp).sum() / mp.sum().clamp_min(1)
        else:
            l_photo = (huber(i_tgt - i_src, cfg["huber_photo"]).mean(-1) * m).sum() / denom
        l_geo = (huber(geo, cfg["huber_geo"]) * m).sum() / denom
        l_prior = (self.grid**2).mean() if cfg["depth_grid"] > 0 else torch.zeros((), device=u.device)
        total = cfg["w_photo"] * l_photo + cfg["w_geo"] * l_geo + cfg["w_prior"] * l_prior + cfg["w_focal"] * (self.logf**2).mean()
        return total, {"photo": l_photo.detach(), "geo": l_geo.detach(), "prior": l_prior.detach(), "visible": m.mean()}


def sampling_probs(img, valid, grad_weight):
    """[N, H*W] sampling probability of source pixels: valid and ~|grad I|^grad_weight."""
    gray = img.mean(1, keepdim=True)
    gx = F.pad(gray[..., :, 2:] - gray[..., :, :-2], (1, 1, 0, 0))
    gy = F.pad(gray[..., 2:, :] - gray[..., :-2, :], (0, 0, 1, 1))
    g = torch.sqrt(gx**2 + gy**2)[:, 0]
    p = (g + 1e-3) ** grad_weight * valid
    return p.reshape(len(p), -1) + 1e-12 * valid.reshape(len(p), -1)


@torch.no_grad()
def eval_photo(model, nbrs, img_flat, n=8192, seed=1):
    """Fixed-sample photometric/geometric error (sharp images) for before/after logging."""
    gen = torch.Generator(device=img_flat.device).manual_seed(seed)
    N, H, W = model.N, model.H, model.W
    probs = model.valid.reshape(N, -1).float()
    pix = torch.multinomial(probs, n, replacement=True, generator=gen)
    u, v = (pix % W).float(), (pix // W).float()
    i_tgt, i_src, geo, mask = model.residuals(u, v, nbrs, img_flat)
    photo = i_tgt - i_src
    m = mask.float()
    return {
        "photo_l1": float((photo.abs().mean(-1) * m).sum() / m.sum()),
        "geo_l1": float((geo.abs() * m).sum() / m.sum()),
        "visible": float(m.mean()),
    }


def refine_scene(image_files, pred, cfg, device):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()

    depth = torch.from_numpy(pred["depth"]).float().to(device)
    N, H, W = depth.shape
    images = load_images(image_files, H, W).to(device)
    conf = torch.from_numpy(pred["conf"]).float().to(device)
    ext = torch.from_numpy(pred["extrinsics"]).float().to(device)
    K = torch.from_numpy(pred["intrinsics"]).float().to(device)
    model = PBA(images, depth, conf, ext, K, cfg).to(device)

    with torch.no_grad():
        R, t = model.poses()
        nbrs, _ = select_neighbors(model.depth0, R, t, model.intrinsics(), cfg["neighbors"], cfg["occ_thresh"], cfg["nbr_mode"], cfg["min_covis"])
    k = nbrs.shape[1]
    img_sharp = images.permute(1, 0, 2, 3).reshape(3, -1)
    info = {"num_frames": N, "neighbors": k}
    before = eval_photo(model, nbrs, img_sharp)
    load_s = time.perf_counter() - start

    groups = [
        {"params": [model.rot], "lr": cfg["lr_rot"]},
        {"params": [model.trans], "lr": cfg["lr_trans"]},
        {"params": [model.logf], "lr": cfg["lr_focal"]},
        {"params": [model.grid], "lr": cfg["lr_depth"]},
        {"params": [model.bright], "lr": cfg["lr_bright"]},
    ]
    levels = [float(s) for s in cfg["levels"].split(",") if s]  # empty: no optimisation (reference run)
    M = cfg["samples"]
    hist = []
    start = time.perf_counter()
    feats = vgg_features(images, cfg["feat_dim"]) if cfg["features"] == "vgg" and levels else images
    for level, sigma in enumerate(levels):
        img = gaussian_blur(feats, sigma)
        img_flat = img.permute(1, 0, 2, 3).reshape(img.shape[1], -1)
        # a pool of candidate pixels per frame, drawn once per level (multinomial over all pixels is slow)
        probs = sampling_probs(gaussian_blur(images, sigma), model.valid, cfg["grad_weight"])
        pool = torch.multinomial(probs, 16 * M, replacement=True)
        stride = cfg["patch_stride"] * max(1.0, sigma)
        n_pix = M if cfg["photo_loss"] == "huber" else M // 9  # with NCC, M samples = M/9 patches of 9 pixels
        if cfg["optimizer"] == "lbfgs":
            pix = pool[:, : cfg["lbfgs_samples"] * n_pix]
            u, v = (pix % W).float(), (pix // W).float()
            params = [p for g in groups for p in g["params"]]
            opt = torch.optim.LBFGS(params, lr=1, max_iter=cfg["iters"], history_size=20, line_search_fn="strong_wolfe",
                                    tolerance_grad=1e-10, tolerance_change=1e-12)
            evals = []

            def closure():
                opt.zero_grad(set_to_none=True)
                loss, parts = model.loss(u, v, nbrs, img_flat, stride)
                loss.backward()
                evals.append(loss.detach())
                return loss

            opt.step(closure)
            hist.append({"level": level, "evals": len(evals), "loss_first": evals[0].item(), "loss": evals[-1].item()})
            continue
        opt = torch.optim.Adam([dict(g) for g in groups])  # fresh state and learning rates per level
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * it / cfg["iters"])))
        for it in range(cfg["iters"]):
            pix = pool.gather(1, torch.randint(0, pool.shape[1], (N, n_pix), device=device))
            u, v = (pix % W).float(), (pix // W).float()
            loss, parts = model.loss(u, v, nbrs, img_flat, stride)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if it % 20 == 0 or it == cfg["iters"] - 1:
                hist.append({"level": level, "it": it, "loss": loss.item(), **{k: v.item() for k, v in parts.items()}})
    if device.type == "cuda":
        torch.cuda.synchronize()
    opt_s = time.perf_counter() - start
    after = eval_photo(model, nbrs, img_sharp)

    with torch.no_grad():
        R, t = model.poses()
        ext_out = torch.cat([R, t[..., None] * model.scale], -1).cpu().numpy()
        K_out = model.intrinsics().cpu().numpy()
        depth_out = (model.dense_depth() * model.scale).cpu().numpy()
        f_scale = torch.exp(model.logf).cpu().numpy()
    info.update(
        load_s=load_s,
        opt_s=opt_s,
        peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
        photo_before=before["photo_l1"],
        photo_after=after["photo_l1"],
        geo_before=before["geo_l1"],
        geo_after=after["geo_l1"],
        visible_before=before["visible"],
        visible_after=after["visible"],
        focal_scale=float(np.mean(f_scale)),
        depth_corr_abs=float(np.abs(model.grid.detach().cpu().numpy()).mean()),
    )
    return {"depth": depth_out.astype(np.float32), "extrinsics": ext_out.astype(np.float32), "intrinsics": K_out.astype(np.float32)}, info, hist


def main():
    if len(sys.argv) < 3 or sys.argv[1].startswith("-"):
        sys.exit(__doc__)
    src, dst, cfg = sys.argv[1], sys.argv[2], parse_overrides(sys.argv[3:])
    sys.stdout.reconfigure(line_buffering=True)
    os.chdir(REPO_ROOT)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "pba_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = os.path.join(src, "model_results")
    datasets = [d for d in cfg["datasets"].split(",") if d] or sorted(os.listdir(results_dir))

    for dataset in datasets:
        data_dir = os.path.join(results_dir, dataset)
        metric_path = os.path.join(dst, "metric_results", f"{dataset}_pba.json")
        hist_path = os.path.join(dst, "metric_results", f"{dataset}_pba_hist.jsonl")
        os.makedirs(os.path.dirname(metric_path), exist_ok=True)
        metrics = json.load(open(metric_path)) if os.path.exists(metric_path) else {}
        metrics.pop("mean", None)  # recomputed after every scene, so an interrupted run can be resumed

        scenes = sorted(os.path.dirname(p) for p, _, _ in os.walk(data_dir) if os.path.basename(p) == "unposed")
        for i, scene_dir in enumerate(scenes[: cfg["max_scenes"] or None]):
            scene = os.path.relpath(scene_dir, data_dir)
            if cfg["scenes"] and scene not in cfg["scenes"].split(","):
                continue
            if scene in metrics:
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
                spread = lambda e: np.linalg.norm((c := -np.einsum("nji,nj->ni", e[:, :, :3], e[:, :, 3])) - c.mean(0), axis=1).mean()  # noqa: E731
                ext[:, :, 3] *= spread(pred["extrinsics"]) / max(spread(ext), 1e-12)
                pred.update(extrinsics=ext, intrinsics=other["intrinsics"].astype(np.float32))

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
