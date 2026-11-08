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

With WANDB_PROJECT set (and WANDB_RUN_ID / WANDB_RESUME=allow to share a run between processes, as pba_run.sh
does), every scene is also logged to wandb: its metrics, running dataset means and loss curve (see wandb_scene).
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
from depth_anything_3.bench.utils import align_to_first_camera, compute_pose, se3_to_relative_pose_error
from depth_anything_3.utils.geometry import as_homogeneous

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = {
    "datasets": "",  # comma-separated, default: all in <src>/model_results
    "max_scenes": 0,  # per dataset, 0 = all
    "first_scene": 0,  # per dataset: skip this many scenes first (held-out = the scenes after the dev subset)
    "scenes": "",  # comma-separated scene names, default: all
    "neighbors": 6,  # covisible target frames per source frame
    "nbr_mode": "top",  # top (most covisible) | spread (evenly over the covisibility ranking, see select_neighbors)
    "min_covis": 0.2,
    "samples": 4096,  # source pixels per frame and iteration
    # coarse-to-fine levels, each [features:]sigma[:iters]: blur sigma (px, 0 = sharp), features default `features`
    # (rgb | vgg | local | mid | final, the last three DA3 backbone tokens, see cache_da3_feats.py), iters default `iters`
    "levels": "8,4,2,1",
    "iters": 100,  # per level (L-BFGS: iterations, each with one or more evaluations in the line search)
    "optimizer": "adam",  # adam (fresh random samples per iteration) | lbfgs (fixed samples per level, strong Wolfe) | lm
    "lbfgs_samples": 4,  # L-BFGS: fixed samples per frame = lbfgs_samples * samples
    "lm_levels": "",  # with optimizer=adam: comma-separated level indices run with LM instead (e.g. a final polish)
    "lm_iters": 10,  # LM iterations of those levels
    "lm_samples": 0,  # source pixels per frame and LM iteration, 0 = samples
    "lm_lambda": 1e-3,  # LM: initial damping, relative to diag(J^T J)
    "lm_lambda_min": 1e-7,  # LM: lower bound of the damping (keeps steps small along flat directions)
    "lm_mu": 0.0,  # LM: additional Levenberg damping mu * mean(diag(J^T J)) * I: suppresses only weakly constrained directions
    "lm_fixed": False,  # LM: same samples for all iterations of a level (deterministic objective) instead of fresh ones
    "lm_chunk": 0,  # LM: source frames per Jacobian chunk, 0 = automatic (~1.5e8 Jacobian entries)
    "lr_rot": 1e-3,  # rad
    "lr_trans": 1e-3,  # in units of the median depth
    "lr_focal": 1e-3,  # log focal
    "lr_depth": 5e-3,  # log depth grid
    "lr_bright": 1e-2,
    "focal": "shared",  # none | shared (one scale for all frames) | per_frame
    "focal_init": "pred",  # pred | median (replace per-frame focals with their median first)
    "depth_grid": 8,  # cells along the longer side of the log-depth correction grid, 0 = depth fixed
    "depth_grid_fine": 0,  # cells of a second, finer log-depth grid added to the first, 0 = none
    "fine_from": 2,  # level index from which the fine grid is optimised
    "w_prior_fine": 1.0,  # w_prior of the fine grid
    "depth_param": "grid",  # grid (log-depth correction grids) | head (fine-tune DA3's depth head per scene, Adam only)
    "lr_head": 1e-5,
    "head_frames": 8,  # head: frames whose depth is recomputed with gradients per iteration
    "head_refresh": 20,  # head: iterations between recomputing all frames' depth (used for the other frames)
    "poses": True,  # optimise poses
    "w_photo": 1.0,
    "w_geo": 1.0,  # multi-view depth consistency
    "w_prior": 1.0,  # keeps the log-depth correction near zero (also fixes the scale gauge)
    "w_scale": -1.0,  # impl/lm_impl analytic, >= 0: prior weight of each frame's mean log-depth correction (its offset) instead of w_prior; w_prior then acts on the rest (the shape)
    "w_smooth": 0.0,  # impl/lm_impl analytic: w_smooth * sum over adjacent cells of (g_a - g_b)^2 / (cells of all frames), a smoothness prior on the grid
    "w_pose": 0.0,  # keeps the pose updates (rotation in rad, translation in median depths) near zero (= DA3's poses)
    "w_lap": 0.0,  # impl/lm_impl analytic: w_lap / (3 N) x^T (L kron I) x, x the pose updates in world axes (w, dC) as in pose_basis, L the Laplacian of the covisibility graph: smooth corrections are free, differences between covisible frames' corrections cost
    "w_focal": 0.0,  # keeps the log focal scale near zero (DA3's focal); focal and depth trade off on narrow-FOV scenes
    "huber_photo": 0.05,  # scale of the photometric kernel (intensities in [0, 1])
    "huber_geo": 0.05,  # log depth
    "occ_thresh": 0.05,  # relative depth; a sample is occluded in the target if it lies further away
    "conf_quantile": 0.1,  # source pixels below this DA3 confidence quantile (per scene) are not sampled
    "depth_edge": 0.0,  # > 0: source pixels within depth_edge_dilate px of a DA3 depth discontinuity (|grad log depth| above this, central differences) are not sampled
    "tgt_edge": 0.0,  # > 0, impl/lm_impl analytic: warps landing within depth_edge_dilate px of a discontinuity of the target's DA3 depth don't count
    "depth_edge_dilate": 2,
    # per pixel (after affine brightness, kernel scale huber_photo, see robust()): huber | l2 | cauchy | gm | tukey;
    # or ncc (1 - NCC of 3x3 patches)
    "photo_loss": "huber",
    "photo_agg": "mean",  # per-pixel losses: mean over the visible neighbours | min (the best neighbour, occlusion-robust)
    # rgb features: photometric term on the grey image (mean of the channels). Default since 2026-11-02 (same quality
    # on SMALL, 5-7% faster); runs before that used the colour image (their pba_config.json has no "gray" key)
    "gray": True,
    "bright": True,  # per-frame affine brightness (per-pixel losses on rgb)
    "bright_rel": False,  # impl=analytic: I_j exp(a_j - a_i) + b_j - b_i - I_i instead of I_j exp(a_j) + b_j - I_i exp(a_i) - b_i, whose loss vanishes as the contrast of all frames but the first goes to 0
    "patch_stride": 1.0,  # px between patch samples, multiplied by max(1, blur sigma)
    "grad_weight": 1.0,  # sampling probability ~ |grad I|^grad_weight (0 = uniform)
    "features": "rgb",  # rgb (photometric) | vgg (feature-metric: VGG16 relu1_2 + relu2_2, PCA-reduced)
    "feat_dim": 16,
    "vgg_fast": False,  # keep the VGG network between scenes, larger batches, PCA by eigendecomposition (see vgg_features)
    "token_dim": 32,  # DA3 token features: leading PCA channels used (cache has 64)
    "token_stride": 7.0,  # DA3 token features: px between NCC patch samples (patch size 14)
    "feat_cache": "",
    "w_match": 0.0,  # reprojection error of mutual-nearest-neighbour DA3 token matches between neighbours (Adam levels only)
    "match_layer": "local",
    "match_levels": "0,1",  # level indices with the match term
    "match_max": 128,  # matches per neighbour pair (best similarity)
    "match_min_sim": 0.5,  # cosine similarity of the (PCA-reduced, normalised) tokens  # DA3 token cache dir (default workspace/tto/feats/<basename of src>)
    "init_noise_deg": 0.0,  # rotate every camera but the first by this angle about a random axis (centre fixed), basin tests
    "init_from": "",  # work dir whose extrinsics/intrinsics replace DA3's (e.g. the sparse-BA output), depth stays DA3's
    "impl": "autograd",  # gradient of the Adam levels: autograd | analytic (pba_analytic.py: same objective, a third of the tensor operations)
    "lm_impl": "jvp",  # Jacobians of the LM levels: jvp (forward-mode AD) | analytic (pba_analytic.py, per-pixel kernels only)
    "lm_solver": "dense",  # lm_impl=analytic: dense (all unknowns) | dense32 (the same in single precision, Jacobi-scaled) | schur (eliminate the depth grid, without the coupling of the cells of different frames) | pcg
    "lm_cg_iters": 20,  # lm_solver=pcg: at most this many conjugate-gradient iterations
    "lm_cg_tol": 1e-3,  # lm_solver=pcg: relative residual
    "lm_refactor": 1,  # lm_impl=analytic, dense solvers: new Gauss-Newton matrix every n iterations, the iterations between reuse its factorisation with a new gradient
    "lm_stop": 0.0,  # lm_impl=analytic: stop the LM level when the cost on fixed held-out pixels falls by less than this fraction in an iteration, 0 = off
    "lm_stop_samples": 2048,  # held-out source pixels per frame of lm_stop
    "lm_photo_loss": "",  # lm_impl=analytic: photo_loss of the LM levels, empty: photo_loss
    "lm_free_depth": False,  # lm_impl=analytic, lm_fixed=true: one free log depth per LM sample (per patch with NCC), eliminated exactly, see pba_analytic.solve_free; the grid stays fixed in these levels
    "w_free_depth": 20.0,  # its prior 0.5 w z^2 per sample, in units where a warp's geometric row at huber_geo=0.05 has curvature 20
    "patch": "",  # after the levels: alignments and geometric solves, "sigma:alignments,...", e.g. 8:2,4:2,2:2,1:2, see pba_patch.py
    "patch_count": 200,  # patches per frame
    "patch_radius": 1,  # pattern of (2 r + 1)^2 pixels
    "patch_lk": 3,  # Lucas-Kanade iterations per alignment
    "patch_inner": 10,  # at most this many LM iterations per alignment
    "patch_tol": 0.0,  # end them when the cost falls by less than this fraction
    "patch_ncc": 0.5,  # warps with a lower NCC after alignment are left out
    "patch_max_disp": 4.0,  # and warps that moved further than this, in strides
    "patch_info": "photo",  # W of a warp: photo (H / S, in intensities, kernel scale huber_photo) | shape (of H only, in strides, kernel scale patch_delta)
    "patch_delta": 1.0,  # patch_info=shape: scale of the kernel
    "patch_iso": 0.01,  # patch_info=shape: isotropic part of W, relative to its trace
    "patch_gate_after": -1,  # without Adam: index of the patch level after which radial_gate decides on the coefficient
    "patch_depth_iters": 0,  # LM iterations of the depth grid, poses fixed, after the patch levels
    "patch_depth_sigma": 1.0,  # their blur
    "lm_w_geo": -1.0,  # lm_impl=analytic: w_geo of the LM levels, < 0: w_geo (with 0 and lm_solver=schur the elimination of the grid is exact)
    "nbr_iter": 0,  # Adam: neighbours per frame used in an iteration (fresh random subset of the k), 0 = all
    "strat_cells": 0,  # source pixels: equal sampling mass per cell of a grid with this many cells along the long side, 0 = off
    "radial": False,  # analytic only: one shared radial distortion coefficient (division model), for images that are not undistorted (ETH3D)
    "w_radial": 1e-3,  # keeps it near zero where the data do not determine it
    "radial_max": 1.0,  # upper bound of the coefficient; 0 allows only barrel distortion
    "radial_gate": 0.0,  # at the first LM level: if the coefficient is smaller than this in magnitude, set it to 0 and leave it out (no distortion in this scene)
    "lr_radial": 1e-3,
    "radial_gate_heldout": False,  # instead of the threshold radial_gate: at the first LM level keep k only if it lowers the cost of fixed held-out pixels (sharp level, lm_stop_samples per frame) against k = 0
    "pose_basis": 0,  # impl=analytic: K > 0 runs the Adam levels on a smooth pose basis + per-frame updates (pba_analytic.PoseBasis)
    "pose_world": False,  # impl=analytic: the Adam levels' per-frame pose updates in world axes (PoseBasis with K = 0); implied by pose_basis > 0
    "nbr_refresh": False,  # recompute the neighbours from the current poses and depth at the start of every level after the first
    "init_fix_gross": False,  # oracle: before the optimisation, replace the frames whose pose is grossly wrong (> 15 deg or > 25 % of the median depth, against GT in DA3's gauge) by their GT pose
    "reloc": False,  # before the optimisation: relocalise low-confidence frames by a discrete search (relocalise())
    "reloc_conf": 0.5,  # frames whose median DA3 confidence is below this fraction of the scene's median are candidates
    "reloc_gain": 1.5,  # the best hypothesis replaces the pose only if its consistency score is this many times the current one
    "cuda_graph": False,  # impl=analytic on a GPU: replay a captured CUDA graph of the Adam iteration (same update, its own random draws); the iterations that log stay eager
    "pp": False,  # impl=analytic: one shared principal-point offset (pixels), optimised in the Adam levels and held fixed in the LM levels
    "lr_pp": 0.05,  # pixels
    "w_pp": 0.0,  # keeps it near zero, per squared pixel
    "init_gt": "",  # start from the ground truth: pose | pose_focal (scale of the translations from DA3's camera spread), to test where the objective's minimum lies
    "pool_once": False,  # draw the candidate pixels once (at the sharpest level's distribution) instead of per level
    "pool_factor": 16,  # Adam: candidate pixels per frame drawn once per level = pool_factor * samples
    "track_every": 0,  # record the poses and a held-out loss every n iterations (<dst>/metric_results/<dataset>_pba_track.jsonl)
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


def robust(r, c, kind="huber"):
    """Robust kernel rho(r) with scale c. All equal 0.5 r^2 / c near 0 (the Huber's quadratic part), so only the
    tails differ: huber (linear), l2 (quadratic), cauchy (logarithmic), gm = Geman-McClure (bounded, -> c / 2),
    tukey (biweight: constant c / 6 beyond c, i.e. no gradient)."""
    if kind == "huber":
        return huber(r, c)
    x2 = (r / c) ** 2
    if kind == "l2":
        return 0.5 * c * x2
    if kind == "cauchy":
        return 0.5 * c * torch.log1p(x2)
    if kind == "gm":
        return 0.5 * c * x2 / (1 + x2)
    if kind == "tukey":
        return c / 6 * (1 - (1 - x2.clamp(max=1)) ** 3)
    raise ValueError(f"unknown kernel {kind}")


def robust_weight(r, c, kind="huber"):
    """IRLS weight rho'(r) / r of robust() (the LM levels minimise sum 0.5 w r^2 with these weights)."""
    if kind == "huber":
        return torch.where(r.abs() < c, 1 / c, 1 / r.abs().clamp_min(1e-12))
    x2 = (r / c) ** 2
    if kind == "l2":
        return torch.full_like(r, 1 / c)
    if kind == "cauchy":
        return 1 / (c * (1 + x2))
    if kind == "gm":
        return 1 / (c * (1 + x2) ** 2)
    if kind == "tukey":
        return (1 - x2).clamp_min(0) ** 2 / c
    raise ValueError(f"unknown kernel {kind}")


# ---------------------------------------------------------------- scene


def load_images(image_files, H, W):
    """RGB in [0, 1], resized to the prediction's resolution (the aspect ratios match exactly)."""
    imgs = [cv2.resize(cv2.imread(str(f))[..., ::-1], (W, H), interpolation=cv2.INTER_AREA) for f in image_files]
    return torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255


_VGG_NET = {}  # vgg_fast: the network of this process, by device


@torch.no_grad()
def vgg_features(images, dim, chunk=2, scale=0.1, fast=False):
    """VGG16 relu1_2 and relu2_2 (upsampled) of images [N, 3, H, W], PCA-reduced to dim channels whose
    per-channel RMS is scale on average (comparable to intensities, so the same Huber threshold applies).
    fast: keep the network between scenes, 8 frames per forward pass, PCA from the eigenvectors of the covariance instead
    of an SVD of all samples (same subspace, the signs of the channels and the sampled pixels differ)."""
    import torchvision

    key = str(images.device)
    if fast and key in _VGG_NET:
        vgg = _VGG_NET[key]
    else:
        vgg = torchvision.models.vgg16(weights="IMAGENET1K_V1").features[:9].to(images.device).eval()
        if fast:
            _VGG_NET[key] = vgg
    if fast:
        chunk = 8
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
    if fast:
        Xc = X - mu
        ev, U = torch.linalg.eigh(Xc.T @ Xc)
        P = U[:, -dim:].flip(1)  # [C, dim], the largest eigenvalues first
        norm = scale / (ev[-dim:] / len(X)).mean().sqrt()
    else:
        _, S, V = torch.linalg.svd(X - mu, full_matrices=False)
        P = V[:dim].T  # [C, dim]
        norm = scale / (S[:dim] ** 2 / len(X)).mean().sqrt()
    out = torch.empty(len(images), dim, H, W, device=images.device)
    for i in range(0, len(images), chunk):
        f = feats(images[i : i + chunk])
        out[i : i + chunk] = torch.einsum("nchw,cd->ndhw", f - mu[:, None, None], P) * norm
    if not fast:
        del vgg
    return out


def token_features(path, layer, dim, H, W, device, scale=0.1):
    """Cached DA3 patch tokens [N, 64, H/14, W/14] of one layer -> leading dim PCA channels, bilinearly upsampled
    to [N, dim, H, W] (token centres at the patch centres) with per-channel RMS scale on average."""
    f = torch.from_numpy(np.load(path)[layer][:, :dim]).float().to(device)
    f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
    return f * (scale / f.pow(2).mean().sqrt())


class HeadDepth:
    """DA3's depth head on frozen backbone features of one scene, for depth_param=head."""

    def __init__(self, api, image_files, head_state):
        from depth_anything_3.utils.constants import EVAL_REF_VIEW_STRATEGY

        self.head = api.model.head
        self.head.load_state_dict(head_state)  # fresh weights for every scene
        imgs_cpu = api._preprocess_inputs(list(image_files), None, None, 504, "upper_bound_resize")[0]
        imgs = api._prepare_model_inputs(imgs_cpu, None, None)[0]
        self.H, self.W = imgs.shape[-2:]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            feats, _ = api.model.backbone(imgs, cam_token=None, export_feat_layers=[], ref_view_strategy=EVAL_REF_VIEW_STRATEGY)
        self.feats = [(f.float(), c.float()) for f, c in feats]
        self.N = imgs.shape[1]

    def __call__(self, idx):
        with torch.autocast(device_type="cuda", enabled=False):
            out = self.head([(f[:, idx], c[:, idx]) for f, c in self.feats], self.H, self.W, patch_start_idx=0)
        return out["depth"].reshape(len(idx), self.H, self.W)

    @torch.no_grad()
    def all(self, chunk=16):
        return torch.cat([self(torch.arange(i, min(i + chunk, self.N), device=self.feats[0][0].device)) for i in range(0, self.N, chunk)])


@torch.no_grad()
def token_matches(path, layer, nbrs, max_matches, min_sim, device):
    """Mutual nearest neighbours of the cached DA3 tokens of each frame and its neighbours nbrs [N, k].
    Returns source and target token centres (px) and a validity mask, each [N, k, max_matches] (best similarity first)."""
    f = torch.from_numpy(np.load(path)[layer]).float().to(device)  # [N, C, h, w]
    N, C, h, w = f.shape
    f = F.normalize(f.flatten(2).transpose(1, 2), dim=-1)  # [N, h*w, C]
    k = nbrs.shape[1]
    centre_u = ((torch.arange(h * w, device=device) % w).float() + 0.5) * 14 - 0.5
    centre_v = ((torch.arange(h * w, device=device) // w).float() + 0.5) * 14 - 0.5
    out = torch.zeros(5, N, k, max_matches, device=device)
    for i in range(N):
        sim = torch.einsum("pc,kqc->kpq", f[i], f[nbrs[i]])  # [k, P, Q]
        s12, nn12 = sim.max(2)  # best target for every source token
        nn21 = sim.argmax(1)  # best source for every target token
        mutual = nn21.gather(1, nn12) == torch.arange(h * w, device=device)[None]
        score = torch.where(mutual & (s12 > min_sim), s12, torch.full_like(s12, -1))
        top, src = score.topk(min(max_matches, h * w), dim=1)  # [k, Mm]
        tgt = nn12.gather(1, src)
        m = top > 0
        out[:, i, :, : src.shape[1]] = torch.stack([centre_u[src], centre_v[src], centre_u[tgt], centre_v[tgt], m.float()])
    return (*out, nbrs)


def bilinear_weights(gu, gv, gh, gw):
    """Flat cell indices [..., 4] and weights [..., 4] of bilinear interpolation on a gh x gw grid (as in bilinear_gather)."""
    u0 = gu.floor().clamp(0, gw - 2)
    v0 = gv.floor().clamp(0, gh - 2)
    du, dv = gu - u0, gv - v0
    base = v0.long() * gw + u0.long()
    idx = torch.stack([base, base + 1, base + gw, base + gw + 1], -1)
    w = torch.stack([(1 - du) * (1 - dv), du * (1 - dv), (1 - du) * dv, du * dv], -1)
    return idx, w


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


def depth_edges(depth, thr, dilate):
    """[N, H, W] bool: pixels within `dilate` px of a discontinuity of the depth [N, H, W] (|grad log depth| > thr, central differences)."""
    ld = torch.log(depth.clamp_min(1e-6))
    gx = F.pad(ld[:, :, 2:] - ld[:, :, :-2], (1, 1))
    gy = F.pad(ld[:, 2:] - ld[:, :-2], (0, 0, 1, 1))
    e = (torch.sqrt(gx**2 + gy**2) > thr).float()[:, None]
    if dilate > 0:
        e = F.max_pool2d(e, 2 * dilate + 1, stride=1, padding=dilate)
    return e[:, 0] > 0


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
        g = max(cfg["depth_grid_fine"], 1)
        gh, gw = (g, max(2, round(g * W / H))) if H > W else (max(2, round(g * H / W)), g)
        self.grid2 = torch.nn.Parameter(torch.zeros(N, gh, gw, device=dev))
        self.bright = torch.nn.Parameter(torch.zeros(N, 2, device=dev))  # log gain, bias
        self.kr = torch.nn.Parameter(torch.zeros(1, device=dev))  # radial coefficient, only used by pba_analytic.py
        self.pp = torch.nn.Parameter(torch.zeros(2, device=dev))  # shared principal-point offset in pixels (pp=true), pba_analytic.py only

        # sampling mask: confident, not at the border
        thr = torch.quantile(conf.reshape(-1)[:: conf.numel() // 2**22 + 1], cfg["conf_quantile"]) if cfg["conf_quantile"] > 0 else -1
        valid = conf > thr
        if cfg["depth_edge"] > 0:
            valid &= ~depth_edges(depth, cfg["depth_edge"], cfg["depth_edge_dilate"])
        valid[(valid.flatten(1).float().mean(1) < 0.01)] = True  # frames with (almost) no confident pixels: use all
        valid[:, :2], valid[:, -2:], valid[:, :, :2], valid[:, :, -2:] = False, False, False, False
        self.register_buffer("valid", valid)
        self.tgt_bad = depth_edges(depth, cfg["tgt_edge"], cfg["depth_edge_dilate"]).reshape(-1) if cfg["tgt_edge"] > 0 else None
        self.match = None  # token matches, see token_matches
        self.pose_L = None  # w_lap: Laplacian of the covisibility graph [N, N]
        self.depth_dense = None  # depth_param=head: current depth [N, H, W] (replaces depth0, with gradients on some frames)

    # --- current state
    def poses(self, rot=None, trans=None):
        rot = self.rot if rot is None else rot
        trans = self.trans if trans is None else trans
        dR = so3_exp(rot * self.free) if self.cfg["poses"] else torch.eye(3, device=self.R0.device).expand(self.N, 3, 3)
        R = dR @ self.R0
        t = (dR @ self.t0[..., None])[..., 0] + (trans * self.free if self.cfg["poses"] else 0)
        return R, t

    def intrinsics(self, logf=None):
        logf = self.logf if logf is None else logf
        f = torch.exp(logf if self.cfg["focal"] != "none" else logf.detach() * 0).expand(self.N)
        scale = 1 + torch.diag_embed(torch.stack([f - 1, f - 1, torch.zeros_like(f)], -1))  # only fx, fy
        K = self.K0 * scale
        if self.cfg["pp"]:
            K = K.clone()
            K[:, 0, 2] += self.pp[0]
            K[:, 1, 2] += self.pp[1]
        return K

    def grid_at(self, idx, u, v):
        """log-depth correction at pixel (u, v) of frame idx (bilinear, grid corners at the image corners)."""
        out = torch.zeros_like(u)
        for grid, on in ((self.grid, self.cfg["depth_grid"] > 0), (self.grid2, self.cfg["depth_grid_fine"] > 0)):
            if on:
                N, gh, gw = grid.shape
                gu, gv = u * (gw - 1) / (self.W - 1), v * (gh - 1) / (self.H - 1)
                out = out + bilinear_gather(grid.reshape(1, -1), idx, gu, gv, gh, gw)[:, 0]
        return out

    def dense_depth(self):
        if self.depth_dense is not None:
            return self.depth_dense
        corr = torch.zeros_like(self.depth0)
        for grid, on in ((self.grid, self.cfg["depth_grid"] > 0), (self.grid2, self.cfg["depth_grid_fine"] > 0)):
            if on:
                corr = corr + F.interpolate(grid[:, None], size=(self.H, self.W), mode="bilinear", align_corners=True)[:, 0]
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
        base = self.depth0 if self.depth_dense is None else self.depth_dense
        d_src = base[idx, v.long(), u.long()] * torch.exp(self.grid_at(idx.reshape(-1), u.reshape(-1), v.reshape(-1)).view_as(u))
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
        d_tgt = bilinear_gather(base.reshape(1, -1), tgt, utf, vtf, H, W)[:, 0] * torch.exp(self.grid_at(tgt, utf, vtf))
        d_tgt = d_tgt.view_as(z)
        visible = inside & (z.detach() < d_tgt.detach() * (1 + self.cfg["occ_thresh"]))
        geo = torch.log(zc) - torch.log(d_tgt.clamp_min(1e-4))

        # affine brightness only for intensities with the Huber loss (NCC and features are invariant to it, and
        # Adam would turn its ~0 gradient into a random walk)
        on = self.free[:, 0] * (self.cfg["features"] == "rgb" and self.cfg["photo_loss"] != "ncc" and self.cfg["bright"])
        a, b = self.bright[:, 0] * on, self.bright[:, 1] * on
        C = img_flat.shape[0]
        i_src = bilinear_gather(img_flat, idx.reshape(-1), u.reshape(-1), v.reshape(-1), H, W).view(N, 1, -1, C)
        if not self.cfg["bright_rel"]:
            i_src = i_src * torch.exp(a)[:, None, None, None] + b[:, None, None, None]
        i_tgt = bilinear_gather(img_flat, tgt, utf, vtf, H, W).view(*z.shape, C)
        if self.cfg["bright_rel"]:
            i_tgt = i_tgt * torch.exp(a[nbrs] - a[:, None])[..., None, None] + (b[nbrs] - b[:, None])[..., None, None]
        else:
            i_tgt = i_tgt * torch.exp(a[nbrs])[..., None, None] + b[nbrs][..., None, None]
        return i_tgt, i_src, geo, visible

    def patch_samples(self, u, v, stride):
        """Patch centres [n, P] -> 3x3 patches [n, 9P] (patch-major)."""
        off = torch.tensor([-1.0, 0.0, 1.0], device=u.device) * stride
        du, dv = off[None, :].expand(3, 3).reshape(-1), off[:, None].expand(3, 3).reshape(-1)
        n = u.shape[0]
        return (u[..., None] + du).clamp(0, self.W - 1).reshape(n, -1), (v[..., None] + dv).clamp(0, self.H - 1).reshape(n, -1)

    def loss(self, u, v, nbrs, img_flat, stride=1.0, match=False):
        cfg = self.cfg
        if cfg["photo_loss"] == "ncc":
            # u, v are patch centres [N, P]; expand to 3x3 patches [N, 9P]
            P = u.shape[1]
            u, v = self.patch_samples(u, v, stride)
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
            e = robust(i_tgt - i_src, cfg["huber_photo"], cfg["photo_loss"]).mean(-1)  # [N, k, M]
            if cfg["photo_agg"] == "min":  # the best visible neighbour of each sample
                any_vis = mask.any(1)
                e_min = torch.where(mask, e, torch.full_like(e, float("inf"))).amin(1)
                l_photo = torch.where(any_vis, e_min, torch.zeros_like(e_min)).sum() / any_vis.sum().clamp_min(1)
            else:
                l_photo = (e * m).sum() / denom
        l_geo = (huber(geo, cfg["huber_geo"]) * m).sum() / denom
        l_prior = (self.grid**2).mean() if cfg["depth_grid"] > 0 else torch.zeros((), device=u.device)
        if cfg["depth_grid_fine"] > 0:
            l_prior = l_prior + cfg["w_prior_fine"] / max(cfg["w_prior"], 1e-12) * (self.grid2**2).mean()
        total = cfg["w_photo"] * l_photo + cfg["w_geo"] * l_geo + cfg["w_prior"] * l_prior + cfg["w_focal"] * (self.logf**2).mean()
        if cfg["w_pose"] > 0:
            total = total + cfg["w_pose"] * ((self.rot * self.free) ** 2 + (self.trans * self.free) ** 2).mean()
        parts = {"photo": l_photo.detach(), "geo": l_geo.detach(), "prior": l_prior.detach(), "visible": m.mean()}
        if match and self.match is not None:
            l_match = self.match_loss()
            total = total + cfg["w_match"] * l_match
            parts["match"] = l_match.detach()
        return total, parts

    def match_loss(self, delta=0.5):
        """Huber reprojection error (in token units, 14 px) of the token matches: source token centres lifted with
        the current depth and projected into the matched neighbour."""
        mu, mv, tu, tv, mm, nb = self.match  # [N, k, Mm] each, nb [N, k]
        N = mu.shape[0]
        R, t = self.poses()
        K = self.intrinsics()
        idx = torch.arange(N, device=mu.device)[:, None, None].expand_as(mu)
        d = self.depth0[idx, mv.long(), mu.long()] * torch.exp(self.grid_at(idx.reshape(-1), mu.reshape(-1), mv.reshape(-1)).view_as(mu))
        x = (mu - K[:, None, None, 0, 2]) / K[:, None, None, 0, 0] * d
        y = (mv - K[:, None, None, 1, 2]) / K[:, None, None, 1, 1] * d
        Xw = (torch.stack([x, y, d], -1) - t[:, None, None]) @ R[:, None]  # [N, k, Mm, 3]
        Xt = Xw @ R[nb].transpose(-1, -2) + t[nb][:, :, None]
        z = Xt[..., 2].clamp_min(1e-4)
        Kt = K[nb]
        ut = Kt[..., None, 0, 0] * Xt[..., 0] / z + Kt[..., None, 0, 2]
        vt = Kt[..., None, 1, 1] * Xt[..., 1] / z + Kt[..., None, 1, 2]
        e = torch.sqrt((ut - tu) ** 2 + (vt - tv) ** 2 + 1e-12) / 14
        return (huber(e, delta) * mm).sum() / mm.sum().clamp_min(1)

    def lm_residuals(self, rot_s, trans_s, rot_t, trans_t, logf, ds, dt, src, u, v, nbrs, img_flat, S):
        """Residuals of the source frames src [n] (u, v [n, P*S], S samples per patch) for Levenberg-Marquardt.

        The poses enter twice, for the source and for the target role, so that the forward-mode derivative w.r.t.
        one component of all frames at once is the derivative w.r.t. that component of each residual's own source
        (or target) frame. ds [n, P] and dt [n, k, P] are per-patch log-depth offsets of the source and target depth
        (zero), whose derivatives are mapped onto the depth grid cells with the bilinear weights of the patch centres.
        Returns the photometric residual [n, k, P, D] (NCC: difference of the normalised patches, so that half its
        squared norm is 1 - NCC; Huber: per channel), the geometric one [n, k, P, S], the visibility [n, k, P, S] and
        the warped pixels [n, k, P*S] (the last two without derivatives).
        """
        H, W = self.H, self.W
        n, M = u.shape
        P, k = M // S, nbrs.shape[1]
        R_s, t_s = self.poses(rot_s, trans_s)
        R_t, t_t = self.poses(rot_t, trans_t)
        K = self.intrinsics(logf)
        idx = src[:, None].expand_as(u)
        logd = self.grid_at(idx.reshape(-1), u.reshape(-1), v.reshape(-1)).view_as(u) + ds.repeat_interleave(S, 1)
        d_src = self.depth0[idx, v.long(), u.long()] * torch.exp(logd)
        Ks = K[src]
        x = (u - Ks[:, None, 0, 2]) / Ks[:, None, 0, 0] * d_src
        y = (v - Ks[:, None, 1, 2]) / Ks[:, None, 1, 1] * d_src
        Xw = (torch.stack([x, y, d_src], -1) - t_s[src][:, None]) @ R_s[src]
        Rt, tt, Kt = R_t[nbrs], t_t[nbrs], K[nbrs]
        Xt = Xw[:, None] @ Rt.transpose(-1, -2) + tt[:, :, None]  # [n, k, M, 3]
        z = Xt[..., 2]
        zc = z.clamp_min(1e-4)
        ut = Kt[..., None, 0, 0] * Xt[..., 0] / zc + Kt[..., None, 0, 2]
        vt = Kt[..., None, 1, 1] * Xt[..., 1] / zc + Kt[..., None, 1, 2]
        inside = (z > 1e-4) & (ut >= 0) & (ut <= W - 1) & (vt >= 0) & (vt <= H - 1)
        ut, vt = ut.clamp(0, W - 1), vt.clamp(0, H - 1)
        tgt = nbrs[:, :, None].expand_as(ut).reshape(-1)
        utf, vtf = ut.reshape(-1), vt.reshape(-1)
        logd_t = self.grid_at(tgt, utf, vtf) + dt.repeat_interleave(S, -1).reshape(-1)
        d_tgt = (bilinear_gather(self.depth0.reshape(1, -1), tgt, utf, vtf, H, W)[:, 0] * torch.exp(logd_t)).view_as(z)
        visible = inside & (z < d_tgt * (1 + self.cfg["occ_thresh"]))
        geo = torch.log(zc) - torch.log(d_tgt.clamp_min(1e-4))
        C = img_flat.shape[0]
        a = bilinear_gather(img_flat, idx.reshape(-1), u.reshape(-1), v.reshape(-1), H, W).reshape(n, 1, P, S * C)
        b = bilinear_gather(img_flat, tgt, utf, vtf, H, W).reshape(n, k, P, S * C)
        if self.cfg["features"] == "rgb" and self.cfg["photo_loss"] != "ncc" and self.cfg["bright"]:  # brightness as in residuals (fixed)
            ga, gb = (self.bright.detach() * self.free).unbind(1)
            a = a * torch.exp(ga[src])[:, None, None, None] + gb[src][:, None, None, None]
            b = b * torch.exp(ga[nbrs])[..., None, None] + gb[nbrs][..., None, None]
        if S > 1:
            a = a - a.mean(-1, keepdim=True)
            b = b - b.mean(-1, keepdim=True)
            a = a / ((a * a).sum(-1, keepdim=True) + 1e-8).sqrt()
            b = b / ((b * b).sum(-1, keepdim=True) + 1e-8).sqrt()
        photo = b - a
        return photo, geo.view(n, k, P, S), visible.view(n, k, P, S).detach(), ut.detach(), vt.detach()


def sampling_probs(img, valid, grad_weight, strat_cells=0):
    """[N, H*W] sampling probability of source pixels: valid and ~|grad I|^grad_weight. With strat_cells > 0 every cell of
    a grid with this many cells along the long side gets the same mass, so that texture can't concentrate the samples."""
    gray = img.mean(1, keepdim=True)
    gx = F.pad(gray[..., :, 2:] - gray[..., :, :-2], (1, 1, 0, 0))
    gy = F.pad(gray[..., 2:, :] - gray[..., :-2, :], (0, 0, 1, 1))
    g = torch.sqrt(gx**2 + gy**2)[:, 0]
    p = (g + 1e-3) ** grad_weight * valid
    if strat_cells > 0:
        H, W = p.shape[-2:]
        s = max(1, math.ceil(max(H, W) / strat_cells))
        mass = F.avg_pool2d(p[:, None], s, ceil_mode=True)
        p = p / F.interpolate(mass, scale_factor=s, mode="nearest")[:, 0, :H, :W].clamp_min(1e-12)
    return p.reshape(len(p), -1) + 1e-12 * valid.reshape(len(p), -1)


class Tracker:
    """Poses and held-out errors during the optimisation (track_every > 0), to see when the accuracy starts to drop.

    The held-out pixels are fixed, drawn uniformly from the valid ones with their own generator (the optimiser's random
    stream stays the same as without tracking) and scored on the sharp images. Its time is left out of opt_s.
    """

    def __init__(self, model, nbrs, img_flat, every, n=2048):
        self.model, self.nbrs, self.img, self.every = model, nbrs, img_flat, every
        gen = torch.Generator(device=img_flat.device).manual_seed(12345)
        pix = torch.multinomial(model.valid.reshape(model.N, -1).float(), n, replacement=True, generator=gen)
        self.u, self.v = (pix % model.W).float(), (pix // model.W).float()
        self.rows, self.spent, self.it = [], 0.0, 0

    @torch.no_grad()
    def record(self, level, force=False):
        if not (force or self.it % self.every == 0):
            return
        start = time.perf_counter()
        m = self.model
        i_tgt, i_src, geo, mask = m.residuals(self.u, self.v, self.nbrs, self.img)
        w = mask.float()
        denom = w.sum().clamp_min(1)
        R, t = m.poses()
        self.rows.append({
            "it": self.it, "level": level,
            "photo": float(((i_tgt - i_src).abs().mean(-1) * w).sum() / denom),
            "geo": float((geo.abs() * w).sum() / denom),
            "visible": float(w.mean()),
            "logf": float(m.logf.mean()),
            "depth_corr": float(m.grid.abs().mean()),
            "ext": torch.cat([R, t[..., None] * m.scale], -1).cpu().numpy(),
        })
        self.spent += time.perf_counter() - start


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
    out = {
        "photo_l1": float((photo.abs().mean(-1) * m).sum() / m.sum()),
        "geo_l1": float((geo.abs() * m).sum() / m.sum()),
        "visible": float(m.mean()),
    }
    # distribution of the photometric residuals of visible (sample, neighbour) pairs, for choosing the kernel:
    # quantiles of |r| (mean over channels), robust sigma (1.4826 MAD of the signed per-channel residuals) and the
    # fraction of pairs beyond 0.05 / 0.1 / 0.2
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


def lm_layout(model):
    """Global parameter indices: poses of frames 1..N-1 (rot, trans), the shared log focal, the depth grid cells.
    Fixed parameters point to the dummy index T (last row/column of the system, dropped before solving)."""
    cfg, N = model.cfg, model.N
    if cfg["focal"] == "per_frame":
        sys.exit("optimizer=lm supports focal=shared|none")
    n_pose = 6 * (N - 1) if cfg["poses"] else 0
    n_grid = model.grid[0].numel() if cfg["depth_grid"] > 0 else 0
    T = n_pose + 1 + N * n_grid  # (the fine depth grid is too large for a dense system and stays fixed in LM levels)
    dev = model.grid.device
    pose_idx = torch.full((N, 6), T, dtype=torch.long, device=dev)
    if cfg["poses"]:
        pose_idx[1:] = torch.arange(n_pose, device=dev).view(N - 1, 6)
    focal_idx = n_pose if cfg["focal"] == "shared" else T
    return {"T": T, "pose": pose_idx, "focal": focal_idx, "grid0": n_pose + 1, "n_grid": n_grid}


def lm_normal_equations(model, u, v, nbrs, img_flat, S, layout, chunk):
    """Gauss-Newton system (J^T W J, J^T W r) of the IRLS-weighted objective of model.loss at the current
    parameters, for patch samples u, v [N, P*S]. Jacobians by forward-mode AD, chunked over source frames."""
    cfg, N, H, W = model.cfg, model.N, model.H, model.W
    T, k = layout["T"], nbrs.shape[1]
    P = u.shape[1] // S
    gh, gw = model.grid.shape[1:]
    kind, agg_min = cfg["photo_loss"], cfg["photo_agg"] == "min" and S == 1
    p = [x.detach() for x in (model.rot, model.trans, model.rot, model.trans, model.logf)]

    # normalisers of the photometric (patches visible in all samples / visible samples) and geometric terms
    chunks = [torch.arange(i, min(i + chunk, N), device=u.device) for i in range(0, N, chunk)]
    n_patch, n_vis = 0.0, 0.0
    with torch.no_grad():
        for src in chunks:
            zero_s, zero_t = u.new_zeros(len(src), P), u.new_zeros(len(src), k, P)
            _, _, vis, _, _ = model.lm_residuals(*p, zero_s, zero_t, src, u[src], v[src], nbrs[src], img_flat, S)
            n_patch += float(vis.all(-1).sum()) if S > 1 else float(vis[..., 0].any(1).sum()) if agg_min else float(vis.sum())
            n_vis += float(vis.sum())
    n_patch, n_vis = max(n_patch, 1.0), max(n_vis, 1.0)

    Hm = torch.zeros((T + 1) * (T + 1), device=u.device)
    g = torch.zeros(T + 1, device=u.device)
    for src in chunks:
        n = len(src)
        zero_s, zero_t = u.new_zeros(n, P), u.new_zeros(n, k, P)
        primals = (*p, zero_s, zero_t)
        f = lambda *q: model.lm_residuals(*q, src, u[src], v[src], nbrs[src], img_flat, S)[:2]  # noqa: E731
        with torch.no_grad():
            photo, geo, vis, ut, vt = model.lm_residuals(*primals, src, u[src], v[src], nbrs[src], img_flat, S)
            tangents = []  # one per column: rot_s, trans_s, rot_t, trans_t (3 each), logf, ds, dt
            for a in range(7):
                for c in range(3 if a < 4 else 1):
                    t = [torch.zeros_like(x) for x in primals]
                    if a < 4:
                        t[a][:, c] = 1
                    else:
                        t[a].fill_(1)
                    tangents.append(t)
            tangents = [torch.stack([t[i] for t in tangents]) for i in range(len(primals))]
            # all 15 forward-mode derivatives in one vectorised pass (2.3x faster than a loop, identical)
            jp, jg = torch.func.vmap(lambda *t: torch.func.jvp(f, primals, t)[1])(*tangents)
            jp, jg = jp.movedim(0, -1), jg.movedim(0, -1)  # [n, k, P, D, 15], [n, k, P, S, 15]
            # weights of 0.5 * w * r^2
            m = vis.float()
            if S > 1:
                wp = (cfg["w_photo"] * m.amin(-1) / n_patch)[..., None].expand_as(photo)
            else:
                if agg_min:  # only the best visible neighbour of each sample (as in model.loss)
                    e = robust(photo, cfg["huber_photo"], kind).mean(-1)  # [n, k, P]
                    best = torch.where(vis[..., 0], e, torch.full_like(e, float("inf"))).argmin(1)  # [n, P]
                    m = m * F.one_hot(best, k).permute(0, 2, 1)[..., None].float()
                wp = cfg["w_photo"] * m / (n_patch * photo.shape[-1]) * robust_weight(photo, cfg["huber_photo"], kind)
            wg = cfg["w_geo"] * vis.float() / n_vis * robust_weight(geo, cfg["huber_geo"])

            # expand the per-patch depth offsets onto the 4 grid cells around the source / warped patch centre
            uc, vc = u[src].view(n, P, S).mean(-1), v[src].view(n, P, S).mean(-1)
            cs, ws = bilinear_weights(uc * (gw - 1) / (W - 1), vc * (gh - 1) / (H - 1), gh, gw)  # [n, P, 4]
            utc, vtc = ut.view(n, k, P, S).mean(-1), vt.view(n, k, P, S).mean(-1)
            ct, wt = bilinear_weights(utc * (gw - 1) / (W - 1), vtc * (gh - 1) / (H - 1), gh, gw)  # [n, k, P, 4]
            ws = ws[:, None].expand(n, k, P, 4)
            nc = 21

            def expand(J):  # [n, k, P, D, 15] -> [n, k, P, D, 21]
                return torch.cat([J[..., :13], J[..., 13:14] * ws[..., None, :], J[..., 14:15] * wt[..., None, :]], -1)

            jp, jg = expand(jp), expand(jg)
            G = torch.einsum("nkpdi,nkpd,nkpdj->nkpij", jp, wp, jp) + torch.einsum("nkpdi,nkpd,nkpdj->nkpij", jg, wg, jg)
            b = torch.einsum("nkpdi,nkpd,nkpd->nkpi", jp, wp, photo) + torch.einsum("nkpdi,nkpd,nkpd->nkpi", jg, wg, geo)

            idx = torch.empty(n, k, P, nc, dtype=torch.long, device=u.device)
            idx[..., :6] = layout["pose"][src][:, None, None]
            idx[..., 6:12] = layout["pose"][nbrs[src]][:, :, None]
            idx[..., 12] = layout["focal"]
            if layout["n_grid"]:
                idx[..., 13:17] = layout["grid0"] + src[:, None, None, None] * layout["n_grid"] + cs[:, None]
                idx[..., 17:21] = layout["grid0"] + nbrs[src][:, :, None, None] * layout["n_grid"] + ct
            else:
                idx[..., 13:] = T
            idx = idx.reshape(-1, nc)
            Hm.index_add_(0, (idx[:, :, None] * (T + 1) + idx[:, None, :]).reshape(-1), G.reshape(-1))
            g.index_add_(0, idx.reshape(-1), b.reshape(-1))
            del jp, jg, G, b, idx

    Hm = Hm.view(T + 1, T + 1)[:T, :T].double()
    g = g[:T].double()
    if layout["n_grid"]:  # prior w_prior * mean(grid^2) = 0.5 * (2 w_prior / numel) * grid^2
        sl = slice(layout["grid0"], T)
        w = 2 * cfg["w_prior"] / model.grid.numel()
        Hm[sl, sl] += w * torch.eye(T - layout["grid0"], device=u.device, dtype=Hm.dtype)
        g[sl] += w * model.grid.detach().reshape(-1).double()
    if cfg["w_pose"] > 0 and cfg["poses"]:  # w_pose * mean(rot^2 + trans^2) over N x 3 entries each
        n6 = 6 * (N - 1)
        w = 2 * cfg["w_pose"] / (3 * N)
        Hm[:n6, :n6] += w * torch.eye(n6, device=u.device, dtype=Hm.dtype)
        g[:n6] += w * torch.cat([model.rot.detach()[1:], model.trans.detach()[1:]], 1).reshape(-1).double()
    if layout["focal"] < T and cfg["w_focal"] > 0:
        Hm[layout["focal"], layout["focal"]] += 2 * cfg["w_focal"]
        g[layout["focal"]] += 2 * cfg["w_focal"] * float(model.logf.detach()[0])
    return Hm, g


@torch.no_grad()
def lm_apply(model, layout, delta):
    """Add delta [T] to the parameters."""
    N, T = model.N, layout["T"]
    delta = delta.float()
    if model.cfg["poses"]:
        d = delta[: 6 * (N - 1)].view(N - 1, 6)
        model.rot[1:] += d[:, :3]
        model.trans[1:] += d[:, 3:]
    if layout["focal"] < T:
        model.logf += delta[layout["focal"]]
    if layout["n_grid"]:
        model.grid += delta[layout["grid0"] :].view_as(model.grid)


def lm_step(model, u, v, nbrs, img_flat, stride, S, layout, chunk, lam, max_tries=6):
    """One Levenberg-Marquardt iteration on fixed samples (patch centres u, v [N, P]). Returns the new damping,
    the loss before and after and whether a step was accepted."""
    U, V = model.patch_samples(u, v, stride) if S > 1 else (u, v)
    with torch.no_grad():
        cost0 = float(model.loss(u, v, nbrs, img_flat, stride)[0])
    Hm, g = lm_normal_equations(model, U, V, nbrs, img_flat, S, layout, chunk)
    diag = Hm.diagonal().clamp_min(1e-12 * float(Hm.diagonal().max()) + 1e-30)
    if model.cfg["lm_mu"] > 0:
        Hm = Hm + model.cfg["lm_mu"] * float(diag.mean()) * torch.eye(len(diag), device=Hm.device, dtype=Hm.dtype)
    params = (model.rot, model.trans, model.logf, model.grid, model.grid2)
    state = [x.detach().clone() for x in params]
    for _ in range(max_tries):
        L, info = torch.linalg.cholesky_ex(Hm + torch.diag(lam * diag))
        if int(info) == 0:
            delta = -torch.cholesky_solve(g[:, None], L)[:, 0]
            lm_apply(model, layout, delta)
            with torch.no_grad():
                cost1 = float(model.loss(u, v, nbrs, img_flat, stride)[0])
            if cost1 < cost0:
                # the Gauss-Newton model overestimates the curvature of the (large-residual) objective: try twice the step
                lm_apply(model, layout, delta)
                with torch.no_grad():
                    cost2 = float(model.loss(u, v, nbrs, img_flat, stride)[0])
                lam_min = model.cfg["lm_lambda_min"]
                if cost2 < cost1:
                    return max(lam / 3, lam_min), cost0, cost2, True
                lm_apply(model, layout, -delta)
                return max(lam / 3, lam_min), cost0, cost1, True
            with torch.no_grad():
                for x, s in zip(params, state):
                    x.copy_(s)
        lam *= 4
    return lam, cost0, cost0, False


def refine_scene(image_files, pred, cfg, device, feat_path=None, api=None, head_state=None):
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
    reloc_info = {}
    if cfg["reloc"]:
        ext, reloc_info = relocalise(images, depth, conf, ext, K, cfg["reloc_conf"], cfg["reloc_gain"])
    model = PBA(images, depth, conf, ext, K, cfg).to(device)

    decode_s = time.perf_counter() - start
    with torch.no_grad():
        R, t = model.poses()
        nbrs, covis = select_neighbors(model.depth0, R, t, model.intrinsics(), cfg["neighbors"], cfg["occ_thresh"], cfg["nbr_mode"], cfg["min_covis"])
        if cfg["w_lap"] > 0:
            Wg = covis.clamp_min(0)
            Wg = 0.5 * (Wg + Wg.T)
            Wg.fill_diagonal_(0)
            model.pose_L = torch.diag(Wg.sum(1)) - Wg
    if device.type == "cuda":
        torch.cuda.synchronize()
    covis_s = time.perf_counter() - start - decode_s
    k = nbrs.shape[1]
    img_sharp = images.permute(1, 0, 2, 3).reshape(3, -1)
    info = {"num_frames": N, "neighbors": k, "decode_s": decode_s, "covis_s": covis_s, **reloc_info}
    before = eval_photo(model, nbrs, img_sharp)
    load_s = time.perf_counter() - start
    tracker = Tracker(model, nbrs, img_sharp, cfg["track_every"]) if cfg["track_every"] else None
    if tracker:
        tracker.record(-1, force=True)

    groups = [
        {"params": [model.rot], "lr": cfg["lr_rot"]},
        {"params": [model.trans], "lr": cfg["lr_trans"]},
        {"params": [model.logf], "lr": cfg["lr_focal"]},
        {"params": [model.grid], "lr": cfg["lr_depth"]},
        {"params": [model.bright], "lr": cfg["lr_bright"]},
        {"params": [model.grid2], "lr": cfg["lr_depth"]},
    ]
    levels = []  # empty: no optimisation (reference run)
    for spec in cfg["levels"].split(","):
        if spec:
            parts = spec.split(":")
            if parts[0].replace(".", "").isdigit():
                parts.insert(0, cfg["features"])
            levels.append((parts[0], float(parts[1]), int(parts[2]) if len(parts) > 2 else cfg["iters"]))
    M = cfg["samples"]
    hist = []
    start = time.perf_counter()
    feat_maps = {"rgb": images.mean(1, keepdim=True) if cfg["gray"] else images}
    head = None
    if cfg["depth_param"] == "head" and levels:
        head = HeadDepth(api, image_files, head_state)
        cache = head.all() / model.scale
        info["head_init_diff"] = float((cache - model.depth0).abs().mean() / model.depth0.abs().mean())
        groups.append({"params": list(head.head.parameters()), "lr": cfg["lr_head"]})
    if cfg["w_match"] > 0 and levels:
        model.match = token_matches(feat_path, cfg["match_layer"], nbrs, cfg["match_max"], cfg["match_min_sim"], device)
        info["matches_per_pair"] = float(model.match[4].sum(-1).mean())
    layout = lm_layout(model) if cfg["optimizer"] == "lm" else None
    lam = max(cfg["lm_lambda"], cfg["lm_lambda_min"])
    kernel = fadam = basis = None
    if cfg["impl"] == "analytic" or cfg["lm_impl"] == "analytic":
        from pba_analytic import FlatAdam, Kernel, PoseBasis

        kernel = Kernel(model)
    if cfg["impl"] == "analytic":
        rest = [model.logf, model.grid, model.bright, model.kr, model.pp], [cfg["lr_focal"], cfg["lr_depth"], cfg["lr_bright"], cfg["lr_radial"], cfg["lr_pp"]]
        if cfg["pose_basis"] > 0 or cfg["pose_world"]:
            basis = PoseBasis(model, covis, cfg["pose_basis"])
            fadam = FlatAdam(basis.params() + rest[0], [cfg["lr_rot"], cfg["lr_trans"], cfg["lr_rot"], cfg["lr_trans"]] + rest[1])
        else:
            fadam = FlatAdam([model.rot, model.trans] + rest[0], [cfg["lr_rot"], cfg["lr_trans"]] + rest[1])
    if cfg["patch"] and (cfg["impl"] != "analytic" or cfg["lm_impl"] != "analytic" or cfg["focal"] == "per_frame" or not cfg["poses"] or cfg["photo_loss"] == "ncc"):
        sys.exit("patch needs impl=analytic, lm_impl=analytic, poses, focal=shared|none and per-pixel kernels")
    if cfg["pp"] and cfg["impl"] != "analytic":
        sys.exit("pp needs impl=analytic")
    if cfg["bright_rel"] and (cfg["impl"] != "analytic" or (cfg["lm_levels"] and cfg["lm_impl"] != "analytic")):
        sys.exit("bright_rel needs impl=analytic and lm_impl=analytic (lm_residuals has the old form)")
    if cfg["lm_free_depth"] and (not cfg["lm_fixed"] or cfg["lm_stop"] > 0 or cfg["lm_impl"] != "analytic"):
        sys.exit("lm_free_depth needs lm_impl=analytic, lm_fixed=true and lm_stop=0")
    blurred, shared_pool = {}, None  # per (features, sigma): levels of the same blur share their images

    def blur(name, sig):
        if (name, sig) not in blurred:
            blurred[(name, sig)] = gaussian_blur(feat_maps[name], sig)
        return blurred[(name, sig)]

    # the sampling probabilities only need the grey image: the mean of the blurred channels is the blurred mean
    feat_maps["_grey"] = images.mean(1, keepdim=True)
    for level, (feat_name, sigma, iters) in enumerate(levels):
        if cfg["nbr_refresh"] and level > 0:
            with torch.no_grad():
                R_, t_ = model.poses()
                nbrs, _ = select_neighbors(model.dense_depth(), R_, t_, model.intrinsics(), cfg["neighbors"], cfg["occ_thresh"], cfg["nbr_mode"], cfg["min_covis"])
        if feat_name not in feat_maps:
            if feat_name == "vgg":
                t_feat = time.perf_counter()
                feat_maps[feat_name] = vgg_features(images, cfg["feat_dim"], fast=cfg["vgg_fast"])
                if device.type == "cuda":
                    torch.cuda.synchronize()
                info["feat_s"] = time.perf_counter() - t_feat
            else:  # DA3 backbone tokens
                feat_maps[feat_name] = token_features(feat_path, feat_name, cfg["token_dim"], H, W, device)
        img = blur(feat_name, sigma)
        img_flat = img.permute(1, 0, 2, 3).reshape(img.shape[1], -1)
        if kernel:
            kernel.set_level(img, cfg["features"] == "rgb" and cfg["photo_loss"] != "ncc",
                             max(cfg["patch_stride"] * max(1.0, sigma), cfg["token_stride"] if feat_name in ("local", "mid", "final") else 0.0))
        # a pool of candidate pixels per frame, drawn once per level (multinomial over all pixels is slow)
        grey = img if feat_name == "rgb" and img.shape[1] == 1 else blur("_grey", sigma)
        probs = sampling_probs(grey, model.valid, cfg["grad_weight"], cfg["strat_cells"])
        if cfg["pool_once"]:
            if shared_pool is None:
                shared_pool = torch.multinomial(sampling_probs(blur("_grey", levels[-1][1]), model.valid, cfg["grad_weight"], cfg["strat_cells"]), cfg["pool_factor"] * M, replacement=True)
            pool = shared_pool
        else:
            pool = torch.multinomial(probs, cfg["pool_factor"] * M, replacement=True)
        stride = cfg["patch_stride"] * max(1.0, sigma)
        if feat_name in ("local", "mid", "final"):
            stride = max(stride, cfg["token_stride"])
        n_pix = M if cfg["photo_loss"] != "ncc" else M // 9  # with NCC, M samples = M/9 patches of 9 pixels
        fine = cfg["depth_grid_fine"] > 0 and level >= cfg["fine_from"]
        next(g for g in groups if g["params"][0] is model.grid2)["lr"] = cfg["lr_depth"] if fine else 0.0
        use_lm = cfg["optimizer"] == "lm" or str(level) in cfg["lm_levels"].split(",")
        if use_lm and kernel and cfg["radial"] and cfg["radial_gate_heldout"] and kernel.radial and "radial_k_gate" not in info:
            info["radial_k_gate"] = float(model.kr)
            with torch.no_grad():
                gen_h = torch.Generator(device=device).manual_seed(cfg["seed"] + 78)
                hp = kernel.make_pool(torch.multinomial(probs, cfg["lm_stop_samples"], replacement=True, generator=gen_h))
                ch = max(1, int((1.6e7 if device.type == "cuda" else 4e6) / (k * cfg["lm_stop_samples"] * hp[0].shape[-1])))
                c_k = kernel.cost(*hp, nbrs, ch)
                k_val = model.kr.data.clone()
                model.kr.zero_()
                c_0 = kernel.cost(*hp, nbrs, ch)
                info["radial_gate_gain"] = (c_0 - c_k) / max(c_0, 1e-12)
                if c_k < c_0:
                    model.kr.data.copy_(k_val)
                else:
                    kernel.radial = False
                    info["radial_gated"] = 1
        if use_lm and kernel and cfg["radial"] and cfg["radial_gate"] > 0 and kernel.radial and not cfg["radial_gate_heldout"]:
            info["radial_k_gate"] = float(model.kr)
            if abs(info["radial_k_gate"]) < cfg["radial_gate"]:
                with torch.no_grad():
                    model.kr.zero_()
                kernel.radial = False
                info["radial_gated"] = 1
        if use_lm:
            layout = layout or lm_layout(model)
            iters = iters if cfg["optimizer"] == "lm" else cfg["lm_iters"]
            lm_loss = cfg["lm_photo_loss"] or cfg["photo_loss"]
            K_lm = kernel
            if lm_loss != cfg["photo_loss"]:
                K_lm = Kernel(model, {**cfg, "photo_loss": lm_loss})
                K_lm.radial = kernel.radial
                K_lm.set_level(img, cfg["features"] == "rgb" and lm_loss != "ncc", stride)
            S = 9 if lm_loss == "ncc" else 1
            D = S * img_flat.shape[0] + S
            if cfg["lm_samples"]:
                n_pix = cfg["lm_samples"] // S
                pool = torch.multinomial(probs, min(4, cfg["pool_factor"]) * cfg["lm_samples"], replacement=True)
            chunk = cfg["lm_chunk"] or max(1, int(1.5e8 / (k * n_pix * D * 21)))
            if cfg["lm_impl"] == "analytic":  # warps per chunk: its dense weight matrices have 48 entries per warp
                chunk = cfg["lm_chunk"] or max(1, int((1.6e7 if device.type == "cuda" else 4e6) / (k * n_pix * D)))
            held_prev = None
            if cfg["lm_stop"] > 0:
                gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 77)
                held = K_lm.make_pool(torch.multinomial(probs, cfg["lm_stop_samples"] // S, replacement=True, generator=gen))
                held = (held[0].flatten(1, 2), held[1].flatten(1, 2)) if S > 1 else held
                held_prev = K_lm.cost(*held, nbrs, chunk)
            for it in range(iters):
                if it == 0 or not cfg["lm_fixed"]:
                    pix = pool.gather(1, torch.randint(0, pool.shape[1], (N, n_pix), device=device))
                    u, v = (pix % W).float(), (pix // W).float()
                    if cfg["lm_impl"] == "analytic":
                        feat, quad = K_lm.make_pool(pix)
                        if S > 1:
                            feat, quad = feat.flatten(1, 2), quad.flatten(1, 2)
                        if cfg["lm_free_depth"]:
                            K_lm.z = torch.zeros(N, n_pix, device=device, dtype=feat.dtype)
                if cfg["lm_impl"] == "analytic":
                    lam, c0, c1, ok = K_lm.lm_step(feat, quad, nbrs, chunk, lam, cfg["lm_solver"], it=it)
                else:
                    lam, c0, c1, ok = lm_step(model, u, v, nbrs, img_flat, stride, S, layout, chunk, lam)
                hist.append({"level": level, "it": it, "loss": c0, "loss_after": c1, "lambda": lam, "accepted": ok, "t": time.perf_counter() - start})
                if tracker:  # every LM iteration: there are few and each one moves a lot
                    tracker.it += 1
                    tracker.record(level, force=True)
                if held_prev is not None:
                    held_cur = K_lm.cost(*held, nbrs, chunk)
                    hist[-1]["heldout"] = held_cur
                    if (held_prev - held_cur) / max(held_prev, 1e-12) < cfg["lm_stop"]:
                        info["lm_stopped_at"] = it + 1
                        break
                    held_prev = held_cur
            if cfg["lm_free_depth"]:
                info["free_depth_abs"] = float(K_lm.z.abs().mean())
                K_lm.z = None
            continue
        if cfg["optimizer"] == "lbfgs":
            pix = pool[:, : cfg["lbfgs_samples"] * n_pix]
            u, v = (pix % W).float(), (pix // W).float()
            params = [p for g in groups for p in g["params"]]
            opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe",
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
        if fadam:  # same samples and update as below (fresh state per level), with the gradient of pba_analytic.py
            fadam.reset()
            if basis:
                basis.reset()
            grads_k = [basis.g_rot, basis.g_trans] + fadam.grads[4:] if basis else fadam.grads
            # (with NCC the pool of 16 M centres is cut to 16 M / 9, each with its 9 samples)
            feat_pool, quad_pool = kernel.make_pool(pool[:, : cfg["pool_factor"] * n_pix])
            const = kernel.prep(nbrs)
            graph = None
            use_graph = cfg["cuda_graph"] and device.type == "cuda" and cfg["nbr_iter"] == 0 and tracker is None

            def body(want_loss):
                idx_ = torch.randint(0, feat_pool.shape[1], (N, n_pix), device=device)
                r_ = kernel.gradient(*kernel.take(feat_pool, quad_pool, idx_), nbrs, const, grads_k, want_loss=want_loss)
                if basis:
                    basis.grads_to(fadam.grads)
                fadam.step_graph()
                if cfg["radial"]:
                    model.kr.data.clamp_(max=cfg["radial_max"])
                if basis:
                    basis.apply()
                return r_

            for it in range(iters if use_graph else 0):
                log = it % 20 == 0 or it == iters - 1
                fadam.set_scalars(0.1 + 0.45 * (1 + math.cos(math.pi * it / iters)))
                if it < 3 or log:  # eager: warm-up (on a side stream, as torch.cuda.graph wants) and the logged iterations
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        res = body(log)
                    torch.cuda.current_stream().wait_stream(side)
                    if log:
                        hist.append({"level": level, "it": it, "loss": res[0], **res[1], "t": time.perf_counter() - start})
                    continue
                if graph is None:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):  # records the iteration without running it; the scalars set above stay for the replay
                        body(False)
                graph.replay()
            if graph is not None:  # free the graph's private memory pool before the next level (the LM levels need it)
                del graph
                torch.cuda.empty_cache()
            for it in range(0 if use_graph else iters):
                idx = torch.randint(0, feat_pool.shape[1], (N, n_pix), device=device)
                log = it % 20 == 0 or it == iters - 1
                nb_it = nbrs
                if cfg["nbr_iter"] > 0:
                    nb_it = nbrs.gather(1, torch.rand(N, k, device=device).argsort(1)[:, : cfg["nbr_iter"]])
                    const = kernel.prep(nb_it)
                res = kernel.gradient(*kernel.take(feat_pool, quad_pool, idx), nb_it, const, grads_k, want_loss=log)
                if basis:
                    basis.grads_to(fadam.grads)
                fadam.step(0.1 + 0.45 * (1 + math.cos(math.pi * it / iters)))
                if basis:
                    basis.apply()
                if cfg["radial"]:
                    model.kr.data.clamp_(max=cfg["radial_max"])
                if log:
                    hist.append({"level": level, "it": it, "loss": res[0], **res[1], "t": time.perf_counter() - start})
                if tracker:
                    tracker.it += 1
                    tracker.record(level)
            continue
        opt = torch.optim.Adam([dict(g) for g in groups])  # fresh state and learning rates per level
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * it / iters)))
        for it in range(iters):
            pix = pool.gather(1, torch.randint(0, pool.shape[1], (N, n_pix), device=device))
            u, v = (pix % W).float(), (pix // W).float()
            if head is not None:
                if it % cfg["head_refresh"] == 0:
                    cache = head.all() / model.scale
                sub = torch.randperm(N, device=device)[: cfg["head_frames"]]
                d_sub = head(sub) / model.scale
                model.depth_dense = cache.index_put((sub,), d_sub)
            nb_it = nbrs
            if cfg["nbr_iter"] > 0:
                nb_it = nbrs.gather(1, torch.rand(N, k, device=device).argsort(1)[:, : cfg["nbr_iter"]])
            loss, parts = model.loss(u, v, nb_it, img_flat, stride, match=str(level) in cfg["match_levels"].split(","))
            if head is not None:  # depth prior relative to DA3's prediction (also fixes the scale gauge)
                loss = loss + cfg["w_prior"] * ((torch.log(d_sub.clamp_min(1e-6)) - torch.log(model.depth0[sub])) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if it % 20 == 0 or it == iters - 1:
                hist.append({"level": level, "it": it, "loss": loss.item(), **{k: v.item() for k, v in parts.items()}, "t": time.perf_counter() - start})
            if tracker:
                tracker.it += 1
                tracker.record(level)
    if head is not None:
        model.depth_dense = head.all() / model.scale
    if cfg["patch"]:
        from pba_patch import run_patch

        radial = cfg["radial"]
        if kernel and levels:  # the gate of the LM level above, as there is none now
            radial = kernel.radial
            if radial and cfg["radial_gate"] > 0 and "radial_k_gate" not in info:
                info["radial_k_gate"] = float(model.kr)
                if abs(info["radial_k_gate"]) < cfg["radial_gate"]:
                    with torch.no_grad():
                        model.kr.zero_()
                    radial = False
                    info["radial_gated"] = 1
        run_patch(model, nbrs, images, blur, cfg, hist, info, tracker, radial, start, device, sampling_probs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    opt_s = time.perf_counter() - start - (tracker.spent if tracker else 0.0)
    if tracker:
        tracker.record(len(levels), force=True)
        info["track"] = tracker.rows
    after = eval_photo(model, nbrs, img_sharp)

    with torch.no_grad():
        R, t = model.poses()
        ext_out = torch.cat([R, t[..., None] * model.scale], -1).cpu().numpy()
        K_out = model.intrinsics().cpu().numpy()
        depth_out = (model.dense_depth() * model.scale).cpu().numpy()
        f_scale = torch.exp(model.logf).cpu().numpy()
    if kernel:  # the same keys for every scene: the evaluator's mean takes them from the first one
        info.update(cg_iters=float(np.mean(kernel.cg_iters)) if kernel.cg_iters else 0.0, schur_failed=kernel.schur_failed,
                    lm_factorisations=kernel.factorisations, lm_chords=kernel.chords)
        info.setdefault("lm_stopped_at", 0)
        info.setdefault("free_depth_abs", 0.0)
        info.setdefault("radial_k_gate", 0.0)
        info.setdefault("radial_gated", 0)
        info.setdefault("radial_gate_gain", 0.0)
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
        pp_x=float(model.pp.detach()[0]),
        pp_y=float(model.pp.detach()[1]),
        depth_corr_abs=float(np.abs(model.grid.detach().cpu().numpy()).mean()),
        bright_gain_abs=float(model.bright.detach()[:, 0].abs().mean()),
        bright_bias_abs=float(model.bright.detach()[:, 1].abs().mean()),
        bright_gain_mean=float(model.bright.detach()[1:, 0].mean()),
    )
    return {"depth": depth_out.astype(np.float32), "extrinsics": ext_out.astype(np.float32), "intrinsics": K_out.astype(np.float32)}, info, hist


@torch.no_grad()
def relocalise(images, depth, conf, ext, K, conf_frac, gain, n_yaw=12, grid=(20, 15), z_tol=0.05, i_tol=0.1):
    """reloc: frames whose median DA3 confidence is below conf_frac x the scene's median (DA3's grossly misplaced frames have
    low confidence, while their depth is as good as the others', see results/failure_modes/DA3-SMALL/REPORT.md) get a discrete
    pose search. Hypotheses: n_yaw rotations about the vertical (the mean of the cameras' up axes; large errors are
    almost pure rotations about it), each with the camera centre at its current place or at the centre of any frame that is
    not a candidate. Score: the fraction of (sample, other frame) pairs, over a grid of the frame's pixels and all non-candidate
    frames, whose depth agrees within z_tol (log) and grey value within i_tol. The best hypothesis replaces the pose if it scores
    gain x the current one. Returns the extrinsics [N, 3, 4] (metric, like ext) and info."""
    N, H, W = depth.shape
    med = conf.flatten(1).median(1).values
    cand = med < conf_frac * med.median()
    cand[0] = False  # the gauge
    info = {"reloc_candidates": float(cand.float().mean()), "reloc_moved": 0.0}
    if not cand.any() or (~cand).sum() < 2:
        return ext, info
    grey = gaussian_blur(images.mean(1, keepdim=True), 2.0)[:, 0]
    R, t = ext[:, :, :3], ext[:, :, 3]
    C = -torch.einsum("nji,nj->ni", R, t)
    up = -R[:, 1, :].mean(0)  # world direction of each camera's -y axis (OpenCV: y points down)
    up = up / up.norm().clamp_min(1e-12)
    good = torch.nonzero(~cand)[:, 0]
    gw, gh = grid
    vs, us = torch.meshgrid(torch.linspace(2, H - 3, gh, device=depth.device), torch.linspace(2, W - 3, gw, device=depth.device), indexing="ij")
    us, vs = us.reshape(-1), vs.reshape(-1)
    Kinv = torch.linalg.inv(K)
    yaws = torch.arange(n_yaw, device=depth.device) * (2 * math.pi / n_yaw)
    kx = torch.zeros(3, 3, device=depth.device)
    kx[0, 1], kx[0, 2], kx[1, 0], kx[1, 2], kx[2, 0], kx[2, 1] = -up[2], up[1], up[2], -up[0], -up[1], up[0]
    Q = torch.eye(3, device=depth.device) + torch.sin(yaws)[:, None, None] * kx + (1 - torch.cos(yaws))[:, None, None] * (kx @ kx)  # [Y, 3, 3]
    Rg, tg, Kg, Dg, Ig = R[good], t[good], K[good], depth[good], grey[good]
    ext = ext.clone()
    moved = 0
    for f in torch.nonzero(cand)[:, 0].tolist():
        d = depth[f, vs.long(), us.long()]
        Xc = (Kinv[f] @ torch.stack([us, vs, torch.ones_like(us)]) * d).T  # [P, 3]
        If = grey[f, vs.long(), us.long()]
        Rcw = Q @ R[f].T  # [Y, 3, 3] camera-to-world rotations
        centres = torch.cat([C[f][None], C[good]])  # [Cn, 3]
        Xw = (Xc @ Rcw.transpose(-1, -2))[:, None] + centres[None, :, None]  # [Y, Cn, P, 3]
        Y, Cn, P = Xw.shape[:3]
        score = torch.zeros(Y, Cn, device=depth.device)
        for g in range(len(good)):
            Xg = Xw @ Rg[g].T + tg[g]
            z = Xg[..., 2]
            u = Kg[g, 0, 0] * Xg[..., 0] / z.clamp_min(1e-6) + Kg[g, 0, 2]
            v = Kg[g, 1, 1] * Xg[..., 1] / z.clamp_min(1e-6) + Kg[g, 1, 2]
            ok = (z > 1e-6) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
            ui, vi = u.clamp(0, W - 1).round().long(), v.clamp(0, H - 1).round().long()
            dz = (torch.log(z.clamp_min(1e-6)) - torch.log(Dg[g][vi, ui].clamp_min(1e-6))).abs()
            score += (ok & (dz < z_tol) & ((Ig[g][vi, ui] - If).abs() < i_tol)).float().mean(-1)
        best = int(score.argmax())
        y, c = best // Cn, best % Cn
        if float(score[y, c]) > gain * float(score[0, 0]) and (y, c) != (0, 0):
            Rw = Rcw[y].T
            ext[f, :, :3], ext[f, :, 3] = Rw, -Rw @ centres[c]
            moved += 1
    info["reloc_moved"] = moved / N
    return ext, info


def fix_gross(pred_ext, gt_ext, median_depth, max_deg=15.0, max_rel=0.25):
    """init_fix_gross (an oracle): the frames whose pose is grossly wrong, as analyse_failure_modes.py labels them (after the
    robust alignment of all cameras: geodesic median of the rotations, LMedS scale and translation of the centres; more than
    max_deg off in rotation or max_rel median depths in the centre), replaced by their GT pose mapped into DA3's frame. Frame 0
    (the gauge) is kept. Returns the extrinsics [N, 3, 4] and the number of replaced frames."""
    from analyse_failure_modes import apply_sim3, robust_rotation_alignment, robust_scale_translation

    def c2w(e):
        T = np.tile(np.eye(4), (len(e), 1, 1))
        T[:, :3] = e[:, :3]
        return np.linalg.inv(T)
    P, G = c2w(pred_ext.astype(np.float64)), c2w(gt_ext.astype(np.float64))
    A = robust_rotation_alignment(P[:, :3, :3], G[:, :3, :3])
    sim = robust_scale_translation(P[:, :3, 3], G[:, :3, 3], A)
    dR = np.einsum("ij,njk,nlk->nil", A, P[:, :3, :3], G[:, :3, :3])
    ang = np.degrees(np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    off = np.linalg.norm(apply_sim3(sim, P[:, :3, 3]) - G[:, :3, 3], axis=1) / (sim[0] * median_depth)
    bad = (ang > max_deg) | (off > max_rel)
    bad[0] = False
    Rg = np.einsum("ji,njk->nik", A, G[:, :3, :3])  # GT camera-to-world rotations and centres in DA3's frame
    cg = ((G[:, :3, 3] - sim[2]) @ A) / sim[0]
    out = pred_ext[:, :3].astype(np.float64).copy()
    Rw = Rg.transpose(0, 2, 1)
    out[bad, :, :3] = Rw[bad]
    out[bad, :, 3] = -np.einsum("nij,nj->ni", Rw[bad], cg[bad])
    return out.astype(np.float32), int(bad.sum())


# ---------------------------------------------------------------- wandb (optional)


def wandb_start(src, dst, cfg):
    """This process's wandb run, or None. Logging is on when WANDB_PROJECT is set: pba_run.sh sets it together with
    WANDB_RUN_ID and WANDB_RESUME=allow, so its per-dataset processes and scripts/wandb_summary.py share one run.
    A failing init (no login, no network) only disables logging."""
    if not os.environ.get("WANDB_PROJECT") or os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb

        config = {k: v for k, v in cfg.items() if k not in ("datasets", "max_scenes", "scenes")}  # differ per process
        config.update(model=os.path.basename(os.path.normpath(src)), run=os.path.basename(os.path.normpath(dst)), dst=dst,
                      slurm_job=os.environ.get("SLURM_JOB_ID", ""), host=os.uname().nodename)
        return wandb.init(resume="allow", config=config)
    except Exception as e:  # noqa: BLE001
        print(f"[run_tto_pba] wandb init failed, not logging: {e!r}")
        return None


def wandb_scene(run, dst, dataset, scene, metrics, hist):
    """One wandb step per optimised scene (step = scenes done in this run, counted from the metric files): the scene's
    metrics, the running means of its dataset and its loss curve (photo/geo of the Adam levels, the LM cost)."""
    import wandb

    n = sum(len([k for k in json.load(open(os.path.join(dst, "metric_results", f))) if k != "mean"])
            for f in os.listdir(os.path.join(dst, "metric_results")) if f.endswith("_pba.json"))
    m, done = metrics[scene], list(metrics.values())
    row = {
        "progress/scenes": n,
        "scene/auc03": m["auc03"], "scene/auc03_init": m["auc03_init"], "scene/auc03_gain": m["auc03"] - m["auc03_init"],
        "scene/auc30": m["auc30"], "scene/auc30_gain": m["auc30"] - m["auc30_init"],
        "scene/photo_ratio": m["photo_after"] / max(m["photo_before"], 1e-12),
        "scene/geo_ratio": m["geo_after"] / max(m["geo_before"], 1e-12),
        "scene/focal_err": m["focal_err"], "scene/opt_s": m["opt_s"], "scene/peak_mem_gib": m["peak_mem_gib"],
        f"{dataset}/scenes": len(done),
        **{f"resid/{k}": v for k, v in m.items() if k.startswith(("photo_q", "photo_sigma", "photo_out"))},
        **{f"{dataset}/{k}_mean": float(np.mean([v[k] for v in done])) for k in ("auc03", "auc03_init", "auc30", "auc30_init", "opt_s")},
    }
    its = {}  # global iteration = offset of the level + iteration within it
    for h in hist:
        if "it" in h:
            its[h["level"]] = max(its.get(h["level"], 0), h["it"] + 1)
    off, o = {}, 0
    for lv in sorted(its):
        off[lv], o = o, o + its[lv]
    series = {}
    for h in hist:
        if "it" in h:
            for key, name in ((("photo", "photo"), ("geo", "geo")) if "photo" in h else (("loss", "LM cost"),)):
                xs, ys = series.setdefault(name, ([], []))
                xs.append(off[h["level"]] + h["it"])
                ys.append(h[key])
    if series:
        names = list(series)
        row[f"curves/{dataset}/{scene.replace('/', '_')}"] = wandb.plot.line_series(
            xs=[series[k][0] for k in names], ys=[series[k][1] for k in names], keys=names, title=f"{dataset} {scene}", xname="iteration")
    run.log(row, step=n)


def main():
    if len(sys.argv) < 3 or sys.argv[1].startswith("-"):
        sys.exit(__doc__)
    src, dst, cfg = sys.argv[1], sys.argv[2], parse_overrides(sys.argv[3:])
    sys.stdout.reconfigure(line_buffering=True)
    os.chdir(REPO_ROOT)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "pba_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    run = wandb_start(src, dst, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    api = head_state = None
    if cfg["depth_param"] == "head":
        from depth_anything_3.api import DepthAnything3

        name = os.path.basename(os.path.normpath(src))
        api = DepthAnything3.from_pretrained(f"depth-anything/{name}").to(device).eval()
        head_state = {k: v.clone() for k, v in api.model.head.state_dict().items()}
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
        first = cfg["first_scene"]
        for i, scene_dir in enumerate(scenes[first : first + cfg["max_scenes"] if cfg["max_scenes"] else None], first):
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

            if cfg["init_gt"]:
                ext = gt["extrinsics"][:, :3].astype(np.float32).copy()
                spread = lambda e: np.linalg.norm((c := -np.einsum("nji,nj->ni", e[:, :, :3], e[:, :, 3])) - c.mean(0), axis=1).mean()  # noqa: E731
                ext[:, :, 3] *= spread(pred["extrinsics"]) / max(spread(ext), 1e-12)
                pred.update(extrinsics=ext)
                if cfg["init_gt"] == "pose_focal":
                    w_orig = cv2.imread(str(gt["image_files"][0])).shape[1]
                    k_ = pred["intrinsics"].astype(np.float32).copy()
                    k_[:, 0, 0] = gt["intrinsics"][:, 0, 0] * pred["depth"].shape[2] / w_orig
                    k_[:, 1, 1] = gt["intrinsics"][:, 1, 1] * pred["depth"].shape[2] / w_orig
                    pred.update(intrinsics=k_)

            n_fixed = 0
            if cfg["init_fix_gross"]:
                ext, n_fixed = fix_gross(pred["extrinsics"], gt["extrinsics"], float(np.median(pred["depth"])))
                pred["extrinsics"] = ext
            if cfg["init_noise_deg"] > 0:
                rng = np.random.default_rng(cfg["seed"] + 1)
                ext = pred["extrinsics"].astype(np.float64).copy()
                for f in range(1, len(ext)):
                    axis = rng.normal(size=3)
                    Rn = cv2.Rodrigues(axis / np.linalg.norm(axis) * np.deg2rad(cfg["init_noise_deg"]))[0]
                    ext[f] = Rn @ ext[f]  # R' = Rn R, t' = Rn t keeps the camera centre
                pred["extrinsics"] = ext.astype(np.float32)
            feat_path = os.path.join(cfg["feat_cache"] or os.path.join("workspace/tto/feats", os.path.basename(os.path.normpath(src))), dataset, scene + ".npz")
            try:
                out, info, hist = refine_scene(gt["image_files"], pred, cfg, device, feat_path, api, head_state)
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
            track = info.pop("track", None)
            if track:
                for row in track:
                    ext_t = torch.from_numpy(as_homogeneous(row.pop("ext")))
                    p = compute_pose(ext_t, gt_ext)
                    row.update(auc03=float(p["auc03"]), auc30=float(p["auc30"]))
                    r_err, t_err = se3_to_relative_pose_error(align_to_first_camera(ext_t), align_to_first_camera(gt_ext), len(ext_t))
                    row.update(rot_med=float(r_err.median()), trans_med=float(t_err.median()), rot_q90=float(r_err.quantile(0.9)), trans_q90=float(t_err.quantile(0.9)))
                with open(os.path.join(dst, "metric_results", f"{dataset}_pba_track.jsonl"), "a") as f:
                    f.write(json.dumps({"scene": scene, "track": track}) + "\n")
            pose0 = compute_pose(torch.from_numpy(as_homogeneous(pred["extrinsics"])), gt_ext)
            pose = compute_pose(torch.from_numpy(as_homogeneous(out["extrinsics"])), gt_ext)
            gt_f = gt["intrinsics"][:, 0, 0] * pred["depth"].shape[2] / cv2.imread(str(gt["image_files"][0])).shape[1]
            focal_err0 = np.abs(pred["intrinsics"][:, 0, 0] / gt_f - 1).mean()
            focal_err = np.abs(out["intrinsics"][:, 0, 0] / gt_f - 1).mean()
            rec = {**pose, **{f"{k}_init": v for k, v in pose0.items()}, "focal_err": focal_err, "focal_err_init": focal_err0, **info}
            if cfg["init_fix_gross"]:
                rec["gross_fixed"] = n_fixed / len(pred["extrinsics"])
            metrics[scene] = Evaluator._to_float_dict(rec)
            Evaluator._dump_json(metric_path, {**metrics, "mean": Evaluator._mean_of_dicts(metrics.values())})
            with open(hist_path, "a") as f:
                f.write(json.dumps({"scene": scene, "hist": hist}) + "\n")
            if run is not None:
                try:
                    wandb_scene(run, dst, dataset, scene, metrics, hist)
                except Exception as e:  # noqa: BLE001
                    print(f"[run_tto_pba] wandb logging failed: {e!r}")
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
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
