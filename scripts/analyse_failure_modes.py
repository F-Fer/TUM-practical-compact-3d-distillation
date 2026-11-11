"""Decompose the pose and depth errors of a DA3 benchmark run into noise, structure and outliers.

Usage:
    uv run scripts/analyse_failure_modes.py DA3-SMALL [--datasets hiroom 7scenes scannetpp dtu]

Reads the unposed predictions in workspace/evaluation/<model>/model_results/ and writes one npz
per scene to workspace/failure_modes/<model>/<dataset>/, which plot_failure_modes.py turns into
figures. Nothing here depends on the model, so other models only need a benchmark run first.

Pose: predictions are aligned with a robust Sim(3) (geodesic median rotation, LMedS scale and
translation, so that a bad camera stands out instead of being smeared over all cameras by least
squares). Each camera's residual is then split along the frame order into a smooth part, jitter
around it, and outliers.

Depth: the error is r = log(pred) - log(gt), i.e. a relative error. It is split into one scale per
scene (removed), an offset per frame, a smooth spatial part and high-frequency residual, whose
tail beyond DEPTH_OUTLIER_SIGMA is called outliers.
"""

import argparse
import glob
import multiprocessing
import os
import sys

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
from depth_anything_3.bench.utils import calculate_auc_np, compute_pose, se3_to_relative_pose_error  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ["hiroom", "7scenes", "scannetpp", "dtu", "eth3d"]

# |x| of isotropic 3D Gaussian noise follows sigma * Maxwell: median = 1.538 sigma, 99.9% = 4.03 sigma
MAXWELL_MEDIAN = 1.538
POSE_OUTLIER_SIGMA = 4.03
DEPTH_OUTLIER_SIGMA = 5.0  # ~6e-7 of Gaussian noise
DEPTH_LOWPASS_FRAC = 1 / 24  # sigma of the smooth part relative to image width (~1.5 patches at 504 px)
PIXELS_PER_SCENE = 200_000  # sampled for the pooled depth statistics
INLIER_THRESHOLD = 0.03  # relative depth error, as in the 3% inlier ratio


# -------------------- Pose -------------------- #


def umeyama(src, dst):
    """Least-squares Sim(3) with dst ~ s * R @ src + t."""
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    x, y = src - mu_src, dst - mu_dst
    U, S, Vt = np.linalg.svd(y.T @ x / len(src))
    D = np.diag([1, 1, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
    R = U @ D @ Vt
    var = (x**2).sum() / len(src)
    s = (S * np.diag(D)).sum() / max(var, 1e-12)
    return s, R, mu_dst - s * R @ mu_src


def apply_sim3(sim3, x):
    s, R, t = sim3
    return s * x @ R.T + t


def robust_scale_translation(src, dst, R, iters=1000, seed=0):
    """Sim(3) with fixed rotation R: LMedS over 2-point samples, then least squares on the inliers."""
    rotated = src @ R.T

    def fit(idx):
        x, y = rotated[idx] - rotated[idx].mean(0), dst[idx] - dst[idx].mean(0)
        s = (x * y).sum() / max((x**2).sum(), 1e-12)
        return s, R, dst[idx].mean(0) - s * rotated[idx].mean(0)

    rng = np.random.default_rng(seed)
    best, best_median = fit(np.arange(len(src))), np.inf
    for _ in range(iters):
        sim3 = fit(rng.choice(len(src), 2, replace=False))
        median = np.median(np.linalg.norm(apply_sim3(sim3, src) - dst, axis=1))
        if median < best_median:
            best, best_median = sim3, median
    inliers = np.linalg.norm(apply_sim3(best, src) - dst, axis=1) < best_median / MAXWELL_MEDIAN * POSE_OUTLIER_SIGMA
    return fit(np.nonzero(inliers)[0]) if inliers.sum() >= 3 else best


def robust_rotation_alignment(R_pred, R_gt, iters=20):
    """Geodesic median A of R_gt @ R_pred^T, so that A @ R_pred ~ R_gt (camera-to-world)."""
    samples = Rotation.from_matrix(R_gt @ R_pred.transpose(0, 2, 1))
    A = samples.mean()
    for _ in range(iters):  # Weiszfeld on SO(3)
        v = (samples * A.inv()).as_rotvec()
        w = 1 / np.maximum(np.linalg.norm(v, axis=1), 1e-9)
        A = Rotation.from_rotvec((v * w[:, None]).sum(0) / w.sum()) * A
    return A.as_matrix()


def rolling_median(x, half_window):
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(x[max(0, i - half_window) : i + half_window + 1], axis=0)
    return out


def split_along_sequence(x):
    """Split per-frame residual vectors [N, 3] into smooth + jitter + outliers along the frame order."""
    half_window = max(2, len(x) // 20)
    smooth = rolling_median(x, half_window)
    high = x - smooth
    norm = np.linalg.norm(high, axis=1)
    sigma = np.median(norm) / MAXWELL_MEDIAN
    # The second condition keeps pure drift, where sigma is ~0, from being flagged
    outlier = (norm > POSE_OUTLIER_SIGMA * sigma) & (norm > np.median(np.linalg.norm(x, axis=1)))
    return smooth, high, outlier, sigma


def von_neumann_ratio(x):
    """Mean squared successive difference over twice the variance: 1 for white noise, -> 0 for smooth."""
    x = x - np.median(x, axis=0)
    return (np.diff(x, axis=0) ** 2).sum(1).mean() / (2 * (x**2).sum(1).mean())


def camera_pose(R_cw, c_w):
    T = np.tile(np.eye(4), (len(R_cw), 1, 1))
    T[:, :3, :3], T[:, :3, 3] = R_cw, c_w
    return T


def relative_errors(pred_c2w, gt_c2w):
    """Pairwise rotation and translation-direction errors of the camera-to-camera poses, in degrees."""
    return [
        x.numpy()
        for x in se3_to_relative_pose_error(torch.from_numpy(pred_c2w), torch.from_numpy(gt_c2w), len(pred_c2w))
    ]


def auc_scores(pred_c2w, gt_c2w, gt_w2c, bench_order):
    """[AUC@3, AUC@30] as the benchmark computes them, then the same from relative_errors.

    The benchmark passes world-to-camera poses to compute_pose, so its inv(T_i) @ T_j is not the
    relative camera pose: the translation part depends on which camera comes first, and the score
    therefore on the frame order. relative_errors doesn't have this problem.
    """
    pred_w2c = np.linalg.inv(pred_c2w)[bench_order]
    bench = compute_pose(torch.from_numpy(pred_w2c), torch.from_numpy(gt_w2c[bench_order]))
    r, t = relative_errors(pred_c2w, gt_c2w)
    return [bench.auc03, bench.auc30, calculate_auc_np(r, t, max_threshold=3)[0], calculate_auc_np(r, t, max_threshold=30)[0]]


def pose_errors(pred_w2c, gt_w2c, depth_scale, bench_order=None):
    """All pose outputs for one scene; poses are [N, 4, 4] in frame order, translation errors in units of depth_scale.

    bench_order reorders the frames into the order the benchmark scored them in.
    """
    bench_order = np.arange(len(pred_w2c)) if bench_order is None else bench_order
    pred_c2w, gt_c2w = np.linalg.inv(pred_w2c), np.linalg.inv(gt_w2c)
    c_pred, c_gt = pred_c2w[:, :3, 3], gt_c2w[:, :3, 3]
    R_pred, R_gt = pred_c2w[:, :3, :3], gt_c2w[:, :3, :3]

    # One gauge for rotations and centres, otherwise relative translation directions (and AUC) change
    gauge = robust_rotation_alignment(R_pred, R_gt)
    sim3 = robust_scale_translation(c_pred, c_gt, gauge)
    c_aligned = apply_sim3(sim3, c_pred)
    c_aligned_ls = apply_sim3(umeyama(c_pred, c_gt), c_pred)
    R_aligned = gauge @ R_pred

    t_res = (c_aligned - c_gt) / depth_scale
    r_res = np.degrees(Rotation.from_matrix(R_aligned @ R_gt.transpose(0, 2, 1)).as_rotvec())

    t_smooth, t_high, t_out, t_sigma = split_along_sequence(t_res)
    r_smooth, r_high, r_out, r_sigma = split_along_sequence(r_res)

    def rebuild(r, t):
        # Applied as corrections to the prediction, because the GT rotations of 7-Scenes aren't orthonormal
        correction = Rotation.from_rotvec(np.radians(r)) * Rotation.from_rotvec(np.radians(r_res)).inv()
        c2w = camera_pose(correction.as_matrix() @ R_aligned, c_aligned + (t - t_res) * depth_scale)
        return auc_scores(c2w, gt_c2w, gt_w2c, bench_order)

    r_in, t_in = ~r_out[:, None], ~t_out[:, None]
    counterfactuals = {
        "as predicted": (r_res, t_res),
        "no outliers": (r_in * r_res + ~r_in * r_smooth, t_in * t_res + ~t_in * t_smooth),
        "no jitter": (r_in * r_smooth + ~r_in * r_res, t_in * t_smooth + ~t_in * t_res),
        "no smooth part": (r_res - r_smooth, t_res - t_smooth),
        "perfect rotation": (0 * r_res, t_res),
        "perfect translation": (r_res, 0 * t_res),
    }

    rot_pairs, trans_pairs = relative_errors(pred_c2w, gt_c2w)
    i, j = np.triu_indices(len(pred_w2c), 1)
    # build_pair_index enumerates pairs in the same lexicographic order as triu_indices
    rot_matrix, trans_matrix = (np.full((len(pred_w2c),) * 2, np.nan, np.float32) for _ in range(2))
    rot_matrix[i, j] = rot_matrix[j, i] = rot_pairs
    trans_matrix[i, j] = trans_matrix[j, i] = trans_pairs
    baseline = np.linalg.norm(c_gt[i] - c_gt[j], axis=1) / depth_scale

    return {
        "sim3_scale": sim3[0],
        "c_gt": c_gt / depth_scale,
        "c_aligned": c_aligned / depth_scale,
        "c_aligned_ls": c_aligned_ls / depth_scale,
        "t_res": t_res,
        "t_smooth": t_smooth,
        "t_high": t_high,
        "t_outlier": t_out,
        "t_sigma": t_sigma,
        "t_vn": von_neumann_ratio(t_res),
        "r_res": r_res,
        "r_smooth": r_smooth,
        "r_high": r_high,
        "r_outlier": r_out,
        "r_sigma": r_sigma,
        "r_vn": von_neumann_ratio(r_res),
        "rot_pairs": rot_matrix,
        "trans_pairs": trans_matrix,
        "pair_baseline": baseline,
        "cf_names": np.array(list(counterfactuals)),
        "cf_auc": np.array([rebuild(r, t) for r, t in counterfactuals.values()], dtype=np.float64),
    }


# -------------------- Depth -------------------- #


def load_gt_depth(dataset, image_file, data_scene):
    """GT depth in the dataset's units with 0 where invalid, plus an optional extra validity mask."""
    if dataset == "hiroom":
        depth_file = image_file.replace("/image/", "/depth/").rsplit(".", 1)[0] + ".png"
        depth = cv2.imread(depth_file, -1) / 65535.0 * 100.0
        aliasing = cv2.imread(depth_file.replace("/depth/", "/aliasing_mask/"), -1)
        return depth, None if aliasing is None else aliasing == 0
    if dataset == "7scenes":
        depth = cv2.imread(image_file.replace(".color.", ".depth."), -1).astype(np.float64)
        depth[depth == 65535] = 0
        return depth / 1000.0, None
    if dataset == "scannetpp":
        depth_file = os.path.join(
            image_file.split("/images/")[0], "render_depth", os.path.basename(image_file)[:-4] + ".png"
        )
        return cv2.imread(depth_file, -1) / 1000.0, None
    if dataset == "dtu":
        idx = int(os.path.basename(image_file).split("_")[1]) - 1
        root = os.path.join(image_file.split("/Rectified/")[0], "depth_raw", "Depths", data_scene)
        depth = cv2.imread(os.path.join(root, f"depth_map_{idx:04d}.pfm"), -1).astype(np.float64)
        return depth, cv2.imread(os.path.join(root, f"depth_visual_{idx:04d}.png"), -1) > 10
    if dataset == "eth3d":
        # Raw float32 at full DSLR resolution, 0 or inf where invalid; mask value 1 marks occluded regions
        scene_dir, name = image_file.split("/images/dslr_images/")
        h, w = cv2.imread(image_file, cv2.IMREAD_REDUCED_GRAYSCALE_8).shape
        depth = np.fromfile(os.path.join(scene_dir, "ground_truth_depth/dslr_images", name), dtype=np.float32).reshape(h * 8, w * 8)
        depth = np.where(np.isfinite(depth), depth, 0).astype(np.float64)
        mask = cv2.imread(os.path.join(scene_dir, "masks_for_images/dslr_images", name.rsplit(".", 1)[0] + ".png"), cv2.IMREAD_GRAYSCALE)
        return depth, None if mask is None else mask != 1
    raise ValueError(dataset)


SPARSE_GT = {"eth3d"}  # laser samples on ~10% of the full-resolution pixels, scattered


def block_mean_depth(depth, valid, size, min_frac=0.03, max_rel_std=0.05):
    """Mean of the valid samples in each output pixel's block; blocks with few samples or mixed depths are invalid."""
    w = valid.astype(np.float64)
    frac = cv2.resize(w, size, interpolation=cv2.INTER_AREA)
    mean = cv2.resize(depth * w, size, interpolation=cv2.INTER_AREA) / np.maximum(frac, 1e-12)
    mean_sq = cv2.resize(depth**2 * w, size, interpolation=cv2.INTER_AREA) / np.maximum(frac, 1e-12)
    std = np.sqrt(np.maximum(mean_sq - mean**2, 0))
    ok = (frac >= min_frac) & (mean > 0) & (std < max_rel_std * mean)
    return np.where(ok, mean, 0), ok


def load_scene(model, dataset, scene, src=None):
    """Predictions and GT of a scene in frame order, with GT depth resampled to the prediction grid. src: a directory with
    model_results/ (e.g. a TTO run) instead of workspace/evaluation/<model>."""
    src = src or os.path.join(REPO_ROOT, "workspace/evaluation", model)
    export = os.path.join(src, "model_results", dataset, scene, "unposed/exports")
    pred = np.load(os.path.join(export, "mini_npz/results.npz"))
    gt = np.load(os.path.join(export, "gt_meta.npz"), allow_pickle=True)
    image_files = [
        f if os.path.isabs(f) else os.path.join(REPO_ROOT, f) for f in gt["image_files"].tolist()
    ]
    order = np.argsort(image_files)  # DTU moves one view to the front; all others are sorted already
    H, W = pred["depth"].shape[1:]

    depths, valids = [], []
    for i in order:
        depth, mask = load_gt_depth(dataset, image_files[i], os.path.basename(scene))
        if dataset in SPARSE_GT:
            depth, valid = block_mean_depth(depth, depth > 0 if mask is None else (depth > 0) & mask, (W, H))
        else:
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            valid = depth > 0
            if mask is not None:
                valid &= cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        depths.append(depth)
        valids.append(valid)

    pred_w2c = np.tile(np.eye(4), (len(order), 1, 1))
    pred_w2c[:, :3] = pred["extrinsics"][order]
    gt_w2c = gt["extrinsics"][order].astype(np.float64)
    if gt_w2c.shape[1] == 3:
        gt_w2c = np.concatenate([gt_w2c, np.tile([[[0, 0, 0, 1.0]]], (len(gt_w2c), 1, 1))], axis=1)
    return {
        "image_files": [image_files[i] for i in order],
        "bench_order": np.argsort(order),
        "pred_depth": pred["depth"][order],
        "conf": pred["conf"][order],
        "pred_w2c": pred_w2c,
        "gt_w2c": gt_w2c,
        "gt_depth": np.stack(depths),
        "valid": np.stack(valids),
    }


def split_depth_error(e, valid):
    """Split one frame's scene-scaled log-depth error into offset, smooth part and high-frequency rest."""
    offset = np.median(e[valid])
    e1 = np.where(valid, e - offset, 0)
    sigma = 1.4826 * np.median(np.abs(e1[valid]))
    # Normalized convolution: holes and gross errors must not leak into the smooth part
    weight = (valid & (np.abs(e1) < DEPTH_OUTLIER_SIGMA * sigma)).astype(np.float32)
    blur_sigma = e.shape[1] * DEPTH_LOWPASS_FRAC
    low = cv2.GaussianBlur(e1 * weight, (0, 0), blur_sigma) / np.maximum(cv2.GaussianBlur(weight, (0, 0), blur_sigma), 1e-3)
    high = np.where(valid, e1 - low, 0)
    high_sigma = 1.4826 * np.median(np.abs(high[valid]))
    return offset, low, high, high_sigma


def edge_distance(gt_depth, valid):
    """Distance in pixels to the nearest depth discontinuity (>5% between neighbours in the GT)."""
    log_d = np.log(np.where(valid, gt_depth, 1))
    edges = np.zeros_like(valid)
    dx = np.abs(np.diff(log_d, axis=1)) > 0.05
    dy = np.abs(np.diff(log_d, axis=0)) > 0.05
    edges[:, 1:] |= dx & valid[:, 1:] & valid[:, :-1]
    edges[1:, :] |= dy & valid[1:, :] & valid[:-1, :]
    return cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3)


def pixels_in_large_regions(mask, min_size=20):
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    return sizes[sizes >= min_size].sum()


def radial_spectrum(x, weight, n_bins=64):
    """Radially averaged power spectrum of a masked error map, in cycles per pixel."""
    H, W = x.shape
    window = np.outer(np.hanning(H), np.hanning(W))
    power = np.abs(np.fft.rfft2(x * weight * window)) ** 2
    fy, fx = np.fft.fftfreq(H), np.fft.rfftfreq(W)
    freq = np.hypot(*np.meshgrid(fy, fx, indexing="ij"))
    bins = np.linspace(0, 0.5, n_bins + 1)
    idx = np.digitize(freq.ravel(), bins) - 1
    keep = idx < n_bins
    total = np.bincount(idx[keep], power.ravel()[keep], n_bins)
    return total / np.maximum(np.bincount(idx[keep], minlength=n_bins), 1) / (window**2 * weight).sum()


def depth_errors(data, rng):
    r = np.log(np.maximum(data["pred_depth"], 1e-6)) - np.log(np.where(data["valid"], data["gt_depth"], 1))
    valid = data["valid"] & np.isfinite(r) & (data["pred_depth"] > 0)
    scene_log_scale = np.median(r[valid])
    e = r - scene_log_scale

    N = len(e)
    fields = {k: [] for k in ["e", "e_frame", "high", "conf", "edge_dist", "log_gt", "frame", "outlier"]}
    per_frame = {k: np.full(N, np.nan) for k in ["offset", "high_sigma", "rms", "out_frac", "low_rms", "valid_frac"]}
    energy = np.zeros(4)  # frame offset, smooth, high-frequency inliers, outliers
    outlier_pixels = np.zeros(3)  # total, in connected regions >= 20 px, and the same for iid scatter of equal density
    inlier_hits = np.zeros(4)  # within 3% at: scene scale, + frame offset, + smooth part, + no outliers
    spectra = []
    n_valid = 0
    for t in range(N):
        v = valid[t]
        if v.sum() < 100:
            continue
        offset, low, high, sigma = split_depth_error(e[t], v)
        outlier = v & (np.abs(high) > DEPTH_OUTLIER_SIGMA * sigma)
        inlier = v & ~outlier

        energy += [v.sum() * offset**2, (low[v] ** 2).sum(), (high[inlier] ** 2).sum(), (high[outlier] ** 2).sum()]
        inlier_hits += [
            (np.abs(e[t][v]) < INLIER_THRESHOLD).sum(),
            (np.abs(e[t][v] - offset) < INLIER_THRESHOLD).sum(),
            (np.abs(high[v]) < INLIER_THRESHOLD).sum(),
            (np.abs(high[inlier]) < INLIER_THRESHOLD).sum() + outlier.sum(),
        ]
        n_valid += v.sum()
        scattered = v & (rng.random(v.shape) < outlier.sum() / v.sum())
        outlier_pixels += [outlier.sum(), pixels_in_large_regions(outlier), pixels_in_large_regions(scattered)]
        per_frame["offset"][t] = offset
        per_frame["high_sigma"][t] = sigma
        per_frame["rms"][t] = np.sqrt((e[t][v] ** 2).mean())
        per_frame["out_frac"][t] = outlier.sum() / v.sum()
        per_frame["low_rms"][t] = np.sqrt((low[v] ** 2).mean())
        per_frame["valid_frac"][t] = v.mean()

        if v.mean() > 0.5:
            e_clean = (e[t] - offset) * inlier
            spectra.append(radial_spectrum(e_clean, inlier.astype(np.float64)))

        ys, xs = np.nonzero(v)
        idx = rng.choice(len(ys), min(len(ys), PIXELS_PER_SCENE // N), replace=False)
        ys, xs = ys[idx], xs[idx]
        fields["e"].append(e[t][ys, xs])
        fields["e_frame"].append(e[t][ys, xs] - offset)
        fields["high"].append(high[ys, xs])
        fields["conf"].append(data["conf"][t][ys, xs])
        fields["edge_dist"].append(edge_distance(data["gt_depth"][t], v)[ys, xs])
        fields["log_gt"].append(np.log(data["gt_depth"][t][ys, xs]))
        fields["frame"].append(np.full(len(ys), t))
        fields["outlier"].append(outlier[ys, xs])

    return {
        "scene_log_scale": scene_log_scale,
        "energy": energy,
        "inlier_ratio_steps": inlier_hits / max(n_valid, 1),
        "outlier_pixels": outlier_pixels,
        "spectrum": np.mean(spectra, axis=0) if spectra else np.full(64, np.nan),
        "median_gt_depth": np.median(data["gt_depth"][valid]),
        **{f"frame_{k}": v for k, v in per_frame.items()},
        **{f"px_{k}": np.concatenate(v).astype(np.float32 if k not in ("frame", "outlier") else np.int16) for k, v in fields.items()},
    }


# -------------------- Driver -------------------- #


def process_scene(args):
    model, dataset, scene, out_path, src = args
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    data = load_scene(model, dataset, scene, src)
    rng = np.random.default_rng(0)
    depth = depth_errors(data, rng)
    pose = pose_errors(data["pred_w2c"], data["gt_w2c"], depth["median_gt_depth"], data["bench_order"])
    # A consistent prediction has one scale: exp(-scene_log_scale) from depth, sim3_scale from poses
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **pose, **depth)
    return dataset, scene


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--src", help="directory with model_results/ to analyse instead of workspace/evaluation/<model> (e.g. a TTO run)")
    parser.add_argument("--name", help="output folder under workspace/failure_modes/ (default: the model)")
    args = parser.parse_args()
    src = os.path.abspath(args.src) if args.src else os.path.join(REPO_ROOT, "workspace/evaluation", args.model)
    name_out = args.name or args.model

    tasks = []
    for dataset in args.datasets:
        root = os.path.join(src, "model_results", dataset)
        for result in sorted(glob.glob(os.path.join(root, "**/unposed/exports/mini_npz/results.npz"), recursive=True)):
            scene = os.path.relpath(result, root).split("/unposed/")[0]
            name = scene.replace("/", "_") + ".npz"
            tasks.append((args.model, dataset, scene, os.path.join(REPO_ROOT, "workspace/failure_modes", name_out, dataset, name), src))
    if not tasks:
        sys.exit(f"No predictions found for {args.model}; run scripts/run_bench.py first")

    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for i, (dataset, scene) in enumerate(pool.imap_unordered(process_scene, tasks), 1):
            print(f"[{i}/{len(tasks)}] {dataset}/{scene}", flush=True)


if __name__ == "__main__":
    main()
