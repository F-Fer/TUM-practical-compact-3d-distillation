"""Refine saved DA3 predictions with bundle adjustment (VGGT tracks + pycolmap).

Usage (from anywhere; paths are resolved relative to the repo root):
    uv run scripts/run_tto_ba.py <src work dir> <dst work dir> [overrides ...]

    uv run scripts/run_tto_ba.py workspace/evaluation/DA3-SMALL workspace/evaluation/DA3-SMALL-ba datasets=hiroom

Outputs go to <dst work dir>, mirroring <src work dir> with only the new data:
    model_results/    refined extrinsics and intrinsics per scene (mini_npz/results.npz)
    metric_results/   <dataset>_pose.json as written by the DA3 evaluator, with the timings added per scene
"""

import contextlib
import io
import json
import os
import sys
import time

# Fragmentation of the CUDA allocator wastes memory on the 8-10 GB GPUs; must be set before CUDA init
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import pycolmap
import torch
from depth_anything_3.bench.evaluator import Evaluator
from depth_anything_3.bench.utils import compute_pose
from depth_anything_3.utils.geometry import as_homogeneous
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap
from vggt.dependency.track_predict import predict_tracks
from vggt.utils.geometry import unproject_depth_map_to_point_map

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = {
    "datasets": "",  # comma-separated, default: all in <src>/model_results
    "max_scenes": 0,  # per dataset, 0 = all
    "query_frames": 8,
    "max_query_pts": 4096,
    "max_points": 163840,  # tracker chunk size, lower it if memory runs out
    "fine_tracking": True,
    "max_reproj": 8.0,  # px at 1024
    "min_inliers": 64,  # per frame, BA is skipped below this
    "vis_thresh": 0.2,
    "shared_camera": False,
    "seed": 0,
}

# pycolmap's BA and reconstruction logs are very verbose
pycolmap.logging.minloglevel = 2


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


def track(image_files, pred, cfg, device):
    """Track keypoints through the scene, initialised with DA3's depth and poses."""
    depth, conf, ext, K = pred["depth"], pred["conf"], pred["extrinsics"], pred["intrinsics"]
    N, H, W = depth.shape
    L = max(H, W)  # the tracker needs square images

    imgs = np.stack([cv2.resize(cv2.imread(str(f))[..., ::-1], (W, H), interpolation=cv2.INTER_AREA) for f in image_files])
    images = torch.zeros(N, 3, L, L)
    images[:, :, :H, :W] = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255

    points = unproject_depth_map_to_point_map(depth[..., None], ext, K)
    conf_sq = np.ones((N, L, L), np.float32);       conf_sq[:, :H, :W] = conf
    points_sq = np.zeros((N, L, L, 3), np.float32); points_sq[:, :H, :W] = points

    # the tracker prints per query frame
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        tracks, vis, _, points3d, colors = predict_tracks(
            images.to(device), conf=conf_sq, points_3d=points_sq, keypoint_extractor="aliked+sp",
            max_query_pts=cfg["max_query_pts"], query_frame_num=min(cfg["query_frames"], N),
            max_points_num=cfg["max_points"], fine_tracking=cfg["fine_tracking"])
    return tracks, vis, points3d, colors, L


def adjust(pred, tracks, vis, points3d, colors, L, cfg):
    """Bundle-adjust cameras and points on the tracks: (extrinsics, intrinsics, #points), None if too few inliers."""
    N = len(tracks)
    rec, _ = batch_np_matrix_to_pycolmap(
        points3d, pred["extrinsics"], pred["intrinsics"], tracks, np.array([L, L]), masks=vis > cfg["vis_thresh"],
        max_reproj_error=cfg["max_reproj"] * L / 1024, min_inlier_per_frame=cfg["min_inliers"],
        shared_camera=cfg["shared_camera"], camera_type="SIMPLE_PINHOLE", points_rgb=colors)
    if rec is None:
        return None
    opts = pycolmap.BundleAdjustmentOptions()
    opts.print_summary = False
    pycolmap.bundle_adjustment(rec, opts)

    ext = np.stack([rec.images[i + 1].cam_from_world.matrix() for i in range(N)]).astype(np.float32)
    cams = [rec.cameras[rec.images[i + 1].camera_id] for i in range(N)]
    K = np.stack([[[c.params[0], 0, c.params[1]], [0, c.params[0], c.params[2]], [0, 0, 1]] for c in cams]).astype(np.float32)
    return ext, K, rec.num_points3D()


def refine_scene(image_files, pred, cfg, device):
    """Track and adjust one scene; falls back to DA3's poses if BA is skipped."""
    # the query keypoints are subsampled at random
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # same keys for every scene, so that the evaluator's mean works
    info = {"num_frames": len(pred["depth"]), "num_tracks": 0, "num_points": 0, "track_s": 0.0, "ba_s": 0.0, "peak_mem_gib": 0.0, "ba": 0}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    tracks, vis, points3d, colors, L = track(image_files, pred, cfg, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        info["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    info.update(num_tracks=int(tracks.shape[1]), track_s=time.perf_counter() - start)

    start = time.perf_counter()
    result = adjust(pred, tracks, vis, points3d, colors, L, cfg)
    if result is None:
        return pred["extrinsics"], pred["intrinsics"], info
    ext, K, num_points = result
    info.update(num_points=num_points, ba_s=time.perf_counter() - start, ba=1)
    return ext, K, info


def main():
    if len(sys.argv) < 3 or sys.argv[1].startswith("-"):
        sys.exit(__doc__)
    src, dst, cfg = sys.argv[1], sys.argv[2], parse_overrides(sys.argv[3:])
    sys.stdout.reconfigure(line_buffering=True)  # per-scene lines show up in redirected logs
    os.chdir(REPO_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    results_dir = os.path.join(src, "model_results")
    datasets = [d for d in cfg["datasets"].split(",") if d] or sorted(os.listdir(results_dir))

    for dataset in datasets:
        data_dir = os.path.join(results_dir, dataset)
        metric_path = os.path.join(dst, "metric_results", f"{dataset}_pose.json")
        metrics = json.load(open(metric_path)) if os.path.exists(metric_path) else {}
        metrics.pop("mean", None)  # recomputed after every scene, so an interrupted run can be resumed

        scenes = sorted(os.path.dirname(p) for p, _, _ in os.walk(data_dir) if os.path.basename(p) == "unposed")
        for i, scene_dir in enumerate(scenes[: cfg["max_scenes"] or None]):
            scene = os.path.relpath(scene_dir, data_dir)
            if scene in metrics:
                continue
            exports = os.path.join(scene_dir, "unposed", "exports")
            gt = dict(np.load(os.path.join(exports, "gt_meta.npz"), allow_pickle=True))
            pred = dict(np.load(os.path.join(exports, "mini_npz", "results.npz")))

            try:
                ext, K, info = refine_scene(gt["image_files"], pred, cfg, device)
            except Exception as e:  # not stored, so a rerun retries it
                print(f"[run_tto_ba] {dataset} {i + 1}/{len(scenes)} {scene} FAILED: {e!r}")
                torch.cuda.empty_cache()
                continue

            out = os.path.join(dst, "model_results", dataset, scene, "unposed", "exports", "mini_npz")
            os.makedirs(out, exist_ok=True)
            np.savez(os.path.join(out, "results.npz"), extrinsics=ext, intrinsics=K)

            # same pose metric and json format as the evaluator
            pose = compute_pose(torch.from_numpy(as_homogeneous(ext)), torch.from_numpy(as_homogeneous(gt["extrinsics"])))
            metrics[scene] = Evaluator._to_float_dict({**pose, **info})
            Evaluator._dump_json(metric_path, {**metrics, "mean": Evaluator._mean_of_dicts(metrics.values())})
            print(f"[run_tto_ba] {dataset} {i + 1}/{len(scenes)} {scene}: {metrics[scene]}")


if __name__ == "__main__":
    main()
