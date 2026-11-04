"""Reconstruction F1 after aligning the fused cloud to the ground truth (similarity ICP), next to the benchmark's own F1.

The benchmark (recon_unposed) fits the scale and the frame from the camera poses only. A prediction whose depth scale
disagrees with its pose scale comes out too small or too large and scores F1 ~ 0 although its shape is right. This
aligns the fused cloud itself to the GT first (point-to-point ICP with scale, from the pose-based start, coarse to fine),
then scores it with each dataset's own metric (threshold, cropping and GT loading unchanged; for ScanNet++, whose eval3d computes the metric inline, the cloud is aligned to samples of the GT mesh and scored as a file pcd_aligned.ply).

Usage (from anywhere; paths are resolved relative to the repo root):
    uv run scripts/eval_recon_aligned.py <work dir> [datasets]

    uv run scripts/eval_recon_aligned.py workspace/tto/DA3-SMALL/roma_turbo_q4 hiroom,7scenes,scannetpp

Reads <work dir>/model_results/<dataset>/<scene>/unposed/exports/fuse/pcd.ply (written by eval_tto.py, so score the run
first; scenes without it are skipped). Writes only <work dir>/metric_results/<dataset>_recon_aligned.json: the metrics
after alignment, plus fscore_raw (without alignment), align_scale and align_shift (m) per scene. DTU is not covered.
"""

import functools
import json
import os
import sys

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

# same speed-up as run_bench.py (identical distances)
KDTree.query = functools.partialmethod(KDTree.query, workers=-1)
os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
from depth_anything_3.bench import utils  # noqa: E402
from depth_anything_3.bench.evaluator import Evaluator  # noqa: E402
from eval_tto import scenes_in  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = "hiroom,7scenes,scannetpp,eth3d"
_original = utils.evaluate_3d_reconstruction
_last = {}  # the alignment of the scene being scored


def align(pred, gt):
    """Similarity transform (4x4, scale included) that takes pred onto gt."""
    diag = np.linalg.norm(gt.get_max_bound() - gt.get_min_bound())
    voxel = diag / 200
    p, g = pred.voxel_down_sample(voxel), gt.voxel_down_sample(voxel)
    T = np.eye(4)
    for frac in (0.08, 0.04, 0.02, 0.01):
        res = o3d.pipelines.registration.registration_icp(
            p, g, frac * diag, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        p = p.transform(res.transformation)
        T = res.transformation @ T
    return T


def evaluate_aligned(pcd_pred, pcd_trgt, threshold=0.05, down_sample=None):
    raw = _original(pcd_pred, pcd_trgt, threshold=threshold, down_sample=down_sample)
    if len(pcd_pred.points) == 0 or len(pcd_trgt.points) == 0:
        return raw
    T = align(pcd_pred, pcd_trgt)
    aligned = o3d.geometry.PointCloud(pcd_pred).transform(T)
    _last.update(align_scale=float(np.cbrt(abs(np.linalg.det(T[:3, :3])))), align_shift=float(np.linalg.norm(T[:3, 3])), fscore_raw=raw["fscore"])
    return _original(aligned, pcd_trgt, threshold=threshold, down_sample=down_sample)


def score_aligned_file(ds, scene, ply):
    """For datasets whose eval3d computes the metric inline (ScanNet++): align the cloud file to the GT mesh samples, score the aligned copy."""
    from depth_anything_3.bench.utils import sample_points_from_mesh

    raw = ds.eval3d(scene, ply)
    gt = sample_points_from_mesh(o3d.io.read_triangle_mesh(ds.get_data(scene).aux.gt_mesh_path), ds.sampling_number)
    pred = o3d.io.read_point_cloud(ply)
    T = align(pred, gt)
    out = ply.replace("pcd.ply", "pcd_aligned.ply")  # a new file next to the original
    o3d.io.write_point_cloud(out, pred.transform(T))
    _last.update(align_scale=float(np.cbrt(abs(np.linalg.det(T[:3, :3])))), align_shift=float(np.linalg.norm(T[:3, 3])), fscore_raw=raw["fscore"])
    return ds.eval3d(scene, out)


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.exit(__doc__)
    work_dir = sys.argv[1]
    datasets = (sys.argv[2] if len(sys.argv) > 2 else DATASETS).split(",")
    sys.stdout.reconfigure(line_buffering=True)
    os.chdir(REPO_ROOT)

    for dataset in datasets:
        scenes = scenes_in(work_dir, dataset)
        ev = Evaluator(work_dir=work_dir, datas=[dataset], modes=["recon_unposed"], scenes=scenes)
        # the dataset modules are imported by now, and each imported the metric by name
        for name, module in list(sys.modules.items()):
            if name.startswith("depth_anything_3.bench.datasets.") and hasattr(module, "evaluate_3d_reconstruction"):
                module.evaluate_3d_reconstruction = evaluate_aligned
        out_path = os.path.join(work_dir, "metric_results", f"{dataset}_recon_aligned.json")
        results = {}
        for scene in scenes:
            ply = os.path.join(work_dir, "model_results", dataset, scene, "unposed", "exports", "fuse", "pcd.ply")
            if not os.path.exists(ply):
                continue
            _last.clear()
            ds = ev.datasets[dataset]
            metrics = score_aligned_file(ds, scene, ply) if dataset == "scannetpp" else ds.eval3d(scene, ply)
            results[scene] = Evaluator._to_float_dict({**metrics, **_last})
            print(f"[eval_recon_aligned] {dataset} {scene}: fscore {metrics['fscore']:.3f} (raw {_last.get('fscore_raw', float('nan')):.3f}), scale {_last.get('align_scale', float('nan')):.3f}")
        if results:
            Evaluator._dump_json(out_path, {**results, "mean": Evaluator._mean_of_dicts(results.values())})


if __name__ == "__main__":
    main()
