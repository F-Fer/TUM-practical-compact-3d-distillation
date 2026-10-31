"""Re-run DA3 conditioned on refined cameras (DA3's posed mode) and keep the refined cameras.

Usage: .venv/bin/python scripts/run_tto_recond.py <model> <src work dir> <dst work dir> [datasets=a,b] [max_scenes=n]

For every scene in <src>/model_results (e.g. a PBA output), feeds its extrinsics and intrinsics (converted from
the processing resolution to the original images) to DA3 together with the images, exactly like the benchmark's
posed mode does with ground-truth cameras, and writes <dst>/model_results/... with DA3's new depth and confidence
(aligned to the input cameras' scale) and the input cameras. The pose metrics are those of <src> (copied).
"""

import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
import da3_lowmem  # noqa: E402
from depth_anything_3.api import DepthAnything3  # noqa: E402
from depth_anything_3.utils.constants import EVAL_REF_VIEW_STRATEGY  # noqa: E402


def main():
    model_name, src, dst = sys.argv[1:4]
    opts = dict(a.split("=", 1) for a in sys.argv[4:])
    os.chdir(REPO_ROOT)
    da3_lowmem.install()
    api = DepthAnything3.from_pretrained(f"depth-anything/{model_name}").to("cuda").eval()
    results_dir = os.path.join(src, "model_results")
    datasets = [d for d in opts.get("datasets", "").split(",") if d] or sorted(os.listdir(results_dir))
    for dataset in datasets:
        data_dir = os.path.join(results_dir, dataset)
        if not os.path.isdir(data_dir):
            continue
        stats = {}
        scenes = sorted(os.path.dirname(p) for p, _, _ in os.walk(data_dir) if os.path.basename(p) == "unposed")
        for scene_dir in scenes[: int(opts.get("max_scenes", 0)) or None]:
            scene = os.path.relpath(scene_dir, data_dir)
            exports = os.path.join(scene_dir, "unposed", "exports")
            gt = np.load(os.path.join(exports, "gt_meta.npz"), allow_pickle=True)
            pred = np.load(os.path.join(exports, "mini_npz", "results.npz"))
            N, H, W = pred["depth"].shape
            h0, w0 = cv2.imread(str(gt["image_files"][0])).shape[:2]
            K = pred["intrinsics"].astype(np.float64).copy()
            K[:, 0] *= w0 / W  # DA3 only resizes (upper_bound_resize): fx, cx scale with the width, fy, cy with the height
            K[:, 1] *= h0 / H
            ext = np.tile(np.eye(4), (N, 1, 1))
            ext[:, :3] = pred["extrinsics"]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = api.inference(list(gt["image_files"]), ext.astype(np.float32), K.astype(np.float32), ref_view_strategy=EVAL_REF_VIEW_STRATEGY)
            torch.cuda.synchronize()
            fwd_s = time.perf_counter() - t0
            assert out.depth.shape == pred["depth"].shape, (out.depth.shape, pred["depth"].shape)
            k_err = float(np.abs(out.intrinsics[:, 0, 0] / pred["intrinsics"][:, 0, 0] - 1).max())
            dst_exports = os.path.join(dst, "model_results", dataset, scene, "unposed", "exports")
            os.makedirs(os.path.join(dst_exports, "mini_npz"), exist_ok=True)
            np.savez(
                os.path.join(dst_exports, "mini_npz", "results.npz"),
                depth=out.depth.astype(np.float32), conf=out.conf.astype(np.float32),
                extrinsics=pred["extrinsics"].astype(np.float32), intrinsics=pred["intrinsics"].astype(np.float32),
            )
            shutil.copy(os.path.join(exports, "gt_meta.npz"), os.path.join(dst_exports, "gt_meta.npz"))
            ratio = float(np.median(out.depth / np.maximum(pred["depth"], 1e-6)))
            stats[scene] = {"forward_s": fwd_s, "depth_ratio_median": ratio, "focal_roundtrip_err": k_err}
            print(f"[run_tto_recond] {dataset} {scene}: forward {fwd_s:.2f}s depth ratio {ratio:.3f} focal roundtrip {k_err:.1e}")
        os.makedirs(os.path.join(dst, "metric_results"), exist_ok=True)
        with open(os.path.join(dst, "metric_results", f"{dataset}_recond.json"), "w") as f:
            json.dump(stats, f, indent=1)
        # cameras are unchanged: keep the source's pose metrics (PBA stats included)
        for name in (f"{dataset}_pba.json", f"{dataset}_pose.json"):
            p = os.path.join(src, "metric_results", name)
            if os.path.exists(p):
                shutil.copy(p, os.path.join(dst, "metric_results", name))


if __name__ == "__main__":
    main()
