"""One stage of an alternating depth / pose refinement (reports/Learned refinement step between DA3 and DeepBA.md, §11).

    .venv/bin/python scripts/lpba_stage.py <state dirs, comma-separated> <dst work dir> [run_tto_pba overrides]

Runs scripts/run_tto_pba.py on DA3-SMALL's scenes (src workspace/evaluation/DA3-SMALL, for the GT and the confidence),
but each scene starts from a saved state instead of DA3's prediction: depth, extrinsics and intrinsics from the
results.npz of an earlier run (the first state dir that has the scene; e.g. the dev and held-out B′ runs). The overrides
choose the stage, e.g.

    depth stage:  poses=false focal=none depth_grid=32 levels=2,1 lm_levels= samples=4096
    pose stage:   depth_grid=0 levels=2,1,1 lm_levels=2

Per scene it also prints the depth error against the GT (abs. rel. after one scale per scene, where GT depth exists, see
lpba_depth_oracle.py) of the input and the output depth.
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_tto_pba as rp  # noqa: E402
from lpba_depth_oracle import gt_depth  # noqa: E402

SRC = "workspace/evaluation/DA3-SMALL"


def depth_err(depth, image_files):
    try:
        N, H, W = depth.shape
        gt = np.stack([gt_depth(f, H, W) for f in image_files])
    except (ValueError, AttributeError, OSError):
        return float("nan")
    v = gt > 0
    if v.sum() < 100:
        return float("nan")
    s = np.median(depth[v]) / np.median(gt[v])
    return float(np.mean(np.abs(depth[v] / s - gt[v]) / gt[v]))


def main():
    states, dst, overrides = sys.argv[1].split(","), sys.argv[2], sys.argv[3:]
    os.chdir(rp.REPO_ROOT)
    index = {}
    for st in states:
        for meta in glob.glob(os.path.join(st, "model_results", "**", "gt_meta.npz"), recursive=True):
            first = str(np.load(meta, allow_pickle=True)["image_files"][0])
            index.setdefault(first, os.path.join(os.path.dirname(meta), "mini_npz", "results.npz"))
    refine = rp.refine_scene

    def patched(image_files, pred, cfg, device):
        st = dict(np.load(index[str(image_files[0])]))
        pred = {**pred, "depth": st["depth"].astype(np.float32), "extrinsics": st["extrinsics"][:, :3].astype(np.float32),
                "intrinsics": st["intrinsics"].astype(np.float32)}
        out, info, hist = refine(image_files, pred, cfg, device)
        info["depth_err_in"] = depth_err(pred["depth"], image_files)
        info["depth_err_out"] = depth_err(out["depth"], image_files)
        print(f"[stage] depth abs.rel. {info['depth_err_in']:.4f} -> {info['depth_err_out']:.4f}")
        return out, info, hist

    rp.refine_scene = patched
    sys.argv = [rp.__file__, SRC, dst] + overrides
    rp.main()


if __name__ == "__main__":
    main()
