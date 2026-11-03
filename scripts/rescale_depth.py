"""Rescale the depth saved by run_tto_ba.py runs from before the scale fix, so that it matches the refined cameras.

Usage (from anywhere; paths are resolved relative to the repo root):
    uv run scripts/rescale_depth.py <src work dir> <dst work dir> [<dst work dir> ...]

    uv run scripts/rescale_depth.py workspace/evaluation/DA3-SMALL workspace/tto/DA3-SMALL-roma-turbo-q4

BA fixes the scale of the scene freely, so its cameras can be a different size than DA3's, while the saved depth is still
DA3's. The evaluator takes the scale from the poses and applies it to the depth, so the fused clouds are too large (and
the TSDF fusion can use all of the RAM). Scenes whose depth already differs from <src>'s are left alone.
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def centres(ext):
    return -np.einsum("nji,nj->ni", ext[:, :3, :3], ext[:, :3, 3])


def main():
    if len(sys.argv) < 3 or sys.argv[1].startswith("-"):
        sys.exit(__doc__)
    os.chdir(REPO_ROOT)
    src = sys.argv[1]
    for dst in sys.argv[2:]:
        done = skipped = 0
        for root, _, files in os.walk(os.path.join(dst, "model_results")):
            if "results.npz" not in files:
                continue
            path = os.path.join(root, "results.npz")
            ref = np.load(os.path.join(src, os.path.relpath(path, dst)))
            new = dict(np.load(path))
            if not np.array_equal(new["depth"], ref["depth"]):
                skipped += 1
                continue
            old_c, new_c = centres(ref["extrinsics"]), centres(new["extrinsics"])
            scale = np.linalg.norm(new_c - new_c.mean(0)) / max(np.linalg.norm(old_c - old_c.mean(0)), 1e-9)
            new["depth"] = (new["depth"] * scale).astype(new["depth"].dtype)
            np.savez(path, **new)
            done += 1
        print(f"{dst}: rescaled {done} scenes, left {skipped} alone")


if __name__ == "__main__":
    main()
