"""Before/after figure of a TTO run: a neighbour frame warped into a reference frame with each set of predictions.

Usage: .venv/bin/python scripts/vis_pba.py <out.png> <dataset> <scene> <work dir> [<work dir> ...] [--frame i] [--nbr j]

Row per work dir: reference image | neighbour warped into it | |difference| (black where not visible).
The first work dir is usually the unrefined predictions (e.g. workspace/tto/DA3-SMALL/init_dev).
"""

import os
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def warp(img_j, depth_i, ext_i, ext_j, K_i, K_j, depth_j, occ=0.05):
    """Backward-warp img_j into frame i with frame i's depth; returns warped image and visibility mask."""
    H, W = depth_i.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    x = (u - K_i[0, 2]) / K_i[0, 0] * depth_i
    y = (v - K_i[1, 2]) / K_i[1, 1] * depth_i
    Xc = np.stack([x, y, depth_i], -1).reshape(-1, 3)
    Xw = (Xc - ext_i[:, 3]) @ ext_i[:, :3]  # R^T (X - t)
    Xj = Xw @ ext_j[:, :3].T + ext_j[:, 3]
    z = np.maximum(Xj[:, 2], 1e-6)
    uj = (K_j[0, 0] * Xj[:, 0] / z + K_j[0, 2]).reshape(H, W).astype(np.float32)
    vj = (K_j[1, 1] * Xj[:, 1] / z + K_j[1, 2]).reshape(H, W).astype(np.float32)
    warped = cv2.remap(img_j, uj, vj, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    dj = cv2.remap(depth_j, uj, vj, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    vis = (Xj[:, 2].reshape(H, W) > 0) & (uj >= 0) & (uj <= W - 1) & (vj >= 0) & (vj <= H - 1)
    vis &= z.reshape(H, W) < dj * (1 + occ)
    return warped, vis


def main():
    args = sys.argv[1:]
    opts = {}
    for key in ("--frame", "--nbr"):
        if key in args:
            i = args.index(key)
            opts[key] = int(args[i + 1])
            del args[i : i + 2]
    out, dataset, scene, runs = args[0], args[1], args[2], args[3:]
    os.chdir(REPO_ROOT)
    rows = []
    for r in runs:
        exports = os.path.join(r, "model_results", dataset, scene, "unposed", "exports")
        p = np.load(os.path.join(exports, "mini_npz", "results.npz"))
        files = np.load(os.path.join(exports, "gt_meta.npz"), allow_pickle=True)["image_files"]
        N, H, W = p["depth"].shape
        i = opts.get("--frame", 0)
        if "--nbr" not in opts:  # the most covisible frame under the first run's predictions
            vis_frac = [warp(np.zeros((H, W), np.float32), p["depth"][i], p["extrinsics"][i], p["extrinsics"][k], p["intrinsics"][i], p["intrinsics"][k], p["depth"][k])[1].mean() if k != i else -1 for k in range(N)]
            opts["--nbr"] = int(np.argmax(vis_frac))
        j = opts["--nbr"]
        imgs = [cv2.resize(cv2.imread(str(files[k])), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 for k in (i, j)]
        warped, vis = warp(imgs[1], p["depth"][i], p["extrinsics"][i], p["extrinsics"][j], p["intrinsics"][i], p["intrinsics"][j], p["depth"][j])
        diff = np.abs(warped - imgs[0]).mean(-1) * vis
        err = diff.sum() / max(vis.sum(), 1)
        diff_vis = cv2.applyColorMap(np.clip(diff / 0.25 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO).astype(np.float32) / 255
        row = np.concatenate([imgs[0], warped * vis[..., None], diff_vis], 1)
        label = f"{os.path.basename(r.rstrip('/'))}  frame {i}<-{j}  mean |diff| {err:.4f}  visible {vis.mean():.2f}"
        cv2.putText(row, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (1, 1, 1), 2)
        rows.append(row)
    cv2.imwrite(out, (np.concatenate(rows, 0) * 255).astype(np.uint8))
    print(out)


if __name__ == "__main__":
    main()
