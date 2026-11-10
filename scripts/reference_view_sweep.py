"""Rerun DA3 on one scene with different reference views and record which frames fail, plus each frame's GT overlap
with the reference view.

For each chosen reference r, the frames are fed as [r, all others in sorted order] with ref_view_strategy="first".
Writes workspace/failure_modes/<model>/ref_sweep/<dataset>/<scene>.npz.

    uv run scripts/reference_view_sweep.py DA3-SMALL scannetpp 5f99900f09 --n-refs 25
"""

import argparse
import os
import time

os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import analyse_failure_modes as analysis  # noqa: E402
import da3_lowmem  # noqa: E402
import reconstruction_errors as recon  # noqa: E402

GROSS_ROT_DEG, GROSS_TRANS = 15.0, 0.25  # same thresholds as plot_failure_modes.gross_frames


def gt_intrinsics(model, dataset, scene, data):
    export = os.path.join(analysis.REPO_ROOT, "workspace/evaluation", model, "model_results", dataset, scene, "unposed/exports")
    K_raw = np.load(os.path.join(export, "gt_meta.npz"), allow_pickle=True)["intrinsics"]
    order = np.argsort(data["bench_order"])
    H, W = data["gt_depth"].shape[1:]
    image_hw = cv2.imread(data["image_files"][0]).shape[:2]
    return np.stack([recon.scale_K(k, image_hw, (H, W)) for k in K_raw[order]])


def overlap_matrix(data, K, stride=6, tol=0.05):
    """O[i, j]: share of frame i's valid pixels that frame j sees too (projects inside j and agrees with j's GT depth)."""
    depth, valid = data["gt_depth"], data["valid"]
    c2w = np.linalg.inv(data["gt_w2c"])
    w2c = data["gt_w2c"]
    N, H, W = depth.shape
    O = np.zeros((N, N))
    for i in range(N):
        ys, xs = np.nonzero(valid[i][::stride, ::stride])
        ys, xs = ys * stride, xs * stride
        if len(ys) == 0:
            continue
        p = recon.backproject(depth[i], K[i], c2w[i, :3, :3], c2w[i, :3, 3], ys, xs)
        for j in range(N):
            q = p @ w2c[j, :3, :3].T + w2c[j, :3, 3]
            z = q[:, 2]
            front = z > 1e-3
            u = np.full(len(z), -1, np.int64)
            v = np.full(len(z), -1, np.int64)
            u[front] = np.round(q[front, 0] / z[front] * K[j][0, 0] + K[j][0, 2]).astype(np.int64)
            v[front] = np.round(q[front, 1] / z[front] * K[j][1, 1] + K[j][1, 2]).astype(np.int64)
            inside = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            seen = np.zeros(len(z), bool)
            uu, vv = u[inside], v[inside]
            ok = valid[j][vv, uu] & (np.abs(depth[j][vv, uu] - z[inside]) < tol * z[inside])
            seen[np.nonzero(inside)[0][ok]] = True
            O[i, j] = seen.mean()
    return O


def frame_errors(pred_w2c, gt_w2c, depth_scale, ref):
    """Per-frame errors after the robust alignment of the failure-mode analysis, and relative to the reference view."""
    pred_c2w, gt_c2w = np.linalg.inv(pred_w2c), np.linalg.inv(gt_w2c)
    gauge = analysis.robust_rotation_alignment(pred_c2w[:, :3, :3], gt_c2w[:, :3, :3])
    sim3 = analysis.robust_scale_translation(pred_c2w[:, :3, 3], gt_c2w[:, :3, 3], gauge)
    c = analysis.apply_sim3(sim3, pred_c2w[:, :3, 3])
    rot = np.degrees(Rotation.from_matrix(gauge @ pred_c2w[:, :3, :3] @ gt_c2w[:, :3, :3].transpose(0, 2, 1)).magnitude())
    trans = np.linalg.norm(c - gt_c2w[:, :3, 3], axis=1) / depth_scale
    # Rotation relative to the reference: no alignment involved
    rel_pred = pred_c2w[ref, :3, :3].T @ pred_c2w[:, :3, :3]
    rel_gt = gt_c2w[ref, :3, :3].T @ gt_c2w[:, :3, :3]
    rot_rel = np.degrees(Rotation.from_matrix(rel_pred @ rel_gt.transpose(0, 2, 1)).magnitude())
    return rot, trans, rot_rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("dataset")
    ap.add_argument("scene")
    ap.add_argument("--n-refs", type=int, default=25)
    args = ap.parse_args()

    os.chdir(analysis.REPO_ROOT)
    data = analysis.load_scene(args.model, args.dataset, args.scene)
    N = len(data["image_files"])
    K = gt_intrinsics(args.model, args.dataset, args.scene, data)
    t0 = time.time()
    O = overlap_matrix(data, K)
    print(f"overlap matrix {time.time() - t0:.0f} s", flush=True)
    depth_scale = float(np.median(data["gt_depth"][data["valid"]]))

    from depth_anything_3.api import DepthAnything3

    da3_lowmem.install()
    api = DepthAnything3.from_pretrained(f"depth-anything/{args.model}").to("cuda")
    refs = np.unique(np.linspace(0, N - 1, args.n_refs).round().astype(int))
    rot, trans, rot_rel = (np.zeros((len(refs), N)) for _ in range(3))
    for k, r in enumerate(refs):
        t0 = time.time()
        perm = np.concatenate([[r], np.delete(np.arange(N), r)])
        pred = api.inference([data["image_files"][i] for i in perm], ref_view_strategy="first")
        w2c = np.tile(np.eye(4), (N, 1, 1))
        w2c[perm, :3] = pred.extrinsics
        rot[k], trans[k], rot_rel[k] = frame_errors(w2c, data["gt_w2c"], depth_scale, r)
        gross = (rot[k] > GROSS_ROT_DEG) | (trans[k] > GROSS_TRANS)
        print(f"ref {r:3d}: {gross.sum():3d} failing frames, {time.time() - t0:.1f} s", flush=True)
        torch.cuda.empty_cache()

    out = os.path.join(analysis.REPO_ROOT, "workspace/failure_modes", args.model, "ref_sweep", args.dataset, args.scene + ".npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, refs=refs, rot=rot, trans=trans, rot_rel=rot_rel, overlap=O, c2w_gt=np.linalg.inv(data["gt_w2c"]),
             depth_scale=depth_scale, gross_rot=GROSS_ROT_DEG, gross_trans=GROSS_TRANS)
    print(out)


if __name__ == "__main__":
    main()
