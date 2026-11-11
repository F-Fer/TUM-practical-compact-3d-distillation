"""One picture of the main pose finding: neighbouring cameras are wrong in the same direction (smooth drift, not jitter).

Shows a short stretch of consecutive cameras of one scene: GT cameras in black, DA3's (after aligning the whole trajectory to
the GT) in blue, and an arrow from each true camera to DA3's, over the faded GT reconstruction.

    uv run scripts/plot_drift_example.py DA3-SMALL 7scenes office --frames 44 50 57 62
"""

import argparse
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import analyse_failure_modes as analysis  # noqa: E402
import reconstruction_errors as recon  # noqa: E402

GT_COLOR, PRED_COLOR, ARROW_COLOR, INK, INK2 = "#111111", "#2a78d6", "#d1342f", "#111111", "#52514e"


def gt_cloud(data, K, R, c, stride):
    H, W = data["pred_depth"].shape[1:]
    pts, rgb, rays = [], [], []
    for t in range(len(R)):
        ys, xs = np.nonzero(data["valid"][t][::stride, ::stride])
        ys, xs = ys * stride, xs * stride
        p = recon.backproject(data["gt_depth"][t], K[t], R[t], c[t], ys, xs)
        img = cv2.resize(cv2.imread(data["image_files"][t]), (W, H), interpolation=cv2.INTER_AREA)
        pts.append(p)
        rgb.append(img[ys, xs, ::-1] / 255.0)
        rays.append(p - c[t])
    rays = np.concatenate(rays)
    return np.concatenate(pts), np.concatenate(rgb), rays / np.linalg.norm(rays, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="DA3-SMALL")
    ap.add_argument("dataset", nargs="?", default="7scenes")
    ap.add_argument("scene", nargs="?", default="office")
    ap.add_argument("--frames", type=int, nargs="+", default=[44, 50, 57, 62], help="frames to show (1-based, in frame order)")
    ap.add_argument("--elev", type=float, default=45.0)
    ap.add_argument("--frustum", type=float, default=0.07, help="frustum depth in m")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = analysis.load_scene(args.model, args.dataset, args.scene)
    saved = dict(np.load(os.path.join(analysis.REPO_ROOT, "workspace/failure_modes", args.model, args.dataset, args.scene + ".npz")))
    cams, _ = recon.camera_variants(data, saved)
    R_gt, c_gt = cams["gt"]
    R_pred, c_pred = cams["pred"]

    export = os.path.join(analysis.REPO_ROOT, "workspace/evaluation", args.model, "model_results", args.dataset, args.scene, "unposed/exports")
    gt_K_raw = np.load(os.path.join(export, "gt_meta.npz"), allow_pickle=True)["intrinsics"]
    order = np.argsort(data["bench_order"])
    H, W = data["pred_depth"].shape[1:]
    image_hw = cv2.imread(data["image_files"][0]).shape[:2]
    K = np.stack([recon.scale_K(k, image_hw, (H, W)) for k in gt_K_raw[order]])

    shown = np.array(args.frames) - 1
    offsets = c_pred[shown] - c_gt[shown]

    # View: from behind and above the cameras, looking across the offsets so the arrows are seen at full length
    up = -R_gt[:, :, 1].mean(0)
    up /= np.linalg.norm(up)
    forward = R_gt[shown, :, 2].mean(0)
    forward -= (forward @ up) * up
    side = offsets.mean(0) - (offsets.mean(0) @ up) * up
    side /= np.linalg.norm(side)
    view_dir = np.cross(up, side)
    view_dir *= np.sign(view_dir @ forward) or 1.0
    e = np.radians(args.elev)
    center = np.concatenate([c_gt[shown], c_pred[shown]]).mean(0)
    extent = np.ptp(np.concatenate([c_gt[shown], c_pred[shown]]), axis=0).max()
    eye = center - 6 * extent * (np.cos(e) * view_dir - np.sin(e) * up)
    R_v = recon.look_at(eye, center, up)

    size = (1000, 640)
    ends = np.concatenate([c_gt[shown], c_pred[shown]])
    for R_set, c_set in ((R_gt, c_gt), (R_pred, c_pred)):  # frame the frustums too
        ends = np.concatenate([ends] + [np.array(seg) for i in shown for seg in recon.frustum_lines(R_set[i], c_set[i], K[i], (H, W), args.frustum)])
    cam = (ends - eye) @ R_v.T
    uv = cam[:, :2] / cam[:, 2:]
    lo, hi = uv.min(0), uv.max(0)
    scale = 0.8 * min(size[0] / (hi - lo)[0], size[1] / (hi - lo)[1])
    offset = np.array(size) / 2 - scale * (lo + hi) / 2
    view = (R_v, eye, scale, offset)

    def proj(x):
        cam = (np.atleast_2d(x) - eye) @ R_v.T
        return cam[:, :2] / cam[:, 2:] * scale + offset

    # Background: the GT reconstruction behind the cameras, faded
    pts, rgb, rays = gt_cloud(data, K, R_gt, c_gt, stride=2)
    depth_center = (center - eye) @ R_v[2]
    keep = (rays @ R_v[2] > -0.05) & ((pts - eye) @ R_v[2] > depth_center + 0.1)
    img = recon.render_points(pts[keep], 0.7 * rgb[keep] + 0.3, view, size)

    fig, ax = plt.subplots(figsize=(10, 6.4))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.imshow(img)
    for i in shown:
        recon.draw_cameras(ax, view, [i], R_gt, c_gt, K, (H, W), args.frustum, GT_COLOR)
        recon.draw_cameras(ax, view, [i], R_pred, c_pred, K, (H, W), args.frustum, PRED_COLOR)
        a, b = proj(c_gt[i])[0], proj(c_pred[i])[0]
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=2.2, mutation_scale=16, shrinkA=0, shrinkB=0))
        ax.annotate(f"{i + 1}", a, xytext=(-10, -12), textcoords="offset points", ha="right", fontsize=11, fontweight="bold", color=GT_COLOR,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_xlim(0, size[0]), ax.set_ylim(size[1], 0)
    handles = [plt.Line2D([], [], color=GT_COLOR, lw=1.6, label="true camera (number = frame)"),
               plt.Line2D([], [], color=PRED_COLOR, lw=1.6, label="DA3-Small's camera"),
               plt.Line2D([], [], color=ARROW_COLOR, lw=2.2, label="error of that camera")]
    ax.legend(handles=handles, loc="lower left", fontsize=10, framealpha=0.9, edgecolor="none")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    out = args.out or os.path.join(analysis.REPO_ROOT, "results/failure_modes", args.model, f"pose_drift_{args.scene}.png")
    fig.savefig(out, dpi=130, facecolor="white")
    cos = offsets @ offsets.mean(0) / (np.linalg.norm(offsets, axis=1) * np.linalg.norm(offsets.mean(0)))
    print(out, f"offsets {np.round(np.linalg.norm(offsets, axis=1) * 100)} cm, cos to mean direction {np.round(cos, 2)}")


if __name__ == "__main__":
    main()
