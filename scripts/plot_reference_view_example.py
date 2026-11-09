"""One picture of finding 2: the reference view decides which frames DA3 misplaces.

Two runs of the same 100 images that differ only in the reference view. The GT reconstruction is tinted where the
reference view sees it; failing frames are drawn as red frustums at their true pose. Needs the output of
reference_view_sweep.py.

    uv run scripts/plot_reference_view_example.py DA3-SMALL scannetpp 5f99900f09 --refs 0 54
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
from reference_view_sweep import gt_intrinsics  # noqa: E402

REF_COLOR, FAIL_COLOR, OK_COLOR, TINT = "#e0a100", "#d1342f", "#333333", np.array([0.98, 0.78, 0.2])


def seen_by(points, w2c, K, depth, valid, tol=0.05):
    q = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = q[:, 2]
    H, W = depth.shape
    out = np.zeros(len(z), bool)
    front = z > 1e-3
    u = np.round(q[front, 0] / z[front] * K[0, 0] + K[0, 2]).astype(np.int64)
    v = np.round(q[front, 1] / z[front] * K[1, 1] + K[1, 2]).astype(np.int64)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx = np.nonzero(front)[0][inside]
    u, v = u[inside], v[inside]
    out[idx] = valid[v, u] & (np.abs(depth[v, u] - z[idx]) < tol * z[idx])
    return out


def cameras(ax, view, idx, R, c, K, hw, size, color, lw):
    n = len(ax.lines)
    recon.draw_cameras(ax, view, idx, R, c, K, hw, size, color)
    for line in ax.lines[n:]:
        line.set_linewidth(lw)
        line.set_zorder(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="DA3-SMALL")
    ap.add_argument("dataset", nargs="?", default="scannetpp")
    ap.add_argument("scene", nargs="?", default="5f99900f09")
    ap.add_argument("--refs", type=int, nargs=2, default=[0, 54])
    ap.add_argument("--elev", type=float, default=60.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sweep = np.load(os.path.join(analysis.REPO_ROOT, "workspace/failure_modes", args.model, "ref_sweep", args.dataset, args.scene + ".npz"))
    gross = (sweep["rot"] > sweep["gross_rot"]) | (sweep["trans"] > sweep["gross_trans"])
    refs = list(sweep["refs"])

    data = analysis.load_scene(args.model, args.dataset, args.scene)
    K = gt_intrinsics(args.model, args.dataset, args.scene, data)
    c2w = np.linalg.inv(data["gt_w2c"])
    R, c = c2w[:, :3, :3], c2w[:, :3, 3]
    H, W = data["gt_depth"].shape[1:]

    stride = 3
    pts, rgb, rays = [], [], []
    for t in range(len(c)):
        ys, xs = np.nonzero(data["valid"][t][::stride, ::stride])
        ys, xs = ys * stride, xs * stride
        p = recon.backproject(data["gt_depth"][t], K[t], R[t], c[t], ys, xs)
        img = cv2.resize(cv2.imread(data["image_files"][t]), (W, H), interpolation=cv2.INTER_AREA)
        pts.append(p), rgb.append(img[ys, xs, ::-1] / 255.0), rays.append(p - c[t])
    pts, rgb, rays = np.concatenate(pts), np.concatenate(rgb), np.concatenate(rays)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)

    # View across the room's long axis, from above
    up = -R[:, :, 1].mean(0)
    up /= np.linalg.norm(up)
    flat = c - np.outer(c @ up, up)
    _, _, Vt = np.linalg.svd(flat - flat.mean(0))
    long_axis = Vt[0] - (Vt[0] @ up) * up
    view_dir = np.cross(up, long_axis / np.linalg.norm(long_axis))
    center = np.median(pts, axis=0)
    e = np.radians(args.elev)
    eye = center - 2.6 * np.percentile(np.linalg.norm(pts - center, axis=1), 90) * (np.cos(e) * view_dir - np.sin(e) * up)
    R_v = recon.look_at(eye, center, up)
    keep = (rays @ R_v[2] > -0.05) & (pts @ up < np.percentile(c @ up, 90))  # drop ceiling and walls facing away
    size = (1000, 560)
    cam = (pts[keep] - eye) @ R_v.T
    uv = cam[:, :2] / cam[:, 2:]
    lo, hi = np.percentile(uv, 1, axis=0), np.percentile(uv, 99, axis=0)
    scale = 0.9 * min(size[0] / (hi - lo)[0], size[1] / (hi - lo)[1])
    view = (R_v, eye, scale, offset := np.array(size) / 2 - scale * (lo + hi) / 2)
    frustum = 0.22 * float(sweep["depth_scale"])

    fig, axes = plt.subplots(2, 1, figsize=(9, 11.2))
    for ax, r in zip(axes, args.refs):
        k = refs.index(r)
        seen = seen_by(pts[keep], data["gt_w2c"][r], K[r], data["gt_depth"][r], data["valid"][r])
        col = 0.5 * rgb[keep] + 0.5
        col[seen] = 0.35 * rgb[keep][seen] + 0.65 * TINT
        ax.imshow(recon.render_points(pts[keep], col, view, size))
        fail = np.nonzero(gross[k])[0]
        fail = fail[fail != r]
        ok = np.setdiff1d(np.arange(len(c)), np.concatenate([fail, [r]]))
        uv_c = (c - eye) @ R_v.T
        uv_c = uv_c[:, :2] / uv_c[:, 2:] * scale + offset
        ax.scatter(*uv_c[ok].T, s=6, color=OK_COLOR, zorder=3, linewidths=0)
        cameras(ax, view, fail, R, c, K, (H, W), frustum, FAIL_COLOR, 1.1)
        cameras(ax, view, [r], R, c, K, (H, W), 2.0 * frustum, REF_COLOR, 2.6)
        ax.text(0.01, 0.99, f"{len(fail)} of {len(c) - 1} frames misplaced", transform=ax.transAxes, va="top", fontsize=20, color="#111111")
        ax.set_xlim(0, size[0]), ax.set_ylim(size[1], 0)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    handles = [
        plt.Line2D([], [], color=REF_COLOR, lw=2.6, label="reference view"),
        matplotlib.patches.Patch(color=0.35 * np.ones(3) * 0.7 + 0.65 * TINT, label="seen by the reference view"),
        plt.Line2D([], [], color=FAIL_COLOR, lw=1.5, label="misplaced frame (true pose)"),
        plt.Line2D([], [], color=OK_COLOR, marker="o", ms=4, lw=0, label="correctly placed frame"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=17, handlelength=1.6, columnspacing=1.2)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.11, hspace=0.03)
    out = args.out or os.path.join(analysis.REPO_ROOT, "results/failure_modes", args.model, f"reference_view_{args.scene}.png")
    fig.savefig(out, dpi=130, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
