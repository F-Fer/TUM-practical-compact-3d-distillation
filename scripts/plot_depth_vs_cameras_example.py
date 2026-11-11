"""One picture of the reconstruction finding: DA3's cameras cost more than its depth maps.

Two reconstructions of one scene from the same viewpoint, coloured by distance to the GT surface:
DA3's depth maps placed with the GT cameras, and the GT depth maps placed with DA3's cameras (intrinsics and poses).

    uv run scripts/plot_depth_vs_cameras_example.py DA3-SMALL hiroom 20241230/828791/cam_sampled_12
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import analyse_failure_modes as analysis  # noqa: E402
import reconstruction_errors as recon  # noqa: E402

PANELS = [("Depth error only", "DA3 depth, true cameras"), ("Camera error only", "True depth, DA3 cameras")]
INK, INK2 = "#111111", "#52514e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="DA3-SMALL")
    ap.add_argument("dataset", nargs="?", default="hiroom")
    ap.add_argument("scene", nargs="?", default="20241230/828791/cam_sampled_12")
    ap.add_argument("--elev", type=float, default=35.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    built = recon.build_clouds(args.model, args.dataset, args.scene, recon.STRIDE["render"])
    clouds, cams = built["clouds"], built["cams"]
    gt = clouds["Ground truth"]

    # Same viewpoint as the 3d_*.png renders of reconstruction_errors.py
    R_gt = cams["gt"][0]
    up = -R_gt[:, :, 1].mean(0)
    up /= np.linalg.norm(up)
    center = np.median(gt, axis=0)
    view_dir = R_gt[:, :, 2].mean(0)
    view_dir -= (view_dir @ up) * up
    view_dir /= np.linalg.norm(view_dir)
    radius = 2.2 * np.percentile(np.linalg.norm(gt - center, axis=1), 90)
    e = np.radians(args.elev)
    eye = center - radius * (np.cos(e) * view_dir - np.sin(e) * up)
    R_v = recon.look_at(eye, center, up)
    keep = built["rays"] @ R_v[2] > -0.05

    size = (900, 640)
    cam = (gt[keep] - eye) @ R_v.T
    uv = cam[:, :2] / cam[:, 2:]
    lo, hi = np.percentile(uv, 1, axis=0), np.percentile(uv, 99, axis=0)
    scale = 0.9 * min(size[0] / (hi - lo)[0], size[1] / (hi - lo)[1])
    offset = np.array(size) / 2 - scale * (lo + hi) / 2
    view = (R_v, eye, scale, offset)

    vmax = recon.TAU[args.dataset] * 2
    factor, unit = recon.DISPLAY[args.dataset]
    tree = cKDTree(gt)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.14, wspace=0.02)
    for ax, (config, label) in zip(axes, PANELS):
        pts = clouds[config][keep]
        dist = tree.query(pts, workers=-1)[0]
        img = recon.render_points(pts, recon.ERR_CMAP(np.clip(dist / vmax, 0, 1))[:, :3], view, size)
        ax.imshow(img)
        ax.set_xlim(0, size[0]), ax.set_ylim(size[1], 0)
        ax.set_axis_off()
        ax.set_title(label, fontsize=14, color=INK)
        print(f"{label}: median distance {np.median(dist) * factor:.1f} {unit}")

    cax = fig.add_axes([0.35, 0.07, 0.3, 0.03])
    sm = plt.cm.ScalarMappable(cmap=recon.ERR_CMAP, norm=plt.Normalize(0, vmax * factor))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    cb.set_label(f"distance to the true surface ({unit})", fontsize=10, color=INK2)
    cb.ax.tick_params(labelsize=9, colors=INK2)
    cb.outline.set_visible(False)

    out = args.out or os.path.join(analysis.REPO_ROOT, "results/failure_modes", args.model, f"depth_vs_cameras_{args.scene.replace('/', '_')}.png")
    fig.savefig(out, dpi=130, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
