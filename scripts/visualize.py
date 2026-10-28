"""Render saved benchmark predictions of one scene for several models.

Usage:
    uv run scripts/visualize.py <dataset> <scene> [--models DA3-SMALL dvlt ...] [--mode unposed]

Reads what the evaluator saved under workspace/evaluation/<model>/model_results/<dataset>/<scene>/<mode>/
(fused point cloud, predicted and GT cameras) and writes two figures to results/figures/:
- <dataset>_<scene>_recon.png: input images, GT and each model's fused point cloud from one viewpoint.
  The unposed clouds were fused with the Umeyama-aligned predicted cameras, so they are in the GT frame.
- <dataset>_<scene>_cameras.png: predicted vs GT camera positions and viewing directions, top-down,
  after the same Sim(3) alignment the evaluator uses for fusion.
Datasets with GT point clouds: hiroom, dtu.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from depth_anything_3.utils.pose_align import align_poses_umeyama
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(REPO_ROOT, "workspace", "evaluation")
OUT_DIR = os.path.join(REPO_ROOT, "results", "figures")

LABELS = {
    "DA3-SMALL": "DA3-Small",
    "DA3-BASE": "DA3-Base",
    "DA3-LARGE-1.1": "DA3-Large-1.1",
    "VGGT-1B": "VGGT",
    "Pi3": "π³",
    "dvlt": "DéjàView",
}
# Per dataset: GT point cloud path, voxel size for rendering (scene units: m for HiRoom, mm for DTU),
# the metric shown under each panel (key in the metric JSON, label, factor) and, for --color error,
# the distance to GT at which the colour map saturates (scene units, display unit, factor), and
# optionally a filter for the points the metric counts.
DATASETS = {
    "hiroom": {
        "gt": lambda s: f"workspace/benchmark_dataset/hiroom/fused_pcd/{'-'.join(s.split('/')[-3:])}.ply",
        "voxel": 0.01,
        "metric": ("fscore", "F1 {:.1f}%", 100),
        "err": (0.10, "cm", 100),
    },
    "dtu": {
        "gt": lambda s: f"workspace/benchmark_dataset/dtu/Points/stl/stl{int(s[4:]):03}_total.ply",
        "voxel": 1.0,
        "metric": ("overall", "CD {:.2f} mm", 1),
        "err": (5.0, "mm", 1),
        "mask": lambda s: dtu_obs_mask(s),
    },
}
UNIFORM_COLOR = (0.62, 0.66, 0.72)
ERR_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("err", ["#c9ced6", "#f2b134", "#d1342f"])


def scene_dir(model, dataset, scene, mode):
    return os.path.join(EVAL_DIR, model, "model_results", dataset, scene, mode, "exports")


def dtu_obs_mask(scene):
    # Official DTU accuracy only counts points inside the observed volume (ObsMask), as in DA3's
    # bench/datasets/dtu.py; without this, table and background show up as large errors.
    from scipy.io import loadmat

    obs = loadmat(f"workspace/benchmark_dataset/dtu/SampleSet/mvs_data/ObsMask/ObsMask{int(scene[4:])}_10.mat")
    mask, bb, res = obs["ObsMask"], obs["BB"].astype(np.float32), obs["Res"]

    def keep(points):
        grid = np.around((points - bb[:1]) / res).astype(np.int64)
        ok = ((grid >= 0) & (grid < np.array(mask.shape))).all(-1)
        ok[ok] = mask[grid[ok, 0], grid[ok, 1], grid[ok, 2]].astype(bool)
        return ok

    return keep


def load_cloud(path, voxel, bbox, keep=None):
    pcd = o3d.io.read_point_cloud(path)
    pcd = pcd.crop(bbox).voxel_down_sample(voxel)
    if keep is not None:
        pcd = pcd.select_by_index(np.flatnonzero(keep(np.asarray(pcd.points))))
    if not pcd.has_colors():
        pcd.paint_uniform_color(UNIFORM_COLOR)
    return pcd


def up_vector(c2w):
    # OpenCV cameras: y points down in the image, so -y averaged over all cameras is "up".
    up = -c2w[:, :3, 1].mean(0)
    return up / np.linalg.norm(up)


def render(pcd, eye, center, up, size, point_size):
    renderer = o3d.visualization.rendering.OffscreenRenderer(*size)
    renderer.scene.set_background([1, 1, 1, 1])
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = point_size
    renderer.scene.add_geometry("pcd", pcd, mat)
    renderer.setup_camera(50, center, eye, up)
    return np.asarray(renderer.render_to_image())


def shade_by_height(pcd, up):
    # Uncoloured clouds (DTU) get a height gradient, otherwise their shape is hard to read.
    h = np.asarray(pcd.points) @ up
    t = (h - np.percentile(h, 2)) / max(np.ptp(np.percentile(h, [2, 98])), 1e-9)
    pcd.colors = o3d.utility.Vector3dVector(plt.cm.cividis(np.clip(t, 0, 1))[:, :3])


def color_by_error(pcd, gt_tree, vmax):
    # Distance of each point to the nearest GT point, the same distance F1 thresholds.
    dist, _ = gt_tree.query(np.asarray(pcd.points), workers=-1)
    pcd.colors = o3d.utility.Vector3dVector(ERR_CMAP(np.clip(dist / vmax, 0, 1))[:, :3])


def input_montage(image_files, n=6):
    idx = np.linspace(0, len(image_files) - 1, min(n, len(image_files))).astype(int)
    ims = [Image.open(os.path.join(REPO_ROOT, image_files[i])).convert("RGB") for i in idx]
    w, h = 320, int(320 * ims[0].height / ims[0].width)
    ims = [im.resize((w, h)) for im in ims]
    cols = 3
    rows = int(np.ceil(len(ims) / cols))
    canvas = Image.new("RGB", (cols * w, rows * h), "white")
    for k, im in enumerate(ims):
        canvas.paste(im, ((k % cols) * w, (k // cols) * h))
    return np.asarray(canvas)


def metric_of(model, dataset, scene, mode):
    import json

    key, fmt, factor = DATASETS[dataset]["metric"]
    path = os.path.join(EVAL_DIR, model, "metric_results", f"{dataset}_recon_{mode}.json")
    with open(path) as f:
        return fmt.format(json.load(f)[scene][key] * factor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=DATASETS)
    ap.add_argument("scene")
    ap.add_argument("--models", nargs="+", default=list(LABELS))
    ap.add_argument("--mode", default="unposed", choices=["unposed", "posed"])
    ap.add_argument("--elev", type=float, default=35, help="viewpoint elevation in degrees")
    ap.add_argument("--dist", type=float, default=1.0, help="viewpoint distance factor")
    ap.add_argument("--point-size", type=float, default=2)
    ap.add_argument("--color", default="rgb", choices=["rgb", "error", "both"],
                    help="model clouds: image colours, distance to GT, or both (one column per model)")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    cfg = DATASETS[args.dataset]
    tag = f"{args.dataset}_{args.scene.replace('/', '-')}"
    suffix = f"_{args.mode}" + ("" if args.color == "rgb" else f"_{args.color}") + f"_elev{args.elev:g}"
    os.makedirs(OUT_DIR, exist_ok=True)

    meta = np.load(os.path.join(scene_dir(args.models[0], args.dataset, args.scene, args.mode), "gt_meta.npz"), allow_pickle=True)
    gt_c2w = np.linalg.inv(meta["extrinsics"].astype(np.float64))
    up = up_vector(gt_c2w)

    gt = o3d.io.read_point_cloud(cfg["gt"](args.scene))
    gt_bbox = gt.get_axis_aligned_bounding_box()
    crop = o3d.geometry.AxisAlignedBoundingBox(
        gt_bbox.get_center() - gt_bbox.get_extent() * 0.75, gt_bbox.get_center() + gt_bbox.get_extent() * 0.75
    )
    keep = cfg["mask"](args.scene) if "mask" in cfg else None
    clouds = [("GT", load_cloud(cfg["gt"](args.scene), cfg["voxel"], crop, keep), "")]
    for m in args.models:
        path = os.path.join(scene_dir(m, args.dataset, args.scene, args.mode), "fuse", "pcd.ply")
        clouds.append((LABELS.get(m, m), load_cloud(path, cfg["voxel"], crop, keep), metric_of(m, args.dataset, args.scene, args.mode)))
    if args.dataset == "dtu":
        for _, pcd, _ in clouds:
            shade_by_height(pcd, up)
    error_clouds = []
    if args.color in ("error", "both"):
        from scipy.spatial import cKDTree

        gt_tree = cKDTree(np.asarray(clouds[0][1].points))
        for name, pcd, sub in clouds[1:]:
            err = o3d.geometry.PointCloud(pcd)
            color_by_error(err, gt_tree, cfg["err"][0])
            error_clouds.append((name, err, sub))
        if args.color == "error":
            clouds = clouds[:1] + error_clouds

    # Viewpoint: behind the mean GT camera, looking at the GT centre, raised by `elev` degrees.
    center = gt_bbox.get_center()
    view = gt_c2w[:, :3, 2].mean(0)
    view -= (view @ up) * up
    view /= np.linalg.norm(view)
    radius = np.linalg.norm(gt_bbox.get_extent()) * args.dist
    e = np.deg2rad(args.elev)
    eye = center - radius * (np.cos(e) * view - np.sin(e) * up)

    size = (800, 600)
    snap = lambda pcd: render(pcd, eye, center, up, size, args.point_size)
    inputs = ("Input views", input_montage(list(meta["image_files"])), "")
    if args.color == "both":
        # Column 0: input views over GT; then one column per model, image colours over error colours.
        top = [inputs] + [(name, snap(pcd), sub) for name, pcd, sub in clouds[1:]]
        bottom = [(clouds[0][0], snap(clouds[0][1]), "")]
        bottom += [(f"{name}: error", snap(pcd), "") for name, pcd, _ in error_clouds]
        cols, rows = len(top), 2
        panels = [p for pair in zip(top, bottom) for p in pair]
        order = [(r, c) for c in range(cols) for r in range(rows)]
    else:
        panels = [inputs] + [(name, snap(pcd), sub) for name, pcd, sub in clouds]
        cols = 4
        rows = int(np.ceil(len(panels) / cols))
        order = [(k // cols, k % cols) for k in range(len(panels))]
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for (r, c), (name, img, sub) in zip(order, panels):
        axes[r, c].imshow(img)
        axes[r, c].set_title(name + (f"\n{sub}" if sub else ""), fontsize=12)
    fig.tight_layout()
    if args.color in ("error", "both"):
        vmax, unit, factor = cfg["err"]
        fig.subplots_adjust(bottom=0.08)
        cax = fig.add_axes([0.35, 0.03, 0.3, 0.015])
        sm = plt.cm.ScalarMappable(cmap=ERR_CMAP, norm=plt.Normalize(0, vmax * factor))
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
        cb.set_label(f"distance to GT ({unit})")
    out = os.path.join(OUT_DIR, f"{tag}_recon{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)

    # Cameras: top-down view in the plane orthogonal to `up`.
    a = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    proj = lambda x: np.stack([x @ a, x @ b], -1)
    gt_xy = proj(gt_c2w[:, :3, 3])
    arrow = 0.15 * np.ptp(gt_xy, 0).max()

    def draw(ax, c2w, color, label):
        xy, d = proj(c2w[:, :3, 3]), proj(c2w[:, :3, 2])
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        ax.scatter(*xy.T, s=14, color=color, label=label, zorder=3)
        ax.quiver(*xy.T, *(d * arrow).T, color=color, angles="xy", scale_units="xy", scale=1, width=0.004, zorder=3)
        return xy

    fig, axes = plt.subplots(1, len(args.models), figsize=(3.4 * len(args.models), 3.6), squeeze=False)
    for ax, m in zip(axes[0], args.models):
        pred = np.load(os.path.join(scene_dir(m, args.dataset, args.scene, args.mode), "mini_npz", "results.npz"))
        ext = np.tile(np.eye(4), (len(pred["extrinsics"]), 1, 1))
        ext[:, :3] = pred["extrinsics"]
        # Same alignment as the evaluator's recon_unposed fusion.
        _, _, _, ext_aligned = align_poses_umeyama(
            meta["extrinsics"].copy(), ext, return_aligned=True, ransac=True, random_state=42
        )
        pred_xy = draw(ax, np.linalg.inv(ext_aligned), "tab:red", "predicted")
        draw(ax, gt_c2w, "black", "GT")
        for p, g in zip(pred_xy, gt_xy):
            ax.plot(*np.stack([p, g]).T, color="tab:red", lw=0.6, alpha=0.5)
        ax.set_title(LABELS.get(m, m), fontsize=12)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0][0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"{tag}_cameras_{args.mode}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
