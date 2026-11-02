"""Figures of the DA3 benchmark against bundle adjustment (BA) for one HiRoom scene.

Usage:
    uv run scripts/visualize_ba.py [scene] [--models DA3-SMALL DA3-LARGE-1.1]

Writes two figures to results/figures/: predicted cameras coloured by rotation error to GT (top-down), and the
fused point cloud coloured by distance to GT, each before and after BA.
Needs workspace/evaluation/<model> (benchmark predictions) and <model>-ba (BA poses from run_tto_ba.py).
BA only changes poses, so the "+ BA" clouds are DA3's depth fused with the BA poses. They and the benchmark
clouds are fused with the evaluator on first use (workspace/evaluation/<model>-ba-recon).
Open3D's offscreen renderer does not run on macOS, so the clouds are drawn with a small numpy renderer.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from depth_anything_3.bench.utils import compute_pose
from depth_anything_3.utils.geometry import as_homogeneous
from depth_anything_3.utils.pose_align import align_poses_umeyama
from PIL import Image
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = {"DA3-SMALL": "DA3-Small", "DA3-BASE": "DA3-Base", "DA3-LARGE-1.1": "DA3-Large-1.1"}
CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("err", ["#c9ced6", "#f2b134", "#d1342f"])
ROT_MAX = 6.0  # degrees at which the camera colours saturate
DIST_MAX = 0.10  # metres at which the cloud colours saturate


def exports(model, scene):
    return f"{ROOT}/workspace/evaluation/{model}/model_results/hiroom/{scene}/unposed/exports"


def to4(ext):
    return np.concatenate([ext, np.tile([[[0, 0, 0, 1.0]]], (len(ext), 1, 1))], 1).astype(np.float64)


def align(target_ext, ext):
    # Sim(3) alignment of `ext` to `target_ext`, as in the evaluator
    return align_poses_umeyama(to4(target_ext), to4(ext), return_aligned=True, ransac=True, random_state=42)[3]


def up_vector(c2w):
    up = -c2w[:, :3, 1].mean(0)  # OpenCV cameras: y points down
    return up / np.linalg.norm(up)


def input_montage(image_files):
    ims = [Image.open(f"{ROOT}/{image_files[i]}").convert("RGB") for i in np.linspace(0, len(image_files) - 1, 6).astype(int)]
    w, h = 320, int(320 * ims[0].height / ims[0].width)
    canvas = Image.new("RGB", (3 * w, 2 * h), "white")
    for k, im in enumerate(ims):
        canvas.paste(im.resize((w, h)), ((k % 3) * w, (k // 3) * h))
    return np.asarray(canvas)


def figure(models):
    # top row: input views and each model; bottom row: GT and each model + BA
    cols = 1 + len(models)
    return plt.subplots(2, cols, figsize=(4 * cols, 7.2), squeeze=False)


def finish(fig, label, vmax):
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08, hspace=0.22)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, cax=fig.add_axes([0.35, 0.03, 0.3, 0.015]), orientation="horizontal", extend="max").set_label(label)


def cameras_figure(models, scene):
    meta = np.load(f"{exports(models[0], scene)}/gt_meta.npz", allow_pickle=True)
    gt_ext = meta["extrinsics"].astype(np.float64)
    gt_c2w = np.linalg.inv(gt_ext)

    # top-down view in the plane orthogonal to the mean GT "up"
    up = up_vector(gt_c2w)
    a = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    proj = lambda x: np.stack([x @ a, x @ b], -1)
    gt_xy = proj(gt_c2w[:, :3, 3])
    arrow = 0.12 * np.ptp(gt_xy, 0).max()
    lo, hi = gt_xy.min(0), gt_xy.max(0)
    margin = 0.35 * (hi - lo).max()

    def draw(ax, c2w, color, z):
        xy, d = proj(c2w[:, :3, 3]), proj(c2w[:, :3, 2])
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        ax.scatter(*xy.T, s=26, color=color, edgecolors="#444444", linewidths=0.4, zorder=z)
        ax.quiver(*xy.T, *(d * arrow).T, color=color, angles="xy", scale_units="xy", scale=1, width=0.004, zorder=z)
        return xy

    def panel(ax, title, ext=None):
        ax.set_facecolor("#ebebeb")
        draw(ax, gt_c2w, "black", 2)
        if ext is not None:
            c2w = np.linalg.inv(align(gt_ext, ext))
            rot_err = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", c2w[:, :3, :3], gt_c2w[:, :3, :3]) - 1) / 2, -1, 1)))
            pred_xy = draw(ax, c2w, CMAP(np.clip(rot_err / ROT_MAX, 0, 1)), 3)
            for p, g in zip(pred_xy, gt_xy):
                ax.plot(*np.stack([p, g]).T, color="#888888", lw=0.6, alpha=0.6, zorder=1)
            pose = compute_pose(torch.from_numpy(as_homogeneous(ext.astype(np.float32))), torch.from_numpy(as_homogeneous(gt_ext.astype(np.float32))))
            title += f"\nAUC3 {100 * float(pose['auc03']):.1f}%"
        ax.set_xlim(lo[0] - margin, hi[0] + margin)
        ax.set_ylim(lo[1] - margin, hi[1] + margin)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=12)

    fig, axes = figure(models)
    axes[0][0].imshow(input_montage(list(meta["image_files"])))
    axes[0][0].axis("off")
    axes[0][0].set_title("Input views", fontsize=12)
    panel(axes[1][0], "GT")
    for k, m in enumerate(models, start=1):
        for row, tag in ((0, ""), (1, "-ba")):
            ext = np.load(f"{exports(m + tag, scene)}/mini_npz/results.npz")["extrinsics"]
            panel(axes[row][k], LABELS[m] + (" + BA" if tag else ""), ext)
    finish(fig, "rotation error to GT (°); black: GT cameras", ROT_MAX)
    return fig


def prepare(models, scene):
    # fuse the benchmark clouds and the BA-pose clouds with the evaluator, if not done yet
    for m in models:
        out = exports(m + "-ba-recon", scene)
        if not os.path.exists(f"{out}/mini_npz/results.npz"):
            base = dict(np.load(f"{exports(m, scene)}/mini_npz/results.npz"))
            ba = np.load(f"{exports(m + '-ba', scene)}/mini_npz/results.npz")
            # BA poses in the benchmark poses' frame and scale, so that they fit the unchanged depth maps
            os.makedirs(f"{out}/mini_npz", exist_ok=True)
            shutil.copy(f"{exports(m, scene)}/gt_meta.npz", f"{out}/gt_meta.npz")
            base["extrinsics"], base["intrinsics"] = align(base["extrinsics"], ba["extrinsics"])[:, :3].astype(np.float32), ba["intrinsics"]
            np.savez(f"{out}/mini_npz/results.npz", **base)
        for name in (m, m + "-ba-recon"):
            if not os.path.exists(f"{exports(name, scene)}/fuse/pcd.ply"):
                subprocess.run(
                    [sys.executable, f"{ROOT}/scripts/run_bench.py", name, "eval.eval_only=true", "eval.datasets=[hiroom]",
                     "eval.modes=[recon_unposed]", f"eval.scenes=[{scene}]", "inference.num_fusion_workers=1"],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def render(pcd, eye, center, up, size=(800, 600), fov=50):
    # perspective projection, near points overwrite far ones
    w, h = size
    pts, cols = np.asarray(pcd.points), np.asarray(pcd.colors)
    f = center - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    rel = pts - eye
    x, y, z = rel @ r, rel @ u, rel @ f
    ok = z > 1e-6
    x, y, z, cols = x[ok], y[ok], z[ok], cols[ok]
    focal = (h / 2) / np.tan(np.deg2rad(fov) / 2)
    px = np.round(w / 2 + focal * x / z).astype(int)
    py = np.round(h / 2 - focal * y / z).astype(int)
    order = np.argsort(-z)
    px, py, cols = px[order], py[order], (np.clip(cols, 0, 1) * 255).astype(np.uint8)[order]
    img = np.full((h, w, 3), 235, np.uint8)
    for dx in range(2):
        for dy in range(2):
            qx, qy = px + dx, py + dy
            m = (qx >= 0) & (qx < w) & (qy >= 0) & (qy < h)
            img[qy[m], qx[m]] = cols[m]
    return img


def error_figure(models, scene):
    prepare(models, scene)
    meta = np.load(f"{exports(models[0], scene)}/gt_meta.npz", allow_pickle=True)
    gt_c2w = np.linalg.inv(meta["extrinsics"].astype(np.float64))
    up = up_vector(gt_c2w)

    gt_path = f"{ROOT}/workspace/benchmark_dataset/hiroom/fused_pcd/{'-'.join(scene.split('/')[-3:])}.ply"
    bbox = o3d.io.read_point_cloud(gt_path).get_axis_aligned_bounding_box()
    crop = o3d.geometry.AxisAlignedBoundingBox(bbox.get_center() - bbox.get_extent() * 0.75, bbox.get_center() + bbox.get_extent() * 0.75)
    load = lambda path: o3d.io.read_point_cloud(path).crop(crop).voxel_down_sample(0.01)
    gt = load(gt_path)
    gt_tree = cKDTree(np.asarray(gt.points))

    # viewpoint: behind the mean GT camera, looking at the GT centre, raised by 35 degrees
    center = bbox.get_center()
    view = gt_c2w[:, :3, 2].mean(0)
    view -= (view @ up) * up
    view /= np.linalg.norm(view)
    e = np.deg2rad(35)
    eye = center - np.linalg.norm(bbox.get_extent()) * (np.cos(e) * view - np.sin(e) * up)

    def error_panel(name, title):
        pcd = load(f"{exports(name, scene)}/fuse/pcd.ply")
        dist, _ = gt_tree.query(np.asarray(pcd.points), workers=-1)
        pcd.colors = o3d.utility.Vector3dVector(CMAP(np.clip(dist / DIST_MAX, 0, 1))[:, :3])
        with open(f"{ROOT}/workspace/evaluation/{name}/metric_results/hiroom_recon_unposed.json") as f:
            return title + f"\nF1 {100 * json.load(f)[scene]['fscore']:.1f}%", render(pcd, eye, center, up)

    fig, axes = figure(models)
    for ax in axes.flat:
        ax.axis("off")
    axes[0][0].imshow(input_montage(list(meta["image_files"])))
    axes[0][0].set_title("Input views", fontsize=12)
    axes[1][0].imshow(render(gt, eye, center, up))
    axes[1][0].set_title("GT", fontsize=12)
    for k, m in enumerate(models, start=1):
        for row, name, title in ((0, m, LABELS[m]), (1, m + "-ba-recon", LABELS[m] + " + BA")):
            title, img = error_panel(name, title)
            axes[row][k].imshow(img)
            axes[row][k].set_title(title, fontsize=12)
    finish(fig, "distance to GT (cm)", DIST_MAX * 100)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default="20241230/828784/cam_sampled_03")
    ap.add_argument("--models", nargs="+", default=["DA3-SMALL", "DA3-LARGE-1.1"])
    args = ap.parse_args()

    os.chdir(ROOT)
    os.makedirs("results/figures", exist_ok=True)
    tag = f"hiroom_{args.scene.replace('/', '-')}"
    for name, fig in (("cameras_ba", cameras_figure(args.models, args.scene)), ("recon_ba_error", error_figure(args.models, args.scene))):
        out = f"results/figures/{tag}_{name}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(out)


if __name__ == "__main__":
    main()
