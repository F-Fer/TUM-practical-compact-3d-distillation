"""How much of the reconstruction error comes from the depth maps and how much from the cameras?

Usage:
    uv run scripts/reconstruction_errors.py DA3-SMALL                       # metrics for all scenes
    uv run scripts/reconstruction_errors.py DA3-SMALL --render auto         # + 3D figures of median and worst scenes
    uv run scripts/reconstruction_errors.py DA3-SMALL --render scannetpp/5f99900f09 --no-metrics

Needs workspace/failure_modes/<model>/ from analyse_failure_modes.py. Every pixel with valid GT depth is
back-projected once per configuration, mixing predicted and GT parts (depth map, intrinsics, rotation,
camera centre), and compared with the GT depth back-projected with GT cameras:
- F1: share of points within TAU of the other cloud, both ways (nearest neighbour, as in the benchmark,
  but without TSDF fusion),
- pixel error: distance between the two back-projections of the same pixel.
Predictions are aligned as in analyse_failure_modes.py; "depth error only" uses the depth's own scene scale
so that a scale mismatch with the poses isn't counted as depth error.

Writes results/failure_modes/<model>/reconstruction.json, reconstruction_parts.png and 3d_<dataset>_<scene>.png.
"""

import argparse
import glob
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import analyse_failure_modes as analysis  # noqa: E402
import plot_failure_modes as plots  # noqa: E402

REPO_ROOT = analysis.REPO_ROOT
STRIDE = {"metrics": 6, "render": 4}  # pixel subsampling of the 378x504 prediction grid
# F1 threshold and colour scale in scene units: m for the indoor datasets, mm for DTU
TAU = {"hiroom": 0.05, "7scenes": 0.05, "scannetpp": 0.05, "dtu": 5.0, "eth3d": 0.25}  # ETH3D as in DA3's benchmark
DISPLAY = {"hiroom": (100, "cm"), "7scenes": (100, "cm"), "scannetpp": (100, "cm"), "dtu": (1, "mm"), "eth3d": (100, "cm")}
# Same colours as scripts/visualize.py on main, so the figures read alike
ERR_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("err", ["#c9ced6", "#f2b134", "#d1342f"])
GT_CAMERA, EST_CAMERA, PANEL_BG = "#111111", "#2a78d6", "#ebebeb"

# Each configuration takes every part from the prediction or the GT: depth map, intrinsics, rotation, camera centre.
# "pred" depth gets the pose alignment's scale, "pred_own_scale" the depth's own (used whenever the cameras are GT).
CONFIGS = {
    "DA3-Small": ("pred", "pred", "pred", "pred"),
    # One predicted part, everything else GT: the damage each part does alone
    "Depth error only": ("pred_own_scale", "gt", "gt", "gt"),
    "Camera error only": ("gt", "pred", "pred", "pred"),
    "Pose error only": ("gt", "gt", "pred", "pred"),
    "Intrinsics error only": ("gt", "pred", "gt", "gt"),
    "Rotation error only": ("gt", "gt", "pred", "gt"),
    "Camera centre error only": ("gt", "gt", "gt", "pred"),
    # Everything predicted except one part: what fixing that part alone would gain
    "Fix depth": ("gt", "pred", "pred", "pred"),
    # Depth at its own scale instead of the poses': removes only the scale mismatch between the two
    "Fix depth-pose scale mismatch": ("pred_own_scale", "pred", "pred", "pred"),
    "Fix intrinsics": ("pred", "gt", "pred", "pred"),
    "Fix poses": ("pred_own_scale", "pred", "gt", "gt"),
    "Fix intrinsics and poses": ("pred_own_scale", "gt", "gt", "gt"),
    # Inside one part, with the others GT: noise, structure or outliers?
    "Depth only: no offset or smooth error": ("no smooth", "gt", "gt", "gt"),
    "Depth only: no outliers": ("no outliers", "gt", "gt", "gt"),
    "Poses only: no outliers": ("gt", "gt", "no outliers", "no outliers"),
    "Poses only: no smooth error": ("gt", "gt", "no smooth part", "no smooth part"),
    "Poses only: no jitter": ("gt", "gt", "no jitter", "no jitter"),
    "Poses only: gross errors fixed": ("gt", "gt", "no gross errors", "no gross errors"),
    "Fix gross pose errors": ("pred", "pred", "no gross errors", "no gross errors"),
}
PANELS = [
    "Ground truth", "DA3-Small", "Depth error only", "Camera error only",
    "Pose error only", "Intrinsics error only", "Depth only: no offset or smooth error", "Poses only: gross errors fixed",
]


def scale_K(K, image_hw, grid_hw):
    K = K.astype(np.float64).copy()
    K[0] *= grid_hw[1] / image_hw[1]
    K[1] *= grid_hw[0] / image_hw[0]
    return K


def backproject(depth, K, R, c, ys, xs):
    """World points of pixels (ys, xs) for camera-to-world rotation R and centre c."""
    z = depth[ys, xs]
    cam = np.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z], -1)
    return cam @ R.T + c


def camera_variants(data, saved):
    """Rotations and centres of the aligned prediction and of each counterfactual, plus the GT."""
    pred_c2w, gt_c2w = np.linalg.inv(data["pred_w2c"]), np.linalg.inv(data["gt_w2c"])
    gauge = analysis.robust_rotation_alignment(pred_c2w[:, :3, :3], gt_c2w[:, :3, :3])
    sim3 = analysis.robust_scale_translation(pred_c2w[:, :3, 3], gt_c2w[:, :3, 3], gauge)
    R_gt, c_gt = gt_c2w[:, :3, :3], gt_c2w[:, :3, 3]
    out = {"gt": (R_gt, c_gt), "pred": (gauge @ pred_c2w[:, :3, :3], analysis.apply_sim3(sim3, pred_c2w[:, :3, 3]))}

    # Counterfactuals from the decomposition saved by analyse_failure_modes.py (translations in depth units)
    depth_scale = float(saved["median_gt_depth"])
    r, t = saved["r_res"], saved["t_res"]
    r_in, t_in = ~saved["r_outlier"][:, None], ~saved["t_outlier"][:, None]
    variants = {
        "no outliers": (r_in * r + ~r_in * saved["r_smooth"], t_in * t + ~t_in * saved["t_smooth"]),
        "no smooth part": (r - saved["r_smooth"], t - saved["t_smooth"]),
        "no jitter": (r_in * saved["r_smooth"] + ~r_in * r, t_in * saved["t_smooth"] + ~t_in * t),
        # Absolute: also catches runs of bad frames, which the local split counts as smooth error
        "no gross errors": tuple(np.where(plots.gross_frames(saved)[:, None], 0, x) for x in (r, t)),
    }
    R_pred, c_pred = out["pred"]
    for name, (r_cf, t_cf) in variants.items():
        # As corrections to the prediction, like the AUC counterfactuals
        corr = Rotation.from_rotvec(np.radians(r_cf)) * Rotation.from_rotvec(np.radians(r)).inv()
        out[name] = (corr.as_matrix() @ R_pred, c_pred + (t_cf - t) * depth_scale)
    return out, sim3[0]


def depth_variants(data, saved, pose_scale):
    pred = data["pred_depth"].astype(np.float64)
    scene_log_scale = float(saved["scene_log_scale"])
    out = {"gt": data["gt_depth"], "pred": pred * pose_scale, "pred_own_scale": pred * np.exp(-scene_log_scale)}
    no_smooth, no_outliers = np.empty_like(pred), np.empty_like(pred)
    for t in range(len(pred)):
        v = data["valid"][t]
        e = np.log(np.maximum(pred[t], 1e-6)) - np.log(np.where(v, data["gt_depth"][t], 1)) - scene_log_scale
        if v.sum() < 100:
            no_smooth[t] = no_outliers[t] = pred[t]
            continue
        offset, low, high, sigma = analysis.split_depth_error(e, v)
        outlier = v & (np.abs(high) > analysis.DEPTH_OUTLIER_SIGMA * sigma)
        no_smooth[t] = pred[t] * np.exp(-(offset + low))
        no_outliers[t] = pred[t] * np.exp(-np.where(outlier, high, 0))
    own = np.exp(-scene_log_scale)
    out["no smooth"] = no_smooth * own
    out["no outliers"] = no_outliers * own
    return out


def build_clouds(model, dataset, scene, stride):
    data = analysis.load_scene(model, dataset, scene)
    name = scene.replace("/", "_")
    saved = dict(np.load(os.path.join(REPO_ROOT, "workspace/failure_modes", model, dataset, name + ".npz")))
    export = os.path.join(REPO_ROOT, "workspace/evaluation", model, "model_results", dataset, scene, "unposed/exports")
    pred_K = np.load(os.path.join(export, "mini_npz/results.npz"))["intrinsics"]
    gt_K_raw = np.load(os.path.join(export, "gt_meta.npz"), allow_pickle=True)["intrinsics"]
    order = np.argsort(data["bench_order"])  # sorted position -> benchmark index
    H, W = data["pred_depth"].shape[1:]
    image_hw = cv2.imread(data["image_files"][0]).shape[:2]
    Ks = {"pred": pred_K[order].astype(np.float64), "gt": np.stack([scale_K(K, image_hw, (H, W)) for K in gt_K_raw[order]])}

    cams, pose_scale = camera_variants(data, saved)
    depths = depth_variants(data, saved, pose_scale)

    pixels, colors = [], []
    for t in range(len(data["valid"])):
        ys, xs = np.nonzero(data["valid"][t][::stride, ::stride])
        ys, xs = ys * stride, xs * stride
        pixels.append((ys, xs))
        colors.append(cv2.resize(cv2.imread(data["image_files"][t]), (W, H), interpolation=cv2.INTER_AREA)[ys, xs, ::-1] / 255.0)

    def cloud(depth, K, rot, cen):
        pts = []
        for t, (ys, xs) in enumerate(pixels):
            pts.append(backproject(depths[depth][t], Ks[K][t], cams[rot][0][t], cams[cen][1][t], ys, xs))
        return np.concatenate(pts)

    gt = cloud("gt", "gt", "gt", "gt")
    frame = np.concatenate([np.full(len(p[0]), t) for t, p in enumerate(pixels)])
    rays = gt - cams["gt"][1][frame]
    clouds = {"Ground truth": gt}
    for cfg_name, (depth, K, rot, cen) in CONFIGS.items():
        clouds[cfg_name] = cloud(depth, K, rot, cen)
    return {
        "clouds": clouds,
        "rgb": np.concatenate(colors),
        "rays": rays / np.linalg.norm(rays, axis=1, keepdims=True),
        "cams": cams,
        "K_gt": Ks["gt"],
        "hw": (H, W),
        "saved": saved,
        "depth_scale": float(saved["median_gt_depth"]),
    }


def scores(clouds, tau):
    gt = clouds["Ground truth"]
    gt_tree = cKDTree(gt)
    out, dist_to_gt = {}, {}
    for name, pts in clouds.items():
        if name == "Ground truth":
            continue
        finite = np.isfinite(pts).all(1)
        acc = np.full(len(pts), np.inf)
        acc[finite] = gt_tree.query(pts[finite], workers=-1)[0]
        comp = cKDTree(pts[finite]).query(gt, workers=-1)[0]
        precision, recall = (acc < tau).mean(), (comp < tau).mean()
        pixel = np.linalg.norm(pts - gt, axis=1)
        out[name] = {
            "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
            "precision": float(precision),
            "recall": float(recall),
            "pixel_error_median": float(np.median(pixel)),
            "pixel_within_tau": float((pixel < tau).mean()),
        }
        dist_to_gt[name] = acc
    return out, dist_to_gt


# -------------------- Rendering -------------------- #


def look_at(eye, center, up):
    f = center - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    return np.stack([r, -np.cross(r, f), f])  # rows: camera x (right), y (down), z (forward)


def render_points(points, colors, view, size):
    """Z-buffered splats; returns an RGB image with PANEL_BG where nothing was drawn."""
    R, eye, scale, offset = view
    cam = (points - eye) @ R.T
    front = cam[:, 2] > 1e-6
    cam, colors = cam[front], colors[front]
    uv = cam[:, :2] / cam[:, 2:] * scale + offset
    W, H = size
    img = np.empty((H, W, 3))
    img[:] = matplotlib.colors.to_rgb(PANEL_BG)
    far_to_near = np.argsort(-cam[:, 2])
    u, v, col = uv[far_to_near, 0], uv[far_to_near, 1], colors[far_to_near]
    # 2x2 splats; numpy keeps the last write per pixel, so nearer points win
    us = np.concatenate([u, u + 1, u, u + 1]).astype(np.int64)
    vs = np.concatenate([v, v, v + 1, v + 1]).astype(np.int64)
    cs = np.concatenate([col] * 4)
    rank = np.tile(np.arange(len(u)), 4)
    order = np.argsort(rank, kind="stable")
    us, vs, cs = us[order], vs[order], cs[order]
    inside = (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
    img[vs[inside], us[inside]] = cs[inside]
    return img


def frustum_lines(R, c, K, hw, size):
    """Line segments (pairs of world points) of a camera pyramid of depth `size`."""
    H, W = hw
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float64)
    rays = np.stack([(corners[:, 0] - K[0, 2]) / K[0, 0], (corners[:, 1] - K[1, 2]) / K[1, 1], np.ones(4)], -1) * size
    world = rays @ R.T + c
    segs = [(c, w) for w in world] + [(world[i], world[(i + 1) % 4]) for i in range(4)]
    return segs


def draw_cameras(ax, view, cams_idx, R_all, c_all, K_all, hw, size, color):
    R_v, eye, scale, offset = view
    for i in cams_idx:
        for a, b in frustum_lines(R_all[i], c_all[i], K_all[i], hw, size):
            pa, pb = (np.stack([a, b]) - eye) @ R_v.T
            if pa[2] <= 0 or pb[2] <= 0:
                continue
            ua = pa[:2] / pa[2] * scale + offset
            ub = pb[:2] / pb[2] * scale + offset
            ax.plot([ua[0], ub[0]], [ua[1], ub[1]], color=color, lw=1.1, solid_capstyle="round")


def render_scene(model, dataset, scene, built, metrics, out, elev=35.0):
    clouds, cams = built["clouds"], built["cams"]
    gt = clouds["Ground truth"]
    gt_c2w_R, gt_c = cams["gt"]
    up = -gt_c2w_R[:, :, 1].mean(0)
    up /= np.linalg.norm(up)
    center = np.median(gt, axis=0)
    view_dir = gt_c2w_R[:, :, 2].mean(0)
    view_dir -= (view_dir @ up) * up
    if np.linalg.norm(view_dir) < 1e-3:
        view_dir = np.cross(up, [1.0, 0, 0])
    view_dir /= np.linalg.norm(view_dir)
    radius = 2.2 * np.percentile(np.linalg.norm(gt - center, axis=1), 90)
    e = np.radians(elev)
    eye = center - radius * (np.cos(e) * view_dir - np.sin(e) * up)
    R_v = look_at(eye, center, up)
    # Only surfaces the GT cameras saw from our side, which drops near walls and ceilings
    keep = built["rays"] @ R_v[2] > -0.05

    size = (900, 640)
    cam = (gt[keep] - eye) @ R_v.T
    uv = cam[:, :2] / cam[:, 2:]
    lo, hi = np.percentile(uv, 1, axis=0), np.percentile(uv, 99, axis=0)
    scale = 0.84 * min(size[0] / (hi - lo)[0], size[1] / (hi - lo)[1])
    offset = np.array(size) / 2 - scale * (lo + hi) / 2
    view = (R_v, eye, scale, offset)

    N = len(gt_c)
    flagged = np.nonzero(plots.gross_frames(built["saved"]))[0]
    if N <= 25:
        shown = np.arange(N)
    else:
        worst = flagged[np.argsort(-plots.norm(built["saved"]["r_res"][flagged]))][:8]
        shown = np.union1d(np.linspace(0, N - 1, 14).astype(int), worst)
    frustum = 0.12 * built["depth_scale"]
    vmax = TAU[dataset] * 2
    factor, unit = DISPLAY[dataset]
    gt_tree = cKDTree(gt)
    pose_source = {name: rot for name, (_, _, rot, _) in CONFIGS.items()}

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.6))
    for ax, name in zip(axes.flat, PANELS):
        pts = clouds[name][keep]
        if name == "Ground truth":
            img = render_points(pts, built["rgb"][keep], view, size)
            subtitle = f"{N} views"
        else:
            finite = np.isfinite(pts).all(1)
            dist = np.full(len(pts), np.inf)
            dist[finite] = gt_tree.query(pts[finite], workers=-1)[0]
            img = render_points(pts, ERR_CMAP(np.clip(dist / vmax, 0, 1))[:, :3], view, size)
            m = metrics[name]
            subtitle = f"F1 {m['f1'] * 100:.1f}%   pixel error {m['pixel_error_median'] * factor:.1f} {unit}"
        ax.imshow(img)
        ax.set_xlim(0, size[0]), ax.set_ylim(size[1], 0)
        draw_cameras(ax, view, shown, *cams["gt"], built["K_gt"], built["hw"], frustum, GT_CAMERA)
        source = pose_source.get(name, "gt")
        if source != "gt":
            draw_cameras(ax, view, shown, *cams[source], built["K_gt"], built["hw"], frustum, EST_CAMERA)
        ax.set_title(f"{name}\n", fontsize=12, color="#111111")
        ax.text(0.5, 1.012, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#52514e")
        ax.set_xticks([]), ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    label = plots.LABELS[dataset]
    fig.suptitle(f"{label} {plots.short(scene.replace('/', '_'))}: where the reconstruction error comes from", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.935, f"Back-projected depth maps of {model}, coloured by distance to the GT cloud (F1 at {TAU[dataset] * factor:g} {unit}). \"Only\" panels take every other part from the GT; the last two also remove one error type.", ha="left", fontsize=10, color="#52514e")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.11, wspace=0.03, hspace=0.22)
    handles = [plt.Line2D([], [], color=GT_CAMERA, lw=1.5, label="ground-truth cameras"), plt.Line2D([], [], color=EST_CAMERA, lw=1.5, label="estimated cameras")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.01), fontsize=10, frameon=False)
    cax = fig.add_axes([0.38, 0.05, 0.24, 0.016])
    sm = plt.cm.ScalarMappable(cmap=ERR_CMAP, norm=plt.Normalize(0, vmax * factor))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    cb.set_label(f"distance to ground truth ({unit})")
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)


def fig_parts(summary, out):
    """Mean F1 per configuration and dataset, in three blocks."""
    blocks = [
        ("One part predicted, rest GT", ["Depth error only", "Camera error only", "Pose error only", "Intrinsics error only", "Rotation error only", "Camera centre error only"]),
        ("Full prediction with one part fixed", ["Fix depth", "Fix depth-pose scale mismatch", "Fix intrinsics", "Fix poses", "Fix intrinsics and poses", "Fix gross pose errors"]),
        ("Inside one part", ["Depth only: no offset or smooth error", "Depth only: no outliers", "Poses only: gross errors fixed", "Poses only: no outliers", "Poses only: no smooth error", "Poses only: no jitter"]),
    ]
    depth_like = lambda n: "epth" in n  # noqa: E731
    datasets = list(summary)
    names = [n for _, ns in blocks for n in ns]
    gap = 0.8
    y, pos = [], 0.0
    for _, ns in blocks:
        for _ in ns:
            y.append(pos)
            pos -= 1
        pos -= gap
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.3 * len(datasets) + 2.4, 7.3), sharey=True, squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        base = summary[dataset]["DA3-Small"]["f1"] * 100
        f1 = [summary[dataset][n]["f1"] * 100 for n in names]
        ax.barh(y, f1, color=[plots.ORANGE if depth_like(n) else plots.BLUE for n in names], height=0.72)
        ax.axvline(base, color=plots.INK, lw=1.2)
        for yi, v in zip(y, f1):
            ax.text(v + 1.5, yi, f"{v:.0f}", va="center", fontsize=8, color=plots.INK2)
        ax.set_xlim(0, 108)
        unit = DISPLAY[dataset]
        ax.set_title(f"{plots.LABELS[dataset]}: {base:.1f}", loc="left")
        ax.set_xlabel(f"F1 at {TAU[dataset] * unit[0]:g} {unit[1]} (%)")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0, 0].set_yticks(y, names)
    top = 0
    for title, ns in blocks:
        axes[0, 0].annotate(title, xy=(0, y[top] + 0.62), xycoords=("axes fraction", "data"), xytext=(-268, 0), textcoords="offset points", fontsize=9, fontweight="bold", color=plots.INK, va="bottom")
        top += len(ns)
    plots.suptitle(fig, "Depth or cameras: what limits the reconstruction?", "Mean F1 over scenes; black line and title: full DA3-Small prediction. Orange involves depth, blue cameras. \"Fix depth\" equals \"camera error only\", \"fix intrinsics and poses\" equals \"depth error only\".")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--datasets", nargs="+", default=analysis.DATASETS)
    parser.add_argument("--render", nargs="*", default=[], help="'auto' (median and worst scene per dataset) or dataset/scene")
    parser.add_argument("--no-metrics", action="store_true", help="only render; reuse reconstruction.json")
    args = parser.parse_args()

    out_dir = os.path.join(REPO_ROOT, "results/failure_modes", args.model)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "reconstruction.json")
    per_scene = json.load(open(json_path))["scenes"] if os.path.exists(json_path) else {}

    if not args.no_metrics:
        for dataset in args.datasets:
            root = os.path.join(REPO_ROOT, "workspace/evaluation", args.model, "model_results", dataset)
            scenes = sorted(os.path.relpath(p, root).split("/unposed/")[0] for p in glob.glob(os.path.join(root, "**/unposed/exports/mini_npz/results.npz"), recursive=True))
            for scene in scenes:
                built = build_clouds(args.model, dataset, scene, STRIDE["metrics"])
                per_scene[f"{dataset}/{scene}"] = scores(built["clouds"], TAU[dataset])[0]
                print(dataset, scene, f"F1 {per_scene[f'{dataset}/{scene}']['DA3-Small']['f1'] * 100:.1f}", flush=True)
        summary = {}
        for dataset in analysis.DATASETS:  # also those computed by earlier runs with other --datasets
            rows = [v for k, v in per_scene.items() if k.split("/")[0] == dataset]
            if rows:
                summary[dataset] = {n: {m: float(np.mean([r[n][m] for r in rows])) for m in rows[0][n]} for n in rows[0]}
        with open(json_path, "w") as f:
            json.dump({"tau": TAU, "summary": summary, "scenes": per_scene}, f, indent=1)
        fig_parts(summary, os.path.join(out_dir, "reconstruction_parts.png"))

    targets = []
    for r in args.render:
        if r == "auto":
            for dataset in args.datasets:
                rows = sorted((v["DA3-Small"]["f1"], k) for k, v in per_scene.items() if k.split("/")[0] == dataset)
                if rows:
                    targets += [rows[len(rows) // 2][1], rows[0][1]]
        else:
            targets.append(r)
    for target in dict.fromkeys(targets):
        dataset, scene = target.split("/", 1)
        built = build_clouds(args.model, dataset, scene, STRIDE["render"])
        metrics = per_scene.get(target) or scores(built["clouds"], TAU[dataset])[0]
        out = os.path.join(out_dir, f"3d_{dataset}_{scene.replace('/', '_')}.png")
        render_scene(args.model, dataset, scene, built, metrics, out)
        print(out, flush=True)


if __name__ == "__main__":
    main()
