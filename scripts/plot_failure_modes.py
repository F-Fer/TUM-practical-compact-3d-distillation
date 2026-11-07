"""Figures for the error decomposition of scripts/analyse_failure_modes.py.

Usage:
    uv run scripts/plot_failure_modes.py DA3-SMALL [--only pose_traces depth_gallery ...]

Writes PNGs and summary.json to results/failure_modes/<model>/.
"""

import argparse
import glob
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import analyse_failure_modes as analysis  # noqa: E402

REPO_ROOT = analysis.REPO_ROOT
DATASETS = analysis.DATASETS
LABELS = {"hiroom": "HiRoom", "7scenes": "7-Scenes", "scannetpp": "ScanNet++", "dtu": "DTU", "eth3d": "ETH3D"}
UNIT = {"hiroom": "synthetic", "7scenes": "Kinect", "scannetpp": "laser mesh", "dtu": "MVS mesh", "eth3d": "laser scan"}

GROSS_ROTATION_DEG, GROSS_TRANSLATION = 15.0, 0.25  # 25% of the median scene depth

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
ROTATION, TRANSLATION = BLUE, ORANGE
CRITICAL = "#d03b3b"
DATASET_COLOR = dict(zip(DATASETS, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]))
# Parts of an error, in the same order and colors everywhere
PART_COLORS = {"frame offset": "#86b6ef", "smooth": BLUE, "fine noise": AQUA, "outliers": CRITICAL}

DIVERGING = LinearSegmentedColormap.from_list("error", ["#184f95", "#3987e5", "#f0efec", "#e66767", "#a32020"])
SEQUENTIAL = LinearSegmentedColormap.from_list("magnitude", ["#f0f5fb", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])
DEPTH = LinearSegmentedColormap.from_list("depth", ["#0d366b", "#256abf", "#86b6ef", "#f0efec"])
DIVERGING.set_bad("#d8d7d0")
DEPTH.set_bad("#d8d7d0")
SEQUENTIAL.set_bad(SURFACE)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "text.color": INK,
        "font.size": 9,
        "figure.dpi": 150,
        "savefig.dpi": 150,
    }
)


def load(model):
    scenes = {}
    for dataset in DATASETS:
        files = sorted(glob.glob(os.path.join(REPO_ROOT, "workspace/failure_modes", model, dataset, "*.npz")))
        if files:
            scenes[dataset] = {os.path.basename(f)[:-4]: dict(np.load(f)) for f in files}
    return scenes


def suptitle(fig, title, subtitle=None):
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", y=0.995)
    if subtitle:
        fig.text(0.01, 0.995 - 0.27 / fig.get_figheight(), subtitle, ha="left", va="top", color=INK2, fontsize=9)


def norm(x):
    return np.linalg.norm(x, axis=-1)


def gross_frames(s):
    """Frames that are far off in absolute terms, as opposed to a spike over small local noise."""
    return (norm(s["r_res"]) > GROSS_ROTATION_DEG) | (norm(s["t_res"]) > GROSS_TRANSLATION)


def by_quality(scenes):
    """Scene names of the best, median and worst AUC@30."""
    names = sorted(scenes, key=lambda n: scenes[n]["cf_auc"][0, 1])
    return names[-1], names[len(names) // 2], names[0]


def short(name):
    parts = name.split("_")
    return parts[1] if parts[0].isdigit() else name


# -------------------- Pose -------------------- #


def fig_trajectories(data, out):
    fig, axes = plt.subplots(len(data), 3, figsize=(10, 3.2 * len(data)), squeeze=False)
    for row, (dataset, scenes) in zip(axes, data.items()):
        for ax, label, name in zip(row, ["best", "median", "worst"], by_quality(scenes)):
            s = scenes[name]
            center = s["c_gt"].mean(0)
            _, _, Vt = np.linalg.svd(s["c_gt"] - center)
            gt, pred = (s["c_gt"] - center) @ Vt[:2].T, (s["c_aligned"] - center) @ Vt[:2].T
            ax.plot(*gt.T, color=AXIS, lw=1, zorder=1)
            ax.plot(*np.moveaxis(np.stack([gt, pred]), -1, 0), color=BLUE, lw=0.7, alpha=0.6, zorder=2)
            ax.scatter(*gt.T, s=9, color=INK2, zorder=3, label="ground truth", linewidths=0)
            ax.scatter(*pred.T, s=9, color=BLUE, zorder=4, label="DA3, aligned", linewidths=0)
            flagged = gross_frames(s)
            ax.scatter(*pred[flagged].T, s=46, facecolors="none", edgecolors=CRITICAL, lw=1.1, zorder=5, label="gross error")
            ax.set_aspect("equal", "datalim")
            ax.grid(False)
            ax.set_xticks([]), ax.set_yticks([])
            ax.spines[["left", "bottom"]].set_visible(False)
            auc3, auc30 = s["cf_auc"][0, :2] * 100
            ax.set_title(f"{LABELS[dataset]}, {label}: {short(name)}", loc="left")
            ax.text(0, 1, f"AUC@3 {auc3:.0f}, AUC@30 {auc30:.0f}, {flagged.sum()}/{len(flagged)} gross", transform=ax.transAxes, fontsize=8, color=INK2, va="top")
    axes[0, 0].legend(loc="lower left", ncols=3, bbox_to_anchor=(0, -0.02), columnspacing=0.8, handletextpad=0.2)
    suptitle(fig, "Camera positions, top-down", "Each line joins a predicted camera to its true position. Gross: over 15° or over 25% of scene depth off. Scenes are the best, median and worst by AUC@30.")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out)


def fig_traces(data, out):
    fig, axes = plt.subplots(len(data), 4, figsize=(13, 2.3 * len(data)), squeeze=False)
    columns = [("r", "median", 1, ROTATION), ("r", "worst", 2, ROTATION), ("t", "median", 1, TRANSLATION), ("t", "worst", 2, TRANSLATION)]
    for row, (dataset, scenes) in zip(axes, data.items()):
        chosen = by_quality(scenes)
        for ax, (kind, label, pick, color) in zip(row, columns):
            s = scenes[chosen[pick]]
            scale = 1 if kind == "r" else 100
            err, smooth = norm(s[f"{kind}_res"]) * scale, norm(s[f"{kind}_smooth"]) * scale
            flags = s[f"{kind}_outlier"]
            x = np.arange(len(err))
            ax.plot(x, err, color=color, lw=1)
            ax.plot(x, smooth, color=INK, lw=1.2)
            ax.scatter(x[flags], err[flags], color=CRITICAL, s=20, zorder=4, linewidths=0)
            ax.set_ylim(bottom=0)
            what = "rotation (°)" if kind == "r" else "translation (% of depth)"
            ax.set_title(f"{LABELS[dataset]} {short(chosen[pick])}, {label}", loc="left", fontsize=9)
            ax.set_ylabel(what)
        row[0].set_xlabel("frame")
    handles = [
        plt.Line2D([], [], color=BLUE, lw=1, label="rotation error"),
        plt.Line2D([], [], color=ORANGE, lw=1, label="translation error"),
        plt.Line2D([], [], color=INK, lw=1.2, label="smooth part"),
        plt.Line2D([], [], color=CRITICAL, marker="o", ms=4, lw=0, label="spike over local noise"),
    ]
    fig.legend(handles=handles, loc="upper right", ncols=4, bbox_to_anchor=(0.995, 1.0))
    suptitle(fig, "Error per camera along the frame sequence", "Flat scatter is jitter, slow waves are structure, isolated spikes are outliers. Y scales differ.")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out)


def fig_pose_qq(data, out):
    """QQ plot of within-scene normalized errors against the Maxwell law of 3D Gaussian noise."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(data), figsize=(3.2 * len(data), 3.6), sharex=True, sharey=True, squeeze=False)
    gaussian = norm(rng.normal(size=(2_000_000, 3)))
    gaussian /= np.median(gaussian)
    for ax, (dataset, scenes) in zip(axes[0], data.items()):
        for kind, color, label in [("r", ROTATION, "rotation"), ("t", TRANSLATION, "translation")]:
            u = np.concatenate([norm(s[f"{kind}_res"]) / np.median(norm(s[f"{kind}_res"])) for s in scenes.values()])
            q = (np.arange(len(u)) + 0.5) / len(u)
            ax.plot(np.quantile(gaussian, q), np.sort(u), color=color, label=label)
        ax.plot([0, 3.5], [0, 3.5], color=MUTED, ls="--", lw=1)
        ax.set_yscale("symlog", linthresh=4)
        ax.set_yticks([0, 1, 2, 3, 4, 10, 40], [0, 1, 2, 3, 4, 10, 40])
        ax.set_xlim(0, 3.2), ax.set_ylim(0, 60)
        ax.set_title(LABELS[dataset], loc="left")
        ax.set_xlabel("expected if Gaussian noise")
    axes[0, 0].set_ylabel("observed error / scene median")
    axes[0, 0].legend(loc="upper left")
    axes[0, 0].text(2.1, 1.6, "Gaussian", color=MUTED, fontsize=8)
    suptitle(fig, "Are pose errors Gaussian noise?", "On the dashed line the errors are 3D Gaussian. Curving up at the right means heavy tails.")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out)


def top_share(x, frac=0.1):
    x = np.sort(x**2)[::-1]
    return x[: max(1, round(frac * len(x)))].sum() / x.sum()


def fig_failure_map(data, out):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(data), figsize=(3.2 * len(data), 3.8), sharex=True, sharey=True, squeeze=False)
    for ax, (dataset, scenes) in zip(axes[0], data.items()):
        N = int(np.median([len(s["t_res"]) for s in scenes.values()]))
        noise = rng.normal(size=(500, N, 3))
        sim_x = [analysis.von_neumann_ratio(n) for n in noise]
        sim_y = [top_share(norm(n)) for n in noise]
        ax.fill_betweenx(
            [np.percentile(sim_y, 1), np.percentile(sim_y, 99)], np.percentile(sim_x, 1), np.percentile(sim_x, 99), color=GRID, zorder=0
        )
        ax.text(np.median(sim_x), np.percentile(sim_y, 1) - 0.015, "white Gaussian", ha="center", va="top", color=MUTED, fontsize=7)
        for kind, color, label in [("r", ROTATION, "rotation"), ("t", TRANSLATION, "translation")]:
            x = [float(s[f"{kind}_vn"]) for s in scenes.values()]
            y = [top_share(norm(s[f"{kind}_res"])) for s in scenes.values()]
            ax.scatter(x, y, color=color, s=22, alpha=0.85, label=label, linewidths=0.5, edgecolors=SURFACE)
        ax.set_title(f"{LABELS[dataset]} ({N} views)", loc="left")
        ax.set_xlabel("smooth  ←  frame-to-frame ratio  →  white")
        ax.set_xlim(0, 1.5), ax.set_ylim(0, 1)
    axes[0, 0].set_ylabel("share of squared error in worst 10% of frames")
    axes[0, 0].legend(loc="lower left", handletextpad=0.1)
    suptitle(fig, "What kind of failure is each scene?", "One dot per scene. Left means slowly varying error, top means a few frames dominate.")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(out)


def fig_counterfactuals(data, out):
    order = ["no jitter", "no smooth part", "no outliers", "perfect rotation", "perfect translation"]
    colors = {"no jitter": AQUA, "no smooth part": BLUE, "no outliers": CRITICAL, "perfect rotation": MUTED, "perfect translation": MUTED}
    fig, axes = plt.subplots(2, len(data), figsize=(3.1 * len(data), 5), sharex="row", squeeze=False)
    for col, (dataset, scenes) in enumerate(data.items()):
        names = list(next(iter(scenes.values()))["cf_names"])
        auc = np.mean([s["cf_auc"] for s in scenes.values()], axis=0) * 100
        for row, (idx, metric) in enumerate([(0, "AUC@3"), (1, "AUC@30")]):
            ax = axes[row, col]
            base = auc[names.index("as predicted"), idx]
            values = [auc[names.index(n), idx] for n in order]
            y = np.arange(len(order))[::-1]
            ax.barh(y, values, color=[colors[n] for n in order], height=0.68)
            ax.axvline(base, color=INK, lw=1.2)
            ax.set_yticks(y, order if col == 0 else [""] * len(order))
            for yi, v in zip(y, values):
                ax.text(v + 1, yi, f"+{v - base:.0f}", va="center", fontsize=8, color=INK2)
            ax.grid(axis="y", visible=False)
            ax.tick_params(axis="y", length=0)
            ax.set_title(f"{LABELS[dataset]}: {base:.1f}" if row == 0 else f"{base:.1f}", loc="left")
            ax.set_xlim(0, 75 if row == 0 else 100)
            ax.set_xlabel(metric)
    suptitle(fig, "What would each fix gain?", "Pose scores if one error type were removed by an oracle. The black line and title show the actual score.")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out)


def fig_pairs(data, out):
    fig, axes = plt.subplots(2, len(data), figsize=(3.0 * len(data), 6.2), squeeze=False)
    for col, (dataset, scenes) in enumerate(data.items()):
        _, median, worst = by_quality(scenes)
        for row, (label, name) in enumerate([("median", median), ("worst", worst)]):
            ax = axes[row, col]
            im = ax.imshow(scenes[name]["rot_pairs"], cmap=SEQUENTIAL, vmin=0, vmax=15, interpolation="nearest")
            ax.grid(False)
            ax.spines[["left", "bottom"]].set_visible(False)
            ax.set_title(f"{LABELS[dataset]} {short(name)}, {label}", loc="left", fontsize=9)
            ax.set_xlabel("frame"), ax.set_ylabel("frame")
    suptitle(fig, "Relative rotation error of every camera pair", "A dark cross is one bad camera. Blocks are groups of cameras that agree with each other. Uniform speckle is jitter.")
    fig.subplots_adjust(top=0.88, left=0.05, right=0.9, hspace=0.3, wspace=0.3)
    fig.colorbar(im, cax=fig.add_axes([0.92, 0.3, 0.012, 0.4]), label="relative rotation error (°)")
    fig.savefig(out)


def fig_correlation_range(data, out):
    """How similar are two cameras' errors, as a function of their true distance?"""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    bins = np.geomspace(0.01, 5, 13)
    for ax, kind, title in [(axes[0], "r", "rotation"), (axes[1], "t", "translation")]:
        for dataset, scenes in data.items():
            d, diff = [], []
            for s in scenes.values():
                i, j = np.triu_indices(len(s["c_gt"]), 1)
                d.append(s["pair_baseline"])
                x = s[f"{kind}_res"]
                diff.append(((x[i] - x[j]) ** 2).sum(1) / np.median((x**2).sum(1)))
            d, diff = np.concatenate(d), np.concatenate(diff)
            idx = np.digitize(d, bins)
            centers, rho = [], []
            for b in range(1, len(bins)):
                if (idx == b).sum() >= 100:
                    centers.append(np.sqrt(bins[b - 1] * bins[b]))
                    rho.append(1 - np.median(diff[idx == b]) / 2)
            ax.plot(centers, rho, color=DATASET_COLOR[dataset], marker="o", ms=4, label=LABELS[dataset])
        ax.axhline(0, color=MUTED, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlim(0.01, 1.2)
        ax.set_xlabel("distance between the two cameras (fraction of scene depth)")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("error similarity (1 = shared, 0 = independent)")
    axes[0].set_ylim(-0.3, 1)
    axes[1].legend(loc="upper right")
    suptitle(fig, "Do nearby cameras share their error?", "Datasets compared at equal camera distance. Values below 0 likely come from the alignment, which centres the errors.")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(out)


def fig_pose_baseline(data, out):
    """Translation-direction error blows up for short baselines even with small absolute errors."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    bins = np.geomspace(0.01, 5, 13)
    for dataset, scenes in data.items():
        d, r, t = [], [], []
        for s in scenes.values():
            i, j = np.triu_indices(len(s["c_gt"]), 1)
            d.append(s["pair_baseline"]), r.append(s["rot_pairs"][i, j]), t.append(s["trans_pairs"][i, j])
        d, r, t = np.concatenate(d), np.concatenate(r), np.concatenate(t)
        idx = np.digitize(d, bins)
        keep = [b for b in range(1, len(bins)) if (idx == b).sum() >= 100]
        centers = [np.sqrt(bins[b - 1] * bins[b]) for b in keep]
        for ax, v in zip(axes, (r, t)):
            ax.plot(centers, [np.median(v[idx == b]) for b in keep], color=DATASET_COLOR[dataset], marker="o", ms=4, label=LABELS[dataset])
    for ax, title in zip(axes, ["relative rotation error", "relative translation direction error"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("distance between the two cameras (fraction of scene depth)")
        ax.set_ylabel("median error (°)")
        ax.axhline(3, color=MUTED, ls="--", lw=1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yticks([1, 2, 3, 5, 10, 20, 50])
        ax.set_title(title, loc="left")
    axes[0].legend(loc="upper left", ncols=2)
    axes[1].text(5, 3.3, "3°", color=MUTED, ha="right", fontsize=8)
    suptitle(fig, "Why AUC@3 is so low", "Median errors sit near or above the 3° threshold at most distances, and translation direction worsens as cameras get closer.")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(out)


def fig_worst_cameras(data, model, out, per_dataset=6):
    fig, axes = plt.subplots(len(data), per_dataset, figsize=(2.4 * per_dataset, 2.25 * len(data)), squeeze=False)
    for row, (dataset, scenes) in zip(axes, data.items()):
        root = os.path.join(REPO_ROOT, "workspace/evaluation", model, "model_results", dataset)
        scene_of = {
            r.replace("/", "_"): r
            for r in (os.path.relpath(os.path.dirname(p), root) for p in glob.glob(os.path.join(root, "**/unposed"), recursive=True))
        }
        frames = sorted(((norm(s["r_res"])[t], norm(s["t_res"])[t], n, t) for n, s in scenes.items() for t in range(len(s["r_res"]))), reverse=True)
        for ax, (r, t_err, name, t) in zip(row, frames[:per_dataset]):
            gt = np.load(os.path.join(root, scene_of[name], "unposed/exports/gt_meta.npz"), allow_pickle=True)
            files = sorted(f if os.path.isabs(f) else os.path.join(REPO_ROOT, f) for f in gt["image_files"].tolist())
            image = cv2.imread(files[t])
            scale = 300 / image.shape[1]
            ax.imshow(cv2.cvtColor(cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            ax.spines[["left", "bottom"]].set_visible(False)
            ax.set_xlabel(f"{r:.0f}°, {t_err * 100:.0f}% of depth", fontsize=8, color=INK)
            ax.set_title(f"{short(name)}, f{t}", loc="left", fontsize=7, fontweight="normal", color=INK2)
        row[0].set_ylabel(LABELS[dataset], fontsize=10, color=INK)
    suptitle(fig, "The cameras with the largest rotation errors", "Each image is one frame, labelled with its rotation and translation error after alignment.")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)


# -------------------- Depth -------------------- #


def fig_depth_gallery(data, model, dataset, out):
    scenes = data[dataset]
    frames = [
        (name, t, s["frame_rms"][t])
        for name, s in scenes.items()
        for t in range(len(s["frame_rms"]))
        if s["frame_valid_frac"][t] > 0.5
    ]
    frames.sort(key=lambda f: f[2])
    picks = [frames[len(frames) // 10], frames[len(frames) // 2], frames[-len(frames) // 20]]
    labels = ["good (10th pct.)", "median", "bad (95th pct.)"]
    fig, axes = plt.subplots(3, 5, figsize=(15, 8.0))
    root = os.path.join(REPO_ROOT, "workspace/evaluation", model, "model_results", dataset)
    scene_dirs = [os.path.relpath(os.path.dirname(p), root) for p in glob.glob(os.path.join(root, "**/unposed"), recursive=True)]
    scene_of = {r.replace("/", "_"): r for r in scene_dirs}
    for row, label, (name, t, rms) in zip(axes, labels, picks):
        d = analysis.load_scene(model, dataset, scene_of[name])
        H, W = d["pred_depth"].shape[1:]
        v = d["valid"][t]
        scene_scale = float(scenes[name]["scene_log_scale"])
        pred = d["pred_depth"][t] * np.exp(-scene_scale)
        e = np.log(np.maximum(d["pred_depth"][t], 1e-6)) - np.log(np.where(v, d["gt_depth"][t], 1)) - scene_scale
        offset, low, high, sigma = analysis.split_depth_error(e, v)
        outliers = v & (np.abs(high) > analysis.DEPTH_OUTLIER_SIGMA * sigma)
        rgb = cv2.cvtColor(cv2.resize(cv2.imread(d["image_files"][t]), (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        gt = np.where(v, d["gt_depth"][t], np.nan)
        lo, hi = np.nanpercentile(gt, [1, 99])

        row[0].imshow(rgb)
        row[1].imshow(gt, cmap=DEPTH, vmin=lo, vmax=hi)
        row[2].imshow(pred, cmap=DEPTH, vmin=lo, vmax=hi)
        im_e = row[3].imshow(np.where(v, e, np.nan), cmap=DIVERGING, vmin=-0.2, vmax=0.2)
        row[4].imshow(np.where(v, high, np.nan), cmap=DIVERGING, vmin=-0.2, vmax=0.2)
        for ax in row:
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            ax.spines[["left", "bottom"]].set_visible(False)
        row[0].set_ylabel(label, fontsize=10, color=INK)
        row[0].text(4, H - 8, f"{short(name)}, frame {t}", color="white", fontsize=7, bbox=dict(fc="black", alpha=0.5, lw=0, pad=1.5))
        row[3].set_xlabel(f"RMS {rms * 100:.1f}%, frame offset {offset * 100:+.1f}%", fontsize=8)
        row[4].set_xlabel(f"σ {sigma * 100:.1f}%, {outliers.sum() / v.sum() * 100:.1f}% of pixels beyond 5σ", fontsize=8)
    for ax, title in zip(axes[0], ["Image", "Ground truth", "DA3, scene scale fixed", "Log depth error", "After removing offset and smooth part"]):
        ax.set_title(title, loc="left")
    suptitle(fig, f"{LABELS[dataset]} ({UNIT[dataset]} ground truth): depth errors", "Frames are the 10th, 50th and 95th percentile by RMS error. Gray has no valid ground truth.")
    fig.subplots_adjust(top=0.9, left=0.03, right=0.92, wspace=0.03, hspace=0.12, bottom=0.03)
    fig.colorbar(im_e, cax=fig.add_axes([0.935, 0.25, 0.01, 0.5]), label="log depth error (0.1 ≈ 10%): blue too close, red too far")
    fig.savefig(out)


def pooled(scenes, key):
    return np.concatenate([s[key] for s in scenes.values()])


def fig_depth_hist(data, out):
    fig, axes = plt.subplots(1, len(data), figsize=(3.2 * len(data), 3.6), sharex=True, sharey=True, squeeze=False)
    bins = np.linspace(-0.5, 0.5, 121)
    centers = (bins[1:] + bins[:-1]) / 2
    for ax, (dataset, scenes) in zip(axes[0], data.items()):
        e = pooled(scenes, "px_high")
        sigma = 1.4826 * np.median(np.abs(e))
        hist, _ = np.histogram(e, bins, density=True)
        ax.fill_between(centers, hist, 1e-4, color="#9ec5f4", lw=0, step="mid")
        ax.step(centers, hist, color=BLUE, where="mid", lw=1, label="DA3")
        gauss = np.exp(-(centers**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
        ax.plot(centers, gauss, color=INK, ls="--", lw=1, label=f"Gaussian, σ {sigma * 100:.1f}%")
        ax.set_yscale("log")
        ax.set_ylim(3e-3, 200), ax.set_xlim(-0.5, 0.5)
        ax.set_title(LABELS[dataset], loc="left")
        ax.set_xlabel("high-frequency log depth error")
        ax.legend(loc="upper right", fontsize=7)
    axes[0, 0].set_ylabel("density (log scale)")
    suptitle(fig, "Depth errors have far heavier tails than Gaussian noise", "A log axis makes a Gaussian look like a parabola. All panels have the frame offset and smooth part removed.")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out)


def fig_depth_parts(data, out):
    parts = list(PART_COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    names = list(data)
    y = np.arange(len(names))[::-1]
    shares = np.array([np.mean([s["energy"] / s["energy"].sum() for s in data[d].values()], axis=0) * 100 for d in names])
    left = np.zeros(len(names))
    for i, part in enumerate(parts):
        axes[0].barh(y, shares[:, i], left=left, color=PART_COLORS[part], height=0.62, label=part, edgecolor=SURFACE, linewidth=1)
        for yi, l, v in zip(y, left, shares[:, i]):
            if v > 6:
                axes[0].text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white" if part != "frame offset" else INK)
        left += shares[:, i]
    axes[0].set_yticks(y, [LABELS[n] for n in names])
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("share of squared depth error (%)")
    axes[0].grid(axis="y", visible=False)
    axes[0].tick_params(axis="y", length=0)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=4, handlelength=1, columnspacing=1)

    steps = ["scene scale fixed", "+ frame offset", "+ smooth part", "+ outliers"]
    axes[1].plot(steps, [np.nan] * 4)
    label_y = {}
    for n in sorted(names, key=lambda n: np.mean([s["inlier_ratio_steps"][-1] for s in data[n].values()])):
        values = np.mean([s["inlier_ratio_steps"] for s in data[n].values()], axis=0) * 100
        axes[1].plot(steps, values, color=DATASET_COLOR[n], marker="o", ms=5)
        label_y[n] = max(values[-1], max(label_y.values(), default=0) + 4.5)  # keep the end labels apart
        axes[1].text(3.08, label_y[n], LABELS[n], va="center", fontsize=8, color=INK2)
    axes[1].set_ylabel("pixels within 3% of ground truth (%)")
    axes[1].set_xlim(-0.2, 3.7)
    axes[1].set_ylim(30, 106)
    axes[1].set_xlabel("error types removed, left to right")
    suptitle(fig, "Depth: outliers carry the squared error, smooth errors affect the most pixels")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out)


def rank_within_scene(scenes, key, n=10):
    out = []
    for s in scenes.values():
        order = np.argsort(np.argsort(s[key]))
        out.append(np.minimum(order * n // len(order), n - 1))
    return np.concatenate(out)


def auroc(score, label):
    ranks = np.argsort(np.argsort(score)) + 1.0
    pos = label.astype(bool)
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def fig_depth_where(data, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    edges = np.array([0, 1, 2, 3, 5, 9, 17, 33, 65, 1000])
    aurocs = {}
    for dataset, scenes in data.items():
        color = DATASET_COLOR[dataset]
        high, out_flag = np.abs(pooled(scenes, "px_high")), pooled(scenes, "px_outlier") > 0
        dist = pooled(scenes, "px_edge_dist")
        idx = np.digitize(dist, edges) - 1
        x = [edges[i] + (edges[i + 1] - edges[i] - 1) / 2 + 0.5 for i in range(len(edges) - 1)]
        x[-1] = 90
        axes[0].plot(x, [out_flag[idx == i].mean() * 100 if (idx == i).sum() else np.nan for i in range(len(x))], color=color, marker="o", ms=4, label=LABELS[dataset])

        conf_rank = rank_within_scene(scenes, "px_conf")
        axes[1].plot(np.arange(10) * 10 + 5, [out_flag[conf_rank == i].mean() * 100 for i in range(10)], color=color, marker="o", ms=4)
        aurocs[dataset] = 1 - auroc(pooled(scenes, "px_conf"), out_flag)

        depth_rank = rank_within_scene(scenes, "px_log_gt")
        axes[2].plot(np.arange(10) * 10 + 5, [out_flag[depth_rank == i].mean() * 100 for i in range(10)], color=color, marker="o", ms=4)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("distance to the nearest depth edge (pixels)")
    axes[0].set_ylabel("outlier pixels (%)")
    axes[0].set_title("Near depth edges?", loc="left")
    axes[0].legend(loc="upper right")
    axes[1].set_xlabel("confidence percentile within scene")
    axes[1].set_title("Flagged by DA3's confidence?", loc="left")
    axes[2].set_xlabel("depth percentile within scene (near → far)")
    axes[2].set_title("Far from the camera?", loc="left")
    for ax in axes:
        ax.set_ylim(bottom=0)
    suptitle(fig, "Where are the depth outliers?", "Outlier: beyond 5σ of the frame's fine noise. Pixels are sampled uniformly from valid ground truth.")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(out)
    return aurocs


def fig_depth_spectrum(data, out):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    freq = (np.linspace(0, 0.5, 65)[1:] + np.linspace(0, 0.5, 65)[:-1]) / 2
    for dataset, scenes in data.items():
        spectrum = np.nanmean([s["spectrum"] for s in scenes.values()], axis=0)
        ax.plot(freq[1:], spectrum[1:], color=DATASET_COLOR[dataset], label=LABELS[dataset])
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.axvline(1 / 14, color=MUTED, ls="--", lw=1)
    ax.text(1 / 14 * 1.04, ax.get_ylim()[1] * 0.5, "1 ViT patch\n(14 px)", color=MUTED, fontsize=8, va="top")
    ax.set_xlabel("spatial frequency (cycles per pixel)")
    ax.set_ylabel("error power")
    ax.legend(loc="lower left")
    suptitle(fig, "Depth error by spatial scale", "Pixel-wise noise would be flat. A falling curve means most error energy sits at large scales. Outliers are excluded.")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out)


# -------------------- Shared -------------------- #


def fig_concentration(data, out):
    """Share of squared error held by the worst x% of frames or pixels."""
    rng = np.random.default_rng(0)
    panels = [("Camera rotation", "frames", lambda s: norm(s["r_res"]), 3), ("Camera translation", "frames", lambda s: norm(s["t_res"]), 3), ("Depth", "pixels", lambda s: s["px_e_frame"], 1)]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
    for ax, (title, unit, get, dof) in zip(axes, panels):
        for dataset, scenes in data.items():
            x = np.concatenate([get(s) / np.median(np.abs(get(s))) for s in scenes.values()]).astype(np.float64) ** 2
            x = np.sort(x)[::-1]
            frac = (np.arange(len(x)) + 1) / len(x)
            ax.plot(frac * 100, np.cumsum(x) / x.sum() * 100, color=DATASET_COLOR[dataset], label=LABELS[dataset])
        g = np.sort((rng.normal(size=(1_000_000, dof)) ** 2).sum(1))[::-1]
        ax.plot((np.arange(len(g)) + 1) / len(g) * 100, np.cumsum(g) / g.sum() * 100, color=INK, ls="--", lw=1, label="Gaussian noise")
        ax.set_xscale("log")
        ax.set_xlim(0.1 if unit == "pixels" else 1, 100)
        ax.set_xticks([1, 10, 100] if unit == "frames" else [0.1, 1, 10, 100], [f"{v:g}%" for v in ([1, 10, 100] if unit == "frames" else [0.1, 1, 10, 100])])
        ax.set_ylim(0, 100)
        ax.set_title(title, loc="left")
        ax.set_xlabel(f"worst x% of {unit}")
    axes[0].set_ylabel("share of squared error (%)")
    axes[0].legend(loc="lower right")
    suptitle(fig, "How concentrated is the error?", "Errors are normalized per scene. Above the dashed line, a few elements hold more of the error than Gaussian noise would.")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(out)


def fig_pose_vs_depth(data, out):
    fig, axes = plt.subplots(1, len(data), figsize=(3.2 * len(data), 3.4), sharex=True, sharey=True, squeeze=False)
    rho = {}
    for ax, (dataset, scenes) in zip(axes[0], data.items()):
        r, d = [], []
        for s in scenes.values():
            keep = np.isfinite(s["frame_rms"])
            r.append(norm(s["r_res"])[keep] / np.median(norm(s["r_res"])))
            d.append(s["frame_rms"][keep] / np.median(s["frame_rms"][keep]))
        r, d = np.concatenate(r), np.concatenate(d)
        rho[dataset] = float(spearmanr(r, d).statistic)
        ax.scatter(r, d, s=5, color=BLUE, alpha=0.3, linewidths=0)
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlim(0.03, 60)
        ax.set_title(f"{LABELS[dataset]}, Spearman {rho[dataset]:.2f}", loc="left")
        ax.set_xlabel("rotation error / scene median")
    axes[0, 0].set_ylabel("depth RMS / scene median")
    suptitle(fig, "Do frames with a bad pose also have bad depth?", "One dot per frame.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out)
    return rho


def summary(data):
    out = {}
    for dataset, scenes in data.items():
        S = list(scenes.values())
        names = list(S[0]["cf_names"])
        r, t = np.concatenate([norm(s["r_res"]) for s in S]), np.concatenate([norm(s["t_res"]) for s in S])
        ro, to = np.concatenate([s["r_outlier"] for s in S]), np.concatenate([s["t_outlier"] for s in S])
        px = pooled(scenes, "px_e")
        high = pooled(scenes, "px_high")
        out[dataset] = {
            "scenes": len(S),
            "views_median": int(np.median([len(s["r_res"]) for s in S])),
            "auc_bench": dict(zip(names, (np.mean([s["cf_auc"][:, :2] for s in S], 0) * 100).round(2).tolist())),
            "auc_gauge_invariant": dict(zip(names, (np.mean([s["cf_auc"][:, 2:] for s in S], 0) * 100).round(2).tolist())),
            "rotation_deg_p50_p90_p99_max": np.percentile(r, [50, 90, 99, 100]).round(2).tolist(),
            "translation_pct_depth_p50_p90_p99_max": (np.percentile(t, [50, 90, 99, 100]) * 100).round(2).tolist(),
            "rotation_outlier_frames_pct": round(float(ro.mean() * 100), 1),
            "translation_outlier_frames_pct": round(float(to.mean() * 100), 1),
            "rotation_outlier_energy_pct": round(float((r[ro] ** 2).sum() / (r**2).sum() * 100), 1),
            "translation_outlier_energy_pct": round(float((t[to] ** 2).sum() / (t**2).sum() * 100), 1),
            "rotation_von_neumann_median": round(float(np.median([s["r_vn"] for s in S])), 2),
            "translation_von_neumann_median": round(float(np.median([s["t_vn"] for s in S])), 2),
            "depth_error_pct_p50_p90_p99": (np.percentile(np.abs(px), [50, 90, 99]) * 100).round(2).tolist(),
            "depth_pixels_beyond_10pct": round(float((np.abs(px) > 0.1).mean() * 100), 1),
            "depth_fine_noise_sigma_pct": round(float(1.4826 * np.median(np.abs(high)) * 100), 2),
            "depth_outlier_pixels_pct": round(float((pooled(scenes, "px_outlier") > 0).mean() * 100), 1),
            "depth_energy_pct": dict(zip(PART_COLORS, (np.mean([s["energy"] / s["energy"].sum() for s in S], 0) * 100).round(1).tolist())),
            "depth_within_3pct_steps": (np.mean([s["inlier_ratio_steps"] for s in S], 0) * 100).round(1).tolist(),
            "gross_frames_pct": round(float(np.concatenate([gross_frames(s) for s in S]).mean() * 100), 1),
            "gross_frames_pct_per_scene_sorted": np.round(sorted((gross_frames(s).mean() * 100 for s in S), reverse=True), 0).tolist(),
            "gross_frames_energy_pct_rotation": round(float(sum((norm(s["r_res"])[gross_frames(s)] ** 2).sum() for s in S) / (r**2).sum() * 100), 1),
            "gross_frames_energy_pct_translation": round(float(sum((norm(s["t_res"])[gross_frames(s)] ** 2).sum() for s in S) / (t**2).sum() * 100), 1),
            "rotation_translation_spearman_per_frame": round(float(spearmanr(r, t).statistic), 2),
            "depth_outliers_within_8px_of_edge_pct": round(float((pooled(scenes, "px_edge_dist")[pooled(scenes, "px_outlier") > 0] <= 8).mean() * 100), 1),
            "depth_pixels_within_8px_of_edge_pct": round(float((pooled(scenes, "px_edge_dist") <= 8).mean() * 100), 1),
            "depth_outlier_abs_error_pct_p10_p50_p90": (np.percentile(np.abs(high[pooled(scenes, "px_outlier") > 0]), [10, 50, 90]) * 100).round(1).tolist(),
            "depth_outliers_in_regions_20px_pct": round(float(sum(s["outlier_pixels"][1] for s in S) / sum(s["outlier_pixels"][0] for s in S) * 100), 1),
            "depth_outliers_in_regions_20px_if_random_pct": round(float(sum(s["outlier_pixels"][2] for s in S) / sum(s["outlier_pixels"][0] for s in S) * 100), 1),
            "pose_vs_depth_scale_ratio_p10_p90": np.percentile([float(s["sim3_scale"]) * np.exp(float(s["scene_log_scale"])) for s in S], [10, 90]).round(3).tolist(),
            "pose_vs_depth_scale_ratio_median": round(float(np.median([float(s["sim3_scale"]) * np.exp(float(s["scene_log_scale"])) for s in S])), 3),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--only", nargs="*", help="figure names to (re)draw; default all")
    args = parser.parse_args()
    data = load(args.model)
    out_dir = os.path.join(REPO_ROOT, "results/failure_modes", args.model)
    os.makedirs(out_dir, exist_ok=True)
    path = lambda name: os.path.join(out_dir, f"{name}.png")  # noqa: E731
    stats = summary(data)

    figures = {
        "pose_concentration": lambda n: fig_concentration(data, path(n)),
        "pose_trajectories": lambda n: fig_trajectories(data, path(n)),
        "pose_traces": lambda n: fig_traces(data, path(n)),
        "pose_qq": lambda n: fig_pose_qq(data, path(n)),
        "pose_failure_map": lambda n: fig_failure_map(data, path(n)),
        "pose_counterfactuals": lambda n: fig_counterfactuals(data, path(n)),
        "pose_pairs": lambda n: fig_pairs(data, path(n)),
        "pose_worst_cameras": lambda n: fig_worst_cameras(data, args.model, path(n)),
        "pose_correlation_range": lambda n: fig_correlation_range(data, path(n)),
        "pose_baseline": lambda n: fig_pose_baseline(data, path(n)),
        "depth_hist": lambda n: fig_depth_hist(data, path(n)),
        "depth_parts": lambda n: fig_depth_parts(data, path(n)),
        "depth_where": lambda n: stats.setdefault("_conf_auroc_for_outliers", fig_depth_where(data, path(n))),
        "depth_spectrum": lambda n: fig_depth_spectrum(data, path(n)),
        "pose_vs_depth": lambda n: stats.setdefault("_pose_depth_spearman", fig_pose_vs_depth(data, path(n))),
    }
    figures.update({f"depth_gallery_{d}": (lambda n, d=d: fig_depth_gallery(data, args.model, d, path(n))) for d in data})
    for name, draw in figures.items():
        if not args.only or name in args.only:
            draw(name)
            plt.close("all")
            print(name, flush=True)

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
