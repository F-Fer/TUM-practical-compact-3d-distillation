"""Figures of results/pba/REPORT3.md: the speed-up of the photometric BA and what limits long runs (DA3-SMALL).

Usage: .venv/bin/python scripts/plot_pba_speed.py <out dir> [pixels] [gt] [sampling] [gtmodel] [compute]

pixels:   AUC@3 along the optimisation for several pixel counts (runs pm_L_M*, LM to convergence; pm_D_M*, long Adam)
gt:       AUC@3 along LM started at the ground-truth poses for two pixel counts (runs pm_B_gt*)
sampling: AUC@3 along LM for gradient-weighted, uniform and stratified pixels (runs pm_L_*)
gtmodel:  AUC@3 along LM from the ground truth with different depth and camera models (runs pm_B_*)
compute:  quality against time per scene on the RTX 5090
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO_ROOT, "workspace/tto/DA3-SMALL")
DATASETS = [("hiroom", "HiRoom"), ("7scenes", "7-Scenes"), ("scannetpp", "ScanNet++"), ("dtu64", "DTU-64"), ("eth3d", "ETH3D")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURFACE = "#fcfcfb"
# one hue, light to dark: pixel counts are ordered magnitudes (sequential ramp of the blue categorical slot)
RAMP = ["#a9c9ee", "#6ba3e4", "#2a78d6", "#14508f"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def tracks(run, dataset):
    p = os.path.join(ROOT, run, "metric_results", f"{dataset}_pba_track.jsonl")
    if not os.path.exists(p):
        return {}
    return {json.loads(line)["scene"]: json.loads(line)["track"] for line in open(p)}


def curve(run, dataset, scenes, key="auc03", start=0):
    t = tracks(run, dataset)
    its = sorted({r["it"] for s in scenes for r in t.get(s, [])})
    xs, ys = [], []
    for it in its:
        if it < start:
            continue
        v = [r[key] for s in scenes for r in t.get(s, []) if r["it"] == it]
        if len(v) == len(scenes):
            xs.append(it - start)
            ys.append(100 * np.mean(v))
    return xs, ys


def common(runs, dataset):
    sets = [set(tracks(r, dataset)) for r in runs]
    sets = [s for s in sets if s]
    return sorted(set.intersection(*sets)) if sets else []


def fig_pixels(out):
    lm = [("pm_L_M1024", "1,024"), ("pm_L_M4096", "4,096"), ("pm_L_M16384", "16,384"), ("pm_L_M65536", "65,536")]
    adam = [("pm_D_M1024", "1,024"), ("pm_D_M4096", "4,096"), ("pm_D_M16384", "16,384")]
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.2), facecolor=SURFACE, sharex="row")
    for col, (d, name) in enumerate(DATASETS):
        ax = axes[0, col]
        style(ax)
        runs = [r for r, _ in lm if tracks(r, d)]
        scenes = common(runs, d)
        for (r, lab), c in zip(lm, RAMP):
            if r in runs and scenes:
                x, y = curve(r, d, scenes, start=200)
                ax.plot(x, y, color=c, linewidth=2, label=lab)
        ax.set_title(f"{name} ({len(scenes)})", fontsize=10, color=INK, loc="left")
        if col == 0:
            ax.set_ylabel("AUC@3, LM to convergence", fontsize=9, color=MUTED)
        ax.set_xlabel("LM iterations after Adam 4×50", fontsize=8, color=MUTED)
        ax = axes[1, col]
        style(ax)
        runs = [r for r, _ in adam if tracks(r, d)]
        scenes = common(runs, d)
        for (r, lab), c in zip(adam, RAMP):
            if r in runs and scenes:
                x, y = curve(r, d, scenes)
                ax.plot(x, y, color=c, linewidth=2, label=lab)
        if col == 0:
            ax.set_ylabel("AUC@3, long Adam", fontsize=9, color=MUTED)
        ax.set_xlabel("Adam iterations (4 levels × 800)", fontsize=8, color=MUTED)
        ax.set_title(f"{name} ({len(scenes)})", fontsize=10, color=INK, loc="left") if scenes else None
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, title="source pixels per frame and iteration", loc="upper right", ncol=4, frameon=False, fontsize=9, title_fontsize=9, labelcolor=MUTED)
    fig.suptitle("64 times more pixels only stabilise HiRoom; on the other datasets the curves coincide", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out, "pixels_drift.png")
    fig.savefig(path, dpi=160)
    print(path)


def fig_gt(out):
    runs = [("pm_B_gt4096", "4,096 px"), ("pm_B_gt16384", "16,384 px")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), facecolor=SURFACE, sharey=True)
    shown = [d for d in DATASETS if d[0] in ("hiroom", "dtu64", "eth3d")]
    for ax, (d, name) in zip(axes, shown):
        style(ax)
        scenes = common([r for r, _ in runs], d)
        for (r, lab), c in zip(runs, RAMP[1:]):
            x, y = curve(r, d, scenes)
            ax.plot(x, y, color=c, linewidth=2, label=lab)
        ax.set_title(f"{name} ({len(scenes)})", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("LM iterations from the ground-truth poses", fontsize=8, color=MUTED)
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("AUC@3", fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=MUTED)
    fig.suptitle("Started at the true poses, the optimiser walks away: the objective's minimum is elsewhere", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(out, "gt_walk.png")
    fig.savefig(path, dpi=160)
    print(path)


def fig_sampling(out):
    """Where the pixels are drawn from, LM to convergence at 4,096 pixels."""
    runs = [("pm_L_M4096", "gradient-weighted (before)", CAT[0]), ("pm_L_G0", "uniform", CAT[1]), ("pm_L_ST16", "stratified, 16 cells", CAT[2])]
    fig, axes = plt.subplots(1, 5, figsize=(13, 3), facecolor=SURFACE)
    for ax, (d, name) in zip(axes, DATASETS):
        style(ax)
        scenes = common([r for r, _, _ in runs], d)
        for r, lab, c in runs:
            x, y = curve(r, d, scenes, start=200)
            ax.plot(x, y, color=c, linewidth=2, label=lab)
        ax.set_title(f"{name} ({len(scenes)})", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("LM iterations after Adam 4×50", fontsize=8, color=MUTED)
    axes[0].set_ylabel("AUC@3", fontsize=9, color=MUTED)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=3, frameon=False, fontsize=9, labelcolor=MUTED)
    fig.suptitle("Spatial coverage does what pixel count does not", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = os.path.join(out, "sampling_drift.png")
    fig.savefig(path, dpi=160)
    print(path)


def fig_gt_model(out):
    """What keeps the optimiser near the true poses: the depth and camera model."""
    runs = [("pm_B_nogrid", "depth frozen", CAT[1]), ("pm_B_gt4096", "8-cell depth grid (before)", CAT[0]), ("pm_B_g16", "16-cell depth grid", CAT[2]), ("pm_B_rad", "8 cells + radial distortion", "#eda100")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), facecolor=SURFACE, sharey=True)
    shown = [d for d in DATASETS if d[0] in ("hiroom", "dtu64", "eth3d")]
    for ax, (d, name) in zip(axes, shown):
        style(ax)
        scenes = common([r for r, _, _ in runs], d)
        for r, lab, c in runs:
            x, y = curve(r, d, scenes)
            ax.plot(x, y, color=c, linewidth=2, label=lab)
        ax.set_title(f"{name} ({len(scenes)})", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("LM iterations from the ground-truth poses", fontsize=8, color=MUTED)
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("AUC@3", fontsize=9, color=MUTED)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=MUTED)
    fig.suptitle("The walk is model error: more freedom in depth and lens keeps the poses closer to the truth", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    path = os.path.join(out, "gt_model.png")
    fig.savefig(path, dpi=160)
    print(path)


POSE = [d for d, _ in DATASETS]
RECON = ["hiroom", "7scenes", "scannetpp"]


def metrics(run):
    """AUC@3 and AUC@30 (mean of the five datasets), F1 (mean of three) and the mean time per scene, without the first
    scene of each dataset (one-off warm-up of a new process)."""

    def load(name):
        p = os.path.join(ROOT, run, "metric_results", f"{name}.json")
        if not os.path.exists(p):
            return {}
        d = json.load(open(p))
        d.pop("mean", None)
        return d

    a, b, t = [], [], []
    for d in POSE:
        x = load(f"{d}_pba")
        if not x:
            return None
        a.append(np.mean([v["auc03"] for v in x.values()]))
        b.append(np.mean([v["auc30"] for v in x.values()]))
        v = [v["opt_s"] for v in x.values()]
        t += v[1:] if len(v) > 1 else v
    f = []
    for d in RECON:
        x = load(f"{d}_recon_unposed")
        if x:
            f.append(np.mean([v["fscore"] for v in x.values()]))
    return 100 * np.mean(a), 100 * np.mean(b), 100 * np.mean(f) if len(f) == len(RECON) else None, float(np.mean(t))


def mean_seeds(runs):
    m = [q for q in (metrics(r) for r in runs) if q]
    return tuple(np.mean([x[i] for x in m]) if all(x[i] is not None for x in m) else None for i in range(4)) if m else None


def fig_compute(out):
    """Quality against the time of the optimisation on the RTX 5090: the earlier Adam-only family and hybrid, the new runs."""
    adam = ["fast_it25", "fast_it50", "fast_it200", "fast_it400", "fast_it800"]
    # name, runs (seeds), colour, label offset, alignment, run with the F1 if the others have none
    points = [
        ("before: Adam 4×50 + LM 5", ["pm_G_I0"], CAT[1], (8, -3), "left", None),
        ("Fast", ["pm_GF_T25_N5_M1024"], CAT[0], (0, -15), "center", None),
        ("recommended", ["pm_GE_g10_s0", "pm_GE_g10_s1", "pm_GE_g10_s2"], CAT[2], (4, 9), "right", "pm_Z_g10stop_s0"),
        ("feature-metric, before", ["pm_GV_old"], CAT[1], (8, -3), "left", "polish50"),
        ("feature-metric, new", ["pm_GX_best"], CAT[2], (-4, 9), "left", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), facecolor=SURFACE)
    titles = ("AUC@3 (mean of 5 datasets)", "AUC@30", "F1 (HiRoom, 7-Scenes, ScanNet++)")
    a = [m for m in (metrics(r) for r in adam) if m]
    xs = [m[3] for m in a]
    for i, ax in enumerate(axes):
        vals = [(m[3], m[i]) for m in a if m[i] is not None]
        ax.plot(*zip(*vals), color="#a8a7a1", linewidth=2, marker="o", ms=4, label="Adam only, before (4×25 to 4×800)", zorder=2)
    for name, runs, c, off, ha, f1run in points:
        q = mean_seeds(runs)
        if q is None:
            print(f"skip {name}")
            continue
        if f1run:
            q = (q[0], q[1], metrics(f1run)[2], q[3])
        xs.append(q[3])
        for j, (ax, v) in enumerate(zip(axes, q[:3])):
            if v is not None:
                ax.plot([q[3]], [v], marker="o", ms=9, color=c, mec=SURFACE, mew=1.5, ls="none", zorder=3)
                ax.annotate(name, (q[3], v), textcoords="offset points", xytext=off, ha=ha, fontsize=8, color=MUTED)
    ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    for ax, title in zip(axes, titles):
        style(ax)
        ax.set_xscale("log")
        lo, hi = min(xs) * 0.4, max(xs) * 1.4
        ax.set_xlim(lo, hi)
        ax.set_xticks([t for t in ticks if lo <= t <= hi])
        ax.set_xticklabels([f"{t:g}" for t in ticks if lo <= t <= hi])
        ax.minorticks_off()
        ax.margins(y=0.12)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("PBA time per scene (s), RTX 5090", fontsize=8, color=MUTED)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", frameon=False, fontsize=9, labelcolor=MUTED)
    fig.suptitle("Dev subset of DA3-SMALL: orange is before, blue and green are now", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(out, "compute_gpu.png")
    fig.savefig(path, dpi=160)
    print(path)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    which = sys.argv[2:] or ["pixels", "gt", "sampling", "gtmodel"]
    if "compute" in which:
        fig_compute(out)
    if "sampling" in which:
        fig_sampling(out)
    if "gtmodel" in which:
        fig_gt_model(out)
    if "pixels" in which:
        fig_pixels(out)
    if "gt" in which:
        fig_gt(out)


if __name__ == "__main__":
    main()
