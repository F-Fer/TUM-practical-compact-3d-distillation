"""Figures of results/pba/REPORT4.md: pose AUC@3 over the Adam iterations T for the three forms of the brightness (DA3-SMALL dev).

Usage: .venv/bin/python scripts/plot_pba_brightness.py results/pba/figures4

Reads workspace/tto/DA3-SMALL/<run>/metric_results/<dataset>_pba.json; a curve point is the mean over the seeds found.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402
import numpy as np  # noqa: E402

ROOT = "workspace/tto/DA3-SMALL"
SETS = ("hiroom", "7scenes", "scannetpp", "dtu64", "eth3d")
TS = (25, 50, 200, 400, 800)
# per form: the runs of each T (all seeds), colour, label
CURVES = {
    "orig": ({25: ["rec_it25"], 50: ["recommended", "pm_GE_g10_s1", "pm_GE_g10_s2"], 200: ["rec_it200", "x1_t200_s1", "x1_t200_s2"], 400: ["rec_it400"],
              800: ["rec_it800", "x1_base_s1", "x1_base_s2"]}, "#c0392b", "brightness as before: $I_j e^{a_j} + b_j - I_i e^{a_i} - b_i$"),
    "none": ({25: ["x1b_nobright_t25_s0"], 50: ["x1b_nobright_t50_s0"], 200: ["x1b_nobright_t200_s0"], 400: ["x1b_nobright_t400_s0"],
              800: ["x1_nobright_s0", "x1_nobright_s1", "x1_nobright_s2"]}, "#7f8c8d", "no brightness"),
    "rel": ({T: [f"x1d_rel_t{T}_s0", f"x5_t{T}_s1", f"x5_t{T}_s2"] for T in TS}, "#1f6fb2", "relative: $I_j e^{a_j - a_i} + b_j - b_i - I_i$"),
}


def metric(run, key="auc03", sets=SETS):
    vals = []
    for d in sets:
        p = f"{ROOT}/{run}/metric_results/{d}_pba.json"
        if not os.path.exists(p):
            return None
        j = json.load(open(p))
        n = len(j) - 1
        if n < {"hiroom": 10, "7scenes": 2, "scannetpp": 3, "dtu64": 4, "eth3d": 3}[d]:
            return None
        vals.append(100 * j["mean"][key])
    return float(np.mean(vals))


def curve(runs_by_t, key, sets):
    xs, ys, lo, hi, ns = [], [], [], [], []
    for T in TS:
        v = [m for r in runs_by_t.get(T, []) if (m := metric(r, key, sets)) is not None]
        if v:
            xs.append(T), ys.append(np.mean(v)), lo.append(min(v)), hi.append(max(v)), ns.append(len(v))
    return np.array(xs), np.array(ys), np.array(lo), np.array(hi), ns


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "results/pba/figures4"
    os.makedirs(out, exist_ok=True)
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), constrained_layout=True)
    panels = (("auc03", SETS, "pose AUC@3, mean of 5 datasets"), ("auc03", ("dtu64",), "pose AUC@3, DTU-64"), ("auc30", SETS, "pose AUC@30, mean of 5 datasets"))
    for ax, (key, sets, title) in zip(axes, panels):
        for name, (runs, colour, label) in CURVES.items():
            x, y, lo, hi, ns = curve(runs, key, sets)
            if not len(x):
                continue
            ax.fill_between(x, lo, hi, color=colour, alpha=0.15, linewidth=0)
            ax.plot(x, y, "-o", color=colour, lw=2, ms=5, label=label)
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(TS)
        ax.set_xticklabels([str(t) for t in TS])
        ax.set_xlabel("Adam iterations per level, T")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("DA3-SMALL, 22 dev scenes. Adam 4 x T, then LM with the stop rule. Band: range over seeds (3 where run, else 1)", fontsize=10)
    path = os.path.join(out, "brightness_over_T.png")
    fig.savefig(path, dpi=150)
    print(path)
    compute_figure(out)


TIMED = {"recommended": "recommended_t"}  # runs whose time comes from a separate GPU run (the quality run was on CPU)


def tto_s(run):
    """Mean TTO time per scene without each dataset's first scene (warm-up), as plot_compute.py."""
    run = TIMED.get(run, run)
    t = []
    for d in SETS:
        j = json.load(open(f"{ROOT}/{run}/metric_results/{d}_pba.json"))
        t += [v["opt_s"] for k, v in list(j.items())[1:] if k != "mean"]
    return float(np.mean(t))


def compute_figure(out):
    """AUC@3 against TTO seconds per scene (RTX 5090): the curves over T and the LM-on-NCC grid, plus the proposal's patch stage."""
    series = [
        ("brightness as before (Recommended over T)", "#c0392b", "-o", ["rec_it25", "recommended", "rec_it200", "rec_it400", "rec_it800"], None),
        ("relative brightness, LM with the stop rule", "#1f6fb2", "-o", [f"x1d_rel_t{T}_s0" for T in TS], None),
        ("relative brightness + 40 LM iterations on NCC", "#16a085", "-s", ["x6_a50_lmncc", "x7_t200_n40", "x7_t400_n40", "x6_a800_lmncc"], None),
        ("T = 800 + 20 / 40 / 80 / 160 LM iterations on NCC", "#16a085", ":^", ["x7_t800_n20", "x6_a800_lmncc", "x7_t800_n80", "x7_t800_n160"], None),
        ("patch alignment + geometric LM (proposal, exp. 3)", "#8e44ad", "D", ["x6_a50_patch", "x6_a800_patch"], None),
        ("no Adam (proposal, exp. 4)", "#7f8c8d", "x", ["x6_na_patch_shape", "x6_na_lmncc"], None),
        ("VGG + NCC, 4 x 50 + LM, stop, radial (REPORT3)", "#d68910", "P", ["pm_GX_best"], None),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    for label, colour, style, runs, _ in series:
        pts = [(tto_s(r), metric(r)) for r in runs if os.path.isdir(f"{ROOT}/{r}") and metric(r) is not None]
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, style, color=colour, lw=1.8, ms=6, label=label, mfc="white" if style.startswith(":") else colour)
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ticks = [0.25, 0.5, 1, 2, 4, 7]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlabel("TTO seconds per scene (RTX 5090), log scale")
    ax.set_ylabel("pose AUC@3, mean of 5 datasets")
    ax.set_title("DA3-SMALL, 22 dev scenes, one seed per point", fontsize=10)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    path = os.path.join(out, "compute_session4.png")
    fig.savefig(path, dpi=150)
    print(path)


if __name__ == "__main__":
    main()
