"""Session 5 figure: AUC@3 change of each one-change arm against its baseline (results/pba/figures5/arms.png).

    uv run scripts/plot_pba_s5.py

Dev: mean over the seeds of the five datasets' mean AUC@3 (runs <arm>_s<seed>), the bar the range of the seed means' differences
from the baseline's mean. Held-out: one seed (runs <arm>_h). Arms whose run is incomplete (out of memory on the 100-frame sets)
are left out.
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = "workspace/tto/DA3-SMALL"
SETS = ("hiroom", "7scenes", "scannetpp", "dtu64", "eth3d")
# (label, arm, baseline arm, held-out arm, held-out baseline); arms are run-name prefixes
ARMS = [
    ("one shared focal (focal_init=median)", "y_fmed", "y_b50", "y_fmed_h", "y_b50_h"),
    ("pose updates in world axes", "y_pw", "y_b50", None, None),
    ("pose basis K=6 (against world axes)", "y_pb6", "y_pw", None, None),
    ("pose basis K=6, T=200 (against world axes)", "y_pb6200", "y_pw200", None, None),
    ("pose Laplacian prior, w=1", "y_lap1", "y_b50", None, None),
    ("depth-edge mask (sampling)", "y_edge", "y_b50", None, None),
    ("depth-edge mask (target)", "y_tedge", "y_b50", None, None),
    ("principal point", "y_pp", "y_b50", None, None),
    ("depth prior: offset weight", "y_wsc", "y_b50", None, None),
    ("depth prior: smoothness", "y_wsm", "y_b50", None, None),
    ("neighbours refreshed", "y_nbrref", "y_b50", None, None),
    ("relocalisation", "y_reloc", "y_b50", None, None),
    ("oracle: gross frames from GT", "y_oracle", "y_b50", None, None),
    ("radial gate by held-out cost", "y_rgh", "y_b50", None, None),
    ("min_covis 0.1", "y_mc01", "y_b50", None, None),
    ("CUDA graph", "y_cg", "y_b50", None, None),
    ("+ pose basis K=3 (on shared focal)", "y_fpb3", "y_fmed", "y_fpb3_h", "y_fmed_h"),
    ("+ pose basis K=6, T=200 (on shared focal)", None, None, "y_fpb6200_h", "y_f200_h"),
    ("shared focal, T=800", None, None, "y_f800_h", "y_b800_h"),
    ("shared focal, T=800 + 40 LM on NCC", "y_n40fmed", "y_n40", "y_n40fmed_h", "y_n40_h"),
]


def mean_auc(run):
    a = []
    for d in SETS:
        p = f"{ROOT}/{run}/metric_results/{d}_pba.json"
        if not os.path.exists(p):
            return None
        j = json.load(open(p))
        j.pop("mean", None)
        a.append(100 * np.mean([v["auc03"] for v in j.values()]))
    return float(np.mean(a))


def seeds(arm):
    runs = sorted(os.path.basename(p) for p in glob.glob(f"{ROOT}/{arm}_s[0-9]"))
    vals = [mean_auc(r) for r in runs]
    return [v for v in vals if v is not None] if all(v is not None for v in vals) else []


def main():
    rows = []
    for label, arm, base, h, hb in ARMS:
        dev = None
        if arm:
            a, b = seeds(arm), seeds(base)
            if a and b:
                dev = (np.mean(a) - np.mean(b), min(a) - np.mean(b), max(a) - np.mean(b))
        held = None
        if h and mean_auc(h) is not None and mean_auc(hb) is not None:
            held = mean_auc(h) - mean_auc(hb)
        if dev or held is not None:
            rows.append((label, dev, held))
    blue, orange, ink, muted, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    y = np.arange(len(rows))[::-1]
    for yi, (label, dev, held) in zip(y, rows):
        if dev:
            ax.plot([dev[1], dev[2]], [yi + 0.12] * 2, color=blue, lw=2, solid_capstyle="round")
            ax.plot(dev[0], yi + 0.12, "o", ms=8, color=blue, mec="#fcfcfb", mew=2)
            ax.annotate(f"{dev[0]:+.1f}", (max(dev[2], dev[0]), yi + 0.12), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7, color=muted)
        if held is not None:
            ax.plot(held, yi - 0.12, "D", ms=7, color=orange, mec="#fcfcfb", mew=2)
            ax.annotate(f"{held:+.1f}", (held, yi - 0.12), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7, color=muted)
    ax.axvline(0, color=muted, lw=1)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8, color=ink)
    ax.set_xlabel("AUC@3 change against the arm's baseline (points, mean of the five datasets)", fontsize=8, color=muted)
    ax.tick_params(axis="x", labelsize=8, colors=muted)
    ax.grid(axis="x", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.plot([], [], "o", color=blue, ms=7, label="dev (22 scenes), mean and range over seeds")
    ax.plot([], [], "D", color=orange, ms=6, label="held-out (59 scenes), one seed")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.06), ncol=2, fontsize=7, frameon=False)
    ax.set_title("DA3-SMALL + photometric BA: one change at a time", fontsize=10, color=ink, loc="left")
    os.makedirs("results/pba/figures5", exist_ok=True)
    fig.tight_layout()
    fig.savefig("results/pba/figures5/arms.png", facecolor=fig.get_facecolor())
    print("results/pba/figures5/arms.png", len(rows), "arms")


if __name__ == "__main__":
    main()
