"""Slide figures for the depth-bottleneck finding (knowledge/ideas/depth-bottleneck.md, 2026-11-22/23).

    .venv/bin/python scripts/plot_depth_bottleneck.py   # -> results/pba/figures_depth/*.png

Numbers are copied from the evidence rows of that idea file (DA3-SMALL, B′ iters=25, seed 0, RTX 3080):
(1) depth oracles with the depth held fixed, mean AUC@3 of HiRoom / ScanNet++ / ETH3D (`lpba_dor_*`);
(2) cameras x depth on all 81 pose scenes (`lpba_tb_*`, `lpba_pa_base_s0`; LARGE cameras + BASE depth run 2026-11-26).
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "pba", "figures_depth")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
RAMP = ["#b7d3f6", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]  # blue 150 ... 650
ORANGE = "#eb6834"

plt.rcParams.update({"font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "font.family": "DejaVu Sans"})


def oracle(ax):
    labels = ["DA3 depth\n(as predicted)", "+ per-frame\nscale", "+ 4-cell\ncorrection", "+ 8-cell\ncorrection",
              "+ 16-cell\ncorrection", "GT depth"]
    vals = [23.3, 24.8, 37.6, 42.9, 46.6, 49.4]
    x = np.arange(len(vals))
    ax.bar(x, vals, width=0.7, color=RAMP, edgecolor=SURF, linewidth=2, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", va="bottom", color=INK, fontsize=13, fontweight="bold")
    ax.axhline(37.9, color=ORANGE, lw=2, ls=(0, (5, 3)), zorder=4)
    ax.text(-0.4, 36.6, "normal PBA: 37.9\n(DA3 depth, depth grid optimised)", color=INK, fontsize=11.5, va="top")
    ax.set_xticks(x, labels, fontsize=11.5)
    ax.set_ylim(0, 56)
    ax.set_ylabel("pose AUC@3 after PBA")
    ax.set_title("Depth held fixed during PBA: corrected towards GT at coarse resolution",
                 loc="left", color=INK, fontsize=14, pad=12)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0)


def swap(ax):
    # rows: cameras from, cols: depth from
    M = np.array([[43.6, 50.7, 52.5], [58.3, 64.0, 69.8]])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", ["#cde2fb", "#104281"])
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=38, vmax=72, aspect="auto")
    for i in range(2):
        for j in range(3):
            v = M[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#f0efec"))
                ax.text(j, i, "not run", ha="center", va="center", color=INK2, fontsize=12)
            else:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=20, fontweight="bold",
                        color="#ffffff" if v > 55 else INK)
    for k in np.arange(-0.5, 3, 1):
        ax.axvline(k, color=SURF, lw=4)
    ax.axhline(0.5, color=SURF, lw=4)
    ax.set_xticks(range(3), ["SMALL", "BASE", "LARGE"])
    ax.set_yticks(range(2), ["SMALL", "LARGE"])
    ax.set_xlabel("depth from", fontsize=13)
    ax.set_ylabel("cameras from", fontsize=13)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, labelsize=13)
    ax.set_title("Swapping DA3 outputs before PBA (81 scenes)", loc="left", color=INK, fontsize=14, pad=40)


def limits(ax):
    """Camera start vs depth on HiRoom / ScanNet++ / ETH3D (seed 0): lpba_dor_da3_s0, lpba_pa_large_s0, lpba_dor_gtfix_s0,
    lpba_tb_largecam_smalldepth, lpba_tb_large_bprime (mean of the three sets)."""
    rows = [("SMALL", "SMALL depth", 37.9), ("SMALL", "LARGE depth", 49.3), ("SMALL", "GT depth", 49.4),
            ("LARGE", "SMALL depth", 58.9), ("LARGE", "LARGE depth", 75.3)]
    col = {"SMALL": "#2a78d6", "LARGE": ORANGE}
    y = np.arange(len(rows))[::-1]
    for yi, (cam, dep, v) in zip(y, rows):
        ax.barh(yi, v, height=0.62, color=col[cam], edgecolor=SURF, linewidth=2, zorder=3)
        ax.text(v + 0.8, yi, f"{v:.1f}", va="center", color=INK, fontsize=14, fontweight="bold")
    ax.set_yticks(y, [f"{c} cameras + {d}" for c, d, _ in rows], fontsize=13)
    ax.axvline(49.4, color=INK2, lw=1.2, ls=(0, (3, 3)), zorder=2)
    ax.text(49.9, y[0] + 0.42, "ceiling with SMALL's cameras", color=INK2, fontsize=11, va="bottom")
    ax.set_xlim(0, 84)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_xlabel("pose AUC@3 after PBA (HiRoom / ScanNet++ / ETH3D)")
    ax.set_title("The starting cameras matter more than the depth", loc="left", color=INK, fontsize=15, pad=12)
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(length=0)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURF)
    oracle(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "depth_oracle.png"), dpi=200)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor=SURF)
    swap(ax)
    fig.text(0.02, 0.03, "SMALL → LARGE gap after PBA (+26.2): depth +8.9, cameras +14.7, interaction +2.6",
             color=INK2, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(OUT, "depth_camera_swap.png"), dpi=200)
    fig, axs = plt.subplots(1, 2, figsize=(17, 5.6), facecolor=SURF, gridspec_kw={"width_ratios": [1.6, 1]})
    oracle(axs[0])
    swap(axs[1])
    fig.tight_layout(w_pad=4)
    fig.savefig(os.path.join(OUT, "depth_bottleneck.png"), dpi=200)
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURF)
    limits(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pba_limits.png"), dpi=200)
    print(os.path.abspath(OUT))


if __name__ == "__main__":
    main()
