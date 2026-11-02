"""Level and optimiser ablations of the RGB PBA on one model (dev subset), as small multiples.

Usage: .venv/bin/python scripts/plot_ablation.py <out.png> [model]   (model default DA3-SMALL)

One column per block of BLOCKS, rows = pose AUC@3 (mean of 5 datasets) and recon F1 (mean of HiRoom, 7-Scenes,
ScanNet++), x = DA3 forward + PBA time per scene (see plot_compute.point). Grey context in every panel: the RGB Adam
4 x T line (fast_it<T>) and the two anchors fast_it50 / fastpolish50, each drawn at the mean of its seeds (SEEDS)
with a box over their range (the seed noise). Blue: the block's runs, labelled by ID (greedy placement around the
point, with a leader line when it has to move further away), and lines through the runs of one family (LINES; an anchor in a line counts with its mean).
Also prints a markdown table of all runs.
"""

import glob
import json
import os
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_compute import REPO_ROOT, point  # noqa: E402

INK, INK2, GREY, SURFACE, ACCENT = "#0b0b0b", "#52514e", "#a8a7a1", "#fcfcfb", "#2a78d6"
ANCHORS = {"fast_it50": ("o", "Adam 4x50"), "fastpolish50": ("P", "Adam 4x50 + LM 5")}
SEEDS = {"fast_it50": ["abl_N1_fast50_s1", "abl_N2_fast50_s2"], "fastpolish50": ["abl_N3_fpol50_s1", "abl_N4_fpol50_s2"]}
BLOCKS = [
    ("Levels (Adam; 200 its in total, L5-L7: more)", {
        "abl_L1_l421": ("L1", "levels 4,2,1 x 67"),
        "abl_L2_l21": ("L2", "levels 2,1 x 100"),
        "abl_L3_l1": ("L3", "one level (1 px) x 200"),
        "abl_L4_l168421": ("L4", "levels 16,8,4,2,1 x 40"),
        "abl_L5_l1x800": ("L5", "one level (1 px) x 800"),
        "abl_L6_l1x400": ("L6", "one level (1 px) x 400"),
        "abl_L7_l1x1600": ("L7", "one level (1 px) x 1600"),
    }),
    ("Adam iterations (4 levels)", {
        "abl_I1_it10": ("I1", "4 x 10"),
        "abl_I2_it100": ("I2", "4 x 100"),
        "abl_I3_coarse": ("I3", "80/60/40/20 (coarse-heavy)"),
        "abl_I4_fine": ("I4", "20/40/60/80 (fine-heavy)"),
    }),
    ("LM polish (at 1 px)", {
        "abl_P1_lm5only": ("P1", "LM 5 only, no Adam"),
        "abl_P2_it10_lm5": ("P2", "Adam 4 x 10 + LM 5"),
        "abl_P3_it25_lm5": ("P3", "Adam 4 x 25 + LM 5"),
        "abl_P4_it50_lm2": ("P4", "Adam 4 x 50 + LM 2"),
        "abl_P5_it50_lm10": ("P5", "Adam 4 x 50 + LM 10"),
        "abl_P6_l21_lm5": ("P6", "Adam 2,1 x 100 + LM 5"),
        "abl_P7_l1_lm5": ("P7", "Adam 1 x 200 + LM 5"),
    }),
    ("LM instead of Adam", {
        "abl_H1_adam84_lm21": ("H1", "Adam 8,4 x 50, LM 5 at 2 and 1"),
        "abl_H2_lm21only": ("H2", "LM only, levels 2,1 x 5"),
        "lm_rgb": ("H3", "LM only, 4 levels x 10 (earlier run)"),
    }),
]
LINES = {  # per block: runs of one family, connected in this order
    0: [["abl_L3_l1", "abl_L6_l1x400", "abl_L5_l1x800", "abl_L7_l1x1600"]],
    2: [["abl_P1_lm5only", "abl_P2_it10_lm5", "abl_P3_it25_lm5", "fastpolish50"]],
}
# label candidates (alignment, dx, dy in points from the point; text baseline at dy), in order of preference
NEAR = [("left", 5, -3), ("left", 5, 3), ("left", 5, -9), ("right", -5, -3), ("right", -5, 3), ("right", -5, -9), ("center", 0, 6), ("center", 0, -13)]
FAR = [(h, 12 * sx, dy) for dy in (12, -12, 19, -19, 26, -26, 33, -33) for h, sx in (("left", 1), ("right", -1))]  # with a leader line
GREY_CANDS = [("center", 0, 5), ("center", 0, -12), ("left", 5, -3), ("right", -5, -3)]


def place_labels(ax, markers, items):
    """Greedy label placement. markers: (x, y) data points to keep clear; items: (x, y, text, fontsize, color, candidates),
    placed in order, each at the first candidate with the least overlap with markers, labels placed before and the axes edge."""
    k = 72 / ax.figure.dpi
    to_pt = lambda x, y: tuple(v * k for v in ax.transData.transform((x, y)))  # noqa: E731
    ex = ax.get_window_extent()
    frame = (ex.x0 * k, ex.y0 * k, ex.x1 * k, ex.y1 * k)
    boxes = [(px - 4.5, py - 4.5, px + 4.5, py + 4.5) for px, py in (to_pt(x, y) for x, y in markers)]

    def overlap(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    for x, y, text, fs, color, cands in items:
        px, py = to_pt(x, y)
        w, h = 0.62 * fs * len(text), 0.75 * fs
        best = None
        for i, (ha, dx, dy) in enumerate(cands):
            x0 = px + dx - {"left": 0, "right": w, "center": w / 2}[ha]
            box = (x0 - 1, py + dy - 1, x0 + w + 1, py + dy + h + 1)
            cost = sum(overlap(box, o) for o in boxes) + 1000 * (box[0] < frame[0] or box[2] > frame[2] or box[1] < frame[1] or box[3] > frame[3]) + 0.5 * i
            if best is None or cost < best[0]:
                best = (cost, ha, dx, dy, box)
        _, ha, dx, dy, box = best
        boxes.append(box)
        leader = (ha, dx, dy) in FAR
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, va="baseline", fontsize=fs, color=color, zorder=6,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=1, shrinkB=4) if leader else None)


def forward_times(model):
    forward = {}
    for line in open(f"workspace/evaluation/{model}/timings.jsonl"):
        r = json.loads(line)
        if not r["posed"] and "forward_s" in r:
            d, s = r["task"].split("/", 1)
            forward[(d, s.rsplit("/unposed", 1)[0])] = r["forward_s"]
    return forward


def main():
    os.chdir(REPO_ROOT)
    model = sys.argv[2] if len(sys.argv) > 2 else "DA3-SMALL"
    forward = forward_times(model)
    pt = lambda name: point(f"workspace/tto/{model}/{name}", forward)  # noqa: E731
    init = pt("init_dev")
    fam = sorted((int(re.search(r"_it(\d+)$", p).group(1)), os.path.basename(p)) for p in glob.glob(f"workspace/tto/{model}/fast_it*") if re.search(r"_it\d+$", p))
    line = [(0, init)] + [(it, pt(n)) for it, n in fam]
    line = [(it, p) for it, p in line if p]
    anchors = {}
    for name in ANCHORS:
        pts = [p for p in [pt(name)] + [pt(s) for s in SEEDS[name]] if p]
        if pts:
            anchors[name] = pts

    def mean_pt(name):
        pts = anchors.get(name) or [p for p in [pt(name)] if p]
        return tuple(sum(p[i] for p in pts) / len(pts) for i in range(3)) if pts else None

    labels = []
    fig, axes = plt.subplots(2, len(BLOCKS), figsize=(4.1 * len(BLOCKS), 7.6), sharey="row", facecolor=SURFACE)
    for col, (title, runs) in enumerate(BLOCKS):
        for row, k in enumerate((1, 2)):
            ax = axes[row, col]
            ax.plot([p[0] for _, p in line], [p[k] for _, p in line], color=GREY, lw=1.5, ls="--", marker="o", ms=4, zorder=2)
            markers = [(p[0], p[k]) for _, p in line]
            items = [(p[0], p[k], "no PBA" if it == 0 else str(it), 7, GREY, GREY_CANDS[1:2] if it == 0 else GREY_CANDS)
                     for it, p in line if it != 50]  # T = 50: the o anchor marks it
            for name, (marker, _) in ANCHORS.items():
                if name in anchors:
                    pts = anchors[name]
                    x = sum(p[0] for p in pts) / len(pts)
                    ys = [p[k] for p in pts]
                    ax.add_patch(Rectangle((x / 1.12, min(ys)), x * 1.12 - x / 1.12, max(max(ys) - min(ys), 0.15), facecolor=GREY, alpha=0.35,
                                           edgecolor="none", zorder=1.5))
                    ax.plot([x], [sum(ys) / len(ys)], marker=marker, ms=8, color=INK2, mec=SURFACE, mew=1, ls="none", zorder=4)
                    markers.append((x, sum(ys) / len(ys)))
            for fam in LINES.get(col, []):
                xy = [mean_pt(n) for n in fam]
                xy = [q for q in xy if q]
                ax.plot([q[0] for q in xy], [q[k] for q in xy], color=ACCENT, lw=1.2, alpha=0.6, zorder=2.5)
            for name, (rid, _) in runs.items():
                p = pt(name)
                if p:
                    ax.plot([p[0]], [p[k]], marker="o", ms=7, color=ACCENT, mec=SURFACE, mew=1.2, ls="none", zorder=5)
                    markers.append((p[0], p[k]))
                    items.append((p[0], p[k], rid, 8, INK, NEAR + FAR))
            labels.append((ax, markers, items))
            ax.set_xscale("symlog", linthresh=1)
            ax.set_facecolor(SURFACE)
            ax.grid(color="#e4e3df", lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GREY)
            ax.tick_params(colors=INK2, labelsize=8)
            if row == 0:
                ax.set_title(title, fontsize=10, color=INK, loc="left")
            else:
                ax.set_xlabel("time per scene (s)", color=INK2, fontsize=9)
                key = "\n".join(f"{rid}  {desc}" for rid, desc in runs.values())
                ax.text(0, -0.22, key, transform=ax.transAxes, va="top", ha="left", fontsize=7.5, color=INK2, family="monospace")
        axes[0, 0].set_ylabel("pose AUC@3 (%)", color=INK2, fontsize=9)
        axes[1, 0].set_ylabel("recon F1 (%)", color=INK2, fontsize=9)
    handles = [
        plt.Line2D([], [], color=GREY, lw=1.5, ls="--", marker="o", ms=4, label="Adam 4 x T, RGB (labels: T)"),
        *[plt.Line2D([], [], color=INK2, marker=m, ms=8, ls="none", label=f"{d} (mean of 3 seeds)") for m, d in ANCHORS.values()],
        Rectangle((0, 0), 1, 1, facecolor=GREY, alpha=0.35, label="range of the 3 seeds (noise)"),
        plt.Line2D([], [], color=ACCENT, marker="o", ms=7, ls="-", lw=1.2, label="ablation run (see key); line: one family"),
    ]
    fig.legend(handles=handles, loc="upper left", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.005, 0.965))
    fig.suptitle(f"{model}: level and optimiser ablations of the RGB PBA, dev subset. AUC@3: mean of 5 datasets, F1: mean of HiRoom, "
                 "7-Scenes, ScanNet++, time: DA3 forward + PBA on an RTX 5090",
                 fontsize=11, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    for ax, markers, items in labels:
        place_labels(ax, markers, items)
    fig.savefig(sys.argv[1], dpi=150)

    print(f"| run | ID | setup | s/scene | A@3 | F1 |\n|---|---|---|---|---|---|")
    rows = [(n, "", d) for n, (_, d) in ANCHORS.items()] + [(s, "seed", "") for v in SEEDS.values() for s in v]
    rows += [(n, rid, d) for _, runs in BLOCKS for n, (rid, d) in runs.items()]
    for n, rid, d in rows:
        p = pt(n)
        print(f"| {n} | {rid} | {d} | " + (f"{p[0]:.2f} | {p[1]:.1f} | {p[2]:.1f} |" if p else "- | - | - |"))
    print(sys.argv[1])


if __name__ == "__main__":
    main()
