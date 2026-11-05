"""Loss ablations of the RGB PBA (Adam 4 x 50 + LM 5 at 1 px) on one model (dev subset), as a dot plot.

Usage: .venv/bin/python scripts/plot_loss_sweep.py <out.png> [model]   (model default DA3-SMALL)

One row per run of RUNS (grouped by block), columns = pose AUC@3 (mean of 5 datasets), recon F1 (mean of HiRoom,
7-Scenes, ScanNet++) and time per scene (as in plot_compute.point). Grey: the baseline config, as the mean (line)
and range (band) of its runs in BASE (same config, different seeds or code versions), i.e. the noise level.
Also prints a markdown table with the per-dataset AUC@3 and F1.
"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_ablation import forward_times  # noqa: E402
from plot_compute import POSE, RECON, REPO_ROOT, load, point  # noqa: E402

INK, INK2, GREY, SURFACE, ACCENT = "#0b0b0b", "#52514e", "#a8a7a1", "#fcfcfb", "#2a78d6"
BASE = ["loss_B0_base", "fastpolish50", "abl_N3_fpol50_s1", "abl_N4_fpol50_s2"]
RUNS = [
    ("Huber scale (base 0.05)", [
        ("loss_D1_d01", "D1", "δ = 0.01"),
        ("loss_D2_d02", "D2", "δ = 0.02"),
        ("loss_D3_d10", "D3", "δ = 0.1"),
        ("loss_D4_d20", "D4", "δ = 0.2"),
    ]),
    ("Kernel (scale c)", [
        ("loss_K1_l2", "K1", "L2"),
        ("loss_K2_cauchy05", "K2", "Cauchy, c = 0.05"),
        ("loss_K3_cauchy10", "K3", "Cauchy, c = 0.1"),
        ("loss_K4_gm05", "K4", "Geman-McClure, c = 0.05"),
        ("loss_K5_gm10", "K5", "Geman-McClure, c = 0.1"),
        ("loss_K6_tukey10", "K6", "Tukey, c = 0.1"),
        ("loss_K7_tukey20", "K7", "Tukey, c = 0.2"),
    ]),
    ("Residual", [
        ("loss_V1_gray", "V1", "grey instead of RGB"),
        ("loss_V2_min", "V2", "min over neighbours"),
        ("loss_V3_nobright", "V3", "no brightness model"),
    ]),
    ("Geometric term (base: Huber 0.05, w 1)", [
        ("loss_G1_hgeo02", "G1", "Huber δ_geo = 0.02"),
        ("loss_G2_hgeo20", "G2", "Huber δ_geo = 0.2"),
        ("loss_G3_wgeo0", "G3", "w_geo = 0 (photometric only)"),
        ("loss_G4_wgeo3", "G4", "w_geo = 3"),
    ]),
    ("Combined: δ = 0.02 + grey", [
        ("loss_C1_d02gray_s0", "C1", "seed 0"),
        ("loss_C1_d02gray_s1", "C1", "seed 1"),
        ("loss_C1_d02gray_s2", "C1", "seed 2"),
    ]),
]


def per_dataset(run):
    out = {}
    for d in POSE:
        r = load(run, f"{d}_pba")
        out[d] = 100 * sum(v["auc03"] for v in r.values()) / len(r) if r else None
    for d in RECON:
        r = load(run, f"{d}_recon_unposed")
        out[f"{d}_f1"] = 100 * sum(v["fscore"] for v in r.values()) / len(r) if r else None
    return out


def main():
    os.chdir(REPO_ROOT)
    model = sys.argv[2] if len(sys.argv) > 2 else "DA3-SMALL"
    forward = forward_times(model)
    pt = lambda name: point(f"workspace/tto/{model}/{name}", forward)  # noqa: E731
    base = [p for p in map(pt, BASE) if p]

    rows, y, ticks, heads = [], 0, [], []
    for title, runs in RUNS:
        heads.append((y, title))
        y += 1
        for name, rid, label in runs:
            rows.append((y, name, rid, label, pt(name)))
            ticks.append((y, f"{rid}  {label}"))
            y += 1
        y += 0.4
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.32 * y + 1.6), sharey=True, facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1, 0.7]})
    for ax, k, title in zip(axes, (1, 2, 0), ("pose AUC@3 (%)", "recon F1 (%)", "time per scene (s)")):
        if base:
            vals = [p[k] for p in base]
            ax.axvspan(min(vals), max(vals), color=GREY, alpha=0.3, lw=0, zorder=1)
            ax.axvline(sum(vals) / len(vals), color=INK2, lw=1, zorder=2)
        for yy, name, rid, label, p in rows:
            if p:
                ax.plot([p[k]], [yy], marker="o", ms=7, color=ACCENT, mec=SURFACE, mew=1.2, ls="none", zorder=4)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color="#e4e3df", lw=0.8)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GREY)
        ax.tick_params(colors=INK2, labelsize=8, left=False)
    axes[0].set_yticks([t for t, _ in ticks], [lab for _, lab in ticks], fontsize=8, color=INK)
    for yy, title in heads:
        axes[0].annotate(title, (0, yy), xycoords=("axes fraction", "data"), xytext=(-250, 0), textcoords="offset points",
                         fontsize=9, color=INK, weight="bold", va="center", annotation_clip=False)
    axes[0].set_ylim(y - 0.4, -0.6)
    fig.suptitle(f"{model}: loss ablations of the RGB PBA (Adam 4x50 + LM 5 at 1 px), dev subset. "
                 f"Grey: baseline (line: mean, band: range of {len(base)} runs)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0.0, 0, 1, 0.97))
    fig.savefig(sys.argv[1], dpi=150)

    cols = POSE + [f"{d}_f1" for d in RECON]
    print("| run | ID | setup | s/scene | A@3 | F1 | " + " | ".join(cols) + " |\n|" + "---|" * (6 + len(cols)))
    table = [(n, "base", "") for n in BASE] + [(n, rid, lab) for _, runs in RUNS for n, rid, lab in runs]
    for n, rid, lab in table:
        p = pt(n)
        if not p:
            print(f"| {n} | {rid} | {lab} | - |")
            continue
        pd = per_dataset(f"workspace/tto/{model}/{n}")
        print(f"| {n} | {rid} | {lab} | {p[0]:.2f} | {p[1]:.1f} | {p[2]:.1f} | " + " | ".join(f"{pd[c]:.1f}" for c in cols) + " |")
    print(sys.argv[1])


if __name__ == "__main__":
    main()
