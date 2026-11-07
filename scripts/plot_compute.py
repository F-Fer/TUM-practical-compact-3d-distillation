"""Quality vs. compute of DA3 models with and without TTO (dev subset).

Usage: .venv/bin/python scripts/plot_compute.py <out.png> [model ...]

Per model: the unrefined run (workspace/tto/<model>/init_dev), the PBA runs workspace/tto/<model>/<family>_it<T>
for each family in FAMILIES (lines over T, the Adam iterations per level), the single PBA runs in POINTS (markers), and
the track-based BA where a model has it: RoMa tracks + BA as a dotted line over the BA iterations k (time and AUC@3 of
ba_fast_it<k>, F1 of time_ba_it<k>, the same predictions before the solver got faster) and VGGSfM tracks + BA
(time_vggsfm, a star on the pose panel: it has no recon scores). A run that lacks one of the metric files is skipped,
PBA markers with a message; run_tto_dev.sh scores recon without ETH3D, so its runs need
`eval_tto.py workspace/evaluation/<model> <run> eth3d recon_unposed --workers 1` first.
x = mean time per scene: DA3 forward (workspace/evaluation/<model>/timings.jsonl) + PBA optimisation (opt_s). If
<run>_t exists (the same configuration re-timed alone on a GPU, e.g. because <run> shared its GPU or ran on the CPU), its
opt_s is used; the first scene of each dataset (one-off warm-up of a fresh process) counts with the dataset's median time;
family runs with T > 200 (timed on a shared GPU) use the opt_s of <family>_it200 scaled by T / 200 (the cost is
linear in T: 0.7 s + 0.08 s x T per scene for v3 on the dev subset), except rec_it<T>, which ran alone on a GPU.
y = AUC@3 averaged over the five pose datasets, and recon F1 averaged over the four F1 datasets of the DA3 benchmark
(HiRoom, 7-Scenes, ScanNet++, ETH3D; DTU is scored by Chamfer distance). The dotted track lines have no ETH3D recon yet:
their F1 is the mean of three datasets, from the unrefined F1 of the same three, so compare them with the other series
on the pose panel only. Other scripts that import point() keep the default RECON (without ETH3D): runs before 2026-11-03
were scored without ETH3D recon (its full-resolution TSDF fusion needs up to ~58 GB of RAM); the compute-plot runs were
rescored with $SCRATCH/jobs/eth3d_rescore.sh.
"""

import glob
import json
import math
import os
import re
import sys

import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# categorical slots 1-3; GIANT in a dark neutral: the palette's slot 4 (yellow) fails the all-pairs floors against
# orange (normal vision dE 13.7), the dark neutral clears them against all three (dE >= 24 normal, >= 25 CVD)
MODELS = {"DA3-SMALL": "#2a78d6", "DA3-BASE": "#eb6834", "DA3-LARGE-1.1": "#1baf7a", "DA3-GIANT-1.1": "#3d3c39"}
POSE = ["hiroom", "7scenes", "scannetpp", "dtu64", "eth3d"]
RECON = ["hiroom", "7scenes", "scannetpp"]
RECON_ALL = RECON + ["eth3d"]  # all F1 datasets (the DA3 paper's recon average)
STYLE = "#a8a7a1"  # legend entries for line style / marker shape (not a series)
# v3: Adam on VGG features + NCC, 4 pyramid levels x T (before the analytic code)
# rec: the recommended configuration of results/pba/REPORT3.md with T changed: analytic Adam on RGB, 1024 px, 4 x T, then LM
# with the stop rule and the gated radial coefficient. Its T = 50 is the run `recommended`
FAMILIES = {"v3": ("-", "Adam, VGG+NCC (4 levels x T)"), "rec": ("--", "Optimised RGB: analytic Adam (4 x T), LM stop rule, radial")}
POINTS = {
    "fastpolish50": ("P", "RGB Adam 4x50 + LM polish (old code)"),
    "recommended": ("D", "Recommended: optimised RGB at T = 50"),
}
TRACK_ITERS = (2, 5, 10, 25)  # BA iterations of the dotted track lines


def load(run, name):
    p = os.path.join(run, "metric_results", f"{name}.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    d.pop("mean", None)
    return d


def point(run, forward, recon=RECON, recon_run=None):
    """(time per scene, AUC@3, F1) of a run; F1 from recon_run if given, NaN if recon is empty; None if metrics are missing."""
    pba = {d: load(run, f"{d}_pba") for d in POSE}
    scenes = {d: list(pba[d]) for d in POSE}
    if not all(scenes.values()):
        return None
    timing, factor = pba, 1.0
    if os.path.isdir(run + "_t"):
        timing = {d: load(run + "_t", f"{d}_pba") for d in POSE}
    elif "/rec_it" not in run and (m := re.search(r"_it(\d+)$", run)) and int(m.group(1)) > 200 and os.path.isdir(run[: m.start()] + "_it200"):
        timing, factor = {d: load(run[: m.start()] + "_it200", f"{d}_pba") for d in POSE}, int(m.group(1)) / 200
    t = []
    for d in POSE:
        opt = [factor * timing[d].get(s, pba[d][s])["opt_s"] for s in scenes[d]]
        if len(opt) > 1:  # the first scene of every dataset (a new process) includes one-off warm-up: use the median
            opt[0] = sorted(opt[1:])[len(opt[1:]) // 2]
        t += [forward[(d, s)] + o for s, o in zip(scenes[d], opt)]
    auc = sum(sum(pba[d][s]["auc03"] for s in scenes[d]) / len(scenes[d]) for d in POSE) / len(POSE)
    f1 = []
    for d in recon:
        r = load(recon_run or run, f"{d}_recon_unposed")
        if not r:
            return None
        f1.append(sum(v["fscore"] for v in r.values()) / len(r))
    return sum(t) / len(t), 100 * auc, 100 * sum(f1) / len(f1) if f1 else math.nan


def missing(run, recon):
    """The metric files point() needs that `run` lacks."""
    names = [f"{d}_pba" for d in POSE] + [f"{d}_recon_unposed" for d in recon]
    return [n for n in names if not os.path.exists(os.path.join(run, "metric_results", f"{n}.json"))]


def main():
    os.chdir(REPO_ROOT)
    models = sys.argv[2:] or list(MODELS)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6), facecolor="#fcfcfb")
    for model in models:
        color = MODELS[model]
        if not os.path.exists(f"workspace/evaluation/{model}/timings.jsonl"):
            print(f"skip {model}: no DA3 timings")
            continue
        forward = {}
        for line in open(f"workspace/evaluation/{model}/timings.jsonl"):
            r = json.loads(line)
            if not r["posed"] and "forward_s" in r:
                d, s = r["task"].split("/", 1)
                forward[(d, s.rsplit("/unposed", 1)[0])] = r["forward_s"]
        root = f"workspace/tto/{model}"
        init = point(f"{root}/init_dev", forward, RECON_ALL)
        for fam, (ls, _) in FAMILIES.items():
            runs = [(int(re.search(r"_it(\d+)$", p).group(1)), p) for p in glob.glob(f"{root}/{fam}_it*") if re.search(r"_it\d+$", p)]
            runs = sorted(runs + ([(50, f"{root}/recommended")] if fam == "rec" else []))
            pts = [(0, init)] + [(it, point(run, forward, RECON_ALL)) for it, run in runs]
            pts = [(it, p) for it, p in pts if p]
            if len(pts) < 2:
                continue
            for ax, k in zip(axes, (1, 2)):
                ax.plot([p[0] for _, p in pts], [p[k] for _, p in pts], color=color, lw=2, ls=ls, marker="o", ms=6, mec="#fcfcfb", mew=1.5, zorder=3)
                for it, p in pts:
                    if fam == "v3" and it:
                        ax.annotate(f"{it}", (p[0], p[k]), textcoords="offset points", xytext=(5, -11), fontsize=7.5, color="#52514e")
                    elif it:  # dashed family: labels above the points, so they don't collide with the solid line's; T = 25 and 50 nearly coincide
                        off, ha = {25: ((-6, 3), "right"), 50: ((7, 5), "left")}.get(it, ((0, 6), "center"))
                        ax.annotate(f"{it}", (p[0], p[k]), textcoords="offset points", xytext=off, ha=ha, fontsize=7.5, color="#52514e", style="italic")
        tracks = [(k, point(f"{root}/ba_fast_it{k}", forward, RECON, recon_run=f"{root}/time_ba_it{k}")) for k in TRACK_ITERS]
        tracks = [(k, p) for k, p in tracks if p]
        start = point(f"{root}/init_dev", forward, RECON)  # the track runs' F1 has three datasets: start from the same three
        if tracks and start:
            for ax, c in zip(axes, (1, 2)):
                ax.plot([start[0]] + [p[0] for _, p in tracks], [start[c]] + [p[c] for _, p in tracks], color=color, lw=2, ls=":", marker="o", ms=5, mec="#fcfcfb", mew=1.2, zorder=3)
                for k, p in (tracks[0], tracks[-1]):  # the iterations hardly change x, so they stack: label the two ends
                    ax.annotate(f"k={k}", (p[0], p[c]), textcoords="offset points", xytext=(6, -3), fontsize=7.5, color="#52514e")
        if init:  # the unrefined DA3 prediction, the start of every line
            for ax, k in zip(axes, (1, 2)):
                ax.plot([init[0]], [init[k]], marker="o", ms=10, mfc="#fcfcfb", mec=color, mew=2.2, ls="none", zorder=5)
        for name, (marker, _) in POINTS.items():
            run = f"{root}/{name}"
            p = point(run, forward, RECON_ALL)
            if p is None and os.path.isdir(run):
                print(f"skip {run}: missing {', '.join(missing(run, RECON_ALL)) or 'scenes'}")
            if p:
                print(f"{model} {name}: {p[0]:.2f} s, AUC@3 {p[1]:.1f}, F1 {p[2]:.1f}")
                for ax, k in zip(axes, (1, 2)):
                    ax.plot([p[0]], [p[k]], color=color, marker=marker, ms=9, mec="#0b0b0b", mew=0.8, ls="none", zorder=6)
        v = point(f"{root}/time_vggsfm", forward, ())  # pose only: its recon scores are not in this workspace
        if v:
            axes[0].plot([v[0]], [v[1]], color=color, marker="*", ms=11, mec="#0b0b0b", mew=0.8, ls="none", zorder=4)
        axes[0].plot([], [], color=color, lw=2, marker="o", ms=6, label=model)
    for ax, ylabel in zip(axes, ("pose AUC@3 (%)", "recon F1 (%)")):
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("time per scene (s): DA3 forward + TTO, RTX 5090", color="#52514e")
        ax.set_ylabel(ylabel, color="#52514e")
        ax.set_facecolor("#fcfcfb")
        ax.grid(color="#e4e3df", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#a8a7a1")
        ax.tick_params(colors="#52514e", labelsize=8)
    axes[0].plot([], [], marker="o", ms=9, mfc="#fcfcfb", mec=STYLE, mew=2.2, ls="none", label="DA3 alone (no TTO)")
    for fam, (ls, desc) in FAMILIES.items():
        axes[0].plot([], [], color=STYLE, lw=2, ls=ls, label=desc)
    axes[0].plot([], [], color=STYLE, lw=2, ls=":", label="RoMa tracks + BA (radial, ba_lm), k iterations; F1 without ETH3D")
    for name, (marker, desc) in POINTS.items():
        axes[0].plot([], [], color=STYLE, marker=marker, ms=7, mec="#0b0b0b", mew=0.8, ls="none", label=desc)
    axes[0].plot([], [], color=STYLE, marker="*", ms=9, mec="#0b0b0b", mew=0.8, ls="none", label="VGGSfM tracks + BA (pose panel only)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(sys.argv[1], dpi=150)
    print(sys.argv[1])


if __name__ == "__main__":
    main()
