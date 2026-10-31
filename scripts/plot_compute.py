"""Quality vs. compute of DA3 models with and without PBA (dev subset).

Usage: .venv/bin/python scripts/plot_compute.py <out.png> [model ...]

Per model: the unrefined run (workspace/tto/<model>/init_dev), the PBA runs workspace/tto/<model>/<family>_it<T>
for each family in FAMILIES (lines over T, the Adam iterations per level), and the single runs in POINTS (markers).
x = mean time per scene: DA3 forward (workspace/evaluation/<model>/timings.jsonl) + PBA optimisation (opt_s). If
<run>_t exists (the same configuration re-timed alone on a GPU, e.g. because <run> shared its GPU), its opt_s is used;
the first scene of each dataset (one-off warm-up of a fresh process) counts with the dataset's median time;
family runs with T > 200 (timed on a shared GPU) use the opt_s of <family>_it200 scaled by T / 200 (the cost is
linear in T: 0.7 s + 0.08 s x T per scene for v3 on the dev subset).
y = AUC@3 averaged over the five pose datasets, and recon F1 averaged over HiRoom, 7-Scenes, ScanNet++.
"""

import glob
import json
import os
import re
import sys

import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {"DA3-SMALL": "#2a78d6", "DA3-BASE": "#eb6834", "DA3-LARGE-1.1": "#1baf7a"}  # categorical slots 1-3
POSE = ["hiroom", "7scenes", "scannetpp", "dtu64", "eth3d"]
RECON = ["hiroom", "7scenes", "scannetpp"]
FAMILIES = {"v3": ("-", "Adam, VGG+NCC (4 levels x T)"), "fast": ("--", "Adam, RGB Huber, 1024 px (4 x T)")}
POINTS = {
    "polish50": ("s", "Adam 4x50 + LM polish"),
    "polish200": ("D", "Adam 4x200 + LM polish"),
    "fastpolish50": ("P", "RGB Adam 4x50 + LM polish"),
    "lm10": ("^", "LM only (4x10)"),
}


def load(run, name):
    p = os.path.join(run, "metric_results", f"{name}.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    d.pop("mean", None)
    return d


def point(run, forward):
    pba = {d: load(run, f"{d}_pba") for d in POSE}
    scenes = {d: list(pba[d]) for d in POSE}
    if not all(scenes.values()):
        return None
    timing, factor = pba, 1.0
    if os.path.isdir(run + "_t"):
        timing = {d: load(run + "_t", f"{d}_pba") for d in POSE}
    elif (m := re.search(r"_it(\d+)$", run)) and int(m.group(1)) > 200 and os.path.isdir(run[: m.start()] + "_it200"):
        timing, factor = {d: load(run[: m.start()] + "_it200", f"{d}_pba") for d in POSE}, int(m.group(1)) / 200
    t = []
    for d in POSE:
        opt = [factor * timing[d].get(s, pba[d][s])["opt_s"] for s in scenes[d]]
        if len(opt) > 1:  # the first scene of every dataset (a new process) includes one-off warm-up: use the median
            opt[0] = sorted(opt[1:])[len(opt[1:]) // 2]
        t += [forward[(d, s)] + o for s, o in zip(scenes[d], opt)]
    auc = sum(sum(pba[d][s]["auc03"] for s in scenes[d]) / len(scenes[d]) for d in POSE) / len(POSE)
    f1 = []
    for d in RECON:
        r = load(run, f"{d}_recon_unposed")
        if not r:
            return None
        f1.append(sum(v["fscore"] for v in r.values()) / len(r))
    return sum(t) / len(t), 100 * auc, 100 * sum(f1) / len(f1)


def main():
    os.chdir(REPO_ROOT)
    models = sys.argv[2:] or list(MODELS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), facecolor="#fcfcfb")
    for model in models:
        color = MODELS[model]
        forward = {}
        for line in open(f"workspace/evaluation/{model}/timings.jsonl"):
            r = json.loads(line)
            if not r["posed"] and "forward_s" in r:
                d, s = r["task"].split("/", 1)
                forward[(d, s.rsplit("/unposed", 1)[0])] = r["forward_s"]
        init = point(f"workspace/tto/{model}/init_dev", forward)
        for fam, (ls, _) in FAMILIES.items():
            runs = sorted((int(re.search(r"_it(\d+)$", p).group(1)), p) for p in glob.glob(f"workspace/tto/{model}/{fam}_it*") if re.search(r"_it\d+$", p))
            pts = [(0, init)] + [(it, point(run, forward)) for it, run in runs]
            pts = [(it, p) for it, p in pts if p]
            if len(pts) < 2:
                continue
            for ax, k in zip(axes, (1, 2)):
                ax.plot([p[0] for _, p in pts], [p[k] for _, p in pts], color=color, lw=2, ls=ls, marker="o", ms=6, mec="#fcfcfb", mew=1.5, zorder=3)
                for it, p in pts:
                    if fam == "v3":
                        ax.annotate("no PBA" if it == 0 else f"{it}", (p[0], p[k]), textcoords="offset points", xytext=(5, -11), fontsize=7.5, color="#52514e")
        for name, (marker, _) in POINTS.items():
            p = point(f"workspace/tto/{model}/{name}", forward)
            if p:
                for ax, k in zip(axes, (1, 2)):
                    ax.plot([p[0]], [p[k]], color=color, marker=marker, ms=8, mec="#0b0b0b", mew=0.8, ls="none", zorder=4)
        axes[0].plot([], [], color=color, lw=2, marker="o", ms=6, label=model)
    for ax, title in zip(axes, ("Pose AUC@3 (%), mean of 5 datasets", "Recon F1 (%), mean of HiRoom, 7-Scenes, ScanNet++")):
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("time per scene (s): DA3 forward + PBA, RTX 5090", color="#52514e")
        ax.set_title(title, fontsize=10, color="#0b0b0b", loc="left")
        ax.set_facecolor("#fcfcfb")
        ax.grid(color="#e4e3df", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#a8a7a1")
        ax.tick_params(colors="#52514e", labelsize=8)
    for fam, (ls, desc) in FAMILIES.items():
        axes[0].plot([], [], color="#52514e", lw=2, ls=ls, label=desc)
    for name, (marker, desc) in POINTS.items():
        axes[0].plot([], [], color="#52514e", marker=marker, ms=7, mec="#0b0b0b", mew=0.8, ls="none", label=desc)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("PBA iterations per level (labels) and optimiser vs. model size, dev subset", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(sys.argv[1], dpi=150)
    print(sys.argv[1])


if __name__ == "__main__":
    main()
