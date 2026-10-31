"""Quality vs. compute of DA3 models with and without PBA (dev subset).

Usage: .venv/bin/python scripts/plot_compute.py <out.png>

Per model: the unrefined run (workspace/tto/<model>/init_dev) and the PBA runs workspace/tto/<model>/<family>_it<T>
for each family in FAMILIES (line style).
x = mean time per scene: DA3 forward (workspace/evaluation/<model>/timings.jsonl) + PBA optimisation (opt_s).
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
FAMILIES = {"v3": ("-", "VGG+NCC, 4096 px/frame"), "fast": ("--", "RGB Huber, 1024 px/frame")}


def load(run, name):
    p = os.path.join(run, "metric_results", f"{name}.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    d.pop("mean", None)
    return d


def point(model, run, forward):
    pba = {d: load(run, f"{d}_pba") for d in POSE}
    scenes = {d: list(pba[d]) for d in POSE}
    if not all(scenes.values()):
        return None
    t = [forward[(d, s)] + pba[d][s]["opt_s"] for d in POSE for s in scenes[d]]
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
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#fcfcfb")
    for model, color in MODELS.items():
        forward = {}
        for line in open(f"workspace/evaluation/{model}/timings.jsonl"):
            r = json.loads(line)
            if not r["posed"] and "forward_s" in r:
                d, s = r["task"].split("/", 1)
                forward[(d, s.rsplit("/unposed", 1)[0])] = r["forward_s"]
        init = point(model, f"workspace/tto/{model}/init_dev", forward)
        for fam, (ls, _) in FAMILIES.items():
            runs = sorted((int(re.search(r"_it(\d+)$", p).group(1)), p) for p in glob.glob(f"workspace/tto/{model}/{fam}_it*"))
            pts = [(0, init)] + [(it, point(model, run, forward)) for it, run in runs]
            pts = [(it, p) for it, p in pts if p]
            if len(pts) < 2:
                continue
            for ax, k in zip(axes, (1, 2)):
                xs, ys = [p[0] for _, p in pts], [p[k] for _, p in pts]
                ax.plot(xs, ys, color=color, lw=2, ls=ls, marker="o", ms=7, mec="#fcfcfb", mew=2, zorder=3)
                for it, p in pts:
                    label = "no PBA" if it == 0 else f"{it}"
                    if it == 0 and fam != "v3":
                        continue
                    ax.annotate(label, (p[0], p[k]), textcoords="offset points", xytext=(6, -12), fontsize=8, color="#52514e")
        axes[0].plot([], [], color=color, lw=2, marker="o", ms=7, label=model)
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
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("PBA iterations per level (labels) vs. model size, dev subset", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=150)
    print(sys.argv[1])


if __name__ == "__main__":
    main()
