"""Pose AUC@3 vs TTO time per scene for sparse BA (sba_fast.py), PBA and hybrids, one panel per model (RTX 3080).

    .venv/bin/python scripts/sba_plot.py <out.png> <spec.json>

spec: {"<model>": [[label, family, run dir, side (<0: label on the left), dy], ...], ...}; family in {"PBA", "sparse", "hybrid", "pycolmap"};
for a hybrid, "run dir" is "dirA+dirB" and the times add (TTO of A, then B started from it).
Env: YLABEL (y-axis label), XSCALE=linear (default log).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sba_sum import SETS, load  # noqa: E402

COL = {"PBA": "#2a78d6", "sparse": "#eb6834", "hybrid": "#1baf7a", "pycolmap": "#8a8984"}
MARK = {"PBA": "o", "sparse": "s", "hybrid": "D", "pycolmap": "^"}


def point(run):
    parts = run.split("+")
    ms = [load(p) for p in parts]
    m = ms[-1]
    if len(m) < 5:
        return None
    auc = np.mean([100 * np.mean([v["auc03"] for v in m[d].values()]) for d in SETS])
    t = 0.0
    for mm in ms:
        allv = [v for d in mm for v in mm[d].values()]
        t += float(np.mean([v["opt_s"] for v in allv]))
    return t, auc


def main():
    out, spec = sys.argv[1], json.load(open(sys.argv[2]))
    models = list(spec)
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models):
        seen = set()
        for item in spec[model]:
            label, fam, run, dx = item[:4]
            dy = item[4] if len(item) > 4 else 0
            p = point(run)
            if p is None:
                print("missing", run)
                continue
            ax.scatter(p[0], p[1], s=64, marker=MARK[fam], color=COL[fam], edgecolor="white", linewidth=1.5, zorder=3,
                       label=fam if fam not in seen else None)
            seen.add(fam)
            ax.annotate(label, p, xytext=(-7 if dx < 0 else 7, dy - 3), textcoords="offset points", fontsize=8, color="#3d3c38",
                        ha="right" if dx < 0 else "left")
        lin = os.environ.get("XSCALE", "log") == "linear"
        if lin:
            ax.set_xlim(left=0)
        else:
            ax.set_xscale("log")
        ax.set_title(model, fontsize=11, color="#1f1e1c")
        ax.set_xlabel("TTO time per scene (s, RTX 3080" + ("" if lin else ", log") + ")", fontsize=9, color="#52514e")
        ax.grid(True, which="major", color="#e4e3dd", linewidth=0.8)
        ax.tick_params(colors="#52514e", labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#b5b4ad")
    axes[0].set_ylabel(os.environ.get("YLABEL", "pose AUC@3, dev (5-set mean)"), fontsize=9, color="#52514e")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
