"""ETH3D per scene: pose AUC@3 of PBA with the pinhole camera (x) against the radial camera (y), one marker per scene
and model (run_pba_radial_eth3d.sh outputs). Points above the diagonal improve.

Usage: .venv/bin/python scripts/plot_radial_eth3d.py [root=workspace/tto/unlimited_eth3d] [out=results/pba/figures6/radial_eth3d.png]
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "workspace/tto/unlimited_eth3d"
out = sys.argv[2] if len(sys.argv) > 2 else "results/pba/figures6/radial_eth3d.png"
MODELS = [("DA3-SMALL", "DA3-Small", "#2a78d6", "o"), ("DA3-LARGE-1.1", "DA3-Large", "#eb6834", "s")]


def load(model, run):
    d = json.load(open(os.path.join(root, model, run, "metric_results", "eth3d_pba.json")))
    d.pop("mean", None)
    return d


fig, ax = plt.subplots(figsize=(4.6, 4.4), dpi=200)
ax.plot([0, 100], [0, 100], color="#9a9a96", lw=1, ls="--", zorder=1)
for model, label, color, marker in MODELS:
    if not os.path.exists(os.path.join(root, model, "radial", "metric_results", "eth3d_pba.json")):
        continue
    b, r = load(model, "base"), load(model, "radial")
    sc = sorted(set(b) & set(r))
    x = [100 * b[s]["auc03"] for s in sc]
    y = [100 * r[s]["auc03"] for s in sc]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    ax.scatter(x, y, s=34, color=color, marker=marker, edgecolor="white", linewidth=1.2, zorder=3,
               label=f"{label}: mean {mx:.1f} → {my:.1f}")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect("equal")
ax.set_xlabel("pose AUC@3 (%), pinhole camera (current PBA)")
ax.set_ylabel("pose AUC@3 (%), radial camera")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#9a9a96")
ax.tick_params(colors="#55554f", labelsize=8)
ax.grid(color="#e6e5e0", lw=0.6, zorder=0)
ax.legend(frameon=False, fontsize=8, loc="lower right", title="ETH3D, 11 scenes", title_fontsize=8)
fig.tight_layout()
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out)
print(out)
