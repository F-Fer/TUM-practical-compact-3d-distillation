"""Plot: released DA3 models at lower input resolution (RoPE stretched, da3_res.py), pose AUC@3 vs forward time,
before and after photometric TTO (T = 25), all 81 pose scenes, RTX 3080. Reads the metrics via da3_res_compare.py.
  .venv/bin/python scripts/distill_steps/plot_da3_res.py [out.png]"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.argv, ARGS = sys.argv[:1], sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402

with redirect_stdout(io.StringIO()):  # the compare script prints its tables at import
    import da3_res_compare as C  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = ARGS[0] if ARGS else os.path.join(REPO, "results", "figures", "da3_res_compute.png")
COLORS = {"DA3-SMALL": "#2a78d6", "DA3-BASE": "#eb6834", "DA3-LARGE-1.1": "#1baf7a"}  # reference palette slots 1-3
NAMES = {"DA3-SMALL": "DA3-Small", "DA3-BASE": "DA3-Base", "DA3-LARGE-1.1": "DA3-Large-1.1"}
BELOW = {("DA3-SMALL", "420"), ("DA3-BASE", "252"), ("DA3-BASE", "504"), ("DA3-LARGE-1.1", "252")}
INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e4e3de"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK})
fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=200)
for m in C.MODELS:
    pts = []
    for c in ["504", "420_rope", "336_rope", "252_rope"]:
        p, t = C.pose(f"{m}/r{c}"), C.tto(f"{m}/r{c}")
        if p and t:
            pts.append((p[1], p[0], t["all"][1], c.split("_")[0]))
    pts.sort()
    x = [q[0] for q in pts]
    col = COLORS[m]
    ax.plot(x, [q[1] for q in pts], ls=(0, (4, 3)), lw=2, color=col, marker="o", ms=8, mfc="white", mew=2,
            label=f"{NAMES[m]}, alone", zorder=3)
    ax.plot(x, [q[2] for q in pts], ls="-", lw=2, color=col, marker="o", ms=8, mec="white", mew=1.5,
            label=f"{NAMES[m]} + TTO", zorder=4)
    for xi, a, b, r in pts:
        below = (m, r) in BELOW  # after-TTO labels that would sit on another line
        ax.annotate(r, (xi, b), xytext=(0, -9 if below else 8), textcoords="offset points", ha="center",
                    va="top" if below else "bottom", fontsize=8, color=INK)
        ax.annotate(r, (xi, a), xytext=(0, -9), textcoords="offset points", ha="center", va="top", fontsize=8,
                    color=MUTED)

ax.set_xscale("log")
ticks = [0.125, 0.25, 0.5, 1, 2, 4]
ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
ax.minorticks_off()
ax.set_xlim(0.09, 6.5)
ax.set_ylim(0, 80)
ax.set_xlabel("Forward time per scene, median (s, RTX 3080, log scale)")
ax.set_ylabel("Pose AUC@3 (%)")
ax.set_title("Released DA3 at lower input resolution (RoPE stretched), before and after photometric TTO",
             fontsize=10.5, loc="left", color=INK)
ax.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.0),
          handlelength=4)
fig.text(0.01, 0.01, "All 81 pose scenes of the DA3 benchmark; TTO = run_tto_pba.py T = 25. Labels: longest input "
         "side (px). DA3-Large times include the da3_lowmem CPU offload.", fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT)
print(OUT)
