"""Plot: pose AUC@3 vs forward time at lower input resolution (RoPE stretched to the native grid), before and after
photometric TTO (T = 25), all 81 pose scenes, RTX 3080:
- released DA3 SMALL / BASE / LARGE-1.1 (da3_res.py, workspace/da3_res/run.sh),
- released DéjàView, K = 16 (eval_student.py DVLT_RES / DVLT_ROPE, ray fit),
- DéjàView after resolution distillation (R2, train_res.py; ray-fit extrinsics + camera-head intrinsics), the points
  of the fine-tuned column of the resolution-distillation table.
  .venv/bin/python scripts/distill_steps/plot_da3_res.py [--linear] [out.png]"""
import json
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W = os.path.join(REPO, "workspace")
LINEAR = "--linear" in sys.argv  # linear time axes (default log)
ARGS = [a for a in sys.argv[1:] if a != "--linear"]
OUT = ARGS[0] if ARGS else os.path.join(REPO, "results", "figures",
                                        "da3_res_compute_linear.png" if LINEAR else "da3_res_compute.png")
DS = ["hiroom", "7scenes", "scannetpp", "dtu64", "eth3d"]

# (legend name, colour (reference palette slots 1-5), [(evaluation dir, tto dir, point label)])
D3 = lambda m, c, lab: (f"evaluation/da3_res/{m}/r{c}", f"tto/da3_res/{m}/r{c}/pho_it25", lab)
DV = lambda n, lab: (f"evaluation/dvlt_fs/{n}", f"tto/dvlt_fs/{n}/pho_it25", lab)
RES = ("504", "420_rope", "336_rope", "252_rope")
SERIES = [
    ("DA3-Small", "#2a78d6", [D3("DA3-SMALL", c, c.split("_")[0]) for c in RES]),
    ("DA3-Base", "#eb6834", [D3("DA3-BASE", c, c.split("_")[0]) for c in RES]),
    ("DA3-Large-1.1", "#1baf7a", [D3("DA3-LARGE-1.1", c, c.split("_")[0]) for c in RES]),
    ("DéjàView K16", "#eda100", [DV("K16_3080", "504"), DV("rope_res420_K16", "420"), DV("rope_res336_K16", "336"),
                                 DV("rope_res252_K16", "252")]),
    ("DéjàView res-distilled", "#e87ba4", [DV("R2_K12_fhk", "504 K12"), DV("R2_res420_K12_fhk", "420 K12"),
                                           DV("R2_res336_K16_fhk", "336 K16")]),
]
# label placement: default TTO above, alone below; overrides (series, label, "tto" | "alone") -> (dx, dy) in points
UP, DOWN, LEFT, RIGHT = (0, 8), (0, -9), (-8, 0), (8, 0)
OFFSET = {
    # alone (left panel)
    ("DéjàView res-distilled", "420 K12", "alone"): LEFT, ("DéjàView res-distilled", "504 K12", "alone"): DOWN,
    ("DA3-Large-1.1", "336", "alone"): DOWN, ("DA3-Large-1.1", "252", "alone"): DOWN,
    ("DéjàView K16", "336", "alone"): LEFT, ("DA3-Large-1.1", "420", "alone"): DOWN,
    ("DéjàView res-distilled", "336 K16", "alone"): LEFT,
    # + TTO (right panel)
    ("DA3-Small", "336", "tto"): LEFT, ("DA3-Base", "252", "tto"): (8, -6), ("DA3-Small", "420", "tto"): UP,
    ("DA3-Large-1.1", "336", "tto"): DOWN, ("DéjàView K16", "420", "tto"): DOWN,
    ("DA3-Large-1.1", "420", "tto"): DOWN, ("DéjàView K16", "252", "tto"): LEFT,
    ("DA3-Large-1.1", "252", "tto"): DOWN, ("DéjàView K16", "336", "tto"): LEFT,
}
OFFSET_LIN = {**OFFSET,  # linear axes crowd the fast DA3 points near 0 s (alone) and 1-1.4 s (+ TTO)
              ("DA3-Small", "252", "alone"): RIGHT, ("DA3-Small", "336", "alone"): LEFT,
              ("DA3-Small", "420", "alone"): UP, ("DA3-Small", "504", "alone"): UP,
              ("DA3-Base", "252", "alone"): (6, -8), ("DéjàView K16", "252", "alone"): DOWN,
              ("DA3-Small", "252", "tto"): RIGHT, ("DA3-Small", "504", "tto"): RIGHT}
INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e4e3de"


def point(ev, tt):
    """(median forward s per scene, AUC@3 alone, AUC@3 after TTO, median TTO s per scene), dataset means over the 5
    pose sets. TTO time = opt_s (features + Adam + LM; image decoding excluded, knowledge/evaluation.md)."""
    alone = [100 * json.load(open(f"{W}/{ev}/metric_results/{d}_pose.json"))["mean"]["auc03"] for d in DS]
    after, opt = [], []
    for d in DS:
        j = json.load(open(f"{W}/{tt}/metric_results/{d}_pba.json"))
        v = [s for k, s in j.items() if isinstance(s, dict) and "auc03" in s and k != "mean"]
        after.append(100 * sum(s["auc03"] for s in v) / len(v))
        opt += [s["opt_s"] for s in v]
    t = [json.loads(line)["forward_s"] for line in open(f"{W}/{ev}/timings.jsonl")]
    return statistics.median(t), sum(alone) / 5, sum(after) / 5, statistics.median(opt)


plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2), dpi=200, sharey=True, gridspec_kw={"wspace": 0.05})
for name, col, runs in SERIES:
    pts = sorted(point(ev, tt) + (lab,) for ev, tt, lab in runs)
    print(name, [(round(p[0], 2), round(p[1], 1), round(p[2], 1), round(p[3], 2), p[4]) for p in pts])
    ax1.plot([q[0] for q in pts], [q[1] for q in pts], ls=(0, (4, 3)), lw=2, color=col, marker="o", ms=8,
             mfc="white", mew=2, label=name, zorder=3)
    ax2.plot([q[0] + q[3] for q in pts], [q[2] for q in pts], ls="-", lw=2, color=col, marker="o", ms=8,
             mec="white", mew=1.5, label=name, zorder=4)  # after TTO: forward + TTO time
    for fwd, a, b, topt, lab in pts:
        for ax, xx, y, kind, ink in ((ax2, fwd + topt, b, "tto", INK), (ax1, fwd, a, "alone", INK)):
            dx, dy = (OFFSET_LIN if LINEAR else OFFSET).get((name, lab, kind), UP)
            ax.annotate(lab, (xx, y), xytext=(dx, dy), textcoords="offset points",
                        ha="right" if dx < 0 else "left" if dx > 0 else "center",
                        va="top" if dy < 0 else "bottom" if dy > 0 else "center", fontsize=7.5, color=ink)

scale = "" if LINEAR else ", log scale"
for ax, lo, title, xl in ((ax1, 0.09, "Alone (dashed, hollow markers)", f"Forward time per scene, median (s{scale})"),
                          (ax2, 0.85, "+ photometric TTO, T = 25 (solid, filled markers)",
                           f"Forward + TTO time per scene, median (s{scale})")):
    if LINEAR:
        ax.set_xlim(0, 5.5)
        ax.set_xticks([0, 1, 2, 3, 4, 5])
    else:
        ax.set_xscale("log")
        ticks = [t for t in (0.125, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5) if t >= lo and (lo < 0.5 or t != 1.5)]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlim(lo, 6)
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.set_xlabel(xl)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
ax1.set_ylim(0, 80)
ax1.set_ylabel("Pose AUC@3 (%)")
ax2.tick_params(labelleft=True)
h, l = ax2.get_legend_handles_labels()
fig.legend(h, l, ncol=5, fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.995),
           handlelength=3)
fig.suptitle("Lower input resolution (RoPE stretched to the native grid), RTX 3080, all 81 pose scenes", x=0.01,
             y=0.94, ha="left", fontsize=11.5, color=INK)
fig.text(0.01, 0.01, "TTO = run_tto_pba.py T = 25, time = opt_s (image decoding excluded). Labels: longest input side "
         "(px); DéjàView res-distilled (R2, fine-tuned) also loop steps K. DA3-Large forward includes the da3_lowmem "
         "CPU offload.", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.92))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT)
print(OUT)
