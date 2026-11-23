"""Plot: pose AUC@3 vs forward time at lower input resolution (RoPE stretched to the native grid), before and after
photometric TTO (T = 25), all 81 pose scenes, RTX 3080:
- released DA3 SMALL / BASE / LARGE-1.1 (da3_res.py, workspace/da3_res/run.sh),
- released DéjàView, K = 16 (eval_student.py DVLT_RES / DVLT_ROPE, ray fit),
- DéjàView after resolution distillation (train_res.py; ray-fit extrinsics + camera-head intrinsics): R2 (the points
  of the fine-tuned column of the resolution-distillation table), S0 (self-teacher) and S1 (DA3-GIANT teacher) of
  knowledge/ideas/dejaview-res-distill-scale.md, and S0b (self-teacher with DL3DV + crop / zoom augmentation,
  camera-head poses: the multi-setting run and the 336 px K = 12 only run),
- DA3-GIANT-1.1 at 504 px as a horizontal reference (doesn't fit on the 3080, so it has no 3080 time).
  .venv/bin/python scripts/distill_steps/plot_da3_res.py [--linear] [out.png]

Machines: everything except S0 / S1 / S0b / GIANT ran on tum1 (RTX 3080). S0, S1 and S0b were evaluated on tum2 (RTX 5090;
metrics in results/pba/runs/dvlt_rs/, copy that folder to tum1 to plot there). Forward time depends on (res, K) and
not on the weights, so their points use the 3080 forward time of the released / R2 run with the same (res, K), and
their 5090 TTO time x TTO_3080_PER_5090. Values read from metric files are stored in CACHE; where a run's files are
missing or come from another GPU (tum2 has 5090 runs under some of the same names), the cached 3080 value is used."""
import json
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LINEAR = "--linear" in sys.argv  # linear time axes (default log)
ARGS = [a for a in sys.argv[1:] if a != "--linear"]
OUT = ARGS[0] if ARGS else os.path.join(REPO, "results", "figures",
                                        "da3_res_compute_linear.png" if LINEAR else "da3_res_compute.png")
CACHE = os.path.join(REPO, "results", "figures", "da3_res_compute_points.json")
DS = ["hiroom", "7scenes", "scannetpp", "dtu64", "eth3d"]
# 3080 / 5090 ratio of the median TTO time per scene, released DéjàView K16 at 504 / 420 / 336 / 252 px:
# 0.61 / 0.200, 0.58 / 0.201, 0.66 / 0.225, 0.76 / 0.254 s (2.9-3.1)
TTO_3080_PER_5090 = 2.95

# (legend name, colour (reference palette slots 1-7), [(evaluation dir, tto dir, point label)])
D3 = lambda m, c, lab: (f"workspace/evaluation/da3_res/{m}/r{c}", f"workspace/tto/da3_res/{m}/r{c}/pho_it25", lab)
DV = lambda n, lab: (f"workspace/evaluation/dvlt_fs/{n}", f"workspace/tto/dvlt_fs/{n}/pho_it25", lab)
RES = ("504", "420_rope", "336_rope", "252_rope")
SERIES = [
    ("DA3-Small", "#2a78d6", [D3("DA3-SMALL", c, c.split("_")[0]) for c in RES]),
    ("DA3-Base", "#eb6834", [D3("DA3-BASE", c, c.split("_")[0]) for c in RES]),
    ("DA3-Large-1.1", "#1baf7a", [D3("DA3-LARGE-1.1", c, c.split("_")[0]) for c in RES]),
    ("DéjàView K16", "#eda100", [DV("K16_3080", "504"), DV("rope_res420_K16", "420"), DV("rope_res336_K16", "336"),
                                 DV("rope_res252_K16", "252")]),
    ("DéjàView R2 (res-distilled)", "#e87ba4", [DV("R2_K12_fhk", "504 K12"), DV("R2_res420_K12_fhk", "420 K12"),
                                                DV("R2_res336_K16_fhk", "336 K16")]),
]
# S0 / S1 / S0b (tum2): (res, K) -> 3080 evaluation dir with the same (res, K), whose forward time is used. 252 K12 has
# no 3080 run, so it is left out (S0 46.5 -> 58.5, S1 42.2 -> 58.3; 5090 forward 0.22 s vs 0.24 s at K16). S0b uses
# camera-head poses, i.e. no RANSAC ray fit (5090: 0.09 s per scene median, 0.02 s below 25 views to 0.18 s at 100), so
# the ray-fit 3080 time puts its points slightly too far right.
FWD_3080 = {"504_K12": "R2_K12_fhk", "420_K12": "R2_res420_K12_fhk", "420_K8": "rope_res420_K8",
            "336_K16": "R2_res336_K16_fhk", "336_K12": "R2_res336_K12_fhk", "252_K16": "rope_res252_K16"}
# (legend name, colour, run name in results/pba/runs/dvlt_rs, marker); the 336-K12-only run shares the S0b colour
# (same recipe) and is told apart by its diamond marker
# S0 / S1 are left out since S0b (same teacher, more diverse data) supersedes them; to show them again add
# ("DéjàView S0 (self-teacher)", "#008300", "S0", "o"), ("DéjàView S1 (DA3-GIANT teacher)", "#6250d6", "S1", "o")
RS = [("DéjàView S0b (+ DL3DV, crops)", "#e34948", "S0b_mix-head", "o"),
      ("DéjàView S0b, 336 K12 only", "#e34948", "S0b_336K12-head", "D")]
GIANT = ("workspace/evaluation/DA3-GIANT-1.1", "workspace/tto/DA3-GIANT-1.1/pho_it25_all")
# label placement: default TTO above, alone below; overrides (series, label, "tto" | "alone") -> (dx, dy) in points
UP, DOWN, LEFT, RIGHT = (0, 8), (0, -9), (-8, 0), (8, 0)
OFFSET = {
    # alone (left panel)
    ("DéjàView R2 (res-distilled)", "420 K12", "alone"): LEFT,
    ("DéjàView R2 (res-distilled)", "504 K12", "alone"): DOWN,
    ("DA3-Large-1.1", "336", "alone"): DOWN, ("DA3-Large-1.1", "252", "alone"): DOWN,
    ("DéjàView K16", "336", "alone"): LEFT, ("DA3-Large-1.1", "420", "alone"): DOWN,
    ("DéjàView R2 (res-distilled)", "336 K16", "alone"): LEFT,
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
# zoom insets for the crowded distilled students (R2 / S0 / S1 share the (res, K) and hence the time):
# (x0, x1, y0, y1) of the data window, inset position in axes fraction (linear axes only)
ZOOM = {"alone": ((0.6, 1.9, 40, 61), (0.43, 0.06, 0.55, 0.42)),
        "tto": ((1.15, 2.55, 53, 73.4), (0.50, 0.06, 0.48, 0.47))}
DISTILLED = ("DéjàView R2", "DéjàView S0", "DéjàView S1")  # also matches S0b
COL_OFFSET_LOG = {("336 K12", "alone"): (-6, 6), ("336 K12", "tto"): (-6, 6), ("420 K8", "alone"): (-7, -2),
                  ("420 K8", "tto"): (4, -11), ("336 K16", "tto"): (2, 7)}  # log axes: (res, K) column labels
INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e4e3de"

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def gpu(ev):
    p = f"{REPO}/{ev}/runs.jsonl"
    return json.loads(open(p).readlines()[-1]).get("gpu", "") if os.path.exists(p) else ""


def alone(ev):
    return sum(100 * json.load(open(f"{REPO}/{ev}/metric_results/{d}_pose.json"))["mean"]["auc03"] for d in DS) / 5


def fwd(ev):
    return statistics.median(json.loads(line)["forward_s"] for line in open(f"{REPO}/{ev}/timings.jsonl"))


def tto(tt):
    """(AUC@3 after TTO, median TTO s per scene). TTO time = opt_s (features + Adam + LM; image decoding excluded,
    knowledge/evaluation.md)."""
    after, opt = [], []
    for d in DS:
        j = json.load(open(f"{REPO}/{tt}/metric_results/{d}_pba.json"))
        v = [s for k, s in j.items() if isinstance(s, dict) and "auc03" in s and k != "mean"]
        after.append(100 * sum(s["auc03"] for s in v) / len(v))
        opt += [s["opt_s"] for s in v]
    return sum(after) / 5, statistics.median(opt)


def point(ev, tt):
    """(median forward s per scene, AUC@3 alone, AUC@3 after TTO, median TTO s per scene) on the 3080, dataset means
    over the 5 pose sets; from the metric files of a 3080 run, else from CACHE."""
    if "3080" in gpu(ev) and os.path.exists(f"{REPO}/{tt}/metric_results"):
        cache[ev], cache[tt] = dict(fwd=fwd(ev), alone=alone(ev)), dict(zip(("after", "opt"), tto(tt)))
    return cache[ev]["fwd"], cache[ev]["alone"], cache[tt]["after"], cache[tt]["opt"]


def point_rs(run, cfg):
    """S0 / S1 (5090 metrics) on the 3080 time axis: forward time of the 3080 run with the same (res, K)."""
    d = f"results/pba/runs/dvlt_rs/{run}_{cfg}"
    ev = f"workspace/evaluation/dvlt_fs/{FWD_3080[cfg]}"
    if "3080" in gpu(ev):
        cache.setdefault(ev, {})["fwd"] = fwd(ev)
    after, opt = tto(f"{d}/pho_it25")
    return cache[ev]["fwd"], alone(f"{d}/eval"), after, opt * TTO_3080_PER_5090


series = [(name, col, sorted(point(ev, tt) + (lab,) for ev, tt, lab in runs)) for name, col, runs in SERIES]
MARKER = {}
if os.path.isdir(f"{REPO}/results/pba/runs/dvlt_rs"):
    for name, col, run, mk in RS:
        cfgs = [c for c in FWD_3080 if os.path.isdir(f"{REPO}/results/pba/runs/dvlt_rs/{run}_{c}/pho_it25")]
        series.append((name, col, sorted(point_rs(run, c) + (c.replace("_", " "),) for c in cfgs)))
        MARKER[name] = mk
else:
    print("results/pba/runs/dvlt_rs missing (copy it from tum2): S0 / S1 / S0b left out")
giant = None
if all(os.path.exists(f"{REPO}/{p}/metric_results") for p in GIANT):
    cache["giant"] = dict(alone=alone(GIANT[0]), after=tto(GIANT[1])[0])
giant = cache.get("giant")
os.makedirs(os.path.dirname(CACHE), exist_ok=True)
json.dump(cache, open(CACHE, "w"), indent=1, sort_keys=True)

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.9), dpi=200, sharey=True, gridspec_kw={"wspace": 0.05})


def label(ax, lab, xx, y, off, size=7.5):
    dx, dy = off
    ax.annotate(lab, (xx, y), xytext=(dx, dy), textcoords="offset points", fontsize=size, color=INK,
                ha="right" if dx < 0 else "left" if dx > 0 else "center",
                va="top" if dy < 0 else "bottom" if dy > 0 else "center")


def draw(ax, kind, zoom=None, inset=False):
    """All series on ax; kind "alone" (forward time) or "tto" (forward + TTO time). Labels: on the main axes for the
    points outside the zoom window; in the inset for the points inside it (one (res, K) label per column of
    distilled students, above its highest marker)."""
    win = lambda xx, y: bool(zoom) and zoom[0] <= xx <= zoom[1] and zoom[2] <= y <= zoom[3]
    cols = {}
    for name, col, pts in series:
        xs = [q[0] + (q[3] if kind == "tto" else 0) for q in pts]
        ys = [q[2] if kind == "tto" else q[1] for q in pts]
        mk = MARKER.get(name, "o")
        ms = 7 if mk == "D" else 8
        if kind == "tto":
            ax.plot(xs, ys, ls="-", lw=2, color=col, marker=mk, ms=ms, mec="white", mew=1.5, label=name, zorder=4)
        else:
            ax.plot(xs, ys, ls=(0, (4, 3)), lw=2, color=col, marker=mk, ms=ms, mfc="white", mew=2, label=name,
                    zorder=3)
        for xx, y, q in zip(xs, ys, pts):
            if win(xx, y) != inset:
                continue
            if name.startswith(DISTILLED):  # one label per (res, K) column, in the inset or on the main axes
                c = cols.setdefault(q[4], [xx, y])
                c[0], c[1] = (xx, y) if y > c[1] else c
            elif inset:
                label(ax, q[4], xx, y, DOWN)
            elif not name.startswith("DéjàView S1"):  # S1 sits at S0's (res, K) / time
                label(ax, q[4], xx, y, (OFFSET_LIN if LINEAR else OFFSET).get((name, q[4], kind), UP))
    for lab, (xx, y) in cols.items():
        label(ax, lab, xx, y, (0, 7) if zoom else COL_OFFSET_LOG.get((lab, kind), (0, 7)))


for ax, kind in ((ax1, "alone"), (ax2, "tto")):
    zoom = ZOOM[kind][0] if LINEAR and "S0" in "".join(s[0] for s in series) else None
    draw(ax, kind, zoom)
    if zoom:
        axin = ax.inset_axes(ZOOM[kind][1])
        draw(axin, kind, zoom, inset=True)
        axin.set_xlim(*zoom[:2])
        axin.set_ylim(*zoom[2:])
        axin.tick_params(labelsize=7.5)
        axin.grid(True, color=GRID, lw=0.8)
        axin.set_axisbelow(True)
        axin.set_title("zoom into the grey box", fontsize=8, color=MUTED, loc="left")
        for sp in axin.spines.values():
            sp.set_color(MUTED)
        ax.add_patch(plt.Rectangle((zoom[0], zoom[2]), zoom[1] - zoom[0], zoom[3] - zoom[2], fill=False,
                                   ec=MUTED, lw=1, ls=(0, (3, 2)), zorder=5))
if giant:
    for ax, y in ((ax1, giant["alone"]), (ax2, giant["after"])):
        ax.axhline(y, color=MUTED, lw=1, ls=(0, (1, 2)), zorder=2)
        ax.annotate(f"DA3-GIANT-1.1, 504 px: {y:.1f} (no 3080 time)", (0.99, y), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", ha="right", va="bottom", fontsize=7.5, color=MUTED)

scale = "" if LINEAR else ", log scale"
for ax, lo, title, xl in ((ax1, 0.09, "Alone", f"Forward time per scene, median (s{scale})"),
                          (ax2, 0.85, "+ photometric TTO, T = 25",
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
fig.legend(h, l, ncol=4, fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.995),
           handlelength=3)
fig.suptitle("Lower input resolution, RTX 3080, all 81 pose scenes", x=0.01,
             y=0.88 if len(h) > 8 else 0.905, ha="left", fontsize=11.5, color=INK)  # legend rows: 4 per row
fig.subplots_adjust(left=0.065, right=0.99, top=0.82 if len(h) > 8 else 0.845, bottom=0.09, wspace=0.08)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT)
print(OUT)
