"""Two figures of pose AUC@3 vs time per scene, alone and after photometric TTO (T = 25), all 81 pose scenes, RTX 3080
time axis (linear):
1. results/figures/res_distill_dejaview.png, resolution distillation of DéjàView (train_res.py,
   knowledge/ideas/dejaview-speed.md and dejaview-res-distill-scale.md):
   - released DéjàView K = 16 at 504 / 420 / 336 / 252 px (RoPE stretched, ray fit),
   - v0 = R2 (ARKit only, 3 h on the 3080; ray-fit extrinsics + camera-head intrinsics),
   - v1 = S0b_mix (+ DL3DV, crop / zoom, 5 h on a 5090; ray-fit extrinsics + camera-head intrinsics),
   - v1 336 K12 only = S0b_336K12 (same pose mode),
   - v0 DA3-Giant teacher = S1 (ray-fit extrinsics + camera-head intrinsics). S1 was trained on S0's data (ARKit +
     outdoor, fixed clips, ~4x v0's), not on v1's; its like-for-like control is S0. 420 K8 is left out (no other run
     has K8).
2. results/figures/res_distill_vs_da3.png: v1 next to the released DA3 SMALL / BASE / LARGE-1.1 at
   504 / 420 / 336 / 252 px (da3_res.py, RoPE stretched).
  .venv/bin/python scripts/distill_steps/plot_da3_res.py

Machines: the released / v0 / DA3 runs ran on tum1 (RTX 3080). v1 and S1 were evaluated on tum2 (RTX 5090; metrics in
results/pba/runs/dvlt_rs/). Forward time depends on (res, K) and not on the weights, so their points use the 3080
forward time of the released / R2 run with the same (res, K) (with the same ray fit) and their 5090 TTO time x
TTO_3080_PER_5090. All DéjàView points use ray-fit poses; camera-head poses (runs "<name>-head") cost ~2.5-4 AUC@3
alone and ~0-1.5 after TTO. Values read from metric files are stored in
CACHE; where a run's files are missing or come from another GPU, the cached 3080 value is used."""
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(REPO, "results", "figures")
CACHE = os.path.join(FIG, "da3_res_compute_points.json")
DS = ["hiroom", "7scenes", "scannetpp", "dtu64", "eth3d"]
# 3080 / 5090 ratio of the median TTO time per scene, released DéjàView K16 at 504 / 420 / 336 / 252 px:
# 0.61 / 0.200, 0.58 / 0.201, 0.66 / 0.225, 0.76 / 0.254 s (2.9-3.1)
TTO_3080_PER_5090 = 2.95
# tum2 runs: (res, K) -> 3080 evaluation dir with the same (res, K), whose forward time is used
FWD_3080 = {"504_K12": "R2_K12_fhk", "420_K12": "R2_res420_K12_fhk", "336_K16": "R2_res336_K16_fhk",
            "336_K12": "R2_res336_K12_fhk", "252_K16": "rope_res252_K16"}
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


def point(ev, tt, lab):
    """(median forward s per scene, AUC@3 alone, AUC@3 after TTO, median TTO s per scene, label) on the 3080, dataset
    means over the 5 pose sets; from the metric files of a 3080 run, else from CACHE."""
    if "3080" in gpu(ev) and os.path.exists(f"{REPO}/{tt}/metric_results"):
        cache[ev], cache[tt] = dict(fwd=fwd(ev), alone=alone(ev)), dict(zip(("after", "opt"), tto(tt)))
    return cache[ev]["fwd"], cache[ev]["alone"], cache[tt]["after"], cache[tt]["opt"], lab


def da3(model):
    return [point(f"workspace/evaluation/da3_res/{model}/r{c}", f"workspace/tto/da3_res/{model}/r{c}/pho_it25",
                  c.split("_")[0]) for c in ("504", "420_rope", "336_rope", "252_rope")]


def dv(runs):
    return [point(f"workspace/evaluation/dvlt_fs/{n}", f"workspace/tto/dvlt_fs/{n}/pho_it25", lab) for n, lab in runs]


def rs(run, cfgs):
    """tum2 run (5090 metrics in results/pba/runs/dvlt_rs/<run>_<cfg>) on the 3080 time axis."""
    pts = []
    for c in cfgs:
        d = f"results/pba/runs/dvlt_rs/{run}_{c}"
        if not os.path.isdir(f"{REPO}/{d}/pho_it25"):
            print(f"missing {d} (copy results/pba/runs/dvlt_rs from tum2)")
            continue
        ev = f"workspace/evaluation/dvlt_fs/{FWD_3080[c]}"
        if "3080" in gpu(ev):
            cache.setdefault(ev, {})["fwd"] = fwd(ev)
        after, opt = tto(f"{d}/pho_it25")
        pts.append((cache[ev]["fwd"], alone(f"{d}/eval"), after, opt * TTO_3080_PER_5090, c.replace("_", " ")))
    return pts


# (legend name, colour (reference palette slots), marker, points); one colour per entity across both figures
V1 = ("DéjàView v1", "#e34948", "o", rs("S0b_mix", ["252_K16", "336_K12", "336_K16", "420_K12", "504_K12"]))
UP, DOWN, LEFT, RIGHT = (0, 7), (0, -9), (-7, 0), (7, 0)
# label offsets in points: (panel, label) for the v0 / v1 columns (one label per (res, K) above its highest marker),
# (panel, series, label) for the other points; default UP
FIGS = [
    dict(out="res_distill_dejaview.png", title="Resolution distillation of DéjàView", xmax=4.5, ncol=5, series=[
        ("DéjàView (released)", "#eda100", "o",
         dv([("K16_3080", "504 K16"), ("rope_res420_K16", "420 K16"), ("rope_res336_K16", "336 K16"),
             ("rope_res252_K16", "252 K16")])),
        ("DéjàView v0", "#008300", "s",
         dv([("R2_K12_fhk", "504 K12"), ("R2_res420_K12_fhk", "420 K12"), ("R2_res336_K16_fhk", "336 K16")])),
        V1,
        ("DéjàView v1, 336 K12 only", "#e34948", "D", rs("S0b_336K12", ["336_K12"])),
        ("DéjàView v0, DA3-Giant teacher", "#6250d6", "^", rs("S1", ["252_K16", "336_K12", "336_K16", "420_K12"]))],
         offsets={("alone", "336 K12"): LEFT, ("alone", "252 K16"): LEFT, ("tto", "252 K16"): LEFT,
                  ("tto", "336 K12"): LEFT, ("tto", "336 K16"): (2, 8),
                  ("tto", "420 K12"): (5, 6), ("alone", "DéjàView (released)", "336 K16"): RIGHT,
                  ("alone", "DéjàView (released)", "252 K16"): RIGHT,
                  ("tto", "DéjàView (released)", "420 K16"): (6, -6),
                  ("tto", "DéjàView (released)", "336 K16"): RIGHT,
                  ("tto", "DéjàView (released)", "252 K16"): RIGHT}),
    dict(out="res_distill_vs_da3.png", title="DéjàView v1 vs Depth Anything 3 at lower input resolution", xmax=5.5,
         ncol=4, series=[("DA3-Small", "#2a78d6", "o", da3("DA3-SMALL")), ("DA3-Base", "#eb6834", "o", da3("DA3-BASE")),
                         ("DA3-Large-1.1", "#1baf7a", "o", da3("DA3-LARGE-1.1")), V1],
         offsets={("alone", "336 K12"): LEFT, ("tto", "336 K12"): LEFT, ("tto", "336 K16"): (2, 8),
                  ("alone", "252 K16"): LEFT, ("tto", "252 K16"): LEFT,
                  ("tto", "420 K12"): (5, 6), ("alone", "DA3-Small", "252"): RIGHT, ("alone", "DA3-Small", "336"): LEFT,
                  ("alone", "DA3-Base", "252"): (6, -8), ("alone", "DA3-Large-1.1", "252"): DOWN,
                  ("alone", "DA3-Large-1.1", "336"): DOWN, ("alone", "DA3-Large-1.1", "420"): DOWN,
                  ("tto", "DA3-Small", "252"): RIGHT, ("tto", "DA3-Small", "336"): LEFT,
                  ("tto", "DA3-Small", "420"): (5, 5), ("tto", "DA3-Small", "504"): RIGHT,
                  ("tto", "DA3-Base", "252"): (8, -6), ("tto", "DA3-Base", "504"): RIGHT,
                  ("tto", "DA3-Large-1.1", "252"): DOWN, ("tto", "DA3-Large-1.1", "336"): DOWN,
                  ("tto", "DA3-Large-1.1", "420"): DOWN}),
]
os.makedirs(FIG, exist_ok=True)
json.dump(cache, open(CACHE, "w"), indent=1, sort_keys=True)

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK})


def label(ax, lab, xx, y, off):
    dx, dy = off
    ax.annotate(lab, (xx, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=INK,
                ha="right" if dx < 0 else "left" if dx > 0 else "center",
                va="top" if dy < 0 else "bottom" if dy > 0 else "center")


def draw(ax, kind, series, offsets):
    """All series on ax; kind "alone" (forward time) or "tto" (forward + TTO time). Points of the distilled runs (v0 /
    v1) with the same (res, K) share one label (above the highest marker), the others get one label each."""
    top = {}
    for name, col, mk, pts in series:
        pts = sorted(pts)
        xs = [q[0] + (q[3] if kind == "tto" else 0) for q in pts]
        ys = [q[2] if kind == "tto" else q[1] for q in pts]
        ms = {"D": 7, "s": 7, "^": 8.5}.get(mk, 8)
        if kind == "tto":
            ax.plot(xs, ys, ls="-", lw=2, color=col, marker=mk, ms=ms, mec="white", mew=1.5, label=name, zorder=4)
        else:
            ax.plot(xs, ys, ls=(0, (4, 3)), lw=2, color=col, marker=mk, ms=ms, mfc="white", mew=2, label=name,
                    zorder=3)
        for xx, y, q in zip(xs, ys, pts):
            key = (q[4],) if name.startswith("DéjàView v") else (name, q[4])
            if key not in top or y > top[key][1]:
                top[key] = (xx, y)
    for key, (xx, y) in top.items():
        label(ax, key[-1], xx, y, offsets.get((kind, *key), UP))


for f in FIGS:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.4), dpi=200, sharey=True)
    for ax, kind, title, xl in ((ax1, "alone", "Alone", "Forward time per scene, median (s)"),
                                (ax2, "tto", "+ photometric TTO, T = 25", "Forward + TTO time per scene, median (s)")):
        draw(ax, kind, f["series"], f["offsets"])
        ax.set_xlim(0, f["xmax"])
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        ax.set_xlabel(xl)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    ax1.set_ylim(0, 80)
    ax1.set_ylabel("Pose AUC@3 (%)")
    ax2.tick_params(labelleft=True)
    fig.suptitle(f"{f['title']} · RTX 3080, all 81 pose scenes", x=0.01, y=0.985, ha="left", fontsize=12,
                 color=INK)
    h, l = ax2.get_legend_handles_labels()
    fig.legend(h, l, ncol=f["ncol"], fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(0.005, 0.95),
               handlelength=3, columnspacing=2.5)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.86, bottom=0.09, wspace=0.08)
    out = os.path.join(FIG, f["out"])
    fig.savefig(out)
    print(out)
