"""Tables and figures of DA3 quality vs. the number of input views (outputs of run_views.py).

Usage: .venv/bin/python scripts/plot_views.py <out dir> [views root, default workspace/views]

Writes <out dir>/views.md (tables) and views_quality.png, views_targets.png, views_compute.png. A dataset gets a value
for N only if all its scenes have at least N frames (7-Scenes and ScanNet++: all N; HiRoom and ETH3D: N <= 8;
DTU64: N <= 32), plus "all" (the benchmark's own input, at most 100 frames). Figures use 7-Scenes and ScanNet++,
the datasets with 100 frames in every scene.
"""

import json
import os
import statistics
import sys

import matplotlib.pyplot as plt

MODELS = {"DA3-SMALL": "#2a78d6", "DA3-BASE": "#eb6834", "DA3-LARGE-1.1": "#1baf7a"}  # categorical slots 1-3, as plot_compute.py
VIEWS = ["2", "4", "8", "16", "32", "64", "all"]
TARGETS = 8
POSE = {"hiroom": "HiR", "7scenes": "7S", "scannetpp": "SN++", "dtu64": "DTU64", "eth3d": "ETH"}
RECON = {"hiroom": "HiR", "7scenes": "7S", "scannetpp": "SN++"}
FULL = ["7scenes", "scannetpp"]  # 100 frames in every scene
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(root, model, name, data, mode):
    p = os.path.join(root, model, name, "metric_results", f"{data}_{mode}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    d.pop("mean", None)
    return d


def mean(root, model, name, data, mode, key):
    """Mean of key over the dataset's scenes, None unless every scene of the 'all' input is there."""
    d = load(root, model, name, data, mode)
    ref = load(root, model, "nall", data, "pose")
    if not d or not ref or set(d) != set(ref):
        return None
    return 100 * sum(v[key] for v in d.values()) / len(d)


def pooled(root, model, name, mode, key, datas=FULL):
    """Mean over the scenes of several datasets (each scene counts once)."""
    vals = []
    for data in datas:
        d = load(root, model, name, data, mode)
        if not d:
            return None
        vals += [v[key] for v in d.values()]
    return 100 * sum(vals) / len(vals)


def timings(root, model):
    """{N: (median forward s, median peak GiB)} over the 7-Scenes and ScanNet++ scenes (all 378x504)."""
    out = {}
    for line in open(os.path.join(root, model, "timings.jsonl")):
        r = json.loads(line)
        if not r["task"]:  # warm-up
            continue
        parts = r["task"].split("/")  # ../n<N>/model_results/<dataset>/<scene>/unposed
        if parts[3] in FULL:
            out.setdefault(parts[1][1:], []).append((r["forward_s"], r["peak_mem_gib"]))
    return {n: (statistics.median(t for t, _ in v), statistics.median(m for _, m in v)) for n, v in out.items()}


def fmt(x):
    return "-" if x is None else f"{x:.1f}"


def tables(root, models):
    lines = ["## Pose AUC@3 / AUC@30 (%) vs. N input views", ""]
    lines += ["| model | N | " + " | ".join(POSE.values()) + " | 7S+SN++ |", "|---" * (len(POSE) + 3) + "|"]
    for model in models:
        for v in VIEWS:
            n = f"n{v}"
            cells = [f"{fmt(mean(root, model, n, d, 'pose', 'auc03'))} / {fmt(mean(root, model, n, d, 'pose', 'auc30'))}" for d in POSE]
            both = f"{fmt(pooled(root, model, n, 'pose', 'auc03'))} / {fmt(pooled(root, model, n, 'pose', 'auc30'))}"
            lines.append(f"| {model} | {v} | " + " | ".join(cells).replace("- / -", "-") + f" | {both} |")
    lines += ["", "## Recon (unposed) F1 / precision / recall (%) vs. N input views", ""]
    lines += ["| model | N | " + " | ".join(RECON.values()) + " | 7S+SN++ |", "|---" * (len(RECON) + 3) + "|"]
    for model in models:
        for v in VIEWS[1:]:
            n = f"n{v}"
            cells = [" / ".join(fmt(mean(root, model, n, d, "recon_unposed", k)) for k in ("fscore", "precision", "recall")) for d in RECON]
            both = " / ".join(fmt(pooled(root, model, n, "recon_unposed", k)) for k in ("fscore", "precision", "recall"))
            lines.append(f"| {model} | {v} | " + " | ".join(cells).replace("- / - / -", "-") + f" | {both} |")
    lines += ["", f"## The same {TARGETS} target views with N - {TARGETS} more views as context (7S+SN++ pooled)", ""]
    lines += ["| model | N | AUC@3 | AUC@30 | F1 | precision | recall |", "|---" * 7 + "|"]
    for model in models:
        for v in VIEWS[2:]:
            n = f"n{v}" if v == f"{TARGETS}" else f"n{v}_t{TARGETS}"
            cells = [pooled(root, model, n, "pose", k) for k in ("auc03", "auc30")]
            cells += [pooled(root, model, n, "recon_unposed", k) for k in ("fscore", "precision", "recall")]
            lines.append(f"| {model} | {v} | " + " | ".join(fmt(c) for c in cells) + " |")
    lines += ["", "## DA3 forward time and peak GPU memory (median over 7S and SN++ scenes, 378x504, RTX 5090)", ""]
    lines += ["| model | " + " | ".join(f"N={v}" for v in VIEWS) + " |", "|---" * (len(VIEWS) + 1) + "|"]
    for model in models:
        t = timings(root, model)
        lines.append(f"| {model} | " + " | ".join(f"{t[v][0]:.2f} s, {t[v][1]:.1f} GiB" if v in t else "-" for v in VIEWS) + " |")
    return "\n".join(lines) + "\n"


def style(ax, title, xlabel="input views N"):
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 100], ["2", "4", "8", "16", "32", "64", "100"])
    ax.minorticks_off()
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8.5)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#a8a7a1")
    ax.tick_params(colors=MUTED, labelsize=8)


def x_of(v):
    return 100 if v == "all" else int(v)


def quality_figure(root, models, path, targets=False):
    panels = [("pose", "auc03", "Pose AUC@3 (%)"), ("pose", "auc30", "Pose AUC@30 (%)"),
              ("recon_unposed", "fscore", "Recon F1 (%)"), ("recon_unposed", "precision", "Recon precision (%)"),
              ("recon_unposed", "recall", "Recon recall (%)")]
    rows = [("7scenes", "7-Scenes (7 scenes)"), ("scannetpp", "ScanNet++ (20 scenes)")]
    fig, axes = plt.subplots(len(rows), len(panels), figsize=(16, 6.6), facecolor=SURFACE)
    for model in models:
        for (data, label), row in zip(rows, axes):
            for (mode, key, title), ax in zip(panels, row):
                pts = []
                for v in VIEWS[2:] if targets else VIEWS:
                    n = f"n{v}" if not targets or v == f"{TARGETS}" else f"n{v}_t{TARGETS}"
                    y = mean(root, model, n, data, mode, key)
                    if y is not None:
                        pts.append((x_of(v), y))
                ax.plot(*zip(*pts), color=MODELS[model], lw=2, marker="o", ms=5.5, mec=SURFACE, mew=1.2, label=model) if pts else None
                style(ax, f"{label}: {title}")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    if targets:
        fig.suptitle(f"The same {TARGETS} evenly spaced target views, predicted together with N - {TARGETS} more views; "
                     f"scored on the {TARGETS} targets only (pose: their 28 pairs, recon: fused from their depth)", fontsize=10.5, x=0.01, ha="left", color=INK)
    else:
        fig.suptitle("DA3 without TTO on N evenly spaced input views; pose over all N(N-1)/2 pairs, recon fused from the N views "
                     "vs. the full GT scene (recon needs >= 3 views for the Sim(3) alignment)", fontsize=10.5, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=150)


def compute_figure(root, models, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor=SURFACE)
    for model in models:
        t = timings(root, model)
        vs = [v for v in VIEWS if v in t]
        xs = [x_of(v) for v in vs]
        kw = dict(color=MODELS[model], lw=2, marker="o", ms=5.5, mec=SURFACE, mew=1.2, label=model)
        axes[0].plot(xs, [t[v][0] for v in vs], **kw)
        axes[1].plot(xs, [t[v][1] for v in vs], **kw)
        pts = [(t[v][0], pooled(root, model, f"n{v}", "pose", "auc03")) for v in vs]
        pts = [(a, b) for a, b in pts if b is not None]
        axes[2].plot(*zip(*pts), **kw)
        for v, (a, b) in zip(vs, pts):
            axes[2].annotate(str(x_of(v)), (a, b), textcoords="offset points", xytext=(4, -10), fontsize=7, color=MUTED)
    style(axes[0], "DA3 forward time (s)")
    axes[0].set_yscale("log")
    style(axes[1], "Peak GPU memory (GiB, with da3_lowmem)")
    style(axes[2], "Pose AUC@3 (%), 7S+SN++ vs. forward time (labels: N)", xlabel="DA3 forward time per scene (s), RTX 5090")
    axes[2].set_xscale("log")
    axes[2].set_xticks([0.03, 0.1, 0.3, 1, 3], ["0.03", "0.1", "0.3", "1", "3"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)


def main():
    out = sys.argv[1]
    root = sys.argv[2] if len(sys.argv) > 2 else "workspace/views"
    models = [m for m in MODELS if os.path.isdir(os.path.join(root, m))]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "views.md"), "w") as f:
        f.write(tables(root, models))
    quality_figure(root, models, os.path.join(out, "views_quality.png"))
    quality_figure(root, models, os.path.join(out, "views_targets.png"), targets=True)
    compute_figure(root, models, os.path.join(out, "views_compute.png"))
    print(open(os.path.join(out, "views.md")).read())


if __name__ == "__main__":
    main()
