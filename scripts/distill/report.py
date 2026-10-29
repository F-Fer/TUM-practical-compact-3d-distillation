"""Summarize the distillation runs: markdown tables with the official held-out ScanNet++ metrics
(10 val scenes x 100 frames) and 100-frame validation curves.

    uv run scripts/distill/report.py  -> workspace/distill/results.md, workspace/distill/figures/*.png"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO_ROOT, WS  # noqa: E402
from split_metrics import split_metrics  # noqa: E402

# reference categorical palette (fixed order), light surface
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REFS = {
    "DA3-SMALL": ("workspace/evaluation/DA3-SMALL", "depth-anything/DA3-SMALL"),
    "DA3-BASE": ("workspace/evaluation/DA3-BASE", "depth-anything/DA3-BASE"),
    "DA3-LARGE-1.1": ("workspace/evaluation/DA3-LARGE-1.1", "depth-anything/DA3-LARGE-1.1"),
    "DéjàView K=4": ("workspace/distill/eval/dvlt-K4", "dvlt-K4"),
    "DéjàView K=6": ("workspace/distill/eval/dvlt-K6", "dvlt-K6"),
    "DéjàView K=8": ("workspace/distill/eval/dvlt-K8", "dvlt-K8"),
    "DéjàView K=12": ("workspace/evaluation/dvlt", "dvlt-K12"),
}
# DéjàView students trained/validated on cv2-resized images instead of DéjàView's own preprocessing
CV2_INPUTS = {"v2_dvlt6_out", "dvlt4_out", "dvlt4_feat", "dvlt4u_out", "dvlt6_out"}
COLS = ["pose.auc03", "pose.auc30", "recon_unposed.fscore", "recon_posed.fscore"]


def read_jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def fmt(m, k):
    return f"{m[k]:.1f}" if k in m else "-"


def run_desc(cfg):
    views = f"{cfg['min_views']}-{cfg['max_views']} views"
    extra = []
    if cfg.get("w_feat"):
        extra.append(f"feat KD ({cfg.get('feat_match', '')})" if cfg["student"].startswith("dvlt") else "feat KD")
    if cfg.get("scenes", "train") != "train":
        extra.append(cfg["scenes"])
    return f"{cfg['student']} ← {cfg['teacher']}, {views}, {cfg['steps']} steps" + (", " + ", ".join(extra) if extra else "")


def main():
    timings = json.load(open(os.path.join(WS, "timings.json"))) if os.path.exists(os.path.join(WS, "timings.json")) else {}
    out = ["Official DA3-benchmark metrics on the 10 held-out ScanNet++ scenes (100 frames each, seeded as the "
           "evaluator does). Train scenes: the other 10 benchmark scenes. RTX 3080.\n"]
    out += ["## Reference models (no training)\n",
            "| model | params (M) | TFLOPs (24 fr) | s / 100 fr | peak GiB (100 fr) | " + " | ".join(COLS) + " |",
            "|---|---|---|---|---|" + "---|" * len(COLS)]
    for name, (d, tkey) in REFS.items():
        m = split_metrics(os.path.join(REPO_ROOT, d))
        t = timings.get(tkey, {})
        g = lambda k, f: f.format(t[k]) if k in t else "-"  # noqa: E731
        out.append(f"| {name} | {g('params_M', '{:.0f}')} | {g('tflops_24', '{:.1f}')} | {g('sec_100', '{:.2f}')} | "
                   f"{g('peak_gib_100', '{:.1f}')} | " + " | ".join(fmt(m, c) for c in COLS) + " |")

    runs = sorted(glob.glob(os.path.join(WS, "runs", "*")), key=os.path.getmtime)
    for title, pred in [("## Wave 1: 8-16-view training, 32-frame validation (cv2 inputs for all students)",
                         lambda n: not n.startswith(("v2_", "v3_", "verify", "memtest"))),
                        ("## Waves 2/3: many-view training, validation on the evaluator's 100 frames",
                         lambda n: n.startswith(("v2_", "v3_")))]:
        out += ["", title, "", "| run | setup | " + " | ".join(COLS) + " | val AUC@3 start → end | val AbsRel start → end |",
                "|---|---|" + "---|" * (len(COLS) + 2)]
        for r in runs:
            name = os.path.basename(r)
            if not pred(name) or not os.path.exists(os.path.join(r, "config.json")):
                continue
            cfg = json.load(open(os.path.join(r, "config.json")))
            val = read_jsonl(os.path.join(r, "val.jsonl"))
            m = split_metrics(os.path.join(WS, "eval", name))
            v3 = f"{100 * val[0]['auc03']:.1f} → {100 * val[-1]['auc03']:.1f}" if val else ""
            va = f"{val[0]['absrel']:.4f} → {val[-1]['absrel']:.4f}" if val else ""
            note = " (cv2 inputs)" if name in CV2_INPUTS else ""
            out.append(f"| {name} | {run_desc(cfg)}{note} | " + " | ".join(fmt(m, c) for c in COLS) + f" | {v3} | {va} |")
    text = "\n".join(out) + "\n"
    open(os.path.join(WS, "results.md"), "w").write(text)
    print(text)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {}
    for r in runs:
        name = os.path.basename(r)
        val = read_jsonl(os.path.join(r, "val.jsonl"))
        if not name.startswith(("v2_", "v3_")) or name in CV2_INPUTS or len(val) < 2:
            continue
        cfg = json.load(open(os.path.join(r, "config.json")))
        groups.setdefault(cfg["student"].split(":")[0], []).append((name, val))
    os.makedirs(os.path.join(WS, "figures"), exist_ok=True)
    for group, items in groups.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for i, (name, val) in enumerate(items[:8]):
            steps = [v["step"] for v in val]
            for ax, key, scale in zip(axes, ["auc30", "auc03", "absrel"], [100, 100, 1]):
                ax.plot(steps, [scale * v[key] for v in val], color=COLORS[i], lw=2, marker="o", ms=5, label=name)
        for ax, key in zip(axes, ["pose AUC@30 (↑)", "pose AUC@3 (↑)", "depth AbsRel vs GT (↓)"]):
            ax.set_title(key, loc="left", fontsize=11)
            ax.set_xlabel("training step")
            ax.grid(alpha=0.25)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[0].legend(frameon=False, fontsize=9)
        fig.suptitle(f"Validation on the 10 held-out ScanNet++ scenes (evaluator's 100 frames), {group} students",
                     x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(os.path.join(WS, "figures", f"val_curves_{group}.png"), dpi=130)


if __name__ == "__main__":
    main()
