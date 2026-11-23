"""Table for the scaled resolution distillation (S0 / S1, knowledge/ideas/dejaview-res-distill-scale.md): pose AUC@3
alone (official evaluator, 5 datasets) and alone -> after photometric TTO (T = 25) on dev / held-out / all 81 pose
scenes, for every run in workspace/evaluation/dvlt_rs/<run>_<res>_K<K>, plus the DA3-GIANT-1.1 ceiling. Forward times
are the RTX 3080 medians of the same architecture at that (res, K) from dejaview-speed.md (weights don't change them);
"5090 s" is this run's own median.
  .venv/bin/python scripts/distill_steps/compare_rs.py [--md out.md]"""
import json
import os
import re
import statistics
import sys

W = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "workspace")
DS = ["hiroom", "7scenes", "scannetpp", "dtu64", "eth3d"]
DEV = set("20241230/828738/cam_sampled_08,20241230/828744/cam_sampled_04,20241230/828745/cam_sampled_05,"
          "20241230/828749/cam_sampled_12,20241230/828750/cam_sampled_08,20241230/828757/cam_sampled_06,"
          "20241230/828760/cam_sampled_04,20241230/828761/cam_sampled_11,20241230/828763/cam_sampled_12,"
          "20241230/828766/cam_sampled_10,chess,fire,09c1414f1b,1ada7a0617,21d970d8de,scan105,scan114,scan118,"
          "scan122,courtyard,delivery_area,electro".split(","))
FWD_3080 = {(252, 16): 0.73, (336, 16): 1.27, (336, 12): 1.08, (420, 16): 2.13, (420, 12): 1.76, (420, 8): 1.40,
            (504, 16): 3.67, (504, 12): 2.98}
# earlier runs on tum1 (RTX 3080, same evaluator / TTO, fhk unless noted): AUC@3 alone, dev / held-out / all after TTO
REF = [("released 504 K16, fit (teacher)", 504, 16, "61.3", "70.9", "70.7", "70.9"),
       ("released 504 K16, clean K", 504, 16, "61.3", "73.1", "71.6", "72.1"),
       ("released 420 + RoPE K12, fit", 420, 12, "56.0", "67.3", "66.3", "66.7"),
       ("released 336 + RoPE K16, fit", 336, 16, "34.0", "50.5", "48.5", "49.0"),
       ("released 252 + RoPE K16, fit", 252, 16, "9.7", "22.8", "20.5", "21.3"),
       ("R2 336 K16 (200 ARKit videos, 3 h 3080)", 336, 16, "53.9", "68.5", "65.9", "66.7"),
       ("R2 336 K12", 336, 12, "52.0", "66.6", "65.8", "66.0"),
       ("R2 420 K12", 420, 12, "58.5", "72.2", "69.4", "70.3")]


def pose(ev):
    vals = []
    for d in DS:
        f = f"{W}/evaluation/{ev}/metric_results/{d}_pose.json"
        if not os.path.exists(f):
            return None
        vals.append(100 * json.load(open(f))["mean"]["auc03"])
    t = []
    tf = f"{W}/evaluation/{ev}/timings.jsonl"
    if os.path.exists(tf):
        t = [x for x in (json.loads(l).get("forward_s") for l in open(tf)) if x is not None]
    return sum(vals) / 5, statistics.median(t) if t else float("nan"), vals


def tto(path):
    out = {}
    for split in ("dev", "held", "all"):
        b, a = [], []
        for d in DS:
            f = f"{W}/tto/{path}/metric_results/{d}_pba.json"
            if not os.path.exists(f):
                return None
            j = {k: v for k, v in json.load(open(f)).items() if isinstance(v, dict) and "auc03" in v and k != "mean"}
            sel = [v for k, v in j.items() if split == "all" or ((k in DEV) == (split == "dev"))]
            if not sel:
                return None
            b.append(100 * sum(v["auc03_init"] for v in sel) / len(sel))
            a.append(100 * sum(v["auc03"] for v in sel) / len(sel))
        out[split] = (sum(b) / 5, sum(a) / 5)
    return out


def row(label, fwd, p, t, extra=""):
    ts = (lambda s: f"{t[s][0]:.1f} → **{t[s][1]:.1f}**") if t else (lambda s: "—")
    return (f"| {label} | {fwd} | {p[1]:.2f} | **{p[0]:.1f}** | {' / '.join(f'{v:.1f}' for v in p[2])} | {ts('dev')} | "
            f"{ts('held')} | {ts('all')} |{extra}")


lines = ["| model | fwd s (3080) | 5090 s | AUC@3 alone | HiRoom / 7Sc / SN++ / DTU / ETH3D | dev: alone → TTO | "
         "held-out: alone → TTO | all: alone → TTO |", "|---|---|---|---|---|---|---|---|"]
g = pose("DA3-GIANT-1.1")
if g:
    lines.append(row("DA3-GIANT-1.1 504 (S1 teacher)", "OOM at 100 views", g, tto("DA3-GIANT-1.1/pho_it25_all")))
runs = {}
if os.path.isdir(f"{W}/evaluation/dvlt_rs"):
    for tag in sorted(os.listdir(f"{W}/evaluation/dvlt_rs")):
        m = re.fullmatch(r"(.+)_(\d+)_K(\d+)", tag)
        if m:
            runs.setdefault((int(m[2]), int(m[3])), []).append((m[1], tag))
for (res, K) in sorted(runs, key=lambda x: (-x[0], -x[1])):
    for label, r, k, alone, dev, held, al in REF:
        if (r, k) == (res, K):
            lines.append(f"| {label} | {FWD_3080.get((r, k), '—')} | — | {alone} | | {dev} | {held} | {al} |")
    for name, tag in runs[(res, K)]:
        p = pose(f"dvlt_rs/{tag}")
        if p:
            lines.append(row(f"**{name}** {res} K{K}", FWD_3080.get((res, K), "—"), p, tto(f"dvlt_rs/{tag}/pho_it25")))
print("\n".join(lines))
if "--md" in sys.argv:
    open(sys.argv[sys.argv.index("--md") + 1], "w").write("\n".join(lines) + "\n")
