"""Table: DéjàView (released, several K) vs few-step students, pose AUC@3 / AUC@30 alone (official evaluator, 5
datasets) and before -> after photometric TTO (T = 25) on dev / held-out / all pose scenes (dataset means).
  .venv/bin/python scripts/distill_steps/compare.py [--md out.md]"""
import json
import os
import statistics
import sys

W = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "workspace")
DS = ["hiroom", "7scenes", "scannetpp", "dtu64", "eth3d"]
DEV = set("20241230/828738/cam_sampled_08,20241230/828744/cam_sampled_04,20241230/828745/cam_sampled_05,"
          "20241230/828749/cam_sampled_12,20241230/828750/cam_sampled_08,20241230/828757/cam_sampled_06,"
          "20241230/828760/cam_sampled_04,20241230/828761/cam_sampled_11,20241230/828763/cam_sampled_12,"
          "20241230/828766/cam_sampled_10,chess,fire,09c1414f1b,1ada7a0617,21d970d8de,scan105,scan114,scan118,"
          "scan122,courtyard,delivery_area,electro".split(","))
ROWS = [("released K=4", "dvlt_K4", "base_K4"), ("released K=6", "dvlt_K6", "base_K6"),
        ("released K=8", "dvlt_K8", "base_K8"), ("released K=12", "dvlt", "base_K12"),
        ("released K=16", "dvlt_probe/K16", "base_K16")]
for run in sorted(os.listdir(f"{W}/evaluation/dvlt_fs")) if os.path.isdir(f"{W}/evaluation/dvlt_fs") else []:
    ROWS.append((run, f"dvlt_fs/{run}", run))


def pose(ev):
    vals, v30 = [], []
    for d in DS:
        f = f"{W}/evaluation/{ev}/metric_results/{d}_pose.json"
        if not os.path.exists(f):
            return None
        m = json.load(open(f))["mean"]
        vals.append(100 * m["auc03"]), v30.append(100 * m["auc30"])
    t = []
    tf = f"{W}/evaluation/{ev}/timings.jsonl"
    if os.path.exists(tf):
        for line in open(tf):
            r = json.loads(line)
            t.append(r.get("forward_s", r.get("time_s")))
    t = [x for x in t if x is not None]
    return sum(vals) / 5, sum(v30) / 5, statistics.median(t) if t else float("nan"), vals


def tto(name):
    out = {}
    for split in ("dev", "held", "all"):
        b, a, ts = [], [], []
        for d in DS:
            f = f"{W}/tto/dvlt_fs/{name}/pho_it25/metric_results/{d}_pba.json"
            if not os.path.exists(f):
                return None
            j = {k: v for k, v in json.load(open(f)).items() if isinstance(v, dict) and "auc03" in v and k != "mean"}
            sel = [v for k, v in j.items() if split == "all" or ((k in DEV) == (split == "dev"))]
            if not sel:
                return None
            b.append(100 * sum(v["auc03_init"] for v in sel) / len(sel))
            a.append(100 * sum(v["auc03"] for v in sel) / len(sel))
            ts += [v["opt_s"] for v in sel]
        out[split] = (sum(b) / 5, sum(a) / 5, sum(ts) / len(ts))
    return out


lines = ["| model | fwd s | AUC@3 (5 sets) | AUC@30 | HiRoom / 7Sc / SN++ / DTU / ETH3D | dev: alone → TTO | held-out: alone → TTO | all: alone → TTO | TTO s |",
         "|---|---|---|---|---|---|---|---|---|"]
for label, ev, tn in ROWS:
    p = pose(ev)
    if p is None:
        continue
    t = tto(tn)
    ts = (lambda s: f"{t[s][0]:.1f} → **{t[s][1]:.1f}**") if t else (lambda s: "—")
    lines.append(f"| {label} | {p[2]:.2f} | {p[0]:.1f} | {p[1]:.1f} | {' / '.join(f'{v:.1f}' for v in p[3])} | "
                 f"{ts('dev')} | {ts('held')} | {ts('all')} | {t['all'][2]:.2f} |" if t else
                 f"| {label} | {p[2]:.2f} | {p[0]:.1f} | {p[1]:.1f} | {' / '.join(f'{v:.1f}' for v in p[3])} | — | — | — | — |")
print("\n".join(lines))
if "--md" in sys.argv:
    open(sys.argv[sys.argv.index("--md") + 1], "w").write("\n".join(lines) + "\n")
