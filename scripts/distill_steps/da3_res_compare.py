"""Table: released DA3 models at lower input resolution (da3_res.py), naive and with RoPE stretched, pose AUC@3 alone
(official evaluator, 5 datasets, 81 scenes) and before -> after photometric TTO (T = 25), plus the 100-view profile.
Runs come from workspace/da3_res/run.sh.
  .venv/bin/python scripts/distill_steps/da3_res_compare.py [--md out.md]"""
import json
import os
import statistics
import sys

W = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "workspace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DS = ["hiroom", "7scenes", "scannetpp", "dtu64", "eth3d"]
MODELS = ["DA3-SMALL", "DA3-BASE", "DA3-LARGE-1.1"]
CONFIGS = ["504", "420", "420_rope", "336", "336_rope", "252", "252_rope"]
DEV = set("20241230/828738/cam_sampled_08,20241230/828744/cam_sampled_04,20241230/828745/cam_sampled_05,"
          "20241230/828749/cam_sampled_12,20241230/828750/cam_sampled_08,20241230/828757/cam_sampled_06,"
          "20241230/828760/cam_sampled_04,20241230/828761/cam_sampled_11,20241230/828763/cam_sampled_12,"
          "20241230/828766/cam_sampled_10,chess,fire,09c1414f1b,1ada7a0617,21d970d8de,scan105,scan114,scan118,"
          "scan122,courtyard,delivery_area,electro".split(","))


def pose(ev):
    vals = []
    for d in DS:
        f = f"{W}/evaluation/da3_res/{ev}/metric_results/{d}_pose.json"
        if not os.path.exists(f):
            return None
        vals.append(100 * json.load(open(f))["mean"]["auc03"])
    t = [json.loads(line)["forward_s"] for line in open(f"{W}/evaluation/da3_res/{ev}/timings.jsonl")]
    return sum(vals) / 5, statistics.median(t), vals


def tto(ev):
    out = {}
    for split in ("dev", "held", "all"):
        b, a, per = [], [], []
        for d in DS:
            f = f"{W}/tto/da3_res/{ev}/pho_it25/metric_results/{d}_pba.json"
            if not os.path.exists(f):
                return None
            j = {k: v for k, v in json.load(open(f)).items() if isinstance(v, dict) and "auc03" in v and k != "mean"}
            n = len([k for k in json.load(open(f"{W}/evaluation/da3_res/{ev}/metric_results/{d}_pose.json")) if k != "mean"])
            if len(j) < n:  # the TTO writes this file after every scene
                return None
            sel = [v for k, v in j.items() if split == "all" or ((k in DEV) == (split == "dev"))]
            b.append(100 * sum(v["auc03_init"] for v in sel) / len(sel))
            a.append(100 * sum(v["auc03"] for v in sel) / len(sel))
            per.append(100 * sum(v["auc03"] for v in sel) / len(sel))
        out[split] = (sum(b) / 5, sum(a) / 5, per)
    return out


def label(c):
    r, rope = c.split("_")[0], c.endswith("_rope")
    return f"{r} px" + (" + RoPE" if rope else (" (released)" if r == "504" else " naive"))


lines = []
prof = {}
pf = f"{W}/da3_res/profile_3080.jsonl"
if os.path.exists(pf):
    for line in open(pf):
        r = json.loads(line)
        prof[(r["model"], r["res"], r["lowmem"])] = r
    lines += ["### Forward pass at 100 views (7-Scenes chess, RTX 3080, bf16, median of 3)", "",
              "| model | input | tokens / frame | fwd s | peak GiB | fwd s (lowmem) | peak GiB (lowmem) |",
              "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for res in (504, 420, 336, 252):
            a, b = prof.get((m, res, False)), prof.get((m, res, True))
            if not (a or b):
                continue
            ref = b if (a is None or a.get("oom")) else a
            cell = lambda r, k, f: "OOM" if (r is None or r.get("oom")) else f.format(r[k])
            lines.append(f"| {m} | {ref.get('w', '?')}x{ref.get('h', '?')} | {ref.get('tokens', '?')} | "
                         f"{cell(a, 'fwd_s', '{:.2f}')} | {cell(a, 'peak_gib', '{:.1f}')} | "
                         f"{cell(b, 'fwd_s', '{:.2f}')} | {cell(b, 'peak_gib', '{:.1f}')} |")
    lines.append("")
lines += ["### Pose AUC@3, all 81 pose scenes (fwd s = median per scene on the 3080; TTO = run_tto_pba.py T = 25)", "",
          "| model | input | fwd s | AUC@3 alone | HiRoom / 7Sc / SN++ / DTU / ETH3D | + TTO (all) | dev: alone → TTO | held-out: alone → TTO |",
          "|---|---|---|---|---|---|---|---|"]
for m in MODELS:
    for c in CONFIGS:
        p = pose(f"{m}/r{c}")
        if p is None:
            continue
        t = tto(f"{m}/r{c}")
        ts = (lambda s: f"{t[s][0]:.1f} → {t[s][1]:.1f}") if t else (lambda s: "—")
        lines.append(f"| {m} | {label(c)} | {p[1]:.2f} | {p[0]:.1f} | {' / '.join(f'{v:.1f}' for v in p[2])} | "
                     f"{'**%.1f**' % t['all'][1] if t else '—'} | {ts('dev')} | {ts('held')} |")
print("\n".join(lines))
if "--md" in sys.argv:
    open(sys.argv[sys.argv.index("--md") + 1], "w").write("\n".join(lines) + "\n")
