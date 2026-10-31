"""Table of TTO runs (scored by eval_tto.py), averaged over the scenes all runs share.

Usage: .venv/bin/python scripts/compare_tto.py <work dir> [<work dir> ...] [--md out.md]
"""

import json
import os
import sys

COLUMNS = [
    ("hiroom", "pose", "auc03"), ("hiroom", "pose", "auc30"), ("hiroom", "recon_unposed", "fscore"),
    ("7scenes", "pose", "auc03"), ("7scenes", "pose", "auc30"), ("7scenes", "recon_unposed", "fscore"),
    ("scannetpp", "pose", "auc03"), ("scannetpp", "pose", "auc30"), ("scannetpp", "recon_unposed", "fscore"),
    ("dtu64", "pose", "auc03"), ("dtu64", "pose", "auc30"),
    ("eth3d", "pose", "auc03"), ("eth3d", "pose", "auc30"),
    ("dtu", "recon_unposed", "overall"),
]
SHORT = {"hiroom": "HiR", "7scenes": "7S", "scannetpp": "SN++", "dtu64": "DTU64", "eth3d": "ETH", "dtu": "DTU"}


def load(run, dataset, mode):
    p = os.path.join(run, "metric_results", f"{dataset}_{mode}.json")
    if not os.path.exists(p) and mode == "pose":  # run_tto_pba.py's own pose metrics (same compute_pose)
        p = os.path.join(run, "metric_results", f"{dataset}_pba.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    d.pop("mean", None)
    return d


def main():
    args = sys.argv[1:]
    md = None
    if "--md" in args:
        md = args[args.index("--md") + 1]
        args = [a for a in args if a not in ("--md", md)]
    runs = args
    header = ["run"] + [f"{SHORT[d]} {'CD' if k == 'overall' else 'F1' if m.startswith('recon') else k.replace('auc', 'A@')}" for d, m, k in COLUMNS] + ["s/scene"]
    rows = []
    common = {}
    for d, m, _ in COLUMNS:
        sets = [set(load(r, d, m)) for r in runs]
        common[(d, m)] = set.intersection(*sets) if sets else set()
    for r in runs:
        row = [os.path.relpath(r, "workspace/tto") if r.startswith("workspace/tto") else r]
        for d, m, k in COLUMNS:
            res, sc = load(r, d, m), common[(d, m)]
            vals = [res[s][k] for s in sc if k in res[s]]
            scale = 1 if k == "overall" else 100  # DTU Chamfer distance in mm, the rest in %
            row.append(f"{scale * sum(vals) / len(vals):.2f}" if vals and k == "overall" else f"{scale * sum(vals) / len(vals):.1f}" if vals else "-")
        t = []
        for d in SHORT:
            pba = load(r, d, "pba")
            t += [v.get("opt_s", 0) for v in pba.values()]
        row.append(f"{sum(t) / len(t):.1f}" if t else "-")
        rows.append(row)
    n = {f"{SHORT[d]} {m}": len(sc) for (d, m), sc in common.items()}
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]
    out = "\n".join(lines) + f"\n\nscenes per column: {n}\n"
    print(out)
    if md:
        with open(md, "w") as f:
            f.write(out)


if __name__ == "__main__":
    main()
