"""Table of track-BA runs: pose AUC@3, recon F1 as the benchmark scores it and after aligning the cloud to the GT.

Usage: .venv/bin/python scripts/tracks_table.py <work dir> [<work dir> ...] [--md out.md]
       .venv/bin/python scripts/tracks_table.py --seeds <work dir> [<work dir> ...]    (one row: mean +- std over the runs)

Reads metric_results/<dataset>_{pba,recon_unposed,recon_aligned}.json of every run (written by run_tto_ba.py,
eval_tto.py and eval_recon_aligned.py), averaged over the scenes that all runs share. avg A@3 is the mean over the five
pose datasets, avg F1 / F1 al. the mean over HiRoom, 7-Scenes and ScanNet++ (each dataset counts once).
"""

import json
import os
import statistics
import sys

POSE = ["hiroom", "7scenes", "scannetpp", "dtu64", "eth3d"]
RECON = ["hiroom", "7scenes", "scannetpp"]
SHORT = {"hiroom": "HiR", "7scenes": "7S", "scannetpp": "SN++", "dtu64": "DTU64", "eth3d": "ETH"}


def load(run, dataset, kind):
    p = os.path.join(run, "metric_results", f"{dataset}_{kind}.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    d.pop("mean", None)
    return d


def table(runs):
    """{run: {column: value}}, over the scenes all runs share."""
    sources = {("pose", d): ("pba", "auc03") for d in POSE}
    sources.update({("f1", d): ("recon_unposed", "fscore") for d in RECON})
    sources.update({("f1al", d): ("recon_aligned", "fscore") for d in RECON})
    sources[("cd", "dtu")] = ("recon_unposed", "overall")
    loaded = {(r, key): load(r, key[1], kind) for r in runs for key, (kind, _) in sources.items()}
    out = {r: {} for r in runs}
    for key, (_, metric) in sources.items():
        common = set.intersection(*[set(loaded[(r, key)]) for r in runs])
        for r in runs:
            vals = [loaded[(r, key)][s][metric] for s in common if metric in loaded[(r, key)][s]]
            if vals:
                out[r][key] = sum(vals) / len(vals) * (1 if key[0] == "cd" else 100)
    for r in runs:
        t = [v.get("opt_s", 0) for d in POSE for v in load(r, d, "pba").values()]
        out[r]["time"] = sum(t) / len(t) if t else None
        for name, keys in (("avg A@3", [("pose", d) for d in POSE]), ("avg F1", [("f1", d) for d in RECON]), ("avg F1 al.", [("f1al", d) for d in RECON])):
            vals = [out[r][k] for k in keys if k in out[r]]
            out[r][name] = sum(vals) / len(vals) if len(vals) == len(keys) else None
    return out


def fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def main():
    args = sys.argv[1:]
    md = None
    if "--md" in args:
        md = args[args.index("--md") + 1]
        args = [a for a in args if a not in ("--md", md)]
    seeds = "--seeds" in args
    runs = [a for a in args if a != "--seeds"]
    if not runs:
        sys.exit(__doc__)
    cols = [c for d in POSE for c in [("pose", d)] + ([("f1", d), ("f1al", d)] if d in RECON else [])] + [("cd", "dtu")]
    names = [f"{SHORT.get(d, 'DTU')} {'A@3' if k == 'pose' else 'F1' if k == 'f1' else 'F1 al.' if k == 'f1al' else 'CD'}" for k, d in cols]
    header = ["run"] + names + ["avg A@3", "avg F1", "avg F1 al.", "s/scene"]
    res = table(runs)
    rows = []
    if seeds:
        row = ["mean +- std of " + str(len(runs)) + " runs"]
        for key in cols + ["avg A@3", "avg F1", "avg F1 al.", "time"]:
            vals = [res[r].get(key) for r in runs]
            vals = [v for v in vals if v is not None]
            row.append("-" if not vals else f"{statistics.mean(vals):.1f} +- {statistics.stdev(vals) if len(vals) > 1 else 0:.1f}")
        rows.append(row)
    else:
        for r in runs:
            rows.append([os.path.relpath(r, "workspace/tto") if r.startswith("workspace/tto") else r]
                        + [fmt(res[r].get(c), 2 if c[0] == "cd" else 1) for c in cols]
                        + [fmt(res[r][k]) for k in ("avg A@3", "avg F1", "avg F1 al.", "time")])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]
    out = "\n".join(lines) + "\n"
    print(out)
    if md:
        with open(md, "w") as f:
            f.write(out)


if __name__ == "__main__":
    main()
