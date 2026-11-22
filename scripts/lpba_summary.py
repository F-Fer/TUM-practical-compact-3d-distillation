"""Dev pose summary of PBA runs (G0 of reports/Learned refinement step between DA3 and DeepBA.md).

    .venv/bin/python scripts/lpba_summary.py <run dir> ... [--oracle NAME=<run>,<run>,...]

Per run: AUC@3 / AUC@30 (x100) per pose dataset and their mean (each dataset = mean over its scenes), from
metric_results/<dataset>_pba.json. --oracle adds a row that picks, per scene, the run with the best AUC@3 among the given
ones (a scene-level oracle choice of a hyperparameter).
"""

import argparse
import json
import os

import numpy as np

SETS = ("hiroom", "7scenes", "scannetpp", "dtu64", "eth3d")


def load(run):
    out = {}
    for ds in SETS:
        p = os.path.join(run, "metric_results", f"{ds}_pba.json")
        if os.path.exists(p):
            out[ds] = {k: v for k, v in json.load(open(p)).items() if k != "mean"}
    return out


def row(name, per_scene, key="auc03"):
    vals = [100 * np.mean([s[key] for s in per_scene[ds].values()]) if ds in per_scene else np.nan for ds in SETS]
    n = sum(len(per_scene.get(ds, {})) for ds in SETS)
    return f"{name:28s} | " + " | ".join(f"{v:5.1f}" for v in vals) + f" | {np.mean(vals):5.1f} | n={n}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--oracle", action="append", default=[])
    a = ap.parse_args()
    for key in ("auc03", "auc30"):
        print(f"{key} x100{'':18s} | " + " | ".join(f"{d[:5]:>5s}" for d in SETS) + " |  mean")
        for run in a.runs:
            print(row(os.path.basename(run.rstrip("/")), load(run), key))
        if key == "auc03":
            for spec in a.oracle:
                name, runs = spec.split("=", 1)
                data = [load(r) for r in runs.split(",")]
                best = {}
                for ds in SETS:
                    scenes = set.intersection(*[set(d.get(ds, {})) for d in data])
                    if scenes:
                        best[ds] = {s: max((d[ds][s] for d in data), key=lambda m: m["auc03"]) for s in scenes}
                print(row(name, best, key))
        print()


if __name__ == "__main__":
    main()
