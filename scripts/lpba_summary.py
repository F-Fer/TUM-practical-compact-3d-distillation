"""Dev pose summary of PBA runs (G0 of reports/Learned refinement step between DA3 and DeepBA.md).

    .venv/bin/python scripts/lpba_summary.py <run dir> ... [--oracle NAME=<run>,<run>,...]
    .venv/bin/python scripts/lpba_summary.py --table <run dir> ...   # AUC@3, recon F1 and depth error in one row

--table: per run the mean AUC@3 over the five pose datasets, recon F1 (x100) per recon set (HiRoom, 7-Scenes,
ScanNet++, ETH3D, from metric_results/<dataset>_recon_unposed.json of scripts/eval_tto.py) and their mean, and the
depth error after PBA against the GT (abs. rel. %, info depth_err_out of lpba_depth_oracle.py / lpba_field.py /
lpba_stage.py) on HiRoom / ScanNet++ / ETH3D. The first row is DA3 alone (workspace/evaluation/DA3-SMALL, same scenes).

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


RECON = ("hiroom", "7scenes", "scannetpp", "eth3d")


def table(runs, src="workspace/evaluation/DA3-SMALL"):
    print(f"{'run':24s} | AUC@3 |  F1: " + " ".join(f"{d[:5]:>5s}" for d in RECON) + " |  mean | depth err %: hiroo scann eth3d")
    rows = [("DA3 alone", None)] + [(os.path.basename(r.rstrip("/")), r) for r in runs]
    ref = runs[0]
    for name, run in rows:
        pose = load(ref if run is None else run)
        auc = np.mean([100 * np.mean([m["auc03_init" if run is None else "auc03"] for m in pose[d].values()]) for d in SETS if d in pose and pose[d]])
        f1 = []
        for d in RECON:
            scenes = set(pose.get(d, {}))
            p = os.path.join(src if run is None else run, "metric_results", f"{d}_recon_unposed.json")
            if not os.path.exists(p) or not scenes:
                f1.append(np.nan)
                continue
            j = json.load(open(p))
            f1.append(100 * np.mean([j[k]["fscore"] for k in j if k in scenes]) if any(k in scenes for k in j) else np.nan)
        de = []
        for d in ("hiroom", "scannetpp", "eth3d"):
            key = "depth_err_da3" if run is None else "depth_err_out"
            v = [m.get(key, np.nan) for m in pose.get(d, {}).values()]
            de.append(100 * np.nanmean(v) if v and not np.all(np.isnan(v)) else np.nan)
        print(f"{name:24s} | {auc:5.1f} |      " + " ".join(f"{x:5.1f}" for x in f1) + f" | {np.nanmean(f1):5.1f} |              " + " ".join(f"{x:5.2f}" for x in de))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--oracle", action="append", default=[])
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    if a.table:
        return table(a.runs)
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
