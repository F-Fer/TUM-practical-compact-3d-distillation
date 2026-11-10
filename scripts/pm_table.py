"""Table of photometric-BA runs by dataset: pm_table.py <model> <glob of run names> ... [--track 5,20] [--seeds]

Per run: AUC@3 of HiRoom, 7-Scenes, ScanNet++, DTU-64, ETH3D, their mean, AUC@30, mean TTO s per scene without each dataset's first
scene (warm-up, as in plot_compute.py), scenes. With --track: AUC@3 at those LM iterations from <dataset>_pba_track.jsonl, as in the
ground-truth tables of REPORT3. With --seeds: runs named <arm>_s<seed> are averaged per arm, with the spread of the mean.
"""

import argparse
import fnmatch
import json
import os
import re

import numpy as np

SETS = ("hiroom", "7scenes", "scannetpp", "dtu64", "eth3d")
NAMES = ("HiR", "7S", "SN++", "DTU", "ETH")


def load(root, run):
    out = {}
    for d in SETS:
        p = f"{root}/{run}/metric_results/{d}_pba.json"
        if os.path.exists(p):
            j = json.load(open(p))
            out[d] = {k: v for k, v in j.items() if k != "mean"}
    return out


def row(run_data):
    a3 = [100 * np.mean([s["auc03"] for s in run_data[d].values()]) if d in run_data else np.nan for d in SETS]
    a30 = [100 * np.mean([s["auc30"] for s in run_data[d].values()]) if d in run_data else np.nan for d in SETS]
    t = [s["opt_s"] for d in run_data for s in list(run_data[d].values())[1:]]
    return a3, np.nanmean(a3), np.nanmean(a30), float(np.mean(t)) if t else np.nan, sum(len(v) for v in run_data.values())


def track(root, run, its):
    out = {}
    for d in SETS:
        p = f"{root}/{run}/metric_results/{d}_pba_track.jsonl"
        if os.path.exists(p):
            seen = {}
            for line in open(p):
                j = json.loads(line)
                seen[j["scene"]] = {r["it"]: r["auc03"] for r in j["track"]}
            out[d] = [100 * np.mean([s.get(i, s[max(s)]) for s in seen.values()]) for i in its]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("patterns", nargs="+")
    ap.add_argument("--track")
    ap.add_argument("--seeds", action="store_true")
    a = ap.parse_args()
    root = f"workspace/tto/{a.model}"
    runs = [r for p in a.patterns for r in sorted(os.listdir(root)) if fnmatch.fnmatch(r, p)]
    if a.track:
        its = [int(x) for x in a.track.split(",")]
        print("run | " + " | ".join(f"{n} {'/'.join(map(str, its))}" for n, d in zip(NAMES, SETS)))
        for r in runs:
            t = track(root, r, its)
            print(f"{r:24s} | " + " | ".join("/".join(f"{x:5.1f}" for x in t[d]) if d in t else "-" for d in SETS))
        return
    print(f"{'run':26s} " + " ".join(f"{n:>5s}" for n in NAMES) + f" {'A@3':>6s} {'A@30':>6s} {'TTO s':>6s} scenes")
    groups = {}
    for r in runs:
        key = re.sub(r"_s\d+$", "", r) if a.seeds else r
        groups.setdefault(key, []).append(row(load(root, r)))
    for k, rows in groups.items():
        m3 = np.nanmean([x[0] for x in rows], 0)
        tail = f"{np.mean([x[1] for x in rows]):6.1f} {np.mean([x[2] for x in rows]):6.1f} {np.mean([x[3] for x in rows]):6.2f} {sum(x[4] for x in rows):4d}"
        spread = f" ±{np.std([x[1] for x in rows]):.1f} ({len(rows)})" if a.seeds else ""
        print(f"{k:26s} " + " ".join(f"{x:5.1f}" for x in m3) + " " + tail + spread)


if __name__ == "__main__":
    main()
