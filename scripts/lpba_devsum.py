"""AUC@3 (and depth error) of PBA runs on the dev scenes of HiRoom / ScanNet++ / ETH3D (the scenes of the learned-basis
experiments, scripts/lpba_basisnet.py), mean per dataset and over the three sets.

    .venv/bin/python scripts/lpba_devsum.py <run dir> [<run dir> ...]
"""
import json
import os
import sys

import numpy as np

DEV = {"hiroom": 10, "scannetpp": 3, "eth3d": 3}
SRC = "workspace/evaluation/DA3-SMALL/model_results"


def dev_scenes(ds):
    d = os.path.join(SRC, ds)
    sc = sorted(os.path.dirname(p) for p, _, _ in os.walk(d) if os.path.basename(p) == "unposed")
    return [os.path.relpath(s, d) for s in sc[: DEV[ds]]]


def summ(run, key="auc03"):
    out = {}
    for ds in DEV:
        p = os.path.join(run, "metric_results", f"{ds}_pba.json")
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        v = [m[s][key] for s in dev_scenes(ds) if s in m and key in m[s]]
        if len(v) == DEV[ds]:
            out[ds] = 100 * float(np.mean(v)) if key.startswith("auc") else float(np.mean(v))
    return out


if __name__ == "__main__":
    print(f"{'run':40s} {'hiroom':>7s} {'scan++':>7s} {'eth3d':>7s} {'mean':>7s}   depth err out H/S/E")
    for run in sys.argv[1:]:
        a = summ(run)
        e = summ(run, "depth_err_out")
        mean = np.mean(list(a.values())) if len(a) == 3 else float("nan")
        es = "/".join(f"{100 * e[d]:.2f}" for d in DEV if d in e)
        print(f"{os.path.basename(run.rstrip('/')):40s} " + " ".join(f"{a.get(d, float('nan')):7.1f}" for d in DEV) + f" {mean:7.1f}   {es}")
