"""Recon F1 (recon_unposed, 4 sets: HiRoom, 7-Scenes, ScanNet++, ETH3D) of runs scored by eval_tto.py.

    .venv/bin/python scripts/sba_f1.py <run dir> [...]
"""
import json
import os
import sys

import numpy as np

SETS = ["hiroom", "7scenes", "scannetpp", "eth3d"]
for r in sys.argv[1:]:
    vals = {}
    for d in SETS:
        p = os.path.join(r, "metric_results", f"{d}_recon_unposed.json")
        if os.path.exists(p):
            j = json.load(open(p))
            j.pop("mean", None)
            vals[d] = 100 * np.mean([v["fscore"] for v in j.values() if isinstance(v, dict) and "fscore" in v])
    m = np.mean(list(vals.values())) if len(vals) == 4 else float("nan")
    print(f"{r[-40:]:40s} " + " ".join(f"{vals.get(d, float('nan')):6.1f}" for d in SETS) + f"  mean {m:5.1f}")
