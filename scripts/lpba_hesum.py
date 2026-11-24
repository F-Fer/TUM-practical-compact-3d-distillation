"""Mean AUC@3 (and depth error after PBA) per dataset over all scenes of PBA runs, grouped by run name minus _s<seed>.

    .venv/bin/python scripts/lpba_hesum.py <run dir> [...]   (datasets with results in every run of a group)
"""
import json
import os
import re
import sys

import numpy as np


def load(run, ds, key):
    p = os.path.join(run, "metric_results", f"{ds}_pba.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    d.pop("mean", None)
    return [x[key] for x in d.values() if key in x]


groups = {}
for r in sys.argv[1:]:
    groups.setdefault(re.sub(r"_s\d+$", "", os.path.basename(r.rstrip("/"))), []).append(r)
for g, rs in groups.items():
    out = []
    for ds in ("hiroom", "scannetpp", "eth3d", "7scenes", "dtu64"):
        vals = [load(r, ds, "auc03") for r in rs]
        if any(v is None or not v for v in vals):
            continue
        a = [100 * np.mean(v) for v in vals]
        e = [load(r, ds, "depth_err_out") for r in rs]
        es = f" err {100 * np.mean(e[0]):.2f}%" if e[0] else ""
        out.append(f"{ds} {np.mean(a):5.1f} ±{np.std(a):.1f} (n={len(vals[0])}, seeds {' '.join(f'{x:.1f}' for x in a)}){es}")
    print(f"{g:22s} " + " | ".join(out))
