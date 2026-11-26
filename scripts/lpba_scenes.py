"""Per-scene AUC@3 of several runs side by side: .venv/bin/python scripts/lpba_scenes.py <dataset> <run dir> ..."""
import json, os, sys
import numpy as np
ds, runs = sys.argv[1], sys.argv[2:]
M = []
for r in runs:
    d = {}
    for nm in (f"{ds}_pba.json", f"{ds}_ba.json"):
        p = os.path.join(r, "metric_results", nm)
        if os.path.exists(p):
            d = json.load(open(p))
            break
    d.pop("mean", None)
    M.append(d)
scenes = sorted(set().union(*[set(m) for m in M]))
print(f"{'scene':40s} " + " ".join(f"{os.path.basename(r.rstrip('/'))[:12]:>12s}" for r in runs))
for s in scenes:
    print(f"{s[-40:]:40s} " + " ".join(f"{100 * m[s]['auc03']:12.1f}" if s in m else f"{'-':>12s}" for m in M))
common = [s for s in scenes if all(s in m for m in M)]
print(f"{'mean (common ' + str(len(common)) + ')':40s} " + " ".join(f"{100 * np.mean([m[s]['auc03'] for s in common]):12.1f}" for m in M))
