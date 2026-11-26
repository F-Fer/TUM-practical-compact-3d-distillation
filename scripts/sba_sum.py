"""Summarise sparse-BA (sba_fast.py / run_tto_ba.py) and PBA runs: AUC@3 per dataset, 5-set mean, median time per stage.

    .venv/bin/python scripts/sba_sum.py <run dir> [<run dir> ...]
"""
import glob
import json
import os
import sys

import numpy as np

SETS = ["hiroom", "7scenes", "scannetpp", "dtu64", "eth3d"]


def load(run):
    out = {}
    for ds in SETS:
        for name in (f"{ds}_ba.json", f"{ds}_pba.json"):
            p = os.path.join(run, "metric_results", name)
            if os.path.exists(p):
                d = json.load(open(p))
                d.pop("mean", None)
                out[ds] = d
                break
    return out


def main():
    runs = [r for a in sys.argv[1:] for r in sorted(glob.glob(a))]
    print(f"{'run':44s} " + " ".join(f"{d[:6]:>6s}" for d in SETS) + "   mean  init   n   opt_s  feat  match track   ba")
    for r in runs:
        m = load(r)
        if not m:
            continue
        per = {d: 100 * np.mean([v["auc03"] for v in m[d].values()]) for d in m}
        init = {d: 100 * np.mean([v.get("auc03_init", np.nan) for v in m[d].values()]) for d in m}
        allv = [v for d in m for v in m[d].values()]
        med = lambda k: np.median([v.get(k, np.nan) for v in allv])  # noqa: E731
        mean = np.mean(list(per.values())) if len(per) == 5 else np.nan
        im = np.mean(list(init.values())) if len(init) == 5 else np.nan
        print(f"{os.path.basename(r.rstrip('/'))[:44]:44s} " + " ".join(f"{per.get(d, np.nan):6.1f}" for d in SETS) +
              f" {mean:6.1f} {im:5.1f} {len(allv):3d} {np.mean([v.get('opt_s', np.nan) for v in allv]):6.2f} {med('feat_s'):5.2f} {med('match_s'):5.2f} {med('track_s'):5.2f} {med('ba_s'):5.2f}")


if __name__ == "__main__":
    main()
