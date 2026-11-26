"""Match accuracy by initial error (the refiners' capture range) of scripts/gmr_ba.py runs made with eval_matches=1:
    .venv/bin/python scripts/gmr_basin.py <run dir> ...
Per bin of the geometric (initial) match error: share of the matches, median error after refinement, fraction below
1 / 2 px, fraction kept (overlap confidence above vis_thresh), and the fraction below 2 px among the kept ones."""
import json
import os
import sys

for r in sys.argv[1:]:
    for ds in ["hiroom", "scannetpp", "eth3d"]:
        p = os.path.join(r, "metric_results", f"{ds}_pba.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        d.pop("mean", None)
        edges = [0, 1, 2, 4, 8, 16, 32, 64]
        tot = sum(v.get(f"me_{e}_n", 0) for v in d.values() for e in edges)
        if not tot:
            continue
        print(f"{os.path.basename(r.rstrip('/'))} {ds} ({len(d)} scenes): init error bin | share | median after | <1 px | <2 px | kept | kept <2 px | kept median")
        for e in edges:
            n = sum(v[f"me_{e}_n"] for v in d.values())
            if n == 0:
                continue

            def w(k):
                vs = [(v[f"me_{e}_n"], v[f"me_{e}_{k}"]) for v in d.values() if v[f"me_{e}_n"] > 0 and v[f"me_{e}_{k}"] == v[f"me_{e}_{k}"]]
                return sum(a * b for a, b in vs) / max(sum(a for a, _ in vs), 1)

            print(f"   {e:>3}+ px | {100 * n / tot:5.1f}% | {w('med'):6.2f} | {100 * w('lt1'):5.1f} | {100 * w('lt2'):5.1f} | {100 * w('keep'):5.1f} | {100 * w('kept_lt2'):5.1f} | {w('kept_med'):6.2f}")
