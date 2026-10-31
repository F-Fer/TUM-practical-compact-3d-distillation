#!/bin/bash
# Usage: posedev.sh <model> <tag> [pba overrides]: pose-only PBA (no saving) on the dev pose scenes; prints per-dataset AUC@3/@30
cd ~/TUM-practical-compact-3d-distillation
model=$1; tag=$2; shift 2
dst=workspace/tto/posedev/$model/$tag
for r in "hiroom 10" "7scenes 2" "scannetpp 3" "dtu64 4" "eth3d 3"; do
  set -- $r "$@"; d=$1; n=$2; shift 2
  .venv/bin/python ${PBA_SCRIPT:-scripts/run_tto_pba.py} workspace/evaluation/$model $dst datasets=$d max_scenes=$n save=false "$@" > /dev/null 2>&1
done
.venv/bin/python - $dst $tag <<'PY'
import json, sys
row, t = [], []
for d in ["hiroom", "7scenes", "scannetpp", "dtu64", "eth3d"]:
    m = json.load(open(f"{sys.argv[1]}/metric_results/{d}_pba.json"))["mean"]
    row.append(f"{100*m['auc03']:5.1f}/{100*m['auc30']:5.1f}"); t.append(m["opt_s"])
print(f"{sys.argv[2]:28s} " + " | ".join(row) + f" | {sum(t)/len(t):5.1f}s")
PY
