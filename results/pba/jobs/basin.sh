#!/bin/bash
# Usage: basin.sh <model> <tag> [pba overrides]: pose-only PBA (no saving) on hiroom 10, dtu64 4, scannetpp 3, eth3d 3
cd ~/TUM-practical-compact-3d-distillation
model=$1; tag=$2; shift 2
dst=workspace/tto/basin/$model/$tag
echo "== $(hostname) basin $model $tag $*"; date
for r in "hiroom 10" "dtu64 4" "scannetpp 3" "eth3d 3"; do
  set -- $r "$@"; d=$1; n=$2; shift 2
  .venv/bin/python ${PBA_SCRIPT:-scripts/run_tto_pba.py} workspace/evaluation/$model $dst datasets=$d max_scenes=$n save=false "$@" 2>&1 | grep "mean:" | cut -c1-200
done
date
