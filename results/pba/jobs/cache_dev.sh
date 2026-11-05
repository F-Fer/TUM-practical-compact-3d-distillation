#!/bin/bash
# cache DA3 tokens of the dev subset for the three models
cd ~/TUM-practical-compact-3d-distillation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in ${MODELS:-DA3-SMALL DA3-BASE DA3-LARGE-1.1}; do
  for r in "hiroom 10" "7scenes 2" "scannetpp 3" "dtu64 4" "eth3d 3" "dtu 4"; do
    set -- $r
    .venv/bin/python scripts/cache_da3_feats.py $m workspace/evaluation/$m workspace/tto/feats/$m datasets=$1 max_scenes=$2 2>&1 | grep "^\[cache"
  done
done
