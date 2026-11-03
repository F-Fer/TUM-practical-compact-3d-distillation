#!/bin/bash
# Usage: eth3d_rescore.sh <model> <run> ...: ETH3D recon_unposed (F1) of finished PBA runs that were scored without it
# (pba_run.sh before PBA_ETH3D_RECON); one scene at a time (full-resolution TSDF fusion is RAM-heavy). Skips runs that
# already have it; copies the metric file into results/pba/runs/<model>/<run>/.
cd ~/TUM-practical-compact-3d-distillation
PY=.venv/bin/python
model=$1; shift
for name in "$@"; do
  dst=workspace/tto/$model/$name; ref=workspace/tto/$model/init_dev
  [ "$dst" = "$ref" ] && ref=workspace/evaluation/$model
  [ -d $dst/model_results/eth3d ] || { echo "!! no eth3d predictions in $dst"; continue; }
  [ -f $dst/metric_results/eth3d_recon_unposed.json ] && { echo "skip $model/$name"; continue; }
  echo "== eth3d recon $model $name $(date +%T)"
  /usr/bin/time -f "[eth3d recon] %e s, max RSS %M KB" \
    $PY scripts/eval_tto.py $ref $dst eth3d recon_unposed --workers 1 2>&1 | grep -v -i -E "warning|it/s\]|s/it\]"
  o=results/pba/runs/$model/$name/metric_results
  [ -f $dst/metric_results/eth3d_recon_unposed.json ] && mkdir -p $o && cp $dst/metric_results/eth3d_recon_unposed.json $o/
done
