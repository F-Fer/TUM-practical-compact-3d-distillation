#!/bin/bash
# Copies configs and metric JSONs (not predictions or loss logs) of the session-5 runs to results/pba/runs/<model>/<run>/
cd /home/finnf/dev/TUM-practical-compact-3d-distillation
for m in DA3-SMALL DA3-BASE DA3-LARGE-1.1; do
  for d in workspace/tto/$m/y_*; do
    [ -d $d ] || continue
    r=$(basename $d); o=results/pba/runs/$m/$r
    mkdir -p $o/metric_results
    cp $d/pba_config.json $o/ 2>/dev/null; cp $d/snap.txt $o/ 2>/dev/null; cp $d/weak_modes.json $o/ 2>/dev/null
    cp $d/metric_results/*.json $o/metric_results/ 2>/dev/null
  done
  echo "$m: $(ls -d results/pba/runs/$m/y_* 2>/dev/null | wc -l) runs"
done
cp workspace/tto/DA3-SMALL/decomp_*.json results/pba/runs/DA3-SMALL/ 2>/dev/null
