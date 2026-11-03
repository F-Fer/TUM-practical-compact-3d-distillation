#!/bin/bash
# copy the GIANT runs (metrics, configs) and its DA3 pose evaluation (metrics, timings) into results/pba
cd ~/TUM-practical-compact-3d-distillation
m=DA3-GIANT-1.1; D=results/pba/runs/$m
for r in workspace/tto/$m/*/; do
  n=$(basename $r); o=$D/$n; mkdir -p $o/metric_results
  cp $r/pba_config.json $r/comparison.md $o/ 2>/dev/null; cp $r/metric_results/*.json $o/metric_results/ 2>/dev/null
done
for n in chk_v3_it50 chk_fast_it50; do
  r=workspace/tto/DA3-LARGE-1.1/$n; [ -d $r ] || continue
  o=results/pba/runs/DA3-LARGE-1.1/$n; mkdir -p $o/metric_results
  cp $r/pba_config.json $r/comparison.md $o/ 2>/dev/null; cp $r/metric_results/*.json $o/metric_results/ 2>/dev/null
done
o=$D/da3_eval; mkdir -p $o
cp workspace/evaluation/$m/metric_results/*.json workspace/evaluation/$m/timings.jsonl workspace/evaluation/$m/runs.jsonl $o/
for f in pba_run.sh eth3d_rescore.sh rd.sh sync_giant.sh; do cp $SCRATCH/jobs/$f results/pba/jobs/; done
