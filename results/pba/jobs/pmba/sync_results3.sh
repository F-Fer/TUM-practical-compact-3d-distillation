#!/bin/bash
# Copy this session's run configs and metrics into results/pba (no predictions): runs/DA3-SMALL/pm_*, jobs/pmba
cd ~/TUM-practical-compact-3d-distillation
D=results/pba
for r in workspace/tto/DA3-SMALL/pm_*; do
  n=$(basename "$r"); [ -d "$r/metric_results" ] || continue
  case $n in pm_e2e_*) continue;; esac
  o=$D/runs/DA3-SMALL/$n; mkdir -p "$o/metric_results"
  cp "$r/pba_config.json" "$o/" 2>/dev/null; cp "$r/comparison.md" "$o/" 2>/dev/null
  cp "$r"/metric_results/*.json "$o/metric_results/" 2>/dev/null
  case $n in pm_L_*|pm_B_*|pm_D_*) cp "$r"/metric_results/*_track.jsonl "$o/metric_results/" 2>/dev/null;; esac
done
mkdir -p $D/jobs/pmba
J=/scratch/vc17/pr26rb00/jobs/pmba
cp $J/snap.sh $J/run.sh $J/worker.sh $J/score.sh $J/score_when_done.sh $J/sweep_lines.py $J/gpu_lines.py $J/gpu_frontier.py $J/sync_results3.sh $D/jobs/pmba/
du -sh $D $D/runs; git status --short $D | wc -l
