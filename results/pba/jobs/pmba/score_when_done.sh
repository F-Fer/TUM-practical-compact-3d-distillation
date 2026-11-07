#!/bin/bash
# Usage: score_when_done.sh <run> [eth]: waits until the GPU run has all its ETH3D scenes, then scores F1
R=~/TUM-practical-compact-3d-distillation/workspace/tto/${MODEL:-DA3-SMALL}/$1/metric_results/eth3d_pba.json
for i in $(seq 1 360); do [ -f "$R" ] && [ "$(grep -c '"auc03"' "$R")" -ge 4 ] && break; sleep 10; done
bash /scratch/vc17/pr26rb00/jobs/pmba/score.sh "$@"
