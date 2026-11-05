#!/bin/bash
# Usage: submit.sh <model> <run name> <subset> [overrides...]; snapshots run_tto_pba.py so later edits don't affect queued jobs
set -e
mkdir -p $SCRATCH/jobs/snap
snap=$SCRATCH/jobs/snap/$1_$2.py
cp ~/TUM-practical-compact-3d-distillation/scripts/run_tto_pba.py $snap
# the snapshot imports nothing local, but run it from scripts/ semantics: fix REPO_ROOT
sed -i "s#^REPO_ROOT = .*#REPO_ROOT = '$HOME/TUM-practical-compact-3d-distillation'#" $snap
res=""; [ "$3" = dev ] && res="--mem=32G --cpus-per-task=6"
PBA_SCRIPT=$snap sbatch --parsable --export=ALL --job-name=pba_$2 $res $SCRATCH/jobs/pba_run.sh "$@"
