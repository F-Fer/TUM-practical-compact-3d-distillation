#!/bin/bash
# Usage: sbatch [--gres=gpu:1 | CPU only] views.sh <model> <infer|eval> [run_views.py options ...]
#   runs the snapshot $SCRATCH/jobs/snap/run_views.py (with the repo's scripts/ on PYTHONPATH for run_bench)
#SBATCH --job-name=views
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/vc17/pr26rb00/jobs/views_%j.log
set -uo pipefail
cd ~/TUM-practical-compact-3d-distillation
[ "$2" = eval ] && export CUDA_VISIBLE_DEVICES=""  # CPU-only jobs otherwise see (and use) the node's GPU
echo "== $(hostname) $*"; date
PYTHONPATH=scripts .venv/bin/python $SCRATCH/jobs/snap/run_views.py "$@" 2>&1 | grep -v -i -E "warning|it/s\]|s/it\]"
date
