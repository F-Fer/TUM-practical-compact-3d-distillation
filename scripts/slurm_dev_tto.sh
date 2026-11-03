#!/bin/bash
# Track-based BA on the dev subset of results/pba, for scripts/plot_compute.py (see scripts/run_tto_dev.sh).
# Usage: sbatch scripts/slurm_dev_tto.sh <model> [config ...]
#   e.g. sbatch scripts/slurm_dev_tto.sh DA3-SMALL
# One job per model (about 1 h for 4 configs). The GPU is not shared on this partition, so the timings are comparable
# with the PBA runs; BA itself runs on the CPU, so a busy node can make it slower.
#SBATCH --job-name=dev_tto
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log
set -uo pipefail
cd ~/TUM-practical-compact-3d-distillation
echo "== $(hostname) $*"; date
scripts/run_tto_dev.sh "$@" 2>&1 | grep -v -i -E "warning|it/s\]|s/it\]"
date
