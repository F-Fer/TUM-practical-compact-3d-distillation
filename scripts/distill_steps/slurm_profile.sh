#!/bin/bash
# Step 1: time DéjàView's components (encoder, loop step, decoders, pose fit) at 24 / 100 views and 4 resolutions.
#SBATCH --job-name=dvlt_profile
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/vc17/pr26rb00/distill_runs/%x_%j.log
set -uo pipefail
cd ~/TUM-practical-compact-3d-distillation
.venv/bin/python scripts/distill_steps/profile_dvlt.py --out /scratch/vc17/pr26rb00/distill_runs/profile_$SLURM_JOB_ID.jsonl 2>&1 | grep --line-buffered -v -i warning
