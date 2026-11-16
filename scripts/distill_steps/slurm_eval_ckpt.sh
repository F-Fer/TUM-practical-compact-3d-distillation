#!/bin/bash
# Benchmark one train.py checkpoint: pose at K (5 datasets) + photometric TTO (T = 25) on all pose scenes.
# Usage: sbatch scripts/distill_steps/slurm_eval_ckpt.sh <ckpt.pt> <label> [K=4]
#SBATCH --job-name=dvlt_fs_eval
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/vc17/pr26rb00/distill_runs/%x_%j.log
set -uo pipefail
cd ~/TUM-practical-compact-3d-distillation
PY=.venv/bin/python
ck=$1; label=$2; K=${3:-4}
echo "== $(hostname) $ck -> ${label}_K$K"; date
DVLT_CKPT=$ck DVLT_K=$K $PY scripts/distill_steps/eval_student.py nvidia/dvlt \
    "eval.datasets=[hiroom,eth3d,dtu64,7scenes,scannetpp]" "eval.modes=[pose]" \
    "workspace.work_dir=workspace/evaluation/dvlt_fs/${label}_K$K" 2>&1 | grep -v -i -E "warning|it/s\]|s/it\]" | tail -n 8
$PY scripts/run_tto_pba.py workspace/evaluation/dvlt_fs/${label}_K$K workspace/tto/dvlt_fs/${label}_K$K/pho_it25 \
    datasets=hiroom,7scenes,scannetpp,dtu64,eth3d iters=25 2>&1 | grep -E "^\[run_tto_pba\]|Traceback|Error" | tail -n 2
date
