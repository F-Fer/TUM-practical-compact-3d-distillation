#!/bin/bash
# Photometric TTO (T = 25) on the released DéjàView predictions at several K, all pose scenes (baselines for the
# few-step students). Usage: sbatch scripts/distill_steps/slurm_base_tto.sh
#SBATCH --job-name=dvlt_base_tto
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:15:00
#SBATCH --output=/scratch/vc17/pr26rb00/distill_runs/%x_%j.log
set -uo pipefail
cd ~/TUM-practical-compact-3d-distillation
PY=.venv/bin/python
for pair in K4:dvlt_K4 K16:dvlt_probe/K16 K8:dvlt_K8 K12:dvlt K6:dvlt_K6; do
    k=${pair%%:*}; src=workspace/evaluation/${pair#*:}
    echo "== $k $src $(date +%T)"
    $PY scripts/run_tto_pba.py "$src" workspace/tto/dvlt_fs/base_$k/pho_it25 datasets=hiroom,7scenes,scannetpp,dtu64,eth3d \
        iters=25 2>&1 | grep -E "^\[run_tto_pba\]|Traceback|Error" | tail -n 2
done
date
