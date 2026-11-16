#!/bin/bash
# Step 3: distil DéjàView's ViT-B encoder into ViT-S (train_encoder.py), then benchmark DéjàView with the student
# encoder at K = 16 and K = 8: pose on the 5 pose datasets + photometric TTO (T = 25) on all 81 pose scenes.
# Usage: sbatch scripts/distill_steps/slurm_encoder.sh <name> <train hours>
#SBATCH --job-name=dvlt_enc
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/vc17/pr26rb00/distill_runs/%x_%j.log
set -uo pipefail
cd ~/TUM-practical-compact-3d-distillation
PY=.venv/bin/python
name=$1; hours=$2
export DISTILL_ARKIT_LIST=/scratch/vc17/pr26rb00/distill_data/arkit_ready_v2.txt
export DISTILL_VAL_LIST=/scratch/vc17/pr26rb00/distill_data/arkit_val_v1.txt
echo "== $(hostname) $name hours=$hours"; date
$PY scripts/distill_steps/train_encoder.py --name "$name" --hours "$hours" 2>&1 | grep --line-buffered -v -i warning || exit 1
enc=/scratch/vc17/pr26rb00/distill_runs/$name/encoder_final.pt
[ -f "$enc" ] || { echo "no encoder_final.pt"; exit 1; }
for K in 16 8; do
    w=${name}_K$K
    DVLT_K=$K DVLT_ENC=$enc $PY scripts/distill_steps/eval_student.py nvidia/dvlt \
        "eval.datasets=[hiroom,eth3d,dtu64,7scenes,scannetpp]" "eval.modes=[pose]" \
        "workspace.work_dir=workspace/evaluation/dvlt_fs/$w" 2>&1 | grep --line-buffered -E "eval_student|wall time|Traceback|Error"
    $PY scripts/run_tto_pba.py workspace/evaluation/dvlt_fs/$w workspace/tto/dvlt_fs/$w/pho_it25 \
        datasets=hiroom,7scenes,scannetpp,dtu64,eth3d iters=25 2>&1 | grep --line-buffered -E "mean:|Traceback|Error" | cut -c1-120
done
date
