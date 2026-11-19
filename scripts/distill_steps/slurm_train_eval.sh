#!/bin/bash
# Few-step DéjàView distillation: train (train.py), then benchmark pose at K = 4 (+ photometric TTO on all pose
# scenes) and pose at K = 6, 8. Usage: sbatch scripts/distill_steps/slurm_train_eval.sh <name> <train hours> [train args]
#SBATCH --job-name=dvlt_fs
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/vc17/pr26rb00/distill_runs/%x_%j.log
set -uo pipefail
cd ~/TUM-practical-compact-3d-distillation
PY=.venv/bin/python
name=$1; hours=$2; shift 2
export DISTILL_ARKIT_LIST=${DISTILL_ARKIT_LIST:-/scratch/vc17/pr26rb00/distill_data/arkit_ready_v1.txt}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "== $(hostname) $name hours=$hours args=$*"; date; nvidia-smi --query-gpu=name,memory.used --format=csv
$PY scripts/distill_steps/train.py --name "$name" --hours "$hours" "$@" 2>&1 | grep -v -i warning || exit 1
ck=/scratch/vc17/pr26rb00/distill_runs/$name/final.pt
[ -f "$ck" ] || { echo "no final.pt"; exit 1; }
bench() {  # K
    DVLT_CKPT=$ck DVLT_K=$1 $PY scripts/distill_steps/eval_student.py nvidia/dvlt \
        "eval.datasets=[hiroom,eth3d,dtu64,7scenes,scannetpp]" "eval.modes=[pose]" \
        "workspace.work_dir=workspace/evaluation/dvlt_fs/${name}_K$1" 2>&1 | grep -v -i -E "warning|it/s\]|s/it\]" | tail -n 25
}
bench 4
$PY scripts/run_tto_pba.py workspace/evaluation/dvlt_fs/${name}_K4 workspace/tto/dvlt_fs/${name}_K4/pho_it25 \
    datasets=hiroom,7scenes,scannetpp,dtu64,eth3d iters=25 2>&1 | grep -E "^\[run_tto_pba\]|Traceback|Error" | tail -n 3
bench 6
bench 8
date
