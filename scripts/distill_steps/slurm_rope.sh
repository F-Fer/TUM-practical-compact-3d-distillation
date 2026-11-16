#!/bin/bash
# Training-free fix for DéjàView at a lower input resolution: RoPE patch positions stretched to the native grid
# (rope_scale.py). First the RoPE equivalence check and the decoder timing breakdown (rope_dec_check.py), then pose on
# the 5 pose datasets + photometric TTO (T = 25) at 504 px on all 81 pose scenes, as in slurm_res_sweep.sh.
# Usage: sbatch scripts/distill_steps/slurm_rope.sh "16:420 16:336 16:252"
#SBATCH --job-name=dvlt_rope
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/vc17/pr26rb00/distill_runs/%x_%j.log
set -uo pipefail
cd ~/TUM-practical-compact-3d-distillation
PY=.venv/bin/python
echo "== rope / decoder check $(date +%T)"
$PY scripts/distill_steps/rope_dec_check.py 2>&1 | grep --line-buffered -E "^\[|Traceback|Error"
for kr in ${1:-16:420 16:336 16:252}; do
    K=${kr%%:*}; R=${kr#*:}; name=rope_res${R}_K$K
    echo "== $name $(date +%T)"
    DVLT_K=$K DVLT_RES=$R DVLT_ROPE=1 $PY scripts/distill_steps/eval_student.py nvidia/dvlt \
        "eval.datasets=[hiroom,eth3d,dtu64,7scenes,scannetpp]" "eval.modes=[pose]" \
        "workspace.work_dir=workspace/evaluation/dvlt_fs/$name" 2>&1 | grep --line-buffered -E "eval_student|wall time|Traceback|Error"
    $PY scripts/run_tto_pba.py workspace/evaluation/dvlt_fs/$name workspace/tto/dvlt_fs/$name/pho_it25 \
        datasets=hiroom,7scenes,scannetpp,dtu64,eth3d iters=25 2>&1 | grep --line-buffered -E "mean:|Traceback|Error" | cut -c1-120
done
date
