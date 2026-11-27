#!/bin/bash
# Benchmark a resolution-distilled student: pose on all 81 scenes (official evaluator, RoPE stretched, ray-fit
# extrinsics + camera-head intrinsics) and photometric TTO (T = 25) for each (resolution, K); predictions are deleted
# after the TTO. Usage: sbatch scripts/distill_steps/slurm_res_eval.sh <run name> ["252:16 336:16 ..."]
# POSE=head (env): cameras from the camera head (no RANSAC ray fit); results are stored as run "<name>-head".
# Run name "released": the released DéjàView weights (no checkpoint).
#SBATCH --job-name=res_eval
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/vc17/pr26rb00/distill_runs/%x_%j.log
set -uo pipefail
cd ~/TUM-practical-compact-3d-distillation
PY=.venv/bin/python
name=$1
cfgs=${2:-"336:16 420:12 252:16 336:12 252:12 420:8"}
ck=workspace/distill_runs/$name/final.pt
pose=${POSE:-fit_headK}
run=$name; [ "$pose" = head ] && run=${name}-head
if [ "$name" = released ]; then ck=""; else [ -f "$ck" ] || { echo "no $ck"; exit 1; }; fi
echo "== $(hostname) $name pose=$pose"; date
for c in $cfgs; do
    res=${c%:*}; K=${c#*:}; tag=${run}_${res}_K${K}
    ev=workspace/evaluation/dvlt_rs/$tag
    echo "== $tag $(date +%T)"
    DVLT_CKPT=$ck DVLT_K=$K DVLT_RES=$res DVLT_ROPE=1 DVLT_POSE=$pose $PY scripts/distill_steps/eval_student.py \
        nvidia/dvlt "eval.datasets=[hiroom,eth3d,dtu64,7scenes,scannetpp]" "eval.modes=[pose]" \
        "workspace.work_dir=$ev" 2>&1 | grep -E "^\[eval_student\]|^\[run_bench\] wall|Traceback|Error" | tail -n 5
    $PY scripts/run_tto_pba.py "$ev" "workspace/tto/dvlt_rs/$tag/pho_it25" datasets=hiroom,7scenes,scannetpp,dtu64,eth3d \
        iters=25 save=false 2>&1 | grep -E "^\[run_tto_pba\].*(mean|done)|Traceback|Error" | tail -n 6
    [ -f "workspace/tto/dvlt_rs/$tag/pho_it25/metric_results/eth3d_pba.json" ] && rm -rf "$ev/model_results"
done
$PY scripts/distill_steps/compare_rs.py --md workspace/distill_runs/compare_rs.md
date
