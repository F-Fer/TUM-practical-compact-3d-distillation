#!/bin/bash
# Reconstruction quality of one model, alone and after photometric TTO (T = 25): recon_unposed F1 on HiRoom, 7-Scenes,
# ScanNet++, ETH3D (the 4-set average of knowledge/evaluation.md) and Chamfer distance on DTU-49, one fusion worker
# (ETH3D fusion needs up to ~58 GB). Predictions are deleted at the end.
# Usage: sbatch scripts/distill_steps/slurm_res_recon.sh <run> [res:K]
#   with res:K: DéjàView, <run> = "released" or a train_res.py run in workspace/distill_runs/ (ray-fit extrinsics +
#   camera-head intrinsics, RoPE stretched); without: a model whose 504-px benchmark predictions are saved in
#   workspace/evaluation/<run> (e.g. DA3-BASE; those predictions are kept).
#   DATASETS / SCENES (env) restrict the run (smoke tests).
#SBATCH --job-name=res_recon
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/vc17/pr26rb00/distill_runs/%x_%j.log
set -uo pipefail
cd ~/TUM-practical-compact-3d-distillation
PY=.venv/bin/python
name=$1; cfg=${2:-}
DS=${DATASETS:-hiroom,7scenes,scannetpp,eth3d,dtu}
SEL=(); TSEL=(); [ -n "${SCENES:-}" ] && { SEL=("eval.scenes=[$SCENES]"); TSEL=("scenes=$SCENES"); }
RECON=("eval.datasets=[$DS]" "eval.modes=[recon_unposed]" inference.num_fusion_workers=1 "${SEL[@]}")
filt() { grep -E "^\[eval_student\]|^\[run_bench\] wall|^\[run_tto_pba\].*(mean|done)|Traceback|Error|Killed" | tail -n 8; }
echo "== $(hostname) $name ${cfg:-saved predictions} datasets=$DS"; date
if [ -n "$cfg" ]; then
    res=${cfg%:*}; K=${cfg#*:}; tag=${name}_${res}_K${K}; model=nvidia/dvlt
    ck=workspace/distill_runs/$name/final.pt; [ "$name" = released ] && ck=""
    [ -z "$ck" ] || [ -f "$ck" ] || { echo "no $ck"; exit 1; }
    ev=workspace/evaluation/dvlt_rf/$tag
    DVLT_CKPT=$ck DVLT_K=$K DVLT_RES=$res DVLT_ROPE=1 DVLT_POSE=fit_headK $PY scripts/distill_steps/eval_student.py \
        $model "${RECON[@]}" "workspace.work_dir=$ev" 2>&1 | filt
else
    tag=$name; model=depth-anything/$name; ev=workspace/evaluation/$name
    $PY scripts/run_bench.py $model eval.eval_only=true "${RECON[@]}" "workspace.work_dir=$ev" 2>&1 | filt
fi
date
tt=workspace/tto/dvlt_rf/$tag/pho_it25
$PY scripts/run_tto_pba.py "$ev" "$tt" datasets=$DS iters=25 save=true "${TSEL[@]}" 2>&1 | filt
date
$PY scripts/run_bench.py $model eval.eval_only=true "${RECON[@]}" "workspace.work_dir=$tt" 2>&1 | filt
for d in ${DS//,/ }; do  # F1 (Chamfer for DTU), alone -> TTO
    $PY -c "
import json, sys
k = 'overall' if '$d' == 'dtu' else 'fscore'
v = [json.load(open(f'{p}/metric_results/${d}_recon_unposed.json'))['mean'][k] for p in sys.argv[1:]]
print('[recon] $d', k, ' -> '.join(f'{x * (1 if k == \"overall\" else 100):.2f}' for x in v))" "$ev" "$tt" 2>&1 | tail -n 1
done
rm -rf "$tt/model_results"
if [ -n "$cfg" ]; then rm -rf "$ev/model_results"; else find "$ev/model_results" -type d -name fuse -prune -exec rm -rf {} +; fi
date
