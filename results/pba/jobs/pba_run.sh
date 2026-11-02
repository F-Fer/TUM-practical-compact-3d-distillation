#!/bin/bash
# Usage: sbatch pba_run.sh <model> <run name> <subset> [pba overrides ...]
#   subset: dev (hiroom 10, 7scenes 2, scannetpp 3, dtu64 4, eth3d 3) or full
#SBATCH --job-name=pba
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/vc17/pr26rb00/jobs/pba_%j.log
set -uo pipefail
cd ~/TUM-practical-compact-3d-distillation
PY=.venv/bin/python
model=$1; name=$2; subset=$3; shift 3
src=${PBA_SRC:-workspace/evaluation/$model}
dst=workspace/tto/$model/$name
echo "== $(hostname) $model $name $subset $*"; date
# wandb: one run per PBA run, resumed by every run_tto_pba.py process and by wandb_summary.py (WANDB_MODE=disabled: off;
# optional WANDB_RUN_GROUP, e.g. the sweep, and extra WANDB_TAGS)
export WANDB_PROJECT=${WANDB_PROJECT:-compact3d-pba} WANDB_RESUME=allow WANDB_NAME="${model#DA3-}/$name"
export WANDB_RUN_ID=$(printf '%s' "${model}_${name}_${SLURM_JOB_ID:-local}_$(date +%s)" | tr -c 'A-Za-z0-9_-' '_')
export WANDB_TAGS="${model},${subset}${WANDB_TAGS:+,$WANDB_TAGS}"
if [ "$subset" = dev ]; then
  runs=("hiroom 10" "7scenes 2" "scannetpp 3" "dtu64 4" "eth3d 3" "dtu 4")
else
  runs=("hiroom 0" "7scenes 0" "scannetpp 0" "dtu64 0" "eth3d 0" "dtu 0")
fi
for r in "${runs[@]}"; do
  set -- $r "$@"; d=$1; n=$2; shift 2
  [ -n "${PBA_DATASETS:-}" ] && [[ ",$PBA_DATASETS," != *",$d,"* ]] && continue
  $PY ${PBA_SCRIPT:-scripts/run_tto_pba.py} $src $dst datasets=$d max_scenes=$n "$@" 2>&1 | grep -v Warning
done
ref=workspace/tto/$model/init_$subset
[ "$dst" = "$ref" ] && ref=$src
recon=hiroom,7scenes,scannetpp,dtu
$PY scripts/eval_tto.py $ref $dst ${PBA_DATASETS:-$recon} pose,recon_unposed --workers 2 2>&1 | grep -v -i -E "warning|it/s\]|s/it\]"
[ -z "${PBA_DATASETS:-}" ] && $PY scripts/eval_tto.py $ref $dst dtu64,eth3d pose 2>&1 | grep -v -i -E "warning|it/s\]|s/it\]"
$PY scripts/wandb_summary.py $ref $dst 2>&1 | grep -v Warning
date
