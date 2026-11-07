#!/bin/bash
# Usage: run.sh <snap tag> <run name> <subset> <score 0|1|eth> [pba overrides ...]; DA3-SMALL unless MODEL is set
set -uo pipefail
cd ~/TUM-practical-compact-3d-distillation
PY=.venv/bin/python
tag=$1; name=$2; subset=$3; score=$4; shift 4
model=${MODEL:-DA3-SMALL}
src=workspace/evaluation/$model
dst=workspace/tto/$model/$name
code=/scratch/vc17/pr26rb00/jobs/pmba/snap/$tag/run_tto_pba.py
case $subset in
  dev)   runs=("hiroom 10" "7scenes 2" "scannetpp 3" "dtu64 4" "eth3d 3") ;;
  mini)  runs=("hiroom 5" "7scenes 1" "scannetpp 2" "dtu64 2" "eth3d 3") ;;
  drift) runs=("hiroom 5" "dtu64 2" "eth3d 3") ;;
  tiny)  runs=("hiroom 2" "eth3d 1") ;;
  full)  runs=("hiroom 0" "7scenes 0" "scannetpp 0" "dtu64 0" "eth3d 0") ;;
  *) echo "unknown subset $subset"; exit 1 ;;
esac
echo "== $(hostname) job ${SLURM_JOB_ID:-} cuda=[${CUDA_VISIBLE_DEVICES-unset}] $model $name $subset $*"; date
export WANDB_PROJECT=${WANDB_PROJECT:-compact3d-pba} WANDB_RESUME=allow WANDB_NAME="${model#DA3-}/$name"
export WANDB_RUN_ID=$(printf '%s' "${model}_${name}_${SLURM_JOB_ID:-local}_$(date +%s)" | tr -c 'A-Za-z0-9_-' '_')
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-pmba} WANDB_TAGS="${model},${subset},pmba"
for r in "${runs[@]}"; do
  set -- $r "$@"; d=$1; n=$2; shift 2
  $PY $code $src $dst datasets=$d max_scenes=$n "$@" 2>&1 | grep -v Warning
done
ref=workspace/tto/$model/init_dev
if [ "$score" != 0 ]; then
  sets=""
  for r in "${runs[@]}"; do set -- $r; case $1 in hiroom|7scenes|scannetpp) sets=$sets,$1;; esac; done
  [ -n "$sets" ] && $PY scripts/eval_tto.py $ref $dst ${sets#,} recon_unposed --workers 1 2>&1 | grep -v -i -E "warning|it/s\]|s/it\]"
  [ "$score" = eth ] && $PY scripts/eval_tto.py $ref $dst eth3d recon_unposed --workers 1 2>&1 | grep -v -i -E "warning|it/s\]|s/it\]"
fi
$PY scripts/wandb_summary.py $ref $dst 2>&1 | grep -v Warning | tail -1
date
