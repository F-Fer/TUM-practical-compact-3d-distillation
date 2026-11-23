#!/bin/bash
# Resolution distillation, second round (S0b, knowledge/ideas/dejaview-res-distill-scale.md): released DéjàView
# 504 px K = 16 as online teacher, a fresh random clip per step from ARKit (1,200 videos) / DL3DV (prep_dl3dv.py) /
# the 11 outdoor scenes, one centred crop per clip (aspect native / 4:3 / 1:1 / 3:4, half the clips zoomed to 50-100 %
# of the frame, i.e. narrower fields of view). A 3-step smoke test at the largest input (504 px square, K = 16,
# 48 views) checks memory first.
# Usage: sbatch --job-name=<name> scripts/distill_steps/slurm_res2_train.sh <name> --res ... --ks ... [args]
#SBATCH --job-name=res2_train
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/vc17/pr26rb00/distill_runs/%x_%j.log
set -uo pipefail
cd ~/TUM-practical-compact-3d-distillation
PY=.venv/bin/python
name=$1; shift
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA_ARGS="--mix arkit:0.5,dl3dv:0.4,outdoor:0.1 --crop_min 0.5 --p_zoom 0.5 --aspects native,4:3,1:1,3:4 \
--min_views 12 --max_views 48"
echo "== $(hostname) $name args=$*"; date; nvidia-smi --query-gpu=name,memory.used --format=csv
if [ -n "${WAIT_FOR:-}" ]; then  # e.g. the marker of a running download; give up waiting after 1 h
    while [ ! -e "$WAIT_FOR" ] && [ $SECONDS -lt 3600 ]; do sleep 60; done
    [ -e "$WAIT_FOR" ] || { echo "$WAIT_FOR missing after 1 h"; exit 1; }
fi
echo "[data] $(ls workspace/distill_data/dl3dv_frames/*/DONE | wc -l) DL3DV scenes"
$PY scripts/distill_steps/train_res.py --name "${name}_smoke" --smoke 3 $DATA_ARGS --min_views 48 --aspects 1:1 \
    --res 504 --ks 16 --workers 2 2>&1 | grep -v -i warning | tail -n 4
[ "${PIPESTATUS[0]}" = 0 ] || { echo "smoke test failed"; exit 1; }
rm -rf "workspace/distill_runs/${name}_smoke"
$PY scripts/distill_steps/train_res.py --name "$name" $DATA_ARGS --lr 8e-6 --warmup 300 --hours 5.0 --fit_steps 100 \
    --val_every 1000 --workers 6 "$@" 2>&1 | grep -v -i warning
st=${PIPESTATUS[0]}
date
exit $st
