#!/bin/bash
# Cheaper tracking for the track BA on DéjàView predictions: RoMa matches each query frame only to its k most covisible frames
# (pairs=k) and/or uses fewer query frames, instead of all N-1 frames from up to 8 queries.
# Usage: sbatch scripts/slurm_dvlt_tracks_fast.sh <K> [config ...]      (submit from the repo root; default configs below)
#   e.g. sbatch scripts/slurm_dvlt_tracks_fast.sh 8
# Same dev subset, own LM BA (15 iterations) and scoring as slurm_dvlt_tto_lm.sh, so rad_lm15 (all pairs) is the reference. Per
# config of tto_configs.sh it writes workspace/tto/<model>/rad_lm15_<config>; track_s and ba_s per scene are in its pba metrics.
#SBATCH --job-name=dvlt_tracks_fast
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=%x_%j.log
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"   # Slurm runs a copy of this script, so submit from the repository root
[ -f scripts/tto_configs.sh ] || { echo "not in the repository root: $PWD"; exit 1; }
source scripts/tto_configs.sh
k=$1
shift
NAMES=${*:-fast-p10 fast-p5 fast-q2p10 fast-q2p5 fast-q4e0p10}
PY=${PY:-.venv/bin/python}
model=dvlt_K$k
[ "$k" = 12 ] && model=dvlt

DATASETS=hiroom,7scenes,scannetpp,dtu64,eth3d
SCENES=20241230/828738/cam_sampled_08,20241230/828744/cam_sampled_04,20241230/828745/cam_sampled_05,20241230/828749/cam_sampled_12,20241230/828750/cam_sampled_08,20241230/828757/cam_sampled_06,20241230/828760/cam_sampled_04,20241230/828761/cam_sampled_11,20241230/828763/cam_sampled_12,20241230/828766/cam_sampled_10,chess,fire,09c1414f1b,1ada7a0617,21d970d8de,scan105,scan114,scan118,scan122,courtyard,delivery_area,electro
src="workspace/evaluation/$model"
out="workspace/tto/$model"
mkdir -p "$out"
echo "== $(hostname) K=$k src=$src configs: $NAMES"; date

for name in $NAMES; do
    run="rad_lm15_$name"
    # shellcheck disable=SC2086
    BA_MAX_ITERS=15 BA_ITER_LOG="$out/$run.iters.jsonl" $PY scripts/run_tto_ba_truncated.py "$src" "$out/$run" datasets=$DATASETS scenes=$SCENES metric_file=pba \
        ${CONFIGS[$name]} camera=radial depth_fit=grid fit_grid=4 fit_ridge=10 reject=10 2>&1 | tee -a "$out/$run.log" | grep -E "^\[run_tto_ba\]|Traceback|Error"
    $PY scripts/eval_tto.py "$src" "$out/$run" hiroom,7scenes,scannetpp recon_unposed --workers 1 2>&1 | tail -n 6
done
date
