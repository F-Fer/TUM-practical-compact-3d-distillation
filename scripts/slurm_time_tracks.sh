#!/bin/bash
# Times the best track-based points on the RTX 5090: RoMa tracks + radial BA (tracked fresh, so track_s and ba_s are
# measured here), then the depth-only and the full photometric polish started from those cameras. The runs match
# tracks + BA ("rad_frame_rej") and its polish variants ("depth_only_rad", "hyb_rad") of the dev-subset tables.
# Usage: sbatch scripts/slurm_time_tracks.sh <model>     (one job per model; about 1 h for SMALL, whose radial BA is slower)
#   e.g. sbatch scripts/slurm_time_tracks.sh DA3-SMALL
# Writes to workspace/tto/<model>/{time_rad,time_depth_only,time_full_polish} of this checkout only.
# Needs: the DA3 predictions and benchmark data (WORKSPACE_SRC, linked read-only, default: the workspace of
# ~/TUM-practical-compact-3d-distillation) and a Python (PY, default .venv/bin/python) with RoMa v2 (`import romav2`),
# pycolmap 3.10 and the dependencies of scripts/run_tto_ba.py.
#SBATCH --job-name=time_tracks
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"   # Slurm runs a copy of this script, so submit from the repository root
[ -f scripts/tto_configs.sh ] || { echo "not in the repository root: $PWD"; exit 1; }
source scripts/tto_configs.sh
model=$1

WORKSPACE_SRC=${WORKSPACE_SRC:-$HOME/TUM-practical-compact-3d-distillation/workspace}
PY=${PY:-.venv/bin/python}
mkdir -p workspace
for d in evaluation benchmark_dataset; do [ -e "workspace/$d" ] || ln -s "$WORKSPACE_SRC/$d" "workspace/$d"; done

DATASETS=hiroom,7scenes,scannetpp,dtu64,eth3d
SCENES=20241230/828738/cam_sampled_08,20241230/828744/cam_sampled_04,20241230/828745/cam_sampled_05,20241230/828749/cam_sampled_12,20241230/828750/cam_sampled_08,20241230/828757/cam_sampled_06,20241230/828760/cam_sampled_04,20241230/828761/cam_sampled_11,20241230/828763/cam_sampled_12,20241230/828766/cam_sampled_10,chess,fire,09c1414f1b,1ada7a0617,21d970d8de,scan105,scan114,scan118,scan122,courtyard,delivery_area,electro
PBA="nbr_mode=spread neighbors=10 lr_rot=0.003 lr_trans=0.003 lr_focal=0.003 samples=1024 levels=8,4,2,1,1 iters=50 lm_levels=4 lm_iters=5"
case $model in
    DA3-BASE) EXTRA="w_prior=10" ;;
    DA3-LARGE-1.1) EXTRA="w_prior=100" ;;
    *) EXTRA="" ;;
esac

src="workspace/evaluation/$model"
out="workspace/tto/$model"
echo "== $(hostname) $model"; date
mkdir -p "$out"
# shellcheck disable=SC2086
$PY scripts/run_tto_ba.py "$src" "$out/time_rad" datasets=$DATASETS scenes=$SCENES metric_file=pba ${CONFIGS[roma-turbo-q4]} camera=radial depth_fit=grid fit_grid=4 fit_ridge=10 reject=10 2>&1 | tee -a "$out/time_rad.log" | grep -E "^\[run_tto_ba\]|Traceback|Error"
for v in "time_depth_only:poses=false focal=none" "time_full_polish:"; do
    name=${v%%:*}
    # shellcheck disable=SC2086
    $PY scripts/run_tto_pba.py "$src" "$out/$name" datasets=$DATASETS scenes=$SCENES init_from="$out/time_rad" $PBA $EXTRA ${v#*:} 2>&1 | tee -a "$out/$name.log" | grep -E "^\[run_tto_pba\]|Traceback|Error"
done
date
