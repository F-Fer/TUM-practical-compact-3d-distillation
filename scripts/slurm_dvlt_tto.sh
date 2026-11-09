#!/bin/bash
# The TTO methods of the DA3 tables on DéjàView predictions with K refinement steps (scripts/slurm_dvlt_k.sh), dev subset.
# Usage: sbatch scripts/slurm_dvlt_tto.sh <K>      (submit from the repo root; K=12 is workspace/evaluation/dvlt)
#   e.g. for k in 12 8 6 4 3; do sbatch scripts/slurm_dvlt_tto.sh $k; done
# Writes to workspace/tto/<model>/ (model = dvlt_K<K>, or dvlt for K=12), one run directory per method:
#   init         tracker=none: the unrefined predictions, scored by the same pipeline
#   polish       photometric BA + LM polish from the predictions
#   rad          RoMa tracks + radial BA, depth refit (same as time_rad of slurm_time_tracks.sh)
#   depth_only   depth-only polish started from the BA cameras
#   full_polish  full polish started from the BA cameras
# Pose is scored by the run itself, recon_unposed by eval_tto.py (HiRoom, 7-Scenes, ScanNet++). A failing step doesn't stop
# the others; finished scenes are skipped, so the job can be resubmitted. w_prior stays at its default (DA3 used 10 and
# 100 for BASE and LARGE-1.1): pass EXTRA="w_prior=10" in the environment to change it.
#SBATCH --job-name=dvlt_tto
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
PY=${PY:-.venv/bin/python}
EXTRA=${EXTRA:-}
model=dvlt_K$k
[ "$k" = 12 ] && model=dvlt

DATASETS=hiroom,7scenes,scannetpp,dtu64,eth3d
SCENES=20241230/828738/cam_sampled_08,20241230/828744/cam_sampled_04,20241230/828745/cam_sampled_05,20241230/828749/cam_sampled_12,20241230/828750/cam_sampled_08,20241230/828757/cam_sampled_06,20241230/828760/cam_sampled_04,20241230/828761/cam_sampled_11,20241230/828763/cam_sampled_12,20241230/828766/cam_sampled_10,chess,fire,09c1414f1b,1ada7a0617,21d970d8de,scan105,scan114,scan118,scan122,courtyard,delivery_area,electro
PBA="nbr_mode=spread neighbors=10 lr_rot=0.003 lr_trans=0.003 lr_focal=0.003 samples=1024 levels=8,4,2,1,1 iters=50 lm_levels=4 lm_iters=5"
src="workspace/evaluation/$model"
out="workspace/tto/$model"
mkdir -p "$out"
echo "== $(hostname) K=$k src=$src"; date

score() {  # recon_unposed of a run; one fusion worker, the 100-frame scenes need a lot of RAM
    $PY scripts/eval_tto.py "$src" "$out/$1" hiroom,7scenes,scannetpp recon_unposed --workers 1 2>&1 | tail -n 6
}
ba() {  # name, config
    # shellcheck disable=SC2086
    $PY scripts/run_tto_ba.py "$src" "$out/$1" datasets=$DATASETS scenes=$SCENES metric_file=pba ${CONFIGS[$2]} ${3:-} 2>&1 | tee -a "$out/$1.log" | grep -E "^\[run_tto_ba\]|Traceback|Error"
    score "$1"
}
pba() {  # name, extra overrides
    # shellcheck disable=SC2086
    $PY scripts/run_tto_pba.py "$src" "$out/$1" datasets=$DATASETS scenes=$SCENES $PBA $EXTRA $2 2>&1 | tee -a "$out/$1.log" | grep -E "^\[run_tto_pba\]|Traceback|Error"
    score "$1"
}

ba init init-dev
pba polish ""
ba rad roma-turbo-q4 "camera=radial depth_fit=grid fit_grid=4 fit_ridge=10 reject=10"
pba depth_only "init_from=$out/rad poses=false focal=none"
pba full_polish "init_from=$out/rad"
date
