#!/bin/bash
# Track BA with our own LM solver (ba_lm.py, via run_tto_ba_truncated.py) instead of pycolmap, on DéjàView predictions with K steps.
# Usage: sbatch scripts/slurm_dvlt_tto_lm.sh <K> [iterations ...]      (submit from the repo root; default 15 and 100 iterations)
#   e.g. for k in 12 8 6 4 3; do sbatch scripts/slurm_dvlt_tto_lm.sh $k; done
# Same dev subset, tracks and scoring as slurm_dvlt_tto.sh. Per iteration cap N it writes to workspace/tto/<model>/:
#   rad_lm<N>         RoMa tracks + radial BA, depth refit (the `rad` run, BA by ba_lm stopped after N iterations)
#   depth_only_lm<N>  depth-only polish started from it
# rad_lm100 shows whether ba_lm loses quality against pycolmap (`rad`) on DéjàView, rad_lm15 what the cap saves.
#SBATCH --job-name=dvlt_tto_lm
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
ITERS=${*:-15 100}
PY=${PY:-.venv/bin/python}
model=dvlt_K$k
[ "$k" = 12 ] && model=dvlt

DATASETS=hiroom,7scenes,scannetpp,dtu64,eth3d
SCENES=20241230/828738/cam_sampled_08,20241230/828744/cam_sampled_04,20241230/828745/cam_sampled_05,20241230/828749/cam_sampled_12,20241230/828750/cam_sampled_08,20241230/828757/cam_sampled_06,20241230/828760/cam_sampled_04,20241230/828761/cam_sampled_11,20241230/828763/cam_sampled_12,20241230/828766/cam_sampled_10,chess,fire,09c1414f1b,1ada7a0617,21d970d8de,scan105,scan114,scan118,scan122,courtyard,delivery_area,electro
PBA="nbr_mode=spread neighbors=10 lr_rot=0.003 lr_trans=0.003 lr_focal=0.003 samples=1024 levels=8,4,2,1,1 iters=50 lm_levels=4 lm_iters=5"
src="workspace/evaluation/$model"
out="workspace/tto/$model"
mkdir -p "$out"
echo "== $(hostname) K=$k src=$src iterations: $ITERS"; date

score() {  # recon_unposed of a run, as in slurm_dvlt_tto.sh
    $PY scripts/eval_tto.py "$src" "$out/$1" hiroom,7scenes,scannetpp recon_unposed --workers 1 2>&1 | tail -n 6
}

for n in $ITERS; do
    # shellcheck disable=SC2086
    BA_MAX_ITERS=$n BA_ITER_LOG="$out/rad_lm$n.iters.jsonl" $PY scripts/run_tto_ba_truncated.py "$src" "$out/rad_lm$n" datasets=$DATASETS scenes=$SCENES metric_file=pba \
        ${CONFIGS[roma-turbo-q4]} camera=radial depth_fit=grid fit_grid=4 fit_ridge=10 reject=10 2>&1 | tee -a "$out/rad_lm$n.log" | grep -E "^\[run_tto_ba\]|Traceback|Error"
    score "rad_lm$n"
    # shellcheck disable=SC2086
    $PY scripts/run_tto_pba.py "$src" "$out/depth_only_lm$n" datasets=$DATASETS scenes=$SCENES $PBA init_from="$out/rad_lm$n" poses=false focal=none 2>&1 \
        | tee -a "$out/depth_only_lm$n.log" | grep -E "^\[run_tto_pba\]|Traceback|Error"
    score "depth_only_lm$n"
done
date
