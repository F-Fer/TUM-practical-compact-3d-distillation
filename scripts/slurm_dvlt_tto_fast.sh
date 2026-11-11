#!/bin/bash
# The fast photometric TTO (run_tto_pba.py defaults of REPORT7: shared focal, 8-cell depth grid, gated radial coefficient, analytic
# gradients) on DéjàView predictions with K refinement steps (scripts/slurm_dvlt_k.sh), dev subset.
# Usage: sbatch scripts/slurm_dvlt_tto_fast.sh <K> [T ...]      (submit from the repo root; default T = 25, the fastest point of the compute plot)
#   e.g. for k in 12 8 6 4 3; do sbatch scripts/slurm_dvlt_tto_fast.sh $k; done
# Needs the run_tto_pba.py of main (REPORT7), which our branch lacks until it is merged. Writes to workspace/tto/<model>/ (model =
# dvlt_K<K>, or dvlt for K=12), named as in results/pba/runs and read by plot_compute.py:
#   pho_it<T>   photometric: grey Huber at 1,024 px, Adam on 4 levels x T iterations, then LM
#   fmh_it<T>   the same with features=vgg (VGG16 features, 8 PCA channels)
# Pose is scored by the run itself, recon_unposed by eval_tto.py (HiRoom, 7-Scenes, ScanNet++). A failing run doesn't stop the
# others; finished scenes are skipped, so the job can be resubmitted.
#SBATCH --job-name=dvlt_tto_fast
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=%x_%j.log
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"   # Slurm runs a copy of this script, so submit from the repository root
[ -f scripts/tto_configs.sh ] || { echo "not in the repository root: $PWD"; exit 1; }
k=$1
shift
TS=${*:-25}
PY=${PY:-.venv/bin/python}
model=dvlt_K$k
[ "$k" = 12 ] && model=dvlt

DATASETS=hiroom,7scenes,scannetpp,dtu64,eth3d
SCENES=20241230/828738/cam_sampled_08,20241230/828744/cam_sampled_04,20241230/828745/cam_sampled_05,20241230/828749/cam_sampled_12,20241230/828750/cam_sampled_08,20241230/828757/cam_sampled_06,20241230/828760/cam_sampled_04,20241230/828761/cam_sampled_11,20241230/828763/cam_sampled_12,20241230/828766/cam_sampled_10,chess,fire,09c1414f1b,1ada7a0617,21d970d8de,scan105,scan114,scan118,scan122,courtyard,delivery_area,electro
src="workspace/evaluation/$model"
out="workspace/tto/$model"
mkdir -p "$out"
echo "== $(hostname) K=$k src=$src T: $TS"; date

for t in $TS; do
    for fam in pho fmh; do
        extra=""
        [ "$fam" = fmh ] && extra="features=vgg"
        # shellcheck disable=SC2086
        $PY scripts/run_tto_pba.py "$src" "$out/${fam}_it$t" datasets=$DATASETS scenes=$SCENES iters=$t $extra 2>&1 \
            | tee -a "$out/${fam}_it$t.log" | grep -E "^\[run_tto_pba\]|Traceback|Error"
        $PY scripts/eval_tto.py "$src" "$out/${fam}_it$t" hiroom,7scenes,scannetpp recon_unposed --workers 1 2>&1 | tail -n 6
    done
done
date
