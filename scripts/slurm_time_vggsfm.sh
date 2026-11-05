#!/bin/bash
# Times the VGGSfM track BA (config vggsfm-trivial: plain loss, 8 px gate) on the RTX 5090, tracked fresh, on the dev subset.
# Usage: sbatch scripts/slurm_time_vggsfm.sh <model>     (one job per model; the tracker takes about 1 min per scene)
#   e.g. sbatch scripts/slurm_time_vggsfm.sh DA3-SMALL
# Writes to workspace/tto/<model>/time_vggsfm of this checkout only. Submit from the repository root.
# Needs the DA3 predictions in workspace/evaluation, a Python (PY, default .venv/bin/python) with the VGGSfM
# dependencies of scripts/run_tto_ba.py, and the tracker weights in ~/.cache/torch/hub/checkpoints
# (vggsfm_v2_tracker.pt, aliked-n16.pth, superpoint_v1.pth, dinov2_vitb14_reg4_pretrain.pth).
#SBATCH --job-name=time_vggsfm
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"   # Slurm runs a copy of this script, so submit from the repo root
[ -f scripts/tto_configs.sh ] || { echo "not in the repository root: $PWD"; exit 1; }
source scripts/tto_configs.sh
model=$1
export XFORMERS_DISABLED=1   # DINOv2 (query frame selection) would use the xformers kernels, which have no build for the RTX 5090

PY=${PY:-.venv/bin/python}
DATASETS=hiroom,7scenes,scannetpp,dtu64,eth3d
SCENES=20241230/828738/cam_sampled_08,20241230/828744/cam_sampled_04,20241230/828745/cam_sampled_05,20241230/828749/cam_sampled_12,20241230/828750/cam_sampled_08,20241230/828757/cam_sampled_06,20241230/828760/cam_sampled_04,20241230/828761/cam_sampled_11,20241230/828763/cam_sampled_12,20241230/828766/cam_sampled_10,chess,fire,09c1414f1b,1ada7a0617,21d970d8de,scan105,scan114,scan118,scan122,courtyard,delivery_area,electro

src="workspace/evaluation/$model"
out="workspace/tto/$model"
echo "== $(hostname) $model"; date
mkdir -p "$out"
# shellcheck disable=SC2086
$PY scripts/run_tto_ba.py "$src" "$out/time_vggsfm" datasets=$DATASETS scenes=$SCENES metric_file=pba ${CONFIGS[vggsfm-trivial]} 2>&1 | tee -a "$out/time_vggsfm.log" | grep -E "^\[run_tto_ba\]|Traceback|Error"
status=${PIPESTATUS[0]}   # the job fails when the run fails
date
exit $status
