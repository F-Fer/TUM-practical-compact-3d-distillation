#!/bin/bash
# Follow-up of run_pba_unlimited.sh: the recommended PBA configurations with and without the radial camera on all
# ETH3D scenes (the dataset with real lens distortion). Writes workspace/tto/unlimited_eth3d/<model>/{base,radial}/.
# Usage: setsid nohup scripts/run_pba_radial_eth3d.sh > workspace/tto/unlimited_eth3d/run.log 2>&1 &
cd "$(dirname "$0")/.."
while pgrep -f run_pba_unlimited.sh > /dev/null; do sleep 60; done  # one GPU job at a time
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=workspace/tto/unlimited_eth3d
V3="features=vgg photo_loss=ncc nbr_mode=spread neighbors=10 lr_rot=3e-3 lr_trans=3e-3 lr_focal=3e-3 levels=8,4,2,1,1 lm_levels=4 lm_iters=5"
for model in DA3-SMALL DA3-LARGE-1.1; do
  if [[ $model == *LARGE* ]]; then BASE="$V3 iters=50 w_prior=100 w_pose=10"; else BASE="$V3 iters=200"; fi
  for variant in base radial; do
    extra=""; [ $variant == radial ] && extra="camera=radial"
    echo "=== $(date +%T) $model $variant eth3d"
    .venv/bin/python scripts/run_tto_pba.py workspace/evaluation/$model $OUT/$model/$variant $BASE $extra datasets=eth3d | grep -E "\] eth3d [0-9]|FAILED"
  done
done
echo "=== $(date +%T) done"
