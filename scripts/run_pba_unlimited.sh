#!/bin/bash
# "If runtime didn't matter": the current recommended PBA configurations against higher resolution, all frame
# pairs, 16x samples and a radial camera, on a small subset (HiRoom 4, ETH3D 3, DTU-64 2 scenes; 7-Scenes and
# ScanNet++ have 100 frames and don't fit these variants on an 8 GB GPU). Report: results/pba/REPORT3.md.
#
# Usage: scripts/run_pba_unlimited.sh [model ...] [-- variant ...]   (default: DA3-SMALL DA3-LARGE-1.1, all variants)
# Writes workspace/tto/unlimited/<model>/<variant>/, resumable (finished scenes are skipped). Long: start it with
#   setsid nohup scripts/run_pba_unlimited.sh > workspace/tto/unlimited/run.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
OUT=workspace/tto/unlimited

models=(); variants=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do models+=("$1"); shift; done
[ "$1" == "--" ] && shift && variants=("$@")
[ ${#models[@]} -eq 0 ] && models=(DA3-SMALL DA3-LARGE-1.1)
[ ${#variants[@]} -eq 0 ] && variants=(base base_s1 radial hires hires_match allpairs allpix)

declare -A SCENES=(
  [hiroom]=20241230/828738/cam_sampled_08,20241230/828744/cam_sampled_04,20241230/828745/cam_sampled_05,20241230/828749/cam_sampled_12
  [eth3d]=courtyard,delivery_area,electro
  [dtu64]=scan105,scan114
)
V3="features=vgg photo_loss=ncc nbr_mode=spread neighbors=10 lr_rot=3e-3 lr_trans=3e-3 lr_focal=3e-3 levels=8,4,2,1,1 lm_levels=4 lm_iters=5"

for model in "${models[@]}"; do
  # the recommended configurations of results/pba/REPORT2.md
  if [[ $model == *LARGE* ]]; then BASE="$V3 iters=50 w_prior=100 w_pose=10"; else BASE="$V3 iters=200"; fi
  for variant in "${variants[@]}"; do
    case $variant in
      base) extra="" ;;
      base_s1) extra="seed=1" ;;  # seed-to-seed noise
      radial) extra="camera=radial" ;;  # shared k1, k2
      hires) extra="res_scale=2 levels=16,8,4,2,1" ;;  # same blur in scene units, the LM polish on the sharp 2x images
      hires_match) extra="res_scale=2 levels=16,8,4,2,2" ;;  # 2x sampling, every level's blur as in base
      allpairs) extra="neighbors=1000 nbr_mode=top sample_chunk=1024" ;;  # every other frame is a target
      allpix) extra="samples=65536 sample_chunk=4096 lm_samples=16384" ;;  # 16x samples per Adam step, 4x per LM step
      *) echo "unknown variant $variant"; exit 1 ;;
    esac
    dst=$OUT/$model/$variant
    for ds in hiroom eth3d dtu64; do
      echo "=== $(date +%T) $model $variant $ds"
      $PY scripts/run_tto_pba.py workspace/evaluation/$model $dst $BASE $extra datasets=$ds scenes=${SCENES[$ds]} | grep -E "\] $ds [0-9]|mean|FAILED"
    done
    # HiRoom reconstruction (ETH3D's full-resolution fusion needs too much RAM next to other runs; DTU-64 is pose only)
    $PY scripts/eval_tto.py workspace/evaluation/$model $dst hiroom pose,recon_unposed --workers 1 2>&1 | tail -3
  done
done
echo "=== $(date +%T) done"
