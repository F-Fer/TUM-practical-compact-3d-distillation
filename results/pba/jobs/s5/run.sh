#!/bin/bash
# Usage: run.sh <snap tag> <run name> <subset> [pba overrides ...]   (session 5, local RTX 3070; DA3-SMALL unless MODEL is set)
# Runs scripts/run_tto_pba.py from the snapshot workspace/tto/snap/<tag>/ (snap.sh), so editing scripts/ doesn't touch queued runs.
# subset: dev (the 22 dev scenes), heldout (the other 59 scenes of the pose benchmark), drift (the ground-truth-start scenes of REPORT3).
set -uo pipefail
cd /home/finnf/dev/TUM-practical-compact-3d-distillation
PY=.venv/bin/python
tag=$1; name=$2; subset=$3; shift 3
model=${MODEL:-DA3-SMALL}
src=workspace/evaluation/$model
dst=workspace/tto/$model/$name
code=workspace/tto/snap/$tag/scripts/run_tto_pba.py
[ -f $code ] || { echo "no snapshot $tag"; exit 1; }
case $subset in
  dev)     runs=("hiroom 0 10" "7scenes 0 2" "scannetpp 0 3" "dtu64 0 4" "eth3d 0 3") ;;
  heldout) runs=("hiroom 10 0" "7scenes 2 0" "scannetpp 3 0" "dtu64 4 0" "eth3d 3 0") ;;
  drift)   runs=("hiroom 0 5" "dtu64 0 2" "eth3d 0 3") ;;
  tiny)    runs=("hiroom 0 2" "eth3d 0 1") ;;
  *) echo "unknown subset $subset"; exit 1 ;;
esac
echo "== $(hostname) $model $name $subset snap=$tag $*"; date
export WANDB_MODE=disabled
for r in "${runs[@]}"; do
  set -- $r "$@"; d=$1; first=$2; n=$3; shift 3
  $PY $code $src $dst datasets=$d first_scene=$first max_scenes=$n "$@" 2>&1 | grep -v Warning
done
mkdir -p $dst && echo "snap=$tag" > $dst/snap.txt
date
