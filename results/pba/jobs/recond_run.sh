#!/bin/bash
# Usage: recond_run.sh <model> <src run> <dst run> <subset>: DA3 posed mode with the cameras of workspace/tto/<model>/<src run>, then recon scoring
cd ~/TUM-practical-compact-3d-distillation
PY=.venv/bin/python
model=$1; srcrun=$2; dstrun=$3; subset=$4
src=workspace/tto/$model/$srcrun; dst=workspace/tto/$model/$dstrun
echo "== $(hostname) recond $model $srcrun -> $dstrun"; date
$PY scripts/run_tto_recond.py $model $src $dst 2>&1 | grep "^\[run"
$PY scripts/eval_tto.py workspace/tto/$model/init_$subset $dst hiroom,7scenes,scannetpp,dtu recon_unposed --workers 2 2>&1 | grep -v -i -E "warning|it/s\]|s/it\]"
date
