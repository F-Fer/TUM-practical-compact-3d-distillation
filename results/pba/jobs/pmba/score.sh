#!/bin/bash
# Usage: score.sh <run> [eth]: F1 of a run made without scoring (GPU runs), on the CPU workers
cd ~/TUM-practical-compact-3d-distillation
d=workspace/tto/${MODEL:-DA3-SMALL}/$1
ref=workspace/tto/${MODEL:-DA3-SMALL}/init_dev
.venv/bin/python scripts/eval_tto.py $ref $d hiroom,7scenes,scannetpp recon_unposed --workers 1 2>&1 | grep -v -i -E "warning|it/s\]|s/it\]" | tail -4
[ "$2" = eth ] && .venv/bin/python scripts/eval_tto.py $ref $d eth3d recon_unposed --workers 1 2>&1 | grep -v -i -E "warning|it/s\]|s/it\]" | tail -2
