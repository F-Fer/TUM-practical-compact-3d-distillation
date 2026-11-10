#!/bin/bash
# Usage: MODEL=<model> score_clean.sh <run>: recon F1 of a dev run on HiRoom, 7-Scenes, ScanNet++ and ETH3D (CPU workers),
# then delete the run's predictions (model_results/), but only if all four recon files have every dev scene.
# Datasets whose recon file is already complete are not scored again (e.g. after an interrupted worker).
cd ~/TUM-practical-compact-3d-distillation
m=${MODEL:-DA3-SMALL}
d=workspace/tto/$m/$1
ref=workspace/tto/$m/init_dev
echo "== $(hostname) job ${SLURM_JOB_ID:-} $m $1"; date
scenes() {  # number of scenes in the recon file of dataset $1 (0 if missing)
  .venv/bin/python -c "import json; r=json.load(open('$d/metric_results/$1_recon_unposed.json')); r.pop('mean', None); print(len(r))" 2>/dev/null || echo 0
}
declare -A want=([hiroom]=10 [7scenes]=2 [scannetpp]=3 [eth3d]=3)
todo=""
for x in hiroom 7scenes scannetpp; do [ "$(scenes $x)" = "${want[$x]}" ] || todo=$todo,$x; done
[ -n "$todo" ] && .venv/bin/python scripts/eval_tto.py $ref $d ${todo#,} recon_unposed --workers 1 2>&1 | grep -v -i -E "warning|it/s\]|s/it\]" | tail -4
[ "$(scenes eth3d)" = 3 ] || .venv/bin/python scripts/eval_tto.py $ref $d eth3d recon_unposed --workers 1 2>&1 | grep -v -i -E "warning|it/s\]|s/it\]" | tail -2
ok=1
for x in hiroom 7scenes scannetpp eth3d; do
  n=$(scenes $x)
  [ "$n" = "${want[$x]}" ] || { echo "incomplete recon: $x has $n scenes, expected ${want[$x]}"; ok=0; }
done
if [ $ok = 1 ]; then rm -rf "$d/model_results" && echo "deleted $d/model_results"; else echo "kept $d/model_results"; fi
date
