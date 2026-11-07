#!/bin/bash
# Usage: snap.sh <tag>: freeze the PBA code, so that queued lines aren't affected by later edits
set -e
d=/scratch/vc17/pr26rb00/jobs/pmba/snap/$1
[ -e "$d" ] && { echo "snapshot $1 exists"; exit 1; }
mkdir -p "$d"
cp ~/TUM-practical-compact-3d-distillation/scripts/run_tto_pba.py "$d/"
for f in pba_analytic.py bench_pba.py; do [ -f ~/TUM-practical-compact-3d-distillation/scripts/$f ] && cp ~/TUM-practical-compact-3d-distillation/scripts/$f "$d/"; done
sed -i "s#^REPO_ROOT = .*#REPO_ROOT = '$HOME/TUM-practical-compact-3d-distillation'#" "$d/run_tto_pba.py"
echo "$d"
