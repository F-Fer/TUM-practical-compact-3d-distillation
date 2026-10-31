#!/bin/bash
# Usage: chain_submit.sh <name> <list file>; snapshots run_tto_pba.py and the list, then submits chain.sh
set -e
mkdir -p $SCRATCH/jobs/snap
snap=$SCRATCH/jobs/snap/chain_$1.py
cp ~/TUM-practical-compact-3d-distillation/scripts/run_tto_pba.py $snap
sed -i "s#^REPO_ROOT = .*#REPO_ROOT = '$HOME/TUM-practical-compact-3d-distillation'#" $snap
cp $2 $SCRATCH/jobs/snap/chain_$1.txt
PBA_SCRIPT=$snap sbatch --parsable --export=ALL --job-name=pc_$1 $SCRATCH/jobs/chain.sh $SCRATCH/jobs/snap/chain_$1.txt
