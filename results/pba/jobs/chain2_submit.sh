#!/bin/bash
# Usage: chain2_submit.sh <name> <list file> [sbatch options]; snapshots run_tto_pba.py (as PBA_SCRIPT) and the list
set -e
snap=$SCRATCH/jobs/snap/chain_$1.py
cp ~/TUM-practical-compact-3d-distillation/scripts/run_tto_pba.py $snap
sed -i "s#^REPO_ROOT = .*#REPO_ROOT = '$HOME/TUM-practical-compact-3d-distillation'#" $snap
cp $2 $SCRATCH/jobs/snap/chain_$1.txt
name=$1; list=$2; shift 2
PBA_SCRIPT=$snap sbatch --parsable --export=ALL --job-name=pc_$name "$@" $SCRATCH/jobs/chain2.sh $SCRATCH/jobs/snap/chain_$name.txt
