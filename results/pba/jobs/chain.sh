#!/bin/bash
# Usage: sbatch chain.sh <list file>; each line: <model> <run name> <subset> [pba overrides ...], run in turn via pba_run.sh
#SBATCH --job-name=pba_chain
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/vc17/pr26rb00/jobs/chain_%j.log
while read -r line; do
  [ -z "$line" ] && continue
  bash /scratch/vc17/pr26rb00/jobs/pba_run.sh $line
done < "$1"
