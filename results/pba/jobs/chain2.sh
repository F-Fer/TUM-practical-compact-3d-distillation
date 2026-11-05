#!/bin/bash
# Usage: sbatch chain2.sh <list file>; each line is a command run in turn (e.g. "bash $J/pba_run.sh ..." or
# "PBA_SRC=... bash $J/pba_run.sh ..."), with J=/scratch/vc17/pr26rb00/jobs
#SBATCH --job-name=pba_chain
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/vc17/pr26rb00/jobs/chain_%j.log
export J=/scratch/vc17/pr26rb00/jobs
while read -r line; do
  [ -z "$line" ] && continue
  echo ">> $line"
  eval "$line"
done < "$1"
