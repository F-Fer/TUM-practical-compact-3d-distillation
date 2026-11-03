#!/bin/bash
# Usage: PBA_SCRIPT=<snapshot> sbatch rd.sh <D list>: ETH3D recon rescoring of the existing compute-plot runs (foreground,
# CPU, one scene at a time), and in the background the GPU runs of <D list> (timing not used: T > 200 runs, checks),
# started once workspace/tto/DA3-GIANT-1.1/init_dev/.done exists; their ETH3D recon is scored at the end.
#SBATCH --job-name=pc_rd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/vc17/pr26rb00/jobs/chain_%j.log
export J=/scratch/vc17/pr26rb00/jobs
cd ~/TUM-practical-compact-3d-distillation
(
  for i in $(seq 1 480); do [ -f workspace/tto/DA3-GIANT-1.1/init_dev/.done ] && break; sleep 30; done
  [ -f workspace/tto/DA3-GIANT-1.1/init_dev/.done ] || { echo "!! GIANT init_dev not done after 4 h, D list skipped"; exit 0; }
  while read -r line; do [ -z "$line" ] && continue; echo ">> $line"; eval "$line"; done < "$1"
) > $J/rd_D_${SLURM_JOB_ID}.log 2>&1 &
R=$J/eth3d_rescore.sh
bash $R DA3-SMALL init_dev; bash $R DA3-BASE init_dev; bash $R DA3-LARGE-1.1 init_dev
bash $R DA3-SMALL v3_it25 v3_it50 v3_it100 v3_it200 v3_it400 v3_it800 fast_it25 fast_it50 fast_it200 fast_it400 fast_it800 polish50 polish200 fastpolish50 abl_P3_it25_lm5 lm10
bash $R DA3-BASE v3_it50 v3_it200 v3_it400 v3_it800 fast_it50 fast_it200 fast_it400 fast_it800 polish50 fastpolish50 abl_P3_it25_lm5
bash $R DA3-LARGE-1.1 v3_it50 v3_it200 v3_it800 fast_it50 fast_it200 fast_it400 fast_it800 polish50 polish200 fastpolish50 abl_P3_it25_lm5 lm10
echo "== rescoring done $(date +%T), waiting for the D list"
wait
bash $R DA3-GIANT-1.1 v3_it800 fast_it400 fast_it800
date
