#!/bin/bash
# Solv3R benchmark with C solver calls (run_bench.py --solv3r-calls): metrics, time and peak memory per C.
# Usage: sbatch scripts/slurm_solv3r_k.sh <C> [datasets] [modes]
#   e.g. for k in 16 8 6 4 3 2 1; do sbatch scripts/slurm_solv3r_k.sh $k; done
#        sbatch scripts/slurm_solv3r_k.sh 4 eth3d recon_unposed,recon_posed     (split a C that does not fit in 6 h)
# Defaults: all six datasets (dtu64 is DTU's pose run, dtu its recon run), all three modes. Output goes to workspace/evaluation/solv3r_C<C>, per-call time and memory in its timings.jsonl. TTO runs on these predictions afterwards.
#SBATCH --job-name=solv3r_c
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log
set -uo pipefail
cd ~/TUM-dvlt
k=$1
datasets=${2:-hiroom,eth3d,dtu64,dtu,7scenes,scannetpp}
modes=${3:-pose,recon_unposed,recon_posed}
echo "== $(hostname) C=$k datasets=$datasets modes=$modes"; date
# one fusion worker: the recon scoring of the 100-frame scenes and ETH3D needs a lot of RAM
.venv/bin/python scripts/run_bench.py --solv3r-calls "$k" jgkwak/solv3r "eval.datasets=[$datasets]" "eval.modes=[$modes]" \
    inference.num_fusion_workers=1 2>&1 | grep -v -i -E "warning|it/s\]|s/it\]"
date
