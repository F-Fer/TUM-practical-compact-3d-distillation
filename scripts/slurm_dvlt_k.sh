#!/bin/bash
# DéjàView benchmark with K refinement steps (run_bench.py --dvlt-steps): metrics, time and peak memory per K.
# Usage: sbatch scripts/slurm_dvlt_k.sh <K> [datasets] [modes]
#   e.g. for k in 8 6 4 3 2 1; do sbatch scripts/slurm_dvlt_k.sh $k; done
#        sbatch scripts/slurm_dvlt_k.sh 4 eth3d recon_unposed,recon_posed     (split a K that does not fit in 6 h)
# Defaults: all five datasets, all three modes. Output goes to workspace/evaluation/dvlt_K<K> (K=12 is the committed
# dvlt/ run), per-call time and memory in its timings.jsonl. TTO runs on these predictions afterwards.
#SBATCH --job-name=dvlt_k
#SBATCH --partition=rtx_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log
set -uo pipefail
cd ~/TUM-practical-compact-3d-distillation
k=$1
datasets=${2:-hiroom,eth3d,dtu,7scenes,scannetpp}
modes=${3:-pose,recon_unposed,recon_posed}
echo "== $(hostname) K=$k datasets=$datasets modes=$modes"; date
# one fusion worker: the recon scoring of the 100-frame scenes and ETH3D needs a lot of RAM
.venv/bin/python scripts/run_bench.py --dvlt-steps "$k" nvidia/dvlt "eval.datasets=[$datasets]" "eval.modes=[$modes]" \
    inference.num_fusion_workers=1 2>&1 | grep -v -i -E "warning|it/s\]|s/it\]"
date
