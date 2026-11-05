#!/bin/bash
# Usage: sbatch sweep.sh <model> <dataset> <max_scenes> "<cfg1 overrides>" "<cfg2 overrides>" ...  (pose only, no saving)
#SBATCH --job-name=pba_sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/vc17/pr26rb00/jobs/sweep_%j.log
cd ~/TUM-practical-compact-3d-distillation
model=$1; d=$2; n=$3; shift 3
for o in "$@"; do
  tag=$(echo "$o" | sed 's/[ =,]/_/g')
  echo "== $o"
  .venv/bin/python ${PBA_SCRIPT:-scripts/run_tto_pba.py} workspace/evaluation/$model workspace/tto/sweep/$model/$d/$tag datasets=$d max_scenes=$n save=false $o 2>&1 | grep "mean:" | python3 -c "
import sys
for l in sys.stdin:
    kv=dict(zip(l.split()[3::2], map(float,l.split()[4::2])))
    print(f\"   A@3 {kv['auc03_init']*100:5.1f}->{kv['auc03']*100:5.1f}  A@30 {kv['auc30_init']*100:5.1f}->{kv['auc30']*100:5.1f}  focal {kv['focal_err_init']*100:4.1f}->{kv['focal_err']*100:4.1f}%  photo {kv['photo_after']:.4f} geo {kv['geo_after']:.4f}  t {kv['opt_s']:.1f}s mem {kv['peak_mem_gib']:.1f}G\")"
done
