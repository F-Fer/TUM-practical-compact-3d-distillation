#!/bin/bash
# Usage: sbatch worker.sh <queue name> [idle minutes]. Pops lines from <queue>.q (commands, $J = this dir) until the queue
# stays empty for the idle time. CPU workers (--gres=gpu:0) must never see the node's GPU, which belongs to another job:
# only queues named gpu* keep it. Two workers sharing one queue file can run the same line twice (NFS caching, seen with
# two workers on one node, 2026-11-09), so give every worker its own queue (gpuA.q, cpuA.q, ...).
#SBATCH --job-name=pm_worker
#SBATCH --output=/scratch/vc17/pr26rb00/jobs/pmba/logs/worker_%j.log
export J=/scratch/vc17/pr26rb00/jobs/pmba
Q=$J/$1.q; idle=${2:-20}; left=$idle
case "$1" in gpu*) ;; *) export CUDA_VISIBLE_DEVICES="" ;; esac
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "worker $SLURM_JOB_ID on $(hostname) queue $1 cuda=[${CUDA_VISIBLE_DEVICES-unset}]"
while :; do
  until mkdir "$Q.lock" 2>/dev/null; do sleep 1; done
  line=$(head -n 1 "$Q" 2>/dev/null); [ -n "$line" ] && sed -i 1d "$Q"
  rmdir "$Q.lock"
  if [ -z "$line" ]; then
    [ "$left" -le 0 ] && break
    left=$((left-1)); sleep 60; continue
  fi
  left=$idle
  echo ">> $(date +%H:%M:%S) $line"
  eval "$line"
  echo "<< $(date +%H:%M:%S) exit $?"
done
echo "worker done"
