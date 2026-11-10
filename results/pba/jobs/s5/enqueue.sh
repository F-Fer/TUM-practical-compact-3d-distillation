#!/bin/bash
# Usage: enqueue.sh <queue name> <job name> <command ...>: adds a job file (the command line) to the queue. Job names sort
# in queue order, so prefix them with a number (e.g. 010_b50_s0).
cd /home/finnf/dev/TUM-practical-compact-3d-distillation
q=workspace/tto/queue/$1; job=$2; shift 2
mkdir -p $q/pending
f=$q/pending/$job.sh
[ -e $f ] || [ -e $q/done/$job.sh ] || [ -e $q/running/$job.sh ] && { echo "job $job exists"; exit 1; }
printf '%s\n' "$*" > $f && echo "queued $job: $*"
