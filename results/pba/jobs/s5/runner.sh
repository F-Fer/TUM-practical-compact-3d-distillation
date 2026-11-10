#!/bin/bash
# Usage: runner.sh <queue name> <cpu list>: runs the job files in workspace/tto/queue/<name>/pending/ one at a time, in name
# order, pinned to the given CPUs (taskset). A job file is a bash script; its log goes to done/<job>.log. Stops when the file
# workspace/tto/queue/<name>/STOP exists. Start detached: setsid nohup bash runner.sh gpu 0-7 > /dev/null 2>&1 < /dev/null &
cd /home/finnf/dev/TUM-practical-compact-3d-distillation
q=workspace/tto/queue/$1
mkdir -p $q/pending $q/running $q/done
echo $$ > $q/runner.pid
while [ ! -e $q/STOP ]; do
  job=$(ls $q/pending 2>/dev/null | sort | head -1)
  if [ -z "$job" ]; then sleep 15; continue; fi
  mv $q/pending/$job $q/running/$job
  echo "$(date +%F_%T) start $job" >> $q/runner.log
  taskset -c $2 bash $q/running/$job > $q/running/$job.log 2>&1
  rc=$?
  mv $q/running/$job $q/done/$job; mv $q/running/$job.log $q/done/$job.log
  echo "$(date +%F_%T) end $job rc=$rc" >> $q/runner.log
done
rm -f $q/runner.pid
