#!/usr/bin/env bash
# Runs experiments from workspace/distill/queue.txt one by one (train, then official eval).
# Line format: <name> <train.py args...>   (lines starting with # are skipped; append anytime)
# Start: tmux new -d -s distill scripts/distill/queue.sh
cd "$(dirname "$0")/../.."
# bash reads scripts incrementally: run from a frozen copy so the script can be edited meanwhile
if [ -z "$QUEUE_FROZEN" ]; then
  cp "$0" workspace/distill/.queue_frozen.sh
  QUEUE_FROZEN=1 exec bash workspace/distill/.queue_frozen.sh
fi
Q=workspace/distill/queue.txt
LOG=workspace/distill/logs
mkdir -p "$LOG"
touch "$Q"
while true; do
  line=$(grep -v '^#' "$Q" | grep -v '^\s*$' | head -1)
  if [ -z "$line" ]; then sleep 30; continue; fi
  # mark as taken
  python3 - "$Q" "$line" <<'PY'
import sys
q, line = sys.argv[1], sys.argv[2]
lines = open(q).read().split("\n")
i = lines.index(line)
lines[i] = "# started: " + line
open(q, "w").write("\n".join(lines))
PY
  name=${line%% *}
  args=${line#* }
  echo "$(date +%F_%T) start $name" >> "$LOG/queue.log"
  if [ "$args" = "EVAL" ]; then
    uv run scripts/distill/eval_student.py "$name" > "$LOG/eval_${name//[:\/]/-}.log" 2>&1
  elif [[ "$args" == *NOEVAL ]]; then
    uv run scripts/distill/train.py --name "$name" ${args% NOEVAL} > "$LOG/train_$name.log" 2>&1
  else
    uv run scripts/distill/train.py --name "$name" $args > "$LOG/train_$name.log" 2>&1 \
      && uv run scripts/distill/eval_student.py "runs/$name" > "$LOG/eval_$name.log" 2>&1
  fi
  echo "$(date +%F_%T) done $name rc=$?" >> "$LOG/queue.log"
done
