#!/usr/bin/env bash
# Refine the saved DA3 benchmark predictions of each model with bundle adjustment, then compare all runs with the paper.
#
# Usage: scripts/run_tto_loop.sh [overrides ...]
#   e.g. scripts/run_tto_loop.sh datasets=hiroom max_points=40960
# Needs the benchmark run of each model (scripts/run_bench_loop.sh) in workspace/evaluation/<model>.
# Logs: workspace/tto_ba_<model>.log
set -uo pipefail

cd "$(dirname "$0")/.."
for m in DA3-SMALL DA3-BASE DA3-LARGE-1.1; do
    uv run scripts/run_tto_ba.py "workspace/evaluation/$m" "workspace/evaluation/$m-ba" "$@" 2>&1 | tee "workspace/tto_ba_$m.log"
done
uv run scripts/compare_paper.py
