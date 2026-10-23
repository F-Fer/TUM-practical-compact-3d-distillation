#!/usr/bin/env bash
# Run the DA3 benchmark for each model in turn, then compare all runs with the paper.
#
# Usage: scripts/run_bench_loop.sh [evaluator overrides ...]
#   e.g. scripts/run_bench_loop.sh "eval.datasets=[hiroom]"
# Logs: workspace/bench_<model>.log
set -uo pipefail

cd "$(dirname "$0")/.."
for m in DA3-SMALL DA3-BASE; do  # DA3-LARGE-1.1 runs out of GPU memory, see CLAUDE.md
    uv run scripts/run_bench.py "depth-anything/$m" "$@" 2>&1 | tee "workspace/bench_$m.log"
done
uv run scripts/compare_paper.py
