#!/usr/bin/env bash
# Run the DA3 benchmark for each model in turn, then compare all runs with the paper.
#
# Usage: scripts/run_bench_loop.sh [evaluator overrides ...]
#   e.g. scripts/run_bench_loop.sh "eval.datasets=[hiroom]"
# Logs: workspace/bench_<model>.log
set -uo pipefail

# One TSDF fusion at a time: 4 parallel ETH3D recon_posed fusions exceeded 61 GB of RAM
# and got the whole tmux session OOM-killed. Same results, ~10 min slower per model.
FUSION=inference.num_fusion_workers=1

cd "$(dirname "$0")/.."
for m in DA3-SMALL DA3-BASE DA3-LARGE-1.1; do
    uv run scripts/run_bench.py "depth-anything/$m" "$FUSION" "$@" 2>&1 | tee "workspace/bench_$m.log"
done
uv run scripts/compare_paper.py
