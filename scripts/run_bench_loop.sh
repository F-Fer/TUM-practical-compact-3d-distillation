#!/usr/bin/env bash
# Run the DA3 benchmark for each model in turn, then compare all runs with the paper.
#
# Usage: [MODELS="..."] scripts/run_bench_loop.sh [evaluator overrides ...]
#   e.g. scripts/run_bench_loop.sh "eval.datasets=[hiroom]"
#        MODELS="nvidia/dvlt yyfz233/Pi3 facebook/VGGT-1B" scripts/run_bench_loop.sh
# MODELS defaults to the three DA3 models. Logs: workspace/bench_<model>.log
set -uo pipefail

# One TSDF fusion at a time: 4 parallel ETH3D recon_posed fusions exceeded 61 GB of RAM
# and got the whole tmux session OOM-killed. Same results, ~10 min slower per model.
FUSION=inference.num_fusion_workers=1

cd "$(dirname "$0")/.."
MODELS=${MODELS:-depth-anything/DA3-SMALL depth-anything/DA3-BASE depth-anything/DA3-LARGE-1.1}
for m in $MODELS; do
    uv run scripts/run_bench.py "$m" "$FUSION" "$@" 2>&1 | tee "workspace/bench_$(basename "$m").log"
done
uv run scripts/compare_paper.py
