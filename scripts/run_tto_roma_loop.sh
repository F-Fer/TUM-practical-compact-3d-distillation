#!/usr/bin/env bash
# Track-based BA sweep: RoMa v2 settings and VGGSfM with a robust loss, scored with eval_tto.py and tabulated with compare_tto.py.
#
# Usage: scripts/run_tto_roma_loop.sh <model> <datasets> [config ...]
#   e.g. scripts/run_tto_roma_loop.sh DA3-SMALL hiroom                  # all configs below
#        scripts/run_tto_roma_loop.sh DA3-LARGE-1.1 hiroom,eth3d roma-turbo-q4 vggsfm-cauchy
# Needs workspace/evaluation/<model> (scripts/run_bench_loop.sh). Outputs: workspace/tto/<model>-<config>, logs next to them,
# table in workspace/tto/<model>-tracks.md. Finished scenes are skipped, so an interrupted sweep can be restarted.
set -uo pipefail

cd "$(dirname "$0")/.."
model=$1
datasets=$2
shift 2

source scripts/tto_configs.sh
configs=("$@")
[ ${#configs[@]} -eq 0 ] && configs=(roma-turbo-q4 roma-turbo-q8 roma-fast-q4 roma-base-q4 vggsfm-trivial vggsfm-cauchy)

src="workspace/evaluation/$model"
runs=()
mkdir -p workspace/tto
for c in "${configs[@]}"; do
    dst="workspace/tto/$model-$c"
    runs+=("$dst")
    # shellcheck disable=SC2086
    uv run scripts/run_tto_ba.py "$src" "$dst" datasets="$datasets" ${CONFIGS[$c]} 2>&1 | tee -a "$dst.log"
    uv run scripts/eval_tto.py "$src" "$dst" "$datasets" pose,recon_unposed --workers 2 2>&1 | tail -n 20
done
uv run scripts/compare_tto.py "$src" "${runs[@]}" --md "workspace/tto/$model-tracks.md"
