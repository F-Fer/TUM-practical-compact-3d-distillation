#!/usr/bin/env bash
# Track-based BA on the dev subset of results/pba (HiRoom 10, 7-Scenes 2, ScanNet++ 3, DTU-64 4, ETH3D 3 scenes), written
# in the layout that scripts/plot_compute.py reads: workspace/tto/<model>/<config with _ instead of ->.
#
# Usage: scripts/run_tto_dev.sh <model> [config ...]     (configs: scripts/tto_configs.sh, default roma-turbo-q4 roma-turbo-q8 vggsfm-cauchy vggsfm-trivial)
#   e.g. scripts/run_tto_dev.sh DA3-SMALL
# Run it on the machine the PBA runs were timed on (RTX 5090), otherwise the time axis of the plot is not comparable.
# Needs workspace/evaluation/<model> (scripts/run_bench_loop.sh). Finished scenes are skipped, so it can be restarted.
set -uo pipefail

cd "$(dirname "$0")/.."
source scripts/tto_configs.sh
model=$1
shift
configs=("$@")
[ ${#configs[@]} -eq 0 ] && configs=(roma-turbo-q4 roma-turbo-q8 vggsfm-cauchy vggsfm-trivial)

DATASETS=hiroom,7scenes,scannetpp,dtu64,eth3d
SCENES=20241230/828738/cam_sampled_08,20241230/828744/cam_sampled_04,20241230/828745/cam_sampled_05,20241230/828749/cam_sampled_12,20241230/828750/cam_sampled_08,20241230/828757/cam_sampled_06,20241230/828760/cam_sampled_04,20241230/828761/cam_sampled_11,20241230/828763/cam_sampled_12,20241230/828766/cam_sampled_10,chess,fire,09c1414f1b,1ada7a0617,21d970d8de,scan105,scan114,scan118,scan122,courtyard,delivery_area,electro

src="workspace/evaluation/$model"
for c in "${configs[@]}"; do
    dst="workspace/tto/$model/${c//-/_}"
    mkdir -p "$(dirname "$dst")"
    # shellcheck disable=SC2086
    uv run scripts/run_tto_ba.py "$src" "$dst" datasets=$DATASETS scenes=$SCENES metric_file=pba ${CONFIGS[$c]} 2>&1 | tee -a "$dst.log"
    # one fusion worker: the 100-frame scenes of 7-Scenes and ScanNet++ need a lot of RAM
    uv run scripts/eval_tto.py "$src" "$dst" hiroom,7scenes,scannetpp recon_unposed --workers 1 2>&1 | tail -n 5
done
