#!/usr/bin/env bash
# DTU-49 (first 4 scenes, Chamfer distance) for the dev-subset runs of run_tto_dev.sh: same dst dirs, same configs.
#
# Usage: scripts/run_tto_dtu49.sh <model> [config ...]     (default: the four configs of run_tto_dev.sh)
# Run it when run_tto_dev.sh is not using the GPU. Finished scenes are skipped, so it can be restarted.
set -uo pipefail

cd "$(dirname "$0")/.."
source scripts/tto_configs.sh
model=$1
shift
configs=("$@")
[ ${#configs[@]} -eq 0 ] && configs=(roma-turbo-q4 roma-turbo-q8 vggsfm-cauchy vggsfm-trivial)

src="workspace/evaluation/$model"
for c in "${configs[@]}"; do
    dst="workspace/tto/$model/${c//-/_}"
    # shellcheck disable=SC2086
    uv run scripts/run_tto_ba.py "$src" "$dst" datasets=dtu max_scenes=4 metric_file=pba ${CONFIGS[$c]} 2>&1 | tee -a "$dst.log"
    uv run scripts/eval_tto.py "$src" "$dst" dtu recon_unposed --workers 2 2>&1 | tail -n 5
done
