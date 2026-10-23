#!/usr/bin/env bash
# Download and extract the DA3 benchmark (HF dataset depth-anything/DA3-BENCH).
#
# Usage: scripts/download_bench.sh [dataset ...]
#   datasets: hiroom 7scenes dtu dtu64 eth3d scannetpp (default: all)
#
# Data lands in workspace/benchmark_dataset/, the path the DA3 evaluator expects
# when run from the repo root. Zips are deleted after successful extraction.
set -euo pipefail

cd "$(dirname "$0")/.."
DEST=workspace/benchmark_dataset
DATASETS=("$@")
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(hiroom 7scenes dtu64 dtu scannetpp eth3d)

# Prefer the project env's hf CLI, fall back to a throwaway one via uvx.
if [ -x .venv/bin/hf ]; then HF=(.venv/bin/hf); else HF=(uvx --from huggingface_hub hf); fi

mkdir -p "$DEST"
for ds in "${DATASETS[@]}"; do
    if [ -e "$DEST/$ds/.extracted" ]; then
        echo "[skip] $ds already extracted"
        continue
    fi
    echo "[download] $ds"
    "${HF[@]}" download depth-anything/DA3-BENCH "$ds.zip" --repo-type dataset --local-dir "$DEST"
    echo "[extract] $ds"
    unzip -q -o "$DEST/$ds.zip" -d "$DEST"
    touch "$DEST/$ds/.extracted"
    rm "$DEST/$ds.zip"
done
rm -rf "$DEST/.cache"
du -sh "$DEST"/*
