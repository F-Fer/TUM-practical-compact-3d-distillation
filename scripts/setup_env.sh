#!/usr/bin/env bash
# Initialise the submodules and create .venv with the torch group of this machine.
# Usage: scripts/setup_env.sh [cu121|cu128|mac]    (default: mac on macOS, cu128 on the tum2 cluster, cu121 elsewhere)
set -euo pipefail
cd "$(dirname "$0")/.."
git submodule update --init
group=${1:-}
if [ -z "$group" ]; then
    case "$(uname -s)-$(hostname)" in
        Darwin-*) group=mac ;;
        Linux-vcvm* | Linux-VC1*) group=cu128 ;;   # tum2 login and RTX 5090 nodes
        *) group=cu121 ;;
    esac
fi
echo "torch group: $group"
if [ "$group" = mac ]; then
    uv sync --no-default-groups --group mac --no-install-package xformers   # xformers has no macOS wheels
else
    uv sync --no-default-groups --group "$group"
fi
