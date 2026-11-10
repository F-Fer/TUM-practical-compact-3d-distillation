#!/bin/bash
# Usage: snap.sh <tag>: copies scripts/*.py to workspace/tto/snap/<tag>/scripts/ (refuses to overwrite an existing tag). The scripts
# take their repo root from their own location, so the snapshot gets a symlink to the repo's workspace/ next to scripts/.
cd /home/finnf/dev/TUM-practical-compact-3d-distillation
d=workspace/tto/snap/$1
[ -e $d ] && { echo "snapshot $1 exists"; exit 1; }
mkdir -p $d/scripts && cp scripts/*.py $d/scripts/ && ln -s $PWD/workspace $d/workspace && echo "snapshot $d"
