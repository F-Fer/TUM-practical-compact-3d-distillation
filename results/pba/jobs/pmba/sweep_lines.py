#!/usr/bin/env python3
"""Usage: sweep_lines.py <snap> <score> <subset> <base overrides in one string> name:overrides ...  -> queue lines"""
import sys
snap,score,subset,base=sys.argv[1:5]
for spec in sys.argv[5:]:
    name,_,ov=spec.partition(":")
    print(f"bash $J/run.sh {snap} {name} {subset} {score} {base} {ov} > $J/runlogs/{name}.log 2>&1")
