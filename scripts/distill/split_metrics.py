"""Official DA3-benchmark ScanNet++ metrics averaged over the held-out (VAL) scenes only.

    uv run scripts/distill/split_metrics.py [eval_dir ...]   (default: all of workspace/evaluation/*
    and workspace/distill/eval/*)"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO_ROOT, VAL_SCENES  # noqa: E402


def split_metrics(eval_dir, scenes=VAL_SCENES):
    row = {}
    for mode, keys in [("pose", ["auc03", "auc30"]), ("recon_unposed", ["fscore"]), ("recon_posed", ["fscore"])]:
        path = os.path.join(eval_dir, "metric_results", f"scannetpp_{mode}.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        have = [s for s in scenes if s in d]
        if len(have) < len(scenes):
            continue
        for k in keys:
            row[f"{mode}.{k}"] = 100 * sum(d[s][k] for s in have) / len(have)
    return row


if __name__ == "__main__":
    dirs = sys.argv[1:] or sorted(glob.glob(os.path.join(REPO_ROOT, "workspace/evaluation/*"))) + sorted(
        glob.glob(os.path.join(REPO_ROOT, "workspace/distill/eval/*"))
    )
    cols = ["pose.auc03", "pose.auc30", "recon_unposed.fscore", "recon_posed.fscore"]
    print(f"| run | {' | '.join(cols)} |\n|---|" + "---|" * len(cols))
    for d in dirs:
        r = split_metrics(d)
        if r:
            print(f"| {os.path.basename(d)} | " + " | ".join(f"{r[c]:.1f}" if c in r else "-" for c in cols) + " |")
