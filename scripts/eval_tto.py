"""Score refined predictions with the DA3 evaluator and compare them with the unrefined ones, scene by scene.

Usage (from anywhere; paths are resolved relative to the repo root):
    .venv/bin/python scripts/eval_tto.py <src work dir> <dst work dir> [datasets] [modes] [--workers N]

    .venv/bin/python scripts/eval_tto.py workspace/evaluation/DA3-SMALL workspace/tto/DA3-SMALL-pba hiroom,7scenes pose,recon_unposed

Only the scenes present in <dst>/model_results are scored (TTO runs on subsets); the reference values are the
per-scene metrics in <src>/metric_results. Writes <dst>/metric_results/<dataset>_<mode>.json and prints
mean(src) -> mean(dst) over the same scenes; the table is also appended to <dst>/comparison.md.
"""

import functools
import json
import os
import sys

from scipy.spatial import KDTree

# same speed-up as run_bench.py (identical distances)
KDTree.query = functools.partialmethod(KDTree.query, workers=-1)
os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
from depth_anything_3.bench.evaluator import Evaluator  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOWN = {"pose": ["auc03", "auc05", "auc15", "auc30"], "recon": ["fscore", "acc", "comp", "overall"]}


def scenes_in(work_dir, dataset):
    root = os.path.join(work_dir, "model_results", dataset)
    return sorted(
        os.path.relpath(os.path.dirname(p), root)
        for p, _, files in os.walk(root)
        if os.path.basename(p) == "unposed" and os.path.exists(os.path.join(p, "exports", "mini_npz", "results.npz"))
    )


def mean_over(metrics, scenes, key):
    vals = [metrics[s][key] for s in scenes if s in metrics and key in metrics[s]]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    if "--workers" in sys.argv:
        args.remove(str(workers))
    if len(args) < 2:
        sys.exit(__doc__)
    src, dst = args[0], args[1]
    sys.stdout.reconfigure(line_buffering=True)
    os.chdir(REPO_ROOT)
    datasets = args[2].split(",") if len(args) > 2 else sorted(os.listdir(os.path.join(dst, "model_results")))
    modes = args[3].split(",") if len(args) > 3 else ["pose", "recon_unposed"]

    lines = [f"| dataset | mode | n | metric | {os.path.basename(src.rstrip('/'))} | {os.path.basename(dst.rstrip('/'))} |", "|---|---|---|---|---|---|"]
    for dataset in datasets:
        scenes = scenes_in(dst, dataset)
        if not scenes:
            continue
        ev = Evaluator(work_dir=dst, datas=[dataset], modes=modes, scenes=scenes, num_fusion_workers=workers)
        results = {}
        if "pose" in modes:
            results.update({f"{d}_pose": r for d, r in ev._eval_pose()})
        if "recon_unposed" in modes and dataset != "dtu64":
            results.update({f"{d}_recon_unposed": r for d, r in ev._eval_reconstruction("recon_unposed")})
        for key, res in results.items():
            mode = key[len(dataset) + 1 :]
            ref_path = os.path.join(src, "metric_results", f"{key}.json")
            ref = json.load(open(ref_path)) if os.path.exists(ref_path) else {}
            for metric in SHOWN["pose" if mode == "pose" else "recon"]:
                if metric not in res.get("mean", {}):
                    continue
                a, b = mean_over(ref, scenes, metric), mean_over(res, scenes, metric)
                lines.append(f"| {dataset} | {mode} | {len(scenes)} | {metric} | {a:.4f} | {b:.4f} |")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(dst, "comparison.md"), "a") as f:
        f.write(table + "\n\n")


if __name__ == "__main__":
    main()
