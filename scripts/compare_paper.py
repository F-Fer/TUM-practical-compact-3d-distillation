"""Compare DA3 benchmark results against the DA3 paper (arXiv 2511.10647, Tables 2 and 3).

Usage:
    uv run scripts/compare_paper.py [run dir ...] [--save]

Run dirs default to workspace/evaluation/*. With --save, each run's metric JSONs and
timing logs are copied to results/da3_bench/<model>/ together with a comparison.md,
so they can be committed.
"""

import glob
import json
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ["hiroom", "eth3d", "dtu", "7scenes", "scannetpp"]
POSE_DATASET = {"dtu": "dtu64"}  # pose is evaluated on DTU-64, recon on DTU-49

# Paper values in %, DTU Chamfer distance in mm.
# Table 2: (AUC@3, AUC@30). Table 3: (without GT pose, with GT pose).
PAPER = {
    "large": {
        "pose": {"hiroom": (58.7, 94.2), "eth3d": (32.2, 86.9), "dtu": (70.2, 96.7), "7scenes": (29.2, 86.6), "scannetpp": (60.2, 94.7)},
        "recon": {"hiroom": (69.5, 87.1), "eth3d": (65.8, 75.2), "dtu": (2.08, 1.23), "7scenes": (56.3, 49.2), "scannetpp": (67.9, 75.7)},
    },
    "base": {
        "pose": {"hiroom": (19.0, 83.2), "eth3d": (15.1, 74.6), "dtu": (60.1, 95.9), "7scenes": (20.1, 82.9), "scannetpp": (25.1, 83.4)},
        "recon": {"hiroom": (25.9, 71.4), "eth3d": (49.5, 66.7), "dtu": (2.87, 2.36), "7scenes": (49.9, 50.6), "scannetpp": (47.2, 67.8)},
    },
    "small": {
        "pose": {"hiroom": (9.49, 75.2), "eth3d": (8.59, 62.1), "dtu": (30.6, 91.2), "7scenes": (14.0, 78.7), "scannetpp": (10.9, 71.9)},
        "recon": {"hiroom": (18.3, 52.2), "eth3d": (41.6, 63.4), "dtu": (5.83, 2.49), "7scenes": (41.0, 46.8), "scannetpp": (32.3, 53.8)},
    },
}


def paper_entry(model_name):
    name = model_name.lower()
    return next((v for k, v in PAPER.items() if f"-{k}" in name), None)


def load_mean(metric_dir, data, mode):
    path = os.path.join(metric_dir, f"{data}_{mode}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["mean"]


def fmt(ours, paper):
    if ours is None:
        return "–"
    return f"{ours:.2f} ({ours - paper:+.2f})"


def compare(run_dir):
    model = os.path.basename(run_dir.rstrip("/"))
    paper = paper_entry(model)
    metric_dir = os.path.join(run_dir, "metric_results")
    if paper is None or not os.path.isdir(metric_dir):
        return None

    rows = [
        f"## {model}",
        "",
        "Ours (difference to paper). F1 and AUC in %, DTU Chamfer distance in mm.",
        "",
        "| Metric | " + " | ".join(DATASETS) + " |",
        "|---" * (len(DATASETS) + 1) + "|",
    ]
    for label, idx, key in [("Pose AUC@3", 0, "auc03"), ("Pose AUC@30", 1, "auc30")]:
        cells = []
        for d in DATASETS:
            m = load_mean(metric_dir, POSE_DATASET.get(d, d), "pose")
            cells.append(fmt(m and 100 * m[key], paper["pose"][d][idx]))
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    for label, idx, mode in [("Recon w/o GT pose", 0, "recon_unposed"), ("Recon w/ GT pose", 1, "recon_posed")]:
        cells = []
        for d in DATASETS:
            m = load_mean(metric_dir, d, mode)
            value = m and (m["overall"] if d == "dtu" else 100 * m["fscore"])  # DTU is already in mm
            cells.append(fmt(value, paper["recon"][d][idx]))
        rows.append(f"| {label} | " + " | ".join(cells) + " |")

    timings_path = os.path.join(run_dir, "timings.jsonl")
    if os.path.exists(timings_path):
        with open(timings_path) as f:
            timings = [json.loads(line) for line in f]
        if timings:
            rows += [
                "",
                f"{len(timings)} inference calls, forward total {sum(t.get('forward_s', 0) for t in timings) / 60:.1f} min, "
                f"max peak memory {max(t.get('peak_mem_gib', 0) for t in timings):.2f} GiB.",
            ]
    return model, "\n".join(rows) + "\n"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    save = "--save" in sys.argv
    run_dirs = args or sorted(glob.glob(os.path.join(REPO_ROOT, "workspace", "evaluation", "*")))
    for run_dir in run_dirs:
        result = compare(run_dir)
        if result is None:
            continue
        model, table = result
        print(table)
        if save:
            out_dir = os.path.join(REPO_ROOT, "results", "da3_bench", model)
            os.makedirs(out_dir, exist_ok=True)
            for path in glob.glob(os.path.join(run_dir, "metric_results", "*.json")):
                shutil.copy(path, out_dir)
            for name in ["timings.jsonl", "runs.jsonl"]:
                if os.path.exists(os.path.join(run_dir, name)):
                    shutil.copy(os.path.join(run_dir, name), out_dir)
            with open(os.path.join(out_dir, "comparison.md"), "w") as f:
                f.write(table)


if __name__ == "__main__":
    main()
