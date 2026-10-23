"""Run the DA3 benchmark evaluator and log per-scene runtime and peak GPU memory.

Usage (from anywhere; paths are resolved relative to the repo root):
    uv run scripts/run_bench.py <model> [evaluator overrides ...]

    uv run scripts/run_bench.py depth-anything/DA3-SMALL eval.datasets=[hiroom]
    uv run scripts/run_bench.py depth-anything/DA3-SMALL eval.eval_only=true

Outputs go to workspace/evaluation/<model name>/ unless workspace.work_dir is given:
    model_results/, metric_results/   written by the DA3 evaluator
    timings.jsonl                     one line per inference call (scene x posed/unposed)
    runs.jsonl                        one line per invocation (machine, versions, wall time)
"""

import functools
import json
import os
import platform
import runpy
import sys
import time
from datetime import datetime

import torch
from scipy.spatial import KDTree

# Recon scoring spends ~95% of its time in single-threaded KDTree queries (bench/utils.py
# nn_correspondance); querying on all cores gives identical distances, ~7x faster
KDTree.query = functools.partialmethod(KDTree.query, workers=-1)

# DA3's logger reads its level once at import; its per-call INFO timings duplicate timings.jsonl
os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
from depth_anything_3.api import DepthAnything3  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def install_timing_hooks(work_dir):
    """Wrap DepthAnything3.forward/inference to record time and peak memory per call."""
    timings_path = os.path.join(work_dir, "timings.jsonl")
    results_dir = os.path.join(work_dir, "model_results")
    last_forward = {}

    orig_forward = DepthAnything3.forward
    orig_inference = DepthAnything3.inference

    def forward(self, image, *args, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        out = orig_forward(self, image, *args, **kwargs)
        torch.cuda.synchronize()
        last_forward.update(
            forward_s=time.perf_counter() - start,
            peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30,
            num_frames=image.shape[1],
            input_hw=list(image.shape[-2:]),
        )
        return out

    def inference(self, image, extrinsics=None, *args, export_dir=None, **kwargs):
        last_forward.clear()
        start = time.perf_counter()
        out = orig_inference(self, image, extrinsics, *args, export_dir=export_dir, **kwargs)
        append_jsonl(
            timings_path,
            {
                "task": os.path.relpath(export_dir, results_dir) if export_dir else None,
                "posed": extrinsics is not None,
                "inference_s": time.perf_counter() - start,  # incl. image loading and export
                **last_forward,
            },
        )
        return out

    DepthAnything3.forward = forward
    DepthAnything3.inference = inference


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.exit(__doc__)
    model, overrides = sys.argv[1], sys.argv[2:]
    sys.stdout.reconfigure(line_buffering=True)  # per-scene metric lines show up in redirected logs
    os.chdir(REPO_ROOT)  # the evaluator resolves workspace/benchmark_dataset relative to cwd

    work_dir = next(
        (o.split("=", 1)[1] for o in overrides if o.startswith("workspace.work_dir=")),
        os.path.join("workspace", "evaluation", os.path.basename(model.rstrip("/"))),
    )
    os.makedirs(work_dir, exist_ok=True)
    install_timing_hooks(work_dir)

    run = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "model": model,
        "overrides": overrides,
    }
    sys.argv = ["depth_anything_3.bench.evaluator", f"model.path={model}", f"workspace.work_dir={work_dir}"]
    sys.argv += [o for o in overrides if not o.startswith("workspace.work_dir=")]

    start = time.perf_counter()
    runpy.run_module("depth_anything_3.bench.evaluator", run_name="__main__", alter_sys=True)
    run["wall_s"] = time.perf_counter() - start
    append_jsonl(os.path.join(work_dir, "runs.jsonl"), run)
    print(f"[run_bench] wall time {run['wall_s'] / 60:.1f} min, logs in {work_dir}")


if __name__ == "__main__":
    main()
