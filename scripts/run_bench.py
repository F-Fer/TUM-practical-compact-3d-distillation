"""Run the DA3 benchmark evaluator and log per-scene runtime and peak GPU memory.

Usage (from anywhere; paths are resolved relative to the repo root):
    uv run scripts/run_bench.py <model> [evaluator overrides ...]

    uv run scripts/run_bench.py depth-anything/DA3-SMALL eval.datasets=[hiroom]
    uv run scripts/run_bench.py depth-anything/DA3-SMALL eval.eval_only=true
    uv run scripts/run_bench.py facebook/VGGT-1B eval.datasets=[hiroom]
    uv run scripts/run_bench.py --frame-seed 1 depth-anything/DA3-SMALL eval.datasets=[scannetpp] ...

<model> is a DA3 checkpoint or one of the baselines in baselines.MODELS (VGGT, Pi3, DéjàView),
which replace DepthAnything3 in the evaluator.

--frame-seed N replaces the fixed seed (42) of the evaluator's frame sampling, which picks 100
random frames in scenes with more (7-Scenes, ScanNet++). Use it with its own workspace.work_dir.
--ransac-seed N likewise replaces the seed (42) of the RANSAC pose alignment in the recon modes;
with eval.eval_only=true it re-scores existing predictions (copy them into the new work_dir).

Outputs go to workspace/evaluation/<model name>/ unless workspace.work_dir is given:
    model_results/, metric_results/   written by the DA3 evaluator
    timings.jsonl                     one line per inference call (scene x posed/unposed)
    runs.jsonl                        one line per invocation (machine, versions, wall time)
"""

import functools
import json
import os
import platform
import random
import runpy
import sys
import time
from datetime import datetime

# Fragmentation alone made DA3-LARGE-1.1 reserve 9.2 GiB for 7 GiB of tensors; must be set before CUDA init
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
from scipy.spatial import KDTree  # noqa: E402

# Recon scoring spends ~95% of its time in single-threaded KDTree queries (bench/utils.py
# nn_correspondance); querying on all cores gives identical distances, ~7x faster
KDTree.query = functools.partialmethod(KDTree.query, workers=-1)

# DA3's logger reads its level once at import; its per-call INFO timings duplicate timings.jsonl
os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
from depth_anything_3.api import DepthAnything3  # noqa: E402

import baselines  # noqa: E402
import da3_lowmem  # noqa: E402

# Keeps the backbone's output layers in CPU memory (bit-identical outputs); without it
# DA3-LARGE-1.1 runs out of memory on 10 GB, see da3_lowmem.py
da3_lowmem.install()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def install_timing_hooks(cls, work_dir):
    """Wrap cls.forward/inference (DepthAnything3 or a baseline) to record time and peak memory per call."""
    timings_path = os.path.join(work_dir, "timings.jsonl")
    results_dir = os.path.join(work_dir, "model_results")
    last_forward = {}

    orig_forward = cls.forward
    orig_inference = cls.inference

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

    cls.forward = forward
    cls.inference = inference


def set_frame_seed(seed):
    """Make the evaluator's _sample_frames, which calls random.seed(42) before shuffling the frame
    indices, use `seed` instead. Nothing else in depth_anything_3 calls random.seed."""
    orig_seed = random.seed

    def seed_fn(a=None, *args, **kwargs):
        return orig_seed(seed if a == 42 else a, *args, **kwargs)

    random.seed = seed_fn


def set_ransac_seed(seed):
    """Make the RANSAC Sim(3) alignment of the recon modes, which every dataset calls with
    random_state=42 (10 iterations on random halves of the poses), use `seed` instead."""
    from depth_anything_3.utils import pose_align

    orig_ransac = pose_align._ransac_align_sim3

    def ransac(*args, random_state=None, **kwargs):
        return orig_ransac(*args, random_state=seed, **kwargs)

    pose_align._ransac_align_sim3 = ransac


def main():
    args = sys.argv[1:]
    frame_seed = ransac_seed = None
    while args[:1] in (["--frame-seed"], ["--ransac-seed"]):
        if args[0] == "--frame-seed":
            frame_seed = int(args[1])
            set_frame_seed(frame_seed)
        else:
            ransac_seed = int(args[1])
            set_ransac_seed(ransac_seed)
        args = args[2:]
    if not args or args[0].startswith("-"):
        sys.exit(__doc__)
    model, overrides = args[0], args[1:]
    sys.stdout.reconfigure(line_buffering=True)  # per-scene metric lines show up in redirected logs
    os.chdir(REPO_ROOT)  # the evaluator resolves workspace/benchmark_dataset relative to cwd

    work_dir = next(
        (o.split("=", 1)[1] for o in overrides if o.startswith("workspace.work_dir=")),
        os.path.join("workspace", "evaluation", os.path.basename(model.rstrip("/"))),
    )
    os.makedirs(work_dir, exist_ok=True)
    if model in baselines.MODELS:
        # the evaluator builds its model with DepthAnything3.from_pretrained(model.path).to(device)
        DepthAnything3.from_pretrained = staticmethod(baselines.load)
        install_timing_hooks(baselines.MODELS[model], work_dir)
    else:
        install_timing_hooks(DepthAnything3, work_dir)

    run = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "model": model,
        "overrides": overrides,
        "frame_seed": frame_seed,
        "ransac_seed": ransac_seed,
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
