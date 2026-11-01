"""DA3 pose and reconstruction quality vs. the number of input views (no TTO).

Usage: .venv/bin/python scripts/run_views.py <model> <stage> [views=2,4,8,16,32,64,all] [targets=8]
                                             [datasets=a,b] [scenes=a,b] [out=workspace/views] [workers=4]
  model: DA3-SMALL, DA3-BASE, DA3-LARGE-1.1
  stage: infer (GPU: DA3 on every input set) or eval (CPU: the unmodified DA3 evaluator on the saved predictions;
         pose on POSE, recon_unposed on RECON; datasets/modes that already have metric files are skipped)

For every scene, the input for N views is taken from the benchmark's own frames (all of them, or the evaluator's
seeded sample of 100; M frames in capture order): for N <= K (targets) N evenly spaced frames, the first and last
included; for N > K the K evenly spaced frames plus, repeatedly, the frame in the middle of the largest gap. So the
inputs are spread evenly over the sequence, the inputs with N >= K are nested and contain the same K frames, and all
start with the same frame (the reference view of ref_view_strategy=first). "all" is the benchmark's own input (N = M).
Scenes with fewer than N frames are skipped for that N.

Outputs, in the evaluator's layout (model_results/, metric_results/):
  <out>/<model>/n<N>/          the N-view predictions
  <out>/<model>/n<N>_t<K>/     the same predictions restricted to the K target frames (the input of n<K>):
                               the scored camera pairs and the fused views are the same for every N >= K, so these
                               show what the other N - K views do to the cameras and depth of these K views
  <out>/<model>/timings.jsonl  forward time and peak GPU memory per inference call (run_bench.py format; the task is
                               relative to <out>/<model>/model_results, i.e. ../n<N>/model_results/<dataset>/<scene>/unposed)
"""

import bisect
import os
import sys

import run_bench  # noqa: I001  (first: sets the CUDA allocator config, KDTree on all cores, da3_lowmem)
import numpy as np
import torch
from addict import Dict
from depth_anything_3.api import DepthAnything3
from depth_anything_3.bench.evaluator import Evaluator
from depth_anything_3.bench.registries import MV_REGISTRY
from depth_anything_3.utils.constants import EVAL_REF_VIEW_STRATEGY

POSE = ["hiroom", "7scenes", "scannetpp", "dtu64", "eth3d"]
RECON = ["hiroom", "7scenes", "scannetpp"]  # ETH3D's fusion needs ~60 GB of RAM, DTU's ignores frame subsets


def evenly_spaced(m, n):
    return [int(x) for x in np.linspace(0, m - 1, n) + 0.5]  # rounds half up: distinct for n <= m


def input_frames(m, n, k):
    """Indices of the n input frames out of m: n evenly spaced ones if n <= k, otherwise the k evenly spaced ones and
    then repeatedly the middle of the largest gap (the first one on ties)."""
    if n <= k:
        return evenly_spaced(m, n)
    chosen = evenly_spaced(m, k)
    while len(chosen) < n:
        gap, start = max((b - a, -a) for a, b in zip(chosen, chosen[1:]))
        bisect.insort(chosen, -start + gap // 2)
    return chosen


class SubsetEvaluator(Evaluator):
    """Scores only the scenes that have predictions (with N views: the scenes with at least N frames)."""

    def _get_scenes(self, dataset):
        data = next(k for k, v in self.datasets.items() if v is dataset)
        return scenes_with_results(self.work_dir, data, dataset.SCENES)


def scenes_with_results(work_dir, data, scenes):
    return [s for s in scenes if os.path.exists(os.path.join(work_dir, "model_results", data, s, "unposed", "exports", "mini_npz", "results.npz"))]


def subset(scene_data, idx):
    """The GT meta the evaluator saves (_save_gt_meta) for the frames idx."""
    return Dict(
        image_files=[scene_data.image_files[i] for i in idx],
        extrinsics=scene_data.extrinsics[idx],
        intrinsics=scene_data.intrinsics[idx],
    )


def save_mini_npz(export_dir, pred, idx):
    """DA3's mini_npz export (utils/export/npz.py) of the frames idx of a prediction, written synchronously."""
    path = os.path.join(export_dir, "exports", "mini_npz", "results.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        depth=np.round(pred.depth[idx], 8),
        conf=np.round(pred.conf[idx], 2),
        extrinsics=pred.extrinsics[idx],
        intrinsics=pred.intrinsics[idx],
    )


def infer(model, root, views, targets, datasets, scenes):
    ev = Evaluator(work_dir=root, datas=datasets, modes=["pose"], scenes=scenes)
    api = DepthAnything3.from_pretrained(f"depth-anything/{model}").to("cuda")
    run_bench.install_timing_hooks(DepthAnything3, root)
    first = ev.datasets[datasets[0]]
    api.inference(first.get_data(ev._get_scenes(first)[0]).image_files[:8])  # warm-up (CUDA/cuBLAS init), logged with task null
    for data in datasets:
        dataset = ev.datasets[data]
        for scene in ev._get_scenes(dataset):
            scene_data = ev._sample_frames(dataset.get_data(scene), scene)
            m = len(scene_data.image_files)
            target_files = {scene_data.image_files[i] for i in input_frames(m, targets, targets)}
            for v in views:
                n = m if v == "all" else int(v)
                if n > m:
                    continue
                sub = subset(scene_data, input_frames(m, n, targets))
                export_dir = os.path.join(root, f"n{v}", "model_results", data, scene, "unposed")
                pred = api.inference(sub.image_files, export_dir=export_dir, export_format="mini_npz", ref_view_strategy=EVAL_REF_VIEW_STRATEGY)
                ev._save_gt_meta(export_dir, sub)
                if n > targets:
                    keep = [i for i, f in enumerate(sub.image_files) if f in target_files]
                    target_dir = os.path.join(root, f"n{v}_t{targets}", "model_results", data, scene, "unposed")
                    save_mini_npz(target_dir, pred, keep)
                    ev._save_gt_meta(target_dir, subset(sub, keep))
            print(f"[run_views] {model} {data} {scene}: {m} frames")


def evaluate(root, views, targets, datasets, workers):
    names = [f"n{v}" for v in views] + [f"n{v}_t{targets}" for v in views if f"{v}" != f"{targets}"]
    for name in names:
        # recon_unposed aligns the prediction to GT by a Sim(3) fit to the camera centres, undefined for 2 views
        modes = (("pose", POSE),) if name in ("n1", "n2") else (("pose", POSE), ("recon_unposed", RECON))
        work_dir = os.path.join(root, name)
        if not os.path.isdir(work_dir):
            continue
        for mode, mode_datas in modes:
            datas = [
                d for d in datasets
                if d in mode_datas
                and not os.path.exists(os.path.join(work_dir, "metric_results", f"{d}_{mode}.json"))
                and scenes_with_results(work_dir, d, MV_REGISTRY.get(d).SCENES)
            ]
            if datas:
                print(f"[run_views] {name}: {mode} on {datas}")
                SubsetEvaluator(work_dir=work_dir, datas=datas, modes=[mode], num_fusion_workers=workers).eval()


def main():
    model, stage = sys.argv[1:3]
    opts = dict(a.split("=", 1) for a in sys.argv[3:])
    os.chdir(run_bench.REPO_ROOT)  # the evaluator resolves workspace/benchmark_dataset relative to cwd
    sys.stdout.reconfigure(line_buffering=True)
    views = opts.get("views", "2,4,8,16,32,64,all").split(",")
    targets = int(opts.get("targets", 8))
    datasets = opts.get("datasets", ",".join(POSE)).split(",")
    root = os.path.join(opts.get("out", "workspace/views"), model)
    if stage == "infer":
        infer(model, root, views, targets, datasets, [s for s in opts.get("scenes", "").split(",") if s] or None)
    elif stage == "eval":
        evaluate(root, views, targets, datasets, int(opts.get("workers", 4)))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
