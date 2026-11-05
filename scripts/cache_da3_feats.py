"""Cache DA3 backbone (DINOv2) patch tokens of the benchmark scenes, for feature-metric BA (run_tto_pba.py).

Usage: .venv/bin/python scripts/cache_da3_feats.py <model> <src work dir> <out dir> [datasets=a,b] [max_scenes=n]

Re-runs DA3 exactly as the evaluator does (unposed, same frames and reference-view strategy) with
export_feat_layers, and stores per scene <out dir>/<dataset>/<scene>.npz with, for each of three layers
  local: the last layer before cross-view attention starts (alt_start - 1), i.e. per-frame DINOv2-style features
  mid:   halfway between alt_start and the last layer (after some cross-view attention)
  final: the last layer
the L2-normalised tokens, PCA-reduced per scene to 64 channels, as float16 [N, 64, H/14, W/14].
Also checks that the depth maps equal the saved predictions (the tokens must line up with them).
"""

import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
import da3_lowmem  # noqa: E402
from depth_anything_3.api import DepthAnything3  # noqa: E402
from depth_anything_3.utils.constants import EVAL_REF_VIEW_STRATEGY  # noqa: E402

DIM = 64


def pca(feat, dim, n_fit=20000, seed=0):
    """feat [N, h, w, C] -> L2-normalised, PCA-reduced [N, dim, h, w] and explained variance ratio."""
    N, h, w, C = feat.shape
    X = torch.nn.functional.normalize(feat.reshape(-1, C).float(), dim=1)
    gen = torch.Generator(device=X.device).manual_seed(seed)
    fit = X[torch.randperm(len(X), generator=gen, device=X.device)[:n_fit]]
    mu = fit.mean(0)
    _, S, V = torch.linalg.svd(fit - mu, full_matrices=False)
    Y = (X - mu) @ V[:dim].T
    ratio = float((S[:dim] ** 2).sum() / (S**2).sum())
    return Y.reshape(N, h, w, dim).permute(0, 3, 1, 2), ratio


def main():
    model_name, src, out = sys.argv[1:4]
    opts = dict(a.split("=", 1) for a in sys.argv[4:])
    os.chdir(REPO_ROOT)
    da3_lowmem.install()
    device = torch.device("cuda")
    api = DepthAnything3.from_pretrained(f"depth-anything/{model_name}").to(device).eval()
    backbone = api.model.backbone.pretrained
    n_blocks, alt = len(backbone.blocks), backbone.alt_start
    layers = {"local": alt - 1, "mid": (alt + n_blocks - 1) // 2, "final": n_blocks - 1}
    print(f"[cache_da3_feats] {model_name}: {n_blocks} blocks, alt_start {alt}, layers {layers}")

    results_dir = os.path.join(src, "model_results")
    datasets = [d for d in opts.get("datasets", "").split(",") if d] or sorted(os.listdir(results_dir))
    for dataset in datasets:
        data_dir = os.path.join(results_dir, dataset)
        scenes = sorted(os.path.dirname(p) for p, _, _ in os.walk(data_dir) if os.path.basename(p) == "unposed")
        for scene_dir in scenes[: int(opts.get("max_scenes", 0)) or None]:
            scene = os.path.relpath(scene_dir, data_dir)
            path = os.path.join(out, dataset, scene + ".npz")
            if os.path.exists(path):
                continue
            exports = os.path.join(scene_dir, "unposed", "exports")
            gt = np.load(os.path.join(exports, "gt_meta.npz"), allow_pickle=True)
            depth_saved = np.load(os.path.join(exports, "mini_npz", "results.npz"))["depth"]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = api.inference(list(gt["image_files"]), export_feat_layers=list(layers.values()), ref_view_strategy=EVAL_REF_VIEW_STRATEGY)
            torch.cuda.synchronize()
            fwd_s = time.perf_counter() - t0
            rel = float(np.abs(pred.depth - depth_saved).mean() / np.abs(depth_saved).mean())
            arrays, ratios = {}, {}
            for name, layer in layers.items():
                f = torch.as_tensor(np.asarray(pred.aux[f"feat_layer_{layer}"]), device=device)
                f = f[0] if f.dim() == 5 else f  # [N, h, w, C]
                y, ratios[name] = pca(f, DIM)
                arrays[name] = y.half().cpu().numpy()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, **arrays, layers=np.array(list(layers.values())), forward_s=fwd_s, depth_rel_diff=rel)
            print(
                f"[cache_da3_feats] {dataset} {scene}: {arrays['local'].shape} forward {fwd_s:.2f}s depth diff {rel:.1e} "
                f"PCA{DIM} var " + " ".join(f"{k} {v:.2f}" for k, v in ratios.items())
            )


if __name__ == "__main__":
    main()
