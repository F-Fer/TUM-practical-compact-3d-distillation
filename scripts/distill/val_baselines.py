"""Fast validation (32 random frames per held-out scene) of the pretrained models, incl.
DéjàView at different loop counts K. Writes workspace/distill/val_baselines.jsonl."""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WS, append_jsonl, load_da3, load_dvlt, predict_fn_da3, predict_fn_dvlt, val_subsets, validate  # noqa: E402

if __name__ == "__main__":
    subsets = val_subsets()
    out = os.path.join(WS, "val_baselines.jsonl")

    def run(name, fn):
        torch.cuda.synchronize()
        t = time.perf_counter()
        mean, rows = validate(fn, subsets)
        mean["sec"] = time.perf_counter() - t
        print(name, {k: round(v, 4) for k, v in mean.items()}, flush=True)
        append_jsonl(out, dict(model=name, **mean, rows=rows))

    for name in ["depth-anything/DA3-SMALL", "depth-anything/DA3-BASE", "depth-anything/DA3-LARGE-1.1"]:
        net = load_da3(name).cuda().eval()
        run(name, predict_fn_da3(net))
        del net
        torch.cuda.empty_cache()
    model = load_dvlt().cuda().eval()
    for k in [1, 2, 3, 4, 6, 8, 12, 16]:
        run(f"dvlt-K{k}", predict_fn_dvlt(model, k))
