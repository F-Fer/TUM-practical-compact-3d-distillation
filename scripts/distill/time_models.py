"""Forward cost of teachers/students at the benchmark input (378x504): median time and peak memory
for 100 frames, TFLOPs for 24 frames (torch FlopCounterMode). Run on an otherwise idle GPU.

    uv run scripts/distill/time_models.py  -> workspace/distill/timings.json"""

import json
import os
import sys
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from common import WS, imagenet_norm, batch_from_indices, da3_forward, dvlt_cameras, dvlt_forward, load_da3, load_dvlt, load_scene, to_unit, VAL_SCENES  # noqa: E402

import da3_lowmem  # noqa: E402


def measure(fn, x, reps=3):
    fn(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn(x)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2], torch.cuda.max_memory_allocated() / 2**30


def flops(fn, x):
    with FlopCounterMode(display=False) as fc:
        fn(x)
    return fc.get_total_flops() / 1e12


if __name__ == "__main__":
    only_da3 = "--da3" in sys.argv
    d = load_scene(VAL_SCENES[0])
    x100 = to_unit(batch_from_indices(d, list(range(100)))["images"])
    x24 = x100[:, :24]
    res = {}
    for name in ["depth-anything/DA3-SMALL", "depth-anything/DA3-BASE", "depth-anything/DA3-LARGE-1.1"]:
        if "LARGE" in name:
            da3_lowmem.install()  # bit-identical CPU offload; LARGE needs it at 100 frames (adds time)
        net = load_da3(name).cuda().eval()
        params = sum(p.numel() for n, p in net.named_parameters() if not n.startswith("cam_enc")) / 1e6
        def fn(x, net=net):  # DepthAnything3Net.forward as the API runs it (bf16 autocast, ref view first)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                return net(imagenet_norm(x), ref_view_strategy="first")

        t, m = measure(fn, x100)
        res[name] = dict(params_M=params, sec_100=t, peak_gib_100=m, tflops_24=flops(fn, x24))
        print(name, res[name], flush=True)
        del net
        torch.cuda.empty_cache()
    if only_da3:
        json.dump({**(json.load(open(os.path.join(WS, "timings.json"))) if os.path.exists(os.path.join(WS, "timings.json")) else {}), **res},
                  open(os.path.join(WS, "timings.json"), "w"), indent=2)
        sys.exit()
    model = load_dvlt(12).cuda().eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    for K in [2, 3, 4, 6, 8, 12]:
        def fn(x, K=K):
            out = dvlt_forward(model, x, K)
            dvlt_cameras(out["rays"])
        t, m = measure(fn, x100)
        res[f"dvlt-K{K}"] = dict(params_M=params, sec_100=t, peak_gib_100=m, tflops_24=flops(fn, x24))
        print(K, res[f"dvlt-K{K}"], flush=True)
    path = os.path.join(WS, "timings.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    json.dump({**old, **res}, open(path, "w"), indent=2)
