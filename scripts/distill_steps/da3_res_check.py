"""Check and profile da3_res.py on 7-Scenes chess seq-01 frames (RTX 3080 / any local GPU).

1. --check: the float RoPE path at scale 1 is bit-identical to the upstream integer path (depth, conf, cameras).
2. profile: forward time (median of --reps after a warm-up, CUDA-synchronised) and peak allocated memory of one
   DepthAnything3.forward at --views views for each model x resolution, with da3_lowmem installed as in run_bench.py.

  .venv/bin/python scripts/distill_steps/da3_res_check.py --check
  .venv/bin/python scripts/distill_steps/da3_res_check.py --models DA3-SMALL,DA3-BASE,DA3-LARGE-1.1 --res 504,420,336,252
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "distill_steps"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

import da3_lowmem  # noqa: E402
import da3_res  # noqa: E402

if os.environ.get("DA3_LOWMEM", "1") != "0":
    da3_lowmem.install()
da3_res.install()


def frames(api, n):
    files = sorted(glob.glob(f"{REPO}/workspace/benchmark_dataset/7scenes/7Scenes/chess/seq-01/*.color.png"))
    files = files[:: max(len(files) // n, 1)][:n]
    imgs, _, _ = api._preprocess_inputs(files, None, None, 504, "upper_bound_resize")
    return api._prepare_model_inputs(imgs, None, None)[0]  # [1, n, 3, 378, 504]


def run(api, x):
    out = api.forward(x, export_feat_layers=[])
    return {k: out[k].float().cpu() for k in ("depth", "depth_conf", "extrinsics", "intrinsics")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--models", default="DA3-SMALL,DA3-BASE,DA3-LARGE-1.1")
    ap.add_argument("--res", default="504,420,336,252")
    ap.add_argument("--views", type=int, default=100)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    from depth_anything_3.api import DepthAnything3

    print("[gpu]", torch.cuda.get_device_name(), flush=True)
    for m in args.models.split(","):
        api = DepthAnything3.from_pretrained(f"depth-anything/{m}").cuda().eval()
        if args.check:
            x = frames(api, 24)
            da3_res.install(res=0)
            da3_res.FORCE_FLOAT = False
            a = run(api, x)
            da3_res.FORCE_FLOAT = True
            b = run(api, x)
            da3_res.FORCE_FLOAT = False
            for k in a:
                print(f"[check] {m} float RoPE scale 1 vs upstream {k}: equal={torch.equal(a[k], b[k])} "
                      f"maxabs={float((a[k] - b[k]).abs().max()):.3g}", flush=True)
            # low-res wrapper output shapes / intrinsics
            da3_res.install(res=336, rope=True)
            c = run(api, x)
            da3_res.install(res=0, rope=False)
            print(f"[check] {m} 336+RoPE shapes: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in c.items()))
            print(f"[check] {m} K 504 {a['intrinsics'][0, 0].tolist()}\n        K 336->504 {c['intrinsics'][0, 0].tolist()}")
            del api
            torch.cuda.empty_cache()
            continue
        x = frames(api, args.views)
        for r in [int(v) for v in args.res.split(",")]:
            da3_res.install(res=r if r < 504 else 0, rope=True)
            try:
                torch.cuda.synchronize()
                api.forward(x, export_feat_layers=[])  # warm-up
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                ts = []
                for _ in range(args.reps):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    api.forward(x, export_feat_layers=[])
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                peak = torch.cuda.max_memory_allocated() / 2**30
                s = r / 504
                h, w = round(378 * s / 14) * 14, round(504 * s / 14) * 14
                rec = dict(model=m, views=args.views, res=r, lowmem=os.environ.get("DA3_LOWMEM", "1") != "0", h=h, w=w, tokens=(h // 14) * (w // 14),
                           fwd_s=statistics.median(ts), peak_gib=peak, gpu=torch.cuda.get_device_name())
            except torch.OutOfMemoryError:
                rec = dict(model=m, views=args.views, res=r, lowmem=os.environ.get("DA3_LOWMEM", "1") != "0", oom=True)
                torch.cuda.empty_cache()
            print(json.dumps(rec), flush=True)
            if args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        del api, x
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
