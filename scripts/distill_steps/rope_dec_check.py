"""(1) Check that rope_scale's float-position RoPE with scale 1 reproduces the released integer RoPE.
(2) Where the decoder time goes: ray vs depth decoder, transformer front-end vs conv upsampling stack, and the
effect of decode_chunk_size and cudnn.benchmark on the whole _decode (504 px, 24 and 100 views, bf16).

  .venv/bin/python scripts/distill_steps/rope_dec_check.py [--images <folder with >= 100 same-sized images>]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_dvlt import frames, timed  # noqa: E402
from train import load_dvlt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="", help="folder of images instead of an ARKitScenes zip")
    args = ap.parse_args()
    dev = torch.device("cuda")
    from dvlt.model.dvlt.model import _slice_expand_flatten

    model = load_dvlt().to(dev).eval()
    model.inference_steps = 16
    base = frames(100, args.images)
    amp = dict(device_type="cuda", dtype=torch.bfloat16)

    # (2) decoder timing (before the RoPE patch is installed)
    for S in (24, 100):
        img = base[:, torch.linspace(0, 99, S).long()].to(dev)
        H, W = img.shape[-2:]
        with torch.no_grad(), torch.autocast(**amp):
            z0 = model._encode_images(img)
            reg = model.register_token.expand(1, S, -1, -1).reshape(S, model.num_register_tokens, -1)
            cam = _slice_expand_flatten(model.camera_token, 1, S)
            x = torch.cat([cam, reg, z0], dim=1)
            pos = model._get_rope_positions(S, H, W, dev)
            x = model._interval_step(x, 0.0, 1 / 15, pos, 1, S)
            f, p = x[:8], pos[:8]
            kw = dict(H=H, W=W, patch_start_idx=model.patch_start_idx, pos=p)
            rec = dict(views=S)
            for name, dec in (("ray", model.ray_decoder), ("depth", model.depth_decoder)):
                def front(dec=dec):
                    y = dec.proj_in(f)
                    for blk in dec.blocks:
                        y = blk(y, pos=p)
                    return dec.norm(y)
                rec[f"{name}_full_8fr"], _ = timed(lambda dec=dec: dec(f, **kw), 5)
                rec[f"{name}_transformer_8fr"], _ = timed(front, 5)
                rec[f"{name}_params_M"] = sum(q.numel() for q in dec.parameters()) / 1e6
            rec["encoder_params_M"] = sum(q.numel() for q in model.patch_embed_encoder.parameters()) / 1e6
            for bench in (False, True):
                torch.backends.cudnn.benchmark = bench
                for chunk in (8, 16, 32, 64):
                    if chunk > S and chunk != 8:
                        continue
                    model.decode_chunk_size = chunk
                    torch.cuda.reset_peak_memory_stats()
                    try:
                        t, _ = timed(lambda: model._decode(x, H, W, 1, S, pos), 3)
                        rec[f"decode_chunk{chunk}_cudnnbench{int(bench)}"] = t
                        rec[f"peak_gib_chunk{chunk}_cudnnbench{int(bench)}"] = torch.cuda.max_memory_allocated() / 2**30
                    except torch.OutOfMemoryError:
                        rec[f"decode_chunk{chunk}_cudnnbench{int(bench)}"] = "OOM"
                        torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = False
            model.decode_chunk_size = 8
        print("[decoder] " + json.dumps(rec), flush=True)

    # (1) RoPE equivalence on 24 views
    img = base[:, torch.linspace(0, 99, 24).long()].to(dev)
    with torch.no_grad(), torch.autocast(**amp):
        ref = model.forward_inference(img)
        import rope_scale

        rope_scale.install()
        rope_scale.SCALE[:] = [1.0, 1.0]
        new = model.forward_inference(img)
    for k in ("rays", "depth", "depth_conf"):
        d = (ref[k].float() - new[k].float()).abs()
        print(f"[rope check] {k}: max |diff| {d.max().item():.3e}, mean {d.mean().item():.3e}, "
              f"ref mean |x| {ref[k].float().abs().mean().item():.3e}", flush=True)


if __name__ == "__main__":
    main()
