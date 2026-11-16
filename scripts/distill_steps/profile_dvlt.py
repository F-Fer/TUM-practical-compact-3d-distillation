"""Where does DéjàView's forward time go? Times encoder, one loop step, decoders and the ray -> camera fit (as the
benchmark adapter runs them: bf16 autocast, decode chunks of 8, fp32 pose fit) on real ARKitScenes frames, for
several view counts and input resolutions, plus a DINOv2 ViT-S/14 encoder for comparison. Prints a table and
writes JSON lines to --out.

  .venv/bin/python scripts/distill_steps/profile_dvlt.py --out $SCRATCH/distill_runs/profile.jsonl
  uv run scripts/distill_steps/profile_dvlt.py --images <folder with >= 100 images>   # e.g. on a local GPU
"""

import argparse
import io
import json
import os
import sys
import zipfile

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import DATA, index_zip, load_dvlt  # noqa: E402


def frames(n, image_dir=""):
    from PIL import Image

    from dvlt.util.preprocess import preprocess_images

    if image_dir:  # any folder of same-sized images (timing does not depend on content)
        files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        idx = torch.linspace(0, len(files) - 1, n).long().tolist()
        imgs = [Image.open(os.path.join(image_dir, files[i])) for i in idx]
        return preprocess_images(imgs, 504, 14, "cpu")["images"]
    zp = sorted(f for f in os.listdir(os.path.join(DATA, "arkit")) if f.endswith(".zip"))[0]
    zp = os.path.join(DATA, "arkit", zp)
    members = next(iter(index_zip(zp, lambda d: d.endswith("vga_wide")).values()))
    idx = torch.linspace(0, len(members) - 1, n).long().tolist()
    with zipfile.ZipFile(zp) as z:
        imgs = [Image.open(io.BytesIO(z.read(members[i]))) for i in idx]
    return preprocess_images(imgs, 504, 14, "cpu")["images"]  # [1, n, 3, 378, 504]


def timed(fn, reps):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        out = fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / 1000)
    return sorted(ts)[len(ts) // 2], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="24,100")
    ap.add_argument("--res", default="504,420,336,252")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="")
    ap.add_argument("--images", default="", help="folder of images instead of an ARKitScenes zip")
    args = ap.parse_args()
    dev = torch.device("cuda")

    from dvlt.common.rays import rays_to_pose
    from dvlt.model.dvlt.model import _PATCH_EMBED_BUILDERS, _slice_expand_flatten

    model = load_dvlt().to(dev).eval()
    vits = _PATCH_EMBED_BUILDERS["dinov2_vits14_reg"](img_size=518, patch_size=14, num_register_tokens=4,
                                                       interpolate_antialias=True, interpolate_offset=0.0,
                                                       block_chunks=0, init_values=1.0).to(dev).eval()
    print(f"[gpu] {torch.cuda.get_device_name(0)}", flush=True)
    rows = []
    base = frames(max(int(v) for v in args.views.split(",")), args.images)
    for S in map(int, args.views.split(",")):
        for R in map(int, args.res.split(",")):
            img = base[:, torch.linspace(0, base.shape[1] - 1, S).long()].to(dev)
            H, W = img.shape[-2:]
            s = R / max(H, W)
            h, w = round(H * s / 14) * 14, round(W * s / 14) * 14
            img = resize(img[0], [h, w], antialias=True)[None] if (h, w) != (H, W) else img
            B = 1
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                t_enc, z0 = timed(lambda: model._encode_images(img), args.reps)
                t_vits, _ = timed(lambda: vits(model._normalize_images(img).view(S, 3, h, w), is_training=True)
                                  ["x_norm_patchtokens"], args.reps)
                reg = model.register_token.expand(B, S, -1, -1).reshape(B * S, model.num_register_tokens, -1)
                cam = _slice_expand_flatten(model.camera_token, B, S)
                x = torch.cat([cam, reg, z0], dim=1)
                pos = model._get_rope_positions(B * S, h, w, dev)
                t_step, x1 = timed(lambda: model._interval_step(x, 0.0, 1 / 15, pos, B, S), args.reps)
                t_dec, pred = timed(lambda: model._decode(x1, h, w, B, S, pos), args.reps)
                with torch.autocast("cuda", enabled=False):
                    t_pose, _ = timed(lambda: rays_to_pose(pred["rays"].float(), torch.ones_like(pred["depth_conf"]),
                                                           h, w, model.patch_size), args.reps)
                model.inference_steps = 16
                t_full16, _ = timed(lambda: model.forward_inference(img), max(2, args.reps // 2))
            r = dict(views=S, res=R, H=h, W=w, tokens=S * (h // 14) * (w // 14), enc=t_enc, enc_vits=t_vits,
                     step=t_step, dec=t_dec, pose=t_pose, full_k16_no_pose=t_full16,
                     peak_gib=torch.cuda.max_memory_allocated() / 2**30)
            rows.append(r)
            print(json.dumps(r), flush=True)
            if args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps(r) + "\n")
    print("\n| views | input | encoder ViT-B | (ViT-S) | 1 step | decoders | pose fit | fixed = enc+dec+pose | K=16 est. | peak GiB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        fixed = r["enc"] + r["dec"] + r["pose"]
        print(f"| {r['views']} | {r['W']}x{r['H']} | {r['enc']:.3f} | ({r['enc_vits']:.3f}) | {r['step']:.4f} | "
              f"{r['dec']:.3f} | {r['pose']:.3f} | {fixed:.3f} | {fixed + 16 * r['step']:.3f} | {r['peak_gib']:.1f} |")


if __name__ == "__main__":
    main()
