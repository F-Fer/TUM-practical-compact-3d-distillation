"""Distil DéjàView's image encoder (DINOv2 ViT-B/14 fine-tuned, + affine-free LayerNorm; 86.6M) into DINOv2 ViT-S/14
(registers, 22M, initialised from the public DINOv2 weights) + a linear projection 384 -> 768 + affine-free
LayerNorm, by regressing the per-image patch tokens z_0 that DéjàView's loop consumes (MSE). The loop, heads and
everything else stay the released ones, so the student plugs into DéjàView (and Solv3R, which keeps DéjàView's
encoder) via eval_student.py DVLT_ENC=<encoder.pt>. Single images, no GT: frames from the zips of train.py.

  .venv/bin/python scripts/distill_steps/train_encoder.py --name E_vits --hours 2
Outputs: $SCRATCH/distill_runs/<name>/{encoder_final.pt, encoder_<step>.pt, log.jsonl, config.json}.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import RUNS, Clips, build_index, load_dvlt  # noqa: E402

VITS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class StudentEncoder(nn.Module):
    """Same interface as DéjàView's patch_embed_encoder: forward(normalised images [N, 3, H, W]) ->
    {"x_norm_patchtokens": [N, H/14 * W/14, 768]}."""

    def __init__(self, pretrained=True):
        super().__init__()
        from dvlt.model.dvlt.model import _PATCH_EMBED_BUILDERS

        self.vit = _PATCH_EMBED_BUILDERS["dinov2_vits14_reg"](img_size=518, patch_size=14, num_register_tokens=4,
                                                              interpolate_antialias=True, interpolate_offset=0.0,
                                                              block_chunks=0, init_values=1.0)
        if pretrained:
            self.vit.load_state_dict(torch.hub.load_state_dict_from_url(VITS_URL, map_location="cpu"), strict=False)
        self.vit.mask_token.requires_grad_(False)
        self.proj = nn.Linear(self.vit.embed_dim, 768)
        self.norm = nn.LayerNorm(768, elementwise_affine=False)

    def forward(self, x, is_training=True):
        t = self.vit(x, is_training=True)["x_norm_patchtokens"]
        return {"x_norm_patchtokens": self.norm(self.proj(t))}


def load_student(path):
    m = StudentEncoder(pretrained=False)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"], strict=True)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--lr", type=float, default=2e-4, help="ViT-S; the projection gets 5x")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--frames", type=int, default=8, help="frames per chunk (one scene per chunk)")
    ap.add_argument("--chunks", type=int, default=4, help="chunks (scenes) per optimiser step")
    ap.add_argument("--p_other", type=float, default=0.25)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--max_steps", type=int, default=200000)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = os.path.join(RUNS, args.name)
    os.makedirs(out, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)

    arkit, other = build_index()
    vl = os.environ.get("DISTILL_VAL_LIST")
    vids = set(open(vl).read().split()) if vl else {a[2][6:] for a in arkit[-4:]}
    val_scenes, arkit = [a for a in arkit if a[2][6:] in vids], [a for a in arkit if a[2][6:] not in vids]
    print(f"[data] {len(arkit)} ARKit train scenes, {len(val_scenes)} val, {len(other)} other", flush=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**vars(args), "val_scenes": [v[2] for v in val_scenes],
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=1)

    teacher = load_dvlt().patch_embed_encoder.to(dev).eval().requires_grad_(False)
    student = StudentEncoder().to(dev).train()
    n_s = sum(p.numel() for p in student.parameters()) / 1e6
    n_t = sum(p.numel() for p in teacher.parameters()) / 1e6
    print(f"[model] teacher encoder {n_t:.1f}M, student {n_s:.1f}M", flush=True)
    opt = torch.optim.AdamW([{"params": [p for p in student.vit.parameters() if p.requires_grad], "lr": args.lr},
                             {"params": student.proj.parameters(), "lr": 5 * args.lr}],
                            weight_decay=0.05, betas=(0.9, 0.99))
    base_lrs = [g["lr"] for g in opt.param_groups]
    mean, std = MEAN.to(dev), STD.to(dev)

    n_items = args.max_steps * args.chunks
    ds = Clips(arkit, other, n_items, args.frames, args.frames, args.p_other, args.seed)
    dl = iter(torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=args.workers,
                                          prefetch_factor=8, persistent_workers=True))
    val = [Clips([v], [], 1, 16, 16, 0.0, 1000)[0][0] for v in val_scenes]
    amp = dict(device_type="cuda", dtype=torch.bfloat16)
    log = open(os.path.join(out, "log.jsonl"), "a")
    t0 = time.time()

    def validate(step):
        student.eval()
        mse = cos = 0.0
        with torch.no_grad(), torch.autocast(**amp):
            for v in val:
                x = (v.to(dev) - mean) / std
                zt = teacher(x, is_training=True)["x_norm_patchtokens"].float()
                zs = student(x)["x_norm_patchtokens"].float()
                mse += F.mse_loss(zs, zt).item() / len(val)
                cos += F.cosine_similarity(zs, zt, dim=-1).mean().item() / len(val)
        student.train()
        rec = dict(step=step, val_mse=mse, val_cos=cos, t=time.time() - t0)
        print(f"[val] {rec}", flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    validate(0)
    total, step = args.max_steps, 0
    while True:
        if step == 50:
            per = (time.time() - t_fit) / 40
            total = min(args.max_steps, int((args.hours * 3600 - (time.time() - t0)) / per) + 50)
            print(f"[sched] {per:.3f} s/step -> {total} steps", flush=True)
        if step == 10:
            t_fit = time.time()
        frac = min(1.0, (step + 1) / args.warmup)
        cos_f = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(total, 1))))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * frac * cos_f
        tot = 0.0
        for _ in range(args.chunks):
            images, _name = next(dl)
            x = (images.to(dev, non_blocking=True) - mean) / std
            with torch.no_grad(), torch.autocast(**amp):
                zt = teacher(x, is_training=True)["x_norm_patchtokens"]
            with torch.autocast(**amp):
                zs = student(x)["x_norm_patchtokens"]
            loss = F.mse_loss(zs.float(), zt.float()) / args.chunks
            loss.backward()
            tot += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 100 == 0 or step <= 3:
            rec = dict(step=step, loss=tot, gn=gn.item(), lr=opt.param_groups[0]["lr"], t=time.time() - t0,
                       mem=torch.cuda.max_memory_allocated() / 2**30)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % args.val_every == 0:
            validate(step)
        done = step >= total or (time.time() - t0) > args.hours * 3600 + 600
        if not done and step % max(total // 4, 1) == 0:
            torch.save({"step": step, "model": student.state_dict()}, os.path.join(out, f"encoder_{step:06d}.pt"))
        if done:
            break
    validate(step)
    torch.save({"step": step, "model": student.state_dict()}, os.path.join(out, "encoder_final.pt"))
    print(f"[done] {step} steps in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
