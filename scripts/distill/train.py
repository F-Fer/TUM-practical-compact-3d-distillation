"""Distillation / fine-tuning experiments on ScanNet++ (train scenes), validated on held-out scenes.

    uv run scripts/distill/train.py --name NAME --student da3:depth-anything/DA3-SMALL --teacher dvlt:12
    uv run scripts/distill/train.py --name NAME --student da3:depth-anything/DA3-SMALL --teacher gt
    uv run scripts/distill/train.py --name NAME --student da3:depth-anything/DA3-SMALL \
        --teacher da3:depth-anything/DA3-LARGE-1.1 --w_feat 1
    uv run scripts/distill/train.py --name NAME --student dvlt:4 --teacher dvlt:12 --w_feat 1

Students: da3:<checkpoint> (fine-tune all of DA3) or dvlt:<K> (DéjàView run with K loop steps;
the DINOv2 encoder and the camera/register tokens stay frozen, so encoder tokens are shared with
a DéjàView teacher). Teachers: da3:<checkpoint>, dvlt:<K> or gt (ScanNet++ GT depth + poses).
Outputs in workspace/distill/runs/<name>/: config.json, train.jsonl, val.jsonl, student.pt.
"""

import argparse
import json
import math
import os
import random
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    TRAIN_SCENES,
    WS,
    FeatProjector,
    append_jsonl,
    bench_subsets,
    da3_cameras,
    da3_forward,
    da3_student_loss,
    dvlt_cameras,
    dvlt_forward,
    enable_da3_checkpointing,
    enable_dvlt_checkpointing,
    untie_dvlt,
    gradient_loss,
    load_da3,
    load_dvlt,
    make_targets,
    predict_fn_da3,
    predict_fn_dvlt,
    sample_views,
    to_unit,
    val_subsets,
    validate,
)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--min_views", type=int, default=8)
    p.add_argument("--max_views", type=int, default=16)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--depth_loss", default="l1", help="l1 (mean-normalized) | log (scale-invariant log)")
    p.add_argument("--lr", type=float, default=5e-6, help="backbone / recurrent block")
    p.add_argument("--lr_head", type=float, default=2e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--w_depth", type=float, default=1.0)
    p.add_argument("--w_grad", type=float, default=1.0)
    p.add_argument("--w_cam", type=float, default=1.0)
    p.add_argument("--w_ray", type=float, default=1.0)
    p.add_argument("--w_feat", type=float, default=0.0)
    p.add_argument("--feat_match", default="time", help="dvlt: 'time' (same t) or 'double' (step k <- teacher step 2k)")
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--val_every", type=int, default=250)
    p.add_argument("--val_frames", type=int, default=100, help="100: the evaluator's exact frames; else random subsets")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay of the trainable weights (used for val + saving)")
    p.add_argument("--scenes", default="train", help="train | train+extra (unlabeled 7-Scenes/ETH3D) | train+val_unlabeled (transductive)")
    return p.parse_args()


def kind(spec):
    k, _, arg = spec.partition(":")
    return k, arg


class Teacher:
    def __init__(self, spec):
        self.kind, arg = kind(spec)
        if self.kind == "da3":
            self.net = load_da3(arg).cuda().eval().requires_grad_(False)
        elif self.kind == "dvlt":
            self.K = int(arg)
            self.net = load_dvlt(self.K).cuda().eval().requires_grad_(False)
        else:
            assert self.kind == "gt"

    @torch.no_grad()
    def __call__(self, batch, x01, z0=None, need_feats=False):
        """-> dict(depth, w2c, K, valid[, feats, raw])"""
        if self.kind == "gt":
            depth = batch["gt_depth"].cuda()
            valid = (depth > 0) & (depth < 20)
            return dict(depth=depth, w2c=batch["gt_w2c"].cuda(), K=batch["gt_K"].cuda(), valid=valid)
        if self.kind == "da3":
            out = da3_forward(self.net, x01)
            w2c, K = da3_cameras(out["pose_enc"])
            return dict(depth=out["depth"].float(), w2c=w2c, K=K, valid=None, feats=out["feats"])
        out = dvlt_forward(self.net, x01, self.K, z0=z0, return_feats=need_feats)
        w2c, K = dvlt_cameras(out["rays"])
        return dict(depth=out["depth"], w2c=w2c, K=K, valid=None, raw=out, feats=out.get("feats"))


def lr_lambda(step, args):
    if step < args.warmup:
        return (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return 0.5 * (1 + math.cos(math.pi * t)) * 0.95 + 0.05


def main():
    args = parse()
    run_dir = os.path.join(WS, "runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    for f in ("train.jsonl", "val.jsonl"):
        if os.path.exists(os.path.join(run_dir, f)):
            os.remove(os.path.join(run_dir, f))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    s_kind, s_arg = kind(args.student)
    teacher = Teacher(args.teacher)
    w = dict(depth=args.w_depth, grad=args.w_grad, cam=args.w_cam, ray=args.w_ray, log_depth=args.depth_loss == "log")
    extra_params = []
    if s_kind == "da3":
        student = load_da3(s_arg).cuda().train()
        student.cam_enc.requires_grad_(False)  # only used with GT pose input
        enable_da3_checkpointing(student)
        if args.freeze_backbone:
            student.backbone.requires_grad_(False)
        groups = [
            {"params": [p for p in student.backbone.parameters() if p.requires_grad], "lr": args.lr},
            {"params": list(student.head.parameters()) + list(student.cam_dec.parameters()), "lr": args.lr_head},
        ]
        projector = None
        if args.w_feat > 0:
            c_s = student.head.projects[0].in_channels if hasattr(student.head, "projects") else None
            projector = FeatProjector(c_s, teacher.net.head.projects[0].in_channels).cuda()
            groups.append({"params": list(projector.parameters()), "lr": 1e-3})
        predict = predict_fn_da3(student)
    else:
        untied = s_arg.endswith("u")
        K_s = int(s_arg.rstrip("u"))
        student = load_dvlt(K_s)
        if untied:
            untie_dvlt(student, K_s)
        student = student.cuda()
        student.requires_grad_(False)
        loop = student.untied_blocks if untied else student.recurrent_blocks
        for m in (loop, student.ray_decoder, student.depth_decoder, student.camera_head):
            m.requires_grad_(True)
        enable_dvlt_checkpointing(student)
        student.decode_chunk_size = 4  # checkpointed decoder chunks: peak memory ~ one chunk
        student.train()
        student.patch_embed_encoder.eval()
        groups = [
            {"params": list(loop.parameters()), "lr": args.lr},
            {"params": list(student.ray_decoder.parameters()) + list(student.depth_decoder.parameters()), "lr": args.lr_head},
        ]
        predict = predict_fn_dvlt(student, K_s)
    n_train = sum(p.numel() for g in groups for p in g["params"])
    print(f"[train] {args.name}: {n_train / 1e6:.1f}M trainable params", flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=0.01, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, args))
    subsets = bench_subsets() if args.val_frames == 100 else val_subsets(args.val_frames)
    scenes = list(TRAIN_SCENES)
    if args.scenes == "train+extra":  # half the batches from unlabeled 7-Scenes / ETH3D sequences
        from common import extra_scenes

        assert args.teacher != "gt"
        scenes = [list(TRAIN_SCENES), extra_scenes()]
    if args.scenes == "train+val_unlabeled":
        from common import VAL_SCENES

        scenes += VAL_SCENES
        assert args.teacher != "gt"

    params = [p for g in groups for p in g["params"]]
    ema = [p.detach().clone() for p in params] if args.ema > 0 else None

    def swap_ema():
        if ema is not None:
            for p, e in zip(params, ema):
                tmp = p.detach().clone()
                p.data.copy_(e)
                e.copy_(tmp)

    def run_val(step):
        student.eval()
        swap_ema()
        mean, rows = validate(predict, subsets, "images_dvlt" if s_kind == "dvlt" else "images")
        swap_ema()
        student.train()
        if s_kind == "dvlt":
            student.patch_embed_encoder.eval()
        append_jsonl(os.path.join(run_dir, "val.jsonl"), dict(step=step, **mean, rows=rows))
        print(f"[val] step {step}: " + " ".join(f"{k}={v:.4f}" for k, v in mean.items()), flush=True)
        return mean

    run_val(0)
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        logs_acc = {}
        for _ in range(args.accum):
            batch = sample_views(scenes, rng.randint(args.min_views, args.max_views), rng)
            # each student family sees its own benchmark preprocessing (teacher gets the same input)
            x01 = batch["images_dvlt"].cuda().float() if s_kind == "dvlt" else to_unit(batch["images"])
            if s_kind == "da3":
                # a DéjàView teacher gets its own exact preprocessing (same frames, same 504x378 grid)
                x_t = batch["images_dvlt"].cuda().float() if teacher.kind == "dvlt" else x01
                tout = teacher(batch, x_t)
                tgt = make_targets(tout["depth"], tout["w2c"], tout["K"], tout["valid"])
                pred = da3_forward(student, x01)
                loss, logs = da3_student_loss(pred, tgt, w)
                if projector is not None:
                    l_feat = projector(pred["feats"], tout["feats"])
                    loss = loss + args.w_feat * l_feat
                    logs["feat"] = l_feat.item()
            else:
                tout = teacher(batch, x01, need_feats=args.w_feat > 0)
                z0 = tout["raw"]["z0"]
                pred = dvlt_forward(student, x01, K_s, grad=True, z0=z0, return_feats=args.w_feat > 0)
                d_t = tout["raw"]["depth"]
                scale = d_t.mean()
                l_depth = ((pred["depth"] - d_t).abs() / scale).mean()
                l_grad = gradient_loss(pred["depth"] / scale, d_t / scale, torch.ones_like(d_t, dtype=torch.bool))
                r_t, r_s = tout["raw"]["rays"], pred["rays"]
                l_ray = (r_s[..., :3] - r_t[..., :3]).abs().mean() + ((r_s[..., 3:] - r_t[..., 3:]).abs() / scale).mean()
                loss = args.w_depth * l_depth + args.w_grad * l_grad + args.w_ray * l_ray
                logs = dict(depth=l_depth.item(), grad=l_grad.item(), ray=l_ray.item())
                if args.w_feat > 0:
                    f_t = tout["feats"]
                    l_feat = 0
                    for i, (t_s, fs) in enumerate(pred["feats"]):
                        if args.feat_match == "double":
                            j = min(2 * (i + 1), len(f_t)) - 1
                        else:
                            j = min(range(len(f_t)), key=lambda j: abs(f_t[j][0] - t_s))
                        ft = f_t[j][1].float()
                        fs = fs.float()
                        l_feat = l_feat + F.smooth_l1_loss(F.layer_norm(fs, fs.shape[-1:]), F.layer_norm(ft, ft.shape[-1:]))
                    l_feat = l_feat / len(pred["feats"])
                    loss = loss + args.w_feat * l_feat
                    logs["feat"] = l_feat.item()
            (loss / args.accum).backward()
            logs["loss"] = loss.item()
            for k, v in logs.items():
                logs_acc[k] = logs_acc.get(k, 0) + v / args.accum
        gn = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            with torch.no_grad():
                torch._foreach_lerp_(ema, [p.detach() for p in params], 1 - args.ema)
        if step % 10 == 0:
            rec = dict(step=step, **logs_acc, gnorm=gn.item(), lr=sched.get_last_lr()[0],
                       mem=torch.cuda.max_memory_allocated() / 2**30, sec=time.perf_counter() - t0)
            append_jsonl(os.path.join(run_dir, "train.jsonl"), rec)
            if step % 50 == 0:
                print(f"[train] step {step} " + " ".join(f"{k}={v:.4g}" for k, v in rec.items() if k != "step"), flush=True)
        if step % args.val_every == 0 or step == args.steps:
            run_val(step)
    swap_ema()
    state = {k: v for k, v in student.state_dict().items()}
    torch.save({"student": args.student, "state_dict": state, "config": vars(args)}, os.path.join(run_dir, "student.pt"))
    print(f"[train] done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
