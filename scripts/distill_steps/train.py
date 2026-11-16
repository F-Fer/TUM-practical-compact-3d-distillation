"""Few-step DéjàView by self-distillation (idea 1 of reports/DéjàView and Solv3r distillation ideas.md).

Student: DéjàView run with K_s loop steps on its usual linspace(0, 1, K_s) grid, all weights trained (encoder at
0.1x the learning rate). Teacher: a frozen copy run with K_t = 16 steps on the same images. No ground truth: the
data are unlabeled multi-view clips outside the DA3 benchmark (ARKitScenes vga_wide, Mip-NeRF 360, Tanks and
Temples / Deep Blending), read straight from their zips and preprocessed exactly like the benchmark adapter does.

Loss: L1 on the decoded ray maps + L1 on log depth + 0.1 L1 on log depth confidence (all against the teacher).
--w_tok > 0 adds intermediate token distillation: the student state after each step is matched to the teacher
state at the same t (layer-normed tokens, MSE); with K_s = 4 and K_t = 16 the times 1/3, 2/3, 1 coincide.

  .venv/bin/python scripts/distill_steps/train.py --name A_out_K4 --ks 4 [--w_tok 0.5] [--hours 4.5]
Outputs (checkpoints, log.jsonl, config.json) go to $SCRATCH/distill_runs/<name>/.
"""

import argparse
import copy
import io
import json
import math
import os
import random
import sys
import time
import zipfile

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "third_party", "dvlt", "src"))
SCRATCH = os.environ.get("SCRATCH", os.path.join(REPO, "workspace"))  # tum2: /scratch/vc17/pr26rb00
DATA = os.environ.get("DISTILL_DATA", os.path.join(SCRATCH, "distill_data"))
RUNS = os.path.join(SCRATCH, "distill_runs")
IMG_EXT = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


# ---------------------------------------------------------------------------------------------------- data


def index_zip(path, keep_dir):
    """{scene dir: sorted image members} for the image dirs of a zip that keep_dir(dir) accepts."""
    scenes = {}
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as e:  # still downloading
        print(f"[data] skip {path}: {e}", flush=True)
        return {}
    with z:
        for n in z.namelist():
            if n.endswith(IMG_EXT):
                d = os.path.dirname(n)
                if keep_dir(d):
                    scenes.setdefault(d, []).append(n)

    def key(n):  # ARKit frames are named <video>_<timestamp>.png
        stem = os.path.splitext(os.path.basename(n))[0]
        try:
            return (0, float(stem.split("_")[-1]), n)
        except ValueError:
            return (1, 0.0, n)

    return {d: sorted(m, key=key) for d, m in scenes.items()}


def build_index():
    """Two pools of (zip, members, name): ARKitScenes and the others (outdoor / object-centric)."""
    arkit, other = [], []
    adir = os.path.join(DATA, "arkit")
    lst = os.environ.get("DISTILL_ARKIT_LIST")  # optional file of video ids, so parallel runs see the same scenes
    ids = set(open(lst).read().split()) if lst else None
    for f in sorted(os.listdir(adir)):
        if f.endswith(".zip") and (ids is None or f[:-4] in ids):
            for d, m in index_zip(os.path.join(adir, f), lambda d: d.endswith("vga_wide")).items():
                if len(m) >= 64:
                    arkit.append((os.path.join(adir, f), m, "arkit/" + f[:-4]))
    zdir = os.path.join(DATA, "zips")
    for f, keep in [("360_v2.zip", lambda d: d.endswith("images_4")), ("tandt_db.zip", lambda d: d.endswith("images"))]:
        p = os.path.join(zdir, f)
        if os.path.exists(p):
            for d, m in index_zip(p, keep).items():
                other.append((p, m, d))
    return arkit, other


class Clips(torch.utils.data.Dataset):
    """Item i: a random scene and a random sorted subset of n_views frames, preprocessed as for the benchmark."""

    def __init__(self, arkit, other, length, min_views, max_views, p_other, seed):
        self.arkit, self.other, self.length = arkit, other, length
        self.min_views, self.max_views, self.p_other, self.seed = min_views, max_views, p_other, seed
        self._zips = {}

    def __len__(self):
        return self.length

    def _zip(self, p):
        if p not in self._zips:
            self._zips[p] = zipfile.ZipFile(p)
        return self._zips[p]

    def __getitem__(self, i):
        from PIL import Image

        from dvlt.util.preprocess import preprocess_images

        rng = random.Random(self.seed * 1_000_003 + i)
        pool = self.other if (self.other and rng.random() < self.p_other) else self.arkit
        zp, members, name = pool[rng.randrange(len(pool))]
        n = min(rng.randint(self.min_views, self.max_views), len(members))
        idx = sorted(rng.sample(range(len(members)), n))
        z = self._zip(zp)
        imgs = [Image.open(io.BytesIO(z.read(members[j]))) for j in idx]
        images = preprocess_images(imgs, 504, 14, "cpu")["images"][0]  # [S, 3, H, W] in [0, 1]
        return images, name


# ---------------------------------------------------------------------------------------------------- model


def load_dvlt():
    from accelerate import PartialState
    from huggingface_hub import hf_hub_download

    from dvlt.model.dvlt.model import DVLT

    PartialState()
    with open(hf_hub_download("nvidia/dvlt", "config.json")) as f:
        cfg = json.load(f)
    cfg.update(decode_chunk_size=8, drop_path=0.0)  # no stochastic depth in the fine-tune
    w = DVLT(**cfg)
    w.load_pretrained("nvidia/dvlt", strict=True)
    return w.model


def run(model, images, K):
    """DéjàView forward with K steps on linspace(0, 1, K) (as _solve_inference_linspace_k); returns the decoded
    predictions and the token state after every step."""
    from dvlt.model.dvlt.model import _slice_expand_flatten

    B, S, _, H, W = images.shape
    z0 = model._encode_images(images)
    reg = model.register_token.expand(B, S, -1, -1).reshape(B * S, model.num_register_tokens, -1)
    cam = _slice_expand_flatten(model.camera_token, B, S)
    x = torch.cat([cam, reg, z0], dim=1)
    pos = model._get_rope_positions(B * S, H, W, images.device)
    ts = torch.linspace(0.0, 1.0, K).tolist()
    states = []
    for i in range(K):
        x = model._interval_step(x, ts[i], ts[i + 1] if i + 1 < K else 1.0, pos, B, S)
        states.append(x)
    return model._decode(x, H, W, B, S, pos), states


def matched_steps(ks, kt):
    """(student step, teacher step) pairs whose states sit at the same t; the terminal steps always match."""
    pairs = []
    for i in range(ks - 1):  # after student step i the state is at t = (i + 1) / (ks - 1)
        j = (i + 1) * (kt - 1) / (ks - 1) - 1
        if abs(j - round(j)) < 1e-6:
            pairs.append((i, int(round(j))))
    pairs.append((ks - 1, kt - 1))
    return pairs


def out_losses(ps, pt):
    f = lambda t: t.float()
    l_ray = (f(ps["rays"]) - f(pt["rays"])).abs().mean()
    l_depth = (f(ps["depth"]).clamp_min(1e-6).log() - f(pt["depth"]).clamp_min(1e-6).log()).abs().mean()
    l_conf = (f(ps["depth_conf"]).log() - f(pt["depth_conf"]).log()).abs().mean()
    return l_ray, l_depth, l_conf


def tok_loss(ss, st, pairs):
    ln = lambda x: F.layer_norm(x.float(), (x.shape[-1],))
    return sum(F.mse_loss(ln(ss[i]), ln(st[j])) for i, j in pairs) / len(pairs)


# ---------------------------------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ks", type=int, default=4)
    ap.add_argument("--kt", type=int, default=16)
    ap.add_argument("--w_tok", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--enc_lr_mult", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--min_views", type=int, default=16)
    ap.add_argument("--max_views", type=int, default=48)
    ap.add_argument("--p_other", type=float, default=0.25)
    ap.add_argument("--hours", type=float, default=4.5, help="training budget; the cosine schedule is fitted to it")
    ap.add_argument("--max_steps", type=int, default=100000)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--n_val", type=int, default=4, help="ARKit scenes held out for the validation loss")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = os.path.join(RUNS, args.name)
    os.makedirs(out, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)

    arkit, other = build_index()
    vl = os.environ.get("DISTILL_VAL_LIST")  # optional file of held-out ARKit ids (same val scenes across runs)
    if vl:
        vids = set(open(vl).read().split())
        val_scenes, arkit = [a for a in arkit if a[2][6:] in vids], [a for a in arkit if a[2][6:] not in vids]
    else:
        val_scenes, arkit = arkit[-args.n_val :], arkit[: -args.n_val]
    print(f"[data] {len(arkit)} ARKit train scenes, {len(val_scenes)} val, {len(other)} other: "
          f"{[o[2] for o in other]}", flush=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**vars(args), "train_scenes": [a[2] for a in arkit] + [o[2] for o in other],
                   "val_scenes": [v[2] for v in val_scenes], "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, f,
                  indent=1)

    student = load_dvlt().to(dev)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    student.train()
    student.enable_gradient_checkpointing()  # recurrent block + decoders
    student.patch_embed_encoder.use_checkpoint = True
    student.patch_embed_encoder.use_reentrant = False
    enc = [p for n, p in student.named_parameters() if n.startswith("patch_embed_encoder.") and p.requires_grad]
    rest = [p for n, p in student.named_parameters() if not n.startswith("patch_embed_encoder.") and p.requires_grad]
    print(f"[model] trainable: encoder {sum(p.numel() for p in enc) / 1e6:.1f}M, "
          f"rest {sum(p.numel() for p in rest) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr * args.enc_lr_mult}, {"params": rest, "lr": args.lr}],
                            weight_decay=0.0, betas=(0.9, 0.99))
    base_lrs = [g["lr"] for g in opt.param_groups]
    pairs = matched_steps(args.ks, args.kt)
    print(f"[loss] token pairs (student step, teacher step): {pairs}, w_tok {args.w_tok}", flush=True)

    ds = Clips(arkit, other, args.max_steps, args.min_views, args.max_views, args.p_other, args.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=args.workers,
                                     prefetch_factor=4 if args.workers else None, persistent_workers=args.workers > 0)
    val = [Clips([v], [], 1, 32, 32, 0.0, 1000)[0][0] for v in val_scenes]  # one fixed 32-view clip per val scene

    amp = dict(device_type="cuda", dtype=torch.bfloat16)
    total = args.max_steps
    tok_scale = None
    t0 = time.time()
    log = open(os.path.join(out, "log.jsonl"), "a")

    def validate(step):
        student.eval()
        acc = torch.zeros(4)
        with torch.no_grad(), torch.autocast(**amp):
            for v in val:
                im = v[None].to(dev)
                ps, ss = run(student, im, args.ks)
                pt, st = run(teacher, im, args.kt)
                acc += torch.tensor([*map(float, out_losses(ps, pt)), float(tok_loss(ss, st, pairs))])
        student.train()
        acc /= len(val)
        rec = dict(step=step, val_ray=acc[0].item(), val_depth=acc[1].item(), val_conf=acc[2].item(),
                   val_tok=acc[3].item(), t=time.time() - t0)
        print(f"[val] {rec}", flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    validate(0)
    step = 0
    for images, name in dl:
        images = images[None].to(dev, non_blocking=True)
        if step == 30:  # fit the schedule to the time budget
            per = (time.time() - t_fit) / 20
            total = min(args.max_steps, int((args.hours * 3600 - (time.time() - t0)) / per) + 30)
            print(f"[sched] {per:.2f} s/step -> {total} steps", flush=True)
        if step == 10:
            t_fit = time.time()
        frac = min(1.0, (step + 1) / args.warmup)
        cos = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(total, 1))))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * frac * cos

        with torch.no_grad(), torch.autocast(**amp):
            pt, st = run(teacher, images, args.kt)
        with torch.autocast(**amp):
            ps, ss = run(student, images, args.ks)
        l_ray, l_depth, l_conf = out_losses(ps, pt)
        loss_out = l_ray + l_depth + 0.1 * l_conf
        l_tok = tok_loss(ss, st, pairs) if args.w_tok > 0 else torch.zeros((), device=dev)
        if args.w_tok > 0 and tok_scale is None:  # w_tok = share of the output loss at the start
            tok_scale = (loss_out.detach() / l_tok.detach().clamp_min(1e-8)).item()
            print(f"[loss] token loss scale {tok_scale:.3g}", flush=True)
        loss = loss_out + (args.w_tok * tok_scale * l_tok if args.w_tok > 0 else 0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        step += 1

        if step % 25 == 0 or step <= 5:
            rec = dict(step=step, loss=loss.item(), ray=l_ray.item(), depth=l_depth.item(), conf=l_conf.item(),
                       tok=l_tok.item(), gn=gn.item(), lr=opt.param_groups[1]["lr"], views=images.shape[1],
                       scene=name, t=time.time() - t0,
                       mem=torch.cuda.max_memory_allocated() / 2**30)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % args.val_every == 0:
            validate(step)
        done = step >= total or (time.time() - t0) > args.hours * 3600 + 600
        if not done and step % max(total // 4, 1) == 0:
            torch.save({"step": step, "args": vars(args), "model": student.state_dict()},
                       os.path.join(out, f"ckpt_{step:06d}.pt"))
            print(f"[ckpt] step {step}", flush=True)
        if done:
            break
    validate(step)
    torch.save({"step": step, "args": vars(args), "model": student.state_dict()}, os.path.join(out, "final.pt"))
    print(f"[done] {step} steps in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
