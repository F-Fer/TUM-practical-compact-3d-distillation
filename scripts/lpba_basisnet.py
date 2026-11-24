"""Learned depth basis for PBA: a small net on frozen DA3-SMALL head features predicts, per frame, K maps such that the
GT log-depth correction log(GT / DA3) lies in their span (BA-Net / CodeSLAM-style), so PBA can optimise K coefficients
per frame instead of (or on top of) the 8-cell grid. knowledge/ideas/learned-depth-basis.md.

    # 1. features + GT per clip (frozen DA3, hooks on the main DPT branch), at the resolution of refinenet2 (H / 3.5)
    .venv/bin/python scripts/lpba_basisnet.py prep <src work dir> <data dir> split=held-out|dev [extra_spp=4]
    # 2. train
    .venv/bin/python scripts/lpba_basisnet.py train <data dir> <run dir> K=8 [steps=6000 lowpass=0 ...]
    # 3. per eval scene: the basis at full resolution, the GT-fitted correction in its span (oracle, for
    #    lpba_depth_oracle.py ofile:<out>), and the basis maps (basis.npz, for PBA with free coefficients)
    .venv/bin/python scripts/lpba_basisnet.py export <src work dir> <data dir> <run dir> <out root> [split=dev]

Clips: the benchmark's frames of each scene (the evaluator's sampling, as in model_results), and for ScanNet++ also
`extra_spp` random 40-frame subsets of the scene's iPhone frames (all have rendered GT depth). Split as run_tto_pba.py's
subset (dev = the first 10 / 2 / 3 / 3 scenes of HiRoom / 7-Scenes / ScanNet++ / ETH3D); 7-Scenes is skipped (GT
depth misaligned) and DTU too.

Loss per frame (at the working resolution, GT pixels only): r = log(GT * s) - log(DA3) with s the clip's median ratio;
ridge LS fit of r on [1, B] (each basis map scaled to unit RMS over the frame); loss = mean squared residual, plus
`lowpass` x the mean squared Gaussian-blurred (sigma = H / 16) residual, the part PBA's pose is sensitive to.
"""

import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpba_depth_oracle import fit_smooth, grid_shape, try_gt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = {"hiroom": 10, "7scenes": 2, "scannetpp": 3, "eth3d": 3}
DATASETS = ["hiroom", "scannetpp", "eth3d"]


def scenes_of(src, ds, split):
    data_dir = os.path.join(src, "model_results", ds)
    scenes = sorted(os.path.dirname(p) for p, _, _ in os.walk(data_dir) if os.path.basename(p) == "unposed")
    n = DEV[ds]
    return scenes[:n] if split == "dev" else scenes[n:]


# ----------------------------------------------------------------------------------------------------------------------
# prep


class Hooked:
    def __init__(self, model="depth-anything/DA3-SMALL"):
        from depth_anything_3.api import DepthAnything3

        self.api = DepthAnything3.from_pretrained(model).to("cuda").eval()
        sc = self.api.model.head.scratch
        self.store = {k: [] for k in ("r2", "r3", "r4")}
        for k, mod in (("r2", sc.refinenet2), ("r3", sc.refinenet3), ("r4", sc.refinenet4)):
            mod.register_forward_hook(lambda m, i, o, k=k: self.store[k].append(o.detach().to("cpu", torch.float16)))

    def __call__(self, files):
        from depth_anything_3.utils.constants import EVAL_REF_VIEW_STRATEGY

        for v in self.store.values():
            v.clear()
        pred = self.api.inference([str(f) for f in files], ref_view_strategy=EVAL_REF_VIEW_STRATEGY)
        out = {k: torch.cat(v) for k, v in self.store.items()}
        return out, np.asarray(pred.depth, np.float32), np.asarray(pred.conf, np.float32), pred


def pool_to(x, size):
    """x [N, H, W] float -> area average at size."""
    return F.interpolate(torch.as_tensor(x)[:, None].float(), size=size, mode="area")[:, 0]


def gt_pool(gt, size):
    """log GT at size: mean of log over valid pixels of each block, valid if >= 50 % of the block has GT."""
    g = torch.as_tensor(gt).float()
    v = (g > 0).float()
    lg = torch.where(g > 0, torch.log(g.clamp_min(1e-6)), torch.zeros_like(g))
    s = pool_to(lg * v, size)
    c = pool_to(v, size)
    return torch.where(c > 0.5, s / c.clamp_min(1e-6), torch.zeros_like(s)), c > 0.5


def load_rgb(files, H, W):
    import cv2

    out = []
    for f in files:
        im = cv2.imread(str(f))[:, :, ::-1]
        out.append(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA))
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 255.0


def prep_clip(net, files, out_path):
    f, depth, conf, _ = net(files)
    N, H, W = depth.shape
    gt = try_gt(files, H, W)
    if gt is None or (gt > 0).sum() < 1000:
        return None
    size = tuple(f["r2"].shape[-2:])
    lgt, valid = gt_pool(gt, size)
    logd = torch.log(pool_to(depth, size).clamp_min(1e-6))
    rgb = F.interpolate(load_rgb(files, H, W), size=size, mode="area")
    np.savez(out_path, r2=f["r2"].numpy(), r3=f["r3"].numpy(), r4=f["r4"].numpy(), logd=logd.numpy().astype(np.float16),
             conf=pool_to(conf, size).numpy().astype(np.float16), rgb=rgb.numpy().astype(np.float16),
             lgt=lgt.numpy().astype(np.float32), valid=valid.numpy(), files=np.array([str(x) for x in files]), HW=np.array([H, W]))
    return N


def cmd_prep(src, data, opts):
    split = opts.get("split", "held-out")
    extra = int(opts.get("extra_spp", 4))
    os.makedirs(data, exist_ok=True)
    os.chdir(REPO)
    net = Hooked()
    rng = random.Random(0)
    n_clips = 0
    for ds in DATASETS:
        for scene_dir in scenes_of(src, ds, split):
            scene = os.path.relpath(scene_dir, os.path.join(src, "model_results", ds)).replace("/", "_")
            meta = np.load(os.path.join(scene_dir, "unposed", "exports", "gt_meta.npz"), allow_pickle=True)
            clips = [("bench", list(meta["image_files"]))]
            if ds == "scannetpp" and split != "dev" and extra:
                d = os.path.dirname(str(meta["image_files"][0]))
                rd = d.split("/merge_dslr_iphone/")[0] + "/merge_dslr_iphone/render_depth"
                allf = sorted(f for f in glob.glob(os.path.join(d, "*.jpg"))
                              if os.path.exists(os.path.join(rd, os.path.basename(f)[:-4] + ".png")))
                for j in range(extra):
                    clips.append((f"x{j}", sorted(rng.sample(allf, min(40, len(allf))))))
            for tag, files in clips:
                out = os.path.join(data, f"{split}__{ds}__{scene}__{tag}.npz")
                if os.path.exists(out):
                    continue
                t = time.time()
                n = prep_clip(net, files, out)
                n_clips += n is not None
                print(f"[prep] {ds}/{scene}/{tag}: {n} frames, {time.time() - t:.1f}s", flush=True)
    print(f"[prep] done, {n_clips} clips")


# ----------------------------------------------------------------------------------------------------------------------
# model


def cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.GELU())


class BasisNet(nn.Module):
    """U-Net on [r2 (64) + logd - median + conf + rgb] at r2's resolution with r3 / r4 injected at their scales and two
    more downsamplings for global context; K output maps at r2's resolution."""

    def __init__(self, K, w=64, mean=False):
        super().__init__()
        self.K, self.mean = K, mean
        self.inp = nn.Sequential(cbr(64 + 5, w), cbr(w, w))
        self.d1 = cbr(w, w, 2)
        self.m3 = cbr(w + 64, w)
        self.d2 = cbr(w, w, 2)
        self.m4 = cbr(w + 64, w)
        self.d3 = cbr(w, w, 2)
        self.d4 = cbr(w, w, 2)
        self.bott = nn.Sequential(cbr(w, w), cbr(w, w))
        self.u4 = cbr(2 * w, w)
        self.u3 = cbr(2 * w, w)
        self.u2 = cbr(2 * w, w)
        self.u1 = cbr(2 * w, w)
        self.out = nn.Conv2d(w, K + int(mean), 3, 1, 1)

    def forward(self, r2, r3, r4, logd, conf, rgb):
        ld = logd - logd.flatten(1).median(1).values[:, None, None]
        x0 = self.inp(torch.cat([r2, ld[:, None], torch.log(conf.clamp_min(1e-3))[:, None] / 3, rgb], 1))
        x1 = self.m3(torch.cat([self.d1(x0), F.interpolate(r3, size=self._sz(x0, 2), mode="bilinear", align_corners=False)], 1))
        x2 = self.m4(torch.cat([self.d2(x1), F.interpolate(r4, size=self._sz(x1, 2), mode="bilinear", align_corners=False)], 1))
        x3 = self.d3(x2)
        x4 = self.bott(self.d4(x3))
        up = lambda a, b: F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)  # noqa: E731
        y = self.u4(torch.cat([up(x4, x3), x3], 1))
        y = self.u3(torch.cat([up(y, x2), x2], 1))
        y = self.u2(torch.cat([up(y, x1), x1], 1))
        y = self.u1(torch.cat([up(y, x0), x0], 1))
        o = self.out(y)
        B, m = o[:, : self.K], (0.1 * o[:, self.K] if self.mean else torch.zeros_like(o[:, 0]))
        return B / B.pow(2).mean((2, 3), keepdim=True).add(1e-6).sqrt(), m  # unit RMS per map and frame; mean correction

    @staticmethod
    def _sz(x, f):
        return ((x.shape[-2] + f - 1) // f, (x.shape[-1] + f - 1) // f)


def span_fit(r, m, B, ridge=1e-3):
    """Batched ridge LS of r [n, h, w] on [1, B] (B [n, K, h, w]) over mask m; returns the fit [n, h, w]."""
    n, K = B.shape[:2]
    A = torch.cat([torch.ones_like(B[:, :1]), B], 1).flatten(2)  # [n, K+1, P]
    mw = m.flatten(1).float()
    cnt = mw.sum(1).clamp_min(1)
    AtA = torch.einsum("nkp,nlp->nkl", A * mw[:, None], A) / cnt[:, None, None]
    Atr = torch.einsum("nkp,np->nk", A * mw[:, None], r.flatten(1)) / cnt[:, None]
    reg = ridge * torch.eye(K + 1, device=B.device)
    reg[0, 0] = 1e-6  # (frames without GT pixels)
    x = torch.linalg.solve(AtA + reg, Atr)
    return torch.einsum("nk,nkp->np", x, A).view_as(r), x


def blur(x, sigma):
    r = int(3 * sigma)
    k = torch.exp(-((torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2) / 2)
    k = k / k.sum()
    x = F.conv2d(F.pad(x[:, None], (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return x[:, 0]


def masked_lowpass(res, m, sigma):
    num = blur(res * m, sigma)
    den = blur(m.float(), sigma)
    return num / den.clamp_min(1e-3), den > 0.2


# ----------------------------------------------------------------------------------------------------------------------
# train


def fold_of(name):
    """Cross-validation half (0 / 1) of a held-out clip: alternating over the sorted held-out scenes of its dataset."""
    _, ds, scene, _ = name.split("__")
    return FOLDS.get((ds, scene), -1)


FOLDS = {}


def set_folds(data):
    for ds in DATASETS:
        sc = sorted({os.path.basename(p).split("__")[2] for p in glob.glob(os.path.join(data, f"held-out__{ds}__*.npz"))})
        FOLDS.update({(ds, x): i % 2 for i, x in enumerate(sc)})


def select(data, fold, part):
    """Clip files of the training (part = train) or validation set of a fold: ho = train on held-out, validate on dev;
    cv0 / cv1 = train on dev + the other half of held-out, validate on held-out half 0 / 1."""
    set_folds(data)
    files = sorted(glob.glob(os.path.join(data, "*.npz")))
    out = []
    for p in files:
        name = os.path.basename(p)[:-4]
        split = name.split("__")[0]
        if fold == "ho":
            keep = (split == "held-out") if part == "train" else (split == "dev")
        else:
            h = int(fold[2:])
            is_val = split == "held-out" and fold_of(name) == h
            keep = (not is_val) if part == "train" else is_val
            if part == "val" and not name.endswith("__bench"):
                keep = False
        if keep:
            out.append(p)
    return out


def load_clips(files):
    clips = []
    for p in files:
        z = np.load(p)
        c = {k: torch.from_numpy(z[k]) for k in ("r2", "r3", "r4", "logd", "conf", "rgb", "lgt", "valid")}
        c["name"] = os.path.basename(p)[:-4]
        c["ds"] = c["name"].split("__")[1]
        clips.append(c)
    return clips


def clip_batch(c, idx, dev):
    b = {k: c[k][idx].to(dev).float() for k in ("r2", "r3", "r4", "logd", "conf", "rgb", "lgt")}
    v = c["valid"][idx].to(dev)
    s = torch.median((c["lgt"][c["valid"]] - c["logd"].float()[c["valid"]]))  # clip scale (log)
    r = torch.where(v, b["lgt"] - s.to(dev) - b["logd"], torch.zeros_like(b["lgt"]))
    return b, r, v


def frame_loss(net, b, r, v, cfg):
    B, mc = net(b["r2"], b["r3"], b["r4"], b["logd"], b["conf"], b["rgb"])
    r = r - mc  # what is left after the predicted mean correction (0 without it)
    fit, _ = span_fit(r, v, B, cfg["ridge"])
    res = (r - fit) * v
    cnt = v.flatten(1).float().sum(1).clamp_min(1)
    l2 = (res.pow(2).flatten(1).sum(1) / cnt)
    loss = l2.mean()
    if net.mean and cfg["mean_w"] > 0:  # the corrected depth alone (up to a per-frame scale) should be close to GT
        c = (r * v).flatten(1).sum(1) / cnt
        loss = loss + cfg["mean_w"] * ((((r - c[:, None, None]) * v).pow(2).flatten(1).sum(1)) / cnt).mean()
    lp = torch.zeros(())
    if cfg["lowpass"] > 0:  # the low-frequency residual (what PBA's pose is sensitive to), at H / 16 and H / 8
        lp = 0.0
        for div in (16, 8):
            low, lv = masked_lowpass(res, v, r.shape[-2] / div)
            lp = lp + 0.5 * ((low.pow(2) * lv).flatten(1).sum(1) / lv.flatten(1).float().sum(1).clamp_min(1)).mean()
        loss = loss + cfg["lowpass"] * lp
    return loss, l2.sqrt().mean(), lp


def ref_fit(r, v, kind):
    """RMS left after a smooth:G grid or scale-only fit (the references), per frame mean."""
    out = []
    for i in range(r.shape[0]):
        m = v[i].cpu().numpy()
        if m.sum() < 50:
            continue
        ri = r[i].cpu().numpy()
        f = np.full_like(ri, np.mean(ri[m])) if kind == 0 else fit_smooth(ri, m, kind)
        out.append(np.sqrt(np.mean((ri[m] - f[m]) ** 2)))
    return out


def evaluate(net, clips, cfg, dev, refs=False):
    net.eval()
    res, ref = {}, {}
    with torch.no_grad():
        for c in clips:
            N = c["r2"].shape[0]
            errs, merrs, lerrs = [], [], []
            for s in range(0, N, 16):
                idx = torch.arange(s, min(N, s + 16))
                b, r, v = clip_batch(c, idx, dev)
                B, mc = net(b["r2"], b["r3"], b["r4"], b["logd"], b["conf"], b["rgb"])
                fit, _ = span_fit(r - mc, v, B, cfg["ridge"])
                cnt = v.flatten(1).float().sum(1)
                ok = cnt > 50
                e = (((r - mc - fit) * v).pow(2).flatten(1).sum(1) / cnt.clamp_min(1)).sqrt()
                errs += e[ok].tolist()
                low, lv = masked_lowpass((r - mc - fit) * v, v, r.shape[-2] / 16)
                el = ((low.pow(2) * lv).flatten(1).sum(1) / lv.flatten(1).float().sum(1).clamp_min(1)).sqrt()
                lerrs += el[ok].tolist()
                if net.mean:  # the corrected depth alone, per-frame scale only
                    rr = r - mc
                    cm = (rr * v).flatten(1).sum(1) / cnt.clamp_min(1)
                    e0 = ((((rr - cm[:, None, None]) * v).pow(2).flatten(1).sum(1)) / cnt.clamp_min(1)).sqrt()
                    merrs.extend(e0[ok].tolist())
                if refs:
                    for g in (0, 4, 8, 16):
                        ref.setdefault(c["ds"], {}).setdefault(g, []).extend(ref_fit(r, v, g))
            res.setdefault(c["ds"], []).append(float(np.mean(errs)) if errs else float("nan"))
            res.setdefault(c["ds"] + "_low16", []).append(float(np.mean(lerrs)) if lerrs else float("nan"))
            if merrs:
                res.setdefault(c["ds"] + "_meanonly", []).append(float(np.mean(merrs)))
    net.train()
    out = {ds: float(np.nanmean(v)) for ds, v in res.items()}
    if refs:
        return out, {ds: {f"smooth{g}" if g else "none": float(np.mean(x)) for g, x in d.items()} for ds, d in ref.items()}
    return out


def cmd_train(data, run, opts):
    cfg = {"K": 8, "steps": 6000, "lr": 1e-3, "frames": 8, "ridge": 1e-3, "lowpass": 0.0, "width": 64, "seed": 0,
           "eval_every": 1000, "mean": 0, "mean_w": 1.0, "fold": "ho"}
    for k, v in opts.items():
        cfg[k] = type(cfg[k])(v) if k in cfg else v
    os.makedirs(run, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    dev = "cuda"
    train = load_clips(select(data, cfg["fold"], "train"))
    val = load_clips(select(data, cfg["fold"], "val"))
    print(f"[train] {len(train)} train clips ({sum(c['r2'].shape[0] for c in train)} frames), {len(val)} val clips", flush=True)
    net = BasisNet(cfg["K"], cfg["width"], bool(cfg["mean"])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["steps"], pct_start=0.05)
    # sample clips with equal weight per dataset, then per scene
    by_ds = {}
    for c in train:
        by_ds.setdefault(c["ds"], []).append(c)
    ref_path = os.path.join(data, f"refs_{cfg['fold']}.json")  # the grid references don't depend on the net
    if os.path.exists(ref_path):
        refs = json.load(open(ref_path))
    else:
        _, refs = evaluate(net, val, cfg, dev, refs=True)
        json.dump(refs, open(ref_path, "w"))
    print(f"[train] refs (dev, RMS log err left): {json.dumps(refs)}", flush=True)
    log = {"cfg": cfg, "refs": refs, "val": []}
    t0 = time.time()
    run_l, run_e = 0.0, 0.0
    for step in range(1, cfg["steps"] + 1):
        ds = random.choice(list(by_ds))
        c = random.choice(by_ds[ds])
        N = c["r2"].shape[0]
        idx = torch.tensor(sorted(random.sample(range(N), min(cfg["frames"], N))))
        b, r, v = clip_batch(c, idx, dev)
        if random.random() < 0.5:  # horizontal flip
            b = {k: x.flip(-1) for k, x in b.items()}
            r, v = r.flip(-1), v.flip(-1)
        loss, e, _ = frame_loss(net, b, r, v, cfg)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run_l += float(loss)
        run_e += float(e)
        if step % 100 == 0:
            print(f"[train] step {step} loss {run_l / 100:.5f} rms {run_e / 100:.4f} ({time.time() - t0:.0f}s)", flush=True)
            run_l = run_e = 0.0
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            vv = evaluate(net, val, cfg, dev)
            tr = evaluate(net, train[::7], cfg, dev)
            print(f"[train] step {step} val {json.dumps(vv)} train-sub {json.dumps(tr)}", flush=True)
            log["val"].append({"step": step, "val": vv, "train": tr})
            torch.save({"cfg": cfg, "state": net.state_dict()}, os.path.join(run, "net.pt"))
            json.dump(log, open(os.path.join(run, "log.json"), "w"), indent=1)


# ----------------------------------------------------------------------------------------------------------------------
# export


def cmd_export(src, data, run, out_root, opts):
    ck = torch.load(os.path.join(run, "net.pt"), map_location="cuda")
    cfg = ck["cfg"]
    fold = cfg.get("fold", "ho")
    split = "dev" if fold == "ho" else "held-out"
    set_folds(data)
    net = BasisNet(cfg["K"], cfg["width"], bool(cfg.get("mean", 0))).cuda().eval()
    net.load_state_dict(ck["state"])
    os.chdir(REPO)
    summary = {}
    for ds in DATASETS:
        for scene_dir in scenes_of(src, ds, split):
            scene_rel = os.path.relpath(scene_dir, os.path.join(src, "model_results", ds))
            scene = scene_rel.replace("/", "_")
            p = os.path.join(data, f"{split}__{ds}__{scene}__bench.npz")
            if not os.path.exists(p) or (fold != "ho" and FOLDS.get((ds, scene)) != int(fold[2:])):
                continue
            z = np.load(p)
            exports = os.path.join(scene_dir, "unposed", "exports")
            meta = dict(np.load(os.path.join(exports, "gt_meta.npz"), allow_pickle=True))
            da3 = np.load(os.path.join(exports, "mini_npz", "results.npz"))["depth"].astype(np.float32)
            N, H, W = da3.shape
            gt = try_gt(meta["image_files"], H, W)
            with torch.no_grad():
                Bs, Ms = [], []
                for s in range(0, N, 16):
                    sl = slice(s, min(N, s + 16))
                    t = {k: torch.from_numpy(z[k][sl]).cuda().float() for k in ("r2", "r3", "r4", "logd", "conf", "rgb")}
                    B_, m_ = net(t["r2"], t["r3"], t["r4"], t["logd"], t["conf"], t["rgb"])
                    Bs.append(B_)
                    Ms.append(m_)
                Blow = torch.cat(Bs)  # [N, K, h, w]
                Bfull = (F.interpolate(Blow, size=(H, W), mode="bilinear", align_corners=True) if Blow.shape[1]
                         else Blow.new_zeros(N, 0, H, W))  # as run_tto_pba.py's lookup
                mfull = F.interpolate(torch.cat(Ms)[:, None], size=(H, W), mode="bilinear", align_corners=True)[:, 0].cpu().numpy()
            dpred = (da3 * np.exp(mfull)).astype(np.float32)  # the predicted mean correction (= DA3 without it)
            valid = gt > 0
            s = np.median(da3[valid]) / np.median(gt[valid])
            gts = gt * s
            out = dpred.copy()
            errs, errs8, errs0 = [], [], []
            for i in range(N):
                m = valid[i]
                if m.sum() <= 100:
                    continue
                r = np.where(m, np.log(np.maximum(gts[i], 1e-6)) - np.log(np.maximum(dpred[i], 1e-6)), 0)
                errs0.append(np.sqrt(np.mean((r[m] - np.mean(r[m])) ** 2)))
                rt, mt = torch.from_numpy(r).cuda()[None].float(), torch.from_numpy(m).cuda()[None]
                fit, _ = span_fit(rt, mt, Bfull[i:i + 1], cfg["ridge"])
                fit = fit[0].cpu().numpy()
                fit = np.clip(fit, fit[m].min(), fit[m].max())
                out[i] = dpred[i] * np.exp(fit)
                errs.append(np.sqrt(np.mean((r[m] - fit[m]) ** 2)))
                f8 = fit_smooth(r, m, 8)
                errs8.append(np.sqrt(np.mean((r[m] - f8[m]) ** 2)))
            for kind, dep in (("oracle", out), ("pred", dpred)):  # oracle: GT-fitted in the span; pred: no GT
                d = os.path.join(out_root, kind, "model_results", ds, scene_rel, "unposed", "exports")
                os.makedirs(os.path.join(d, "mini_npz"), exist_ok=True)
                np.savez(os.path.join(d, "gt_meta.npz"), **meta)
                np.savez(os.path.join(d, "mini_npz", "results.npz"), depth=dep.astype(np.float32))
                if kind == "pred" and Blow.shape[1]:
                    np.savez(os.path.join(d, "basis.npz"), basis=Blow.cpu().numpy().astype(np.float16))
            summary[f"{ds}/{scene_rel}"] = {"basis": float(np.mean(errs)), "smooth8_on_pred": float(np.mean(errs8)), "pred_scale_only": float(np.mean(errs0))}
            print(f"[export] {ds}/{scene_rel}: RMS left basis {np.mean(errs):.4f} smooth8 {np.mean(errs8):.4f} pred alone {np.mean(errs0):.4f}", flush=True)
    json.dump(summary, open(os.path.join(out_root, "export_summary.json"), "w"), indent=1)


def main():
    cmd = sys.argv[1]
    pos = [a for a in sys.argv[2:] if "=" not in a]
    opts = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    if cmd == "prep":
        cmd_prep(pos[0], pos[1], opts)
    elif cmd == "train":
        cmd_train(pos[0], pos[1], opts)
    elif cmd == "export":
        cmd_export(pos[0], pos[1], pos[2], pos[3], opts)


if __name__ == "__main__":
    main()
