"""Fine-tune DA3-SMALL's depth head for PBA with a pose-sensitivity loss: penalise the pose shift that the predicted
depth's error would cause in PBA's optimum (first order, at the GT), instead of (or on top of) a depth loss.
knowledge/ideas/learned-depth-basis.md.

    # 1. per training clip and frame subset: the sensitivity operator A at the GT (GT poses, focal and depth)
    .venv/bin/python scripts/lpba_ft_head.py sens <src work dir> <data dir> [split=held-out] [frames=16] [subsets=...]
    # 2. fine-tune the main depth branch (refinenet1-4, output_conv1/2; cameras come from the camera decoder, unchanged)
    .venv/bin/python scripts/lpba_ft_head.py train <data dir> <run dir> [w_sens=1 w_depth=0.1 steps=3000 lr=1e-4]
    # 3. predictions of the fine-tuned model in the evaluator's layout (cameras as before), for run_tto_pba.py
    .venv/bin/python scripts/lpba_ft_head.py export <src work dir> <run dir> <out work dir> [split=dev]

Sensitivity: with the depth fixed, PBA's Gauss-Newton optimum moves by dp = -Hpp^-1 Hgp^T c when the log depth of the
frames carries the error sum_n c_in hat_n (hat functions of a 16-cell grid, run_tto_pba.py depth_basis machinery);
A = -Hpp^-1 Hgp^T [np, F nc] per clip subset. In training, the error of the network's log depth (minus GT, one scene
scale) is projected on the hat functions per frame (ridge LS on GT pixels at H/4) and L_sens = 1e3 (|d rot|^2 +
|d trans|^2) / (F - 1) + 1e3 |d log f|^2 (rad, median depths). L_depth = per-frame scale-invariant squared log error.
scripts/lpba_sens.py checks the first-order model against the depth oracles.
"""

import glob
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_tto_pba as rp  # noqa: E402
from lpba_depth_oracle import grid_shape, try_gt  # noqa: E402

DEV = {"hiroom": 10, "scannetpp": 3, "eth3d": 3}
TRAIN_PREFIX = ("refinenet1.", "refinenet2.", "refinenet3.", "refinenet4.", "output_conv1.", "output_conv2.")
G = 16  # hat grid of the sensitivity operator


def scenes_of(src, ds, split):
    d = os.path.join(src, "model_results", ds)
    sc = sorted(os.path.dirname(p) for p, _, _ in os.walk(d) if os.path.basename(p) == "unposed")
    return sc[: DEV[ds]] if split == "dev" else sc[DEV[ds]:]


def load_api():
    from depth_anything_3.api import DepthAnything3

    return DepthAnything3.from_pretrained("depth-anything/DA3-SMALL").to("cuda").eval()


def forward(api, files, grad=False):
    imgs_cpu, _, _ = api._preprocess_inputs([str(f) for f in files], None, None, 504, "upper_bound_resize")
    imgs, _, _ = api._prepare_model_inputs(imgs_cpu, None, None)
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
        out = api.model(imgs, None, None, [], False, False, "first")
    return out


def hats(H, W, size):
    gh, gw = grid_shape(G, H, W)
    eye = torch.eye(gh * gw).reshape(gh * gw, 1, gh, gw)
    return F.interpolate(eye, size=size, mode="bilinear", align_corners=True)[:, 0]


def gt_pool(gt, size):
    g = torch.as_tensor(gt).float()
    v = (g > 0).float()
    lg = torch.where(g > 0, torch.log(g.clamp_min(1e-6)), torch.zeros_like(g))
    s = F.interpolate((lg * v)[:, None], size=size, mode="area")[:, 0]
    c = F.interpolate(v[:, None], size=size, mode="area")[:, 0]
    return torch.where(c > 0.5, s / c.clamp_min(1e-6), torch.zeros_like(s)), c > 0.5


# ----------------------------------------------------------------------------------------------------------------------


def sens_operator(files, g, ext_gt, fxy, depth_da3, conf, pred_K, samples=1024):
    """Hpp [np, np] and Hgp^T [np, F nc] at the GT state of these frames (np = 6 (F - 1) + 1: poses of frames 1..,
    log focal). g: GT depth [F, H, W] (0 = none), ext_gt: GT world-to-camera [F, 3, 4] (metric, as g), fxy [F, 2] at W."""
    dev = torch.device("cuda")
    N, H, W = depth_da3.shape
    valid = g > 0
    s = np.median(depth_da3[valid]) / np.median(g[valid])
    depth = g * s
    for i in range(N):
        v = valid[i]
        k = np.median(depth[i][v] / depth_da3[i][v]) if v.sum() > 100 else 1.0
        depth[i][~v] = depth_da3[i][~v] * k
    ext = ext_gt[:, :3].astype(np.float32).copy()
    ext[:, :, 3] *= s  # GT cameras in the scale of the (scaled) GT depth
    Kp = pred_K.astype(np.float32).copy()
    Kp[:, 0, 0], Kp[:, 1, 1] = fxy[:, 0], fxy[:, 1]
    confm = np.where(valid, conf, conf.min() - 1).astype(np.float32)
    cfg = rp.parse_overrides(["radial=false", "focal_init=pred"])
    hb, wb = 48, 64
    basis = hats(H, W, (hb, wb))[None].expand(N, -1, -1, -1).contiguous().to(dev)
    images = rp.load_images(files, H, W).to(dev)
    m = rp.PBA(images, torch.from_numpy(depth.astype(np.float32)).to(dev), torch.from_numpy(confm).to(dev),
               torch.from_numpy(ext).to(dev), torch.from_numpy(Kp).to(dev), cfg, None, basis).to(dev)
    with torch.no_grad():
        R, t = m.poses()
        nbrs = rp.select_neighbors(m.depth0, R, t, m.intrinsics(), min(cfg["neighbors"], N - 1), cfg["occ_thresh"], cfg["nbr_mode"], cfg["min_covis"])
        kern = rp.Kernel(m)
        grey = images.mean(1, keepdim=True)
        kern.set_level(grey, True, 1.0)
        gen = torch.Generator(device=dev).manual_seed(0)
        pix = torch.multinomial(rp.sampling_probs(grey, m.valid), samples, replacement=True, generator=gen)
        feat, quad = kern.make_pool(pix)
        Hpp, Hgp, *_ = kern.normal_eq(feat, quad, nbrs, 1)
        nc = Hgp.shape[1]
        Bt = Hgp.reshape(N * nc, -1).T  # [np, N nc]
    return Hpp.float().cpu().numpy(), Bt.float().cpu().numpy(), valid


def arkit_clips(root):
    """(name, files, GT depth loader, w2c [F, 3, 4], K [F, 4] at 640 px) of the ARKit windows of lpba_prep_arkit_pose.py."""
    for d in sorted(glob.glob(os.path.join(root, "*_*"))):
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "meta.npz")):
            continue
        m = np.load(os.path.join(d, "meta.npz"))
        E = []
        for p_ in m["traj"]:
            R = cv2.Rodrigues(p_[:3].astype(np.float64))[0]
            E.append(np.concatenate([R, p_[3:, None]], 1))
        files = [os.path.join(d, f"{t}.jpg") for t in m["ts"]]
        deps = [os.path.join(d, f"{t}_depth.png") for t in m["ts"]]
        yield os.path.basename(d), files, deps, np.array(E, np.float32), m["K"].astype(np.float32)


def cmd_sens_arkit(root, data, opts):
    os.makedirs(data, exist_ok=True)
    os.chdir(rp.REPO_ROOT)
    api = load_api()
    for name, files, deps, E, Kc in arkit_clips(root):
        out = os.path.join(data, f"arkit__arkit__{name}__0.npz")
        if os.path.exists(out):
            continue
        t0 = time.time()
        o = forward(api, files)
        d = o["depth"][0].float().cpu().numpy()
        N, H, W = d.shape
        g = np.stack([cv2.resize(cv2.imread(p_, -1).astype(np.float32) / 1000.0, (W, H), interpolation=cv2.INTER_NEAREST) for p_ in deps])
        if (g > 0).mean() < 0.2:
            continue
        fxy = Kc[:, :2] * W / 640.0
        try:
            Hpp, Bt, _ = sens_operator(files, g, E, fxy, d, o["depth_conf"][0].float().cpu().numpy(), o["intrinsics"][0].float().cpu().numpy())
        except Exception as e:  # noqa: BLE001
            print(f"[sens] arkit/{name}: failed {e!r}", flush=True)
            continue
        lgt, vlow = gt_pool(g, (H // 4, W // 4))
        np.savez(out, Hpp=Hpp, Bt=Bt, files=np.array(files), lgt=lgt.numpy(), valid=vlow.numpy(), HW=np.array([H, W]))
        print(f"[sens] arkit/{name}: {N} frames, {time.time() - t0:.1f}s", flush=True)
        torch.cuda.empty_cache()


def cmd_sens(src, data, opts):
    split = opts.get("split", "held-out")
    if split == "arkit":
        return cmd_sens_arkit(src, data, opts)
    Fr = int(opts.get("frames", 16))
    os.makedirs(data, exist_ok=True)
    os.chdir(rp.REPO_ROOT)
    api = load_api()
    rng = random.Random(0)
    for ds in DEV:
        for sd in scenes_of(src, ds, split):
            scene = os.path.relpath(sd, os.path.join(src, "model_results", ds))
            meta = dict(np.load(os.path.join(sd, "unposed", "exports", "gt_meta.npz"), allow_pickle=True))
            files = list(meta["image_files"])
            n = len(files)
            n_sub = 1 if n <= Fr else min(6, max(2, n // Fr))
            for j in range(n_sub):
                out = os.path.join(data, f"{split}__{ds}__{scene.replace('/', '_')}__{j}.npz")
                if os.path.exists(out):
                    continue
                idx = np.arange(n) if n <= Fr else np.array(sorted(rng.sample(range(n), Fr)))
                fs = [files[i] for i in idx]
                t0 = time.time()
                o = forward(api, fs)
                d = o["depth"][0].float().cpu().numpy()
                conf = o["depth_conf"][0].float().cpu().numpy()
                Kp = o["intrinsics"][0].float().cpu().numpy()
                try:
                    N_, H_, W_ = d.shape
                    w0 = cv2.imread(str(fs[0])).shape[1]
                    fxy = meta["intrinsics"][idx][:, [0, 1], [0, 1]] * W_ / w0
                    Hpp, Bt, valid = sens_operator(fs, try_gt(fs, H_, W_), meta["extrinsics"][idx], fxy, d, conf, Kp)
                except Exception as e:  # noqa: BLE001
                    print(f"[sens] {ds}/{scene}/{j}: failed {e!r}", flush=True)
                    continue
                N, H, W = d.shape
                size = (H // 4, W // 4)
                g = try_gt(fs, H, W)
                lgt, vlow = gt_pool(g, size)
                np.savez(out, Hpp=Hpp, Bt=Bt, files=np.array([str(f) for f in fs]), lgt=lgt.numpy(), valid=vlow.numpy(), HW=np.array([H, W]))
                print(f"[sens] {ds}/{scene}/{j}: {N} frames, Bt {Bt.shape}, {time.time() - t0:.1f}s", flush=True)
                torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------------------------------


def head_params(api):
    sc = api.model.head.scratch
    ps = []
    for name, p in sc.named_parameters():
        p.requires_grad_(name.startswith(TRAIN_PREFIX))
        if p.requires_grad:
            ps.append(p)
    for p in api.model.parameters():
        if not any(p is q for q in ps):
            p.requires_grad_(False)
    return ps


def losses(api, item, cfg, dev, grad=True):
    out = forward(api, item["files"], grad=grad)
    d = out["depth"][0].float()
    N, H, W = d.shape
    size = tuple(item["lgt"].shape[-2:])
    ld = torch.log(F.interpolate(d[:, None], size=size, mode="area")[:, 0].clamp_min(1e-6))
    lgt, v = item["lgt"].to(dev), item["valid"].to(dev)
    vf = v.float()
    s = ((ld - lgt) * vf).sum() / vf.sum().clamp_min(1)
    e = (ld - lgt - s) * vf  # error in log depth, one scene scale
    cnt = vf.flatten(1).sum(1).clamp_min(1)
    cm = e.flatten(1).sum(1) / cnt
    l_depth = ((((e - cm[:, None, None]) * vf) ** 2).flatten(1).sum(1) / cnt).mean()
    # projection of e on the hat functions per frame (ridge LS on GT pixels)
    Hb = item["hats"].to(dev)  # [nc, h, w]
    nc = Hb.shape[0]
    Hf = Hb.flatten(1)  # [nc, P]
    M = torch.einsum("kp,np,lp->nkl", Hf, vf.flatten(1), Hf) / cnt[:, None, None]
    b = torch.einsum("kp,np->nk", Hf, e.flatten(1)) / cnt[:, None]
    c = torch.linalg.solve(M + 1e-3 * torch.eye(nc, device=dev), b)  # [N, nc]
    dp = item["A"].to(dev) @ c.reshape(-1)  # [np]
    n = 6 * (N - 1)
    q = dp[:n].view(N - 1, 6)
    l_sens = 1e3 * ((q ** 2).sum() / max(N - 1, 1) + dp[n] ** 2)
    rot_deg = float(q[:, :3].detach().norm(dim=1).mean() * 180 / np.pi)
    if cfg.get("w_photo", 0) > 0 and "meta" in item:
        l_sens = l_sens * 0 + l_sens.detach()  # (reported, not trained, with the photometric loss)
        item["_photo"] = photo_loss(item, d, dev)
    return l_sens, l_depth, rot_deg


def operator(Hpp, Bt, lam):
    """A = -(Hpp + lam diag(Hpp))^-1 Bt: the (LM-damped) first-order pose shift per unit of hat-function error."""
    H = torch.from_numpy(Hpp).double()
    H = H + torch.diag(lam * H.diagonal() + 1e-9 * H.diagonal().mean())
    return (-torch.linalg.solve(H, torch.from_numpy(Bt).double())).float()


def photo_loss(item, d, dev, nbr=(-6, -3, 3, 6), res=2):
    """Self-supervised photometric loss with the GT poses (ARKit windows): every frame's depth (one clip scale from the
    LiDAR median) warps its neighbours' images into it; per pixel the minimum L1 over the neighbours (occlusion, as
    Monodepth2), on a 1/res grid."""
    meta = item["meta"]
    N, H, W = d.shape
    h, w = H // res, W // res
    if "imgs" not in item:
        item["imgs"] = torch.from_numpy(np.stack([cv2.resize(cv2.imread(f)[:, :, ::-1], (W, H), interpolation=cv2.INTER_AREA)
                                                  for f in item["files"]])).permute(0, 3, 1, 2).float() / 255.0
    imgs = item["imgs"].to(dev)
    lg = item["lgt"].to(dev)
    v = item["valid"].to(dev)
    ld = torch.log(F.interpolate(d[:, None], size=lg.shape[-2:], mode="area")[:, 0].clamp_min(1e-6))
    s = torch.exp(((lg - ld) * v).sum() / v.sum().clamp_min(1)).detach()  # metric scale of the clip
    D = F.interpolate(d[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0] * s
    E = torch.from_numpy(meta["E"]).to(dev)  # [N, 3, 4] world-to-camera
    K = torch.from_numpy(meta["K"]).to(dev) * (W / 640.0) / res  # fx fy cx cy at h x w
    ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32) + 0.5, torch.arange(w, device=dev, dtype=torch.float32) + 0.5, indexing="ij")
    src = F.interpolate(imgs, size=(h, w), mode="area")
    errs = []
    R4 = torch.eye(4, device=dev).repeat(N, 1, 1)
    R4[:, :3] = E
    for o in nbr:
        i = torch.arange(N, device=dev)
        j = (i + o).clamp(0, N - 1)
        ok_f = (i + o >= 0) & (i + o < N)
        T = R4[j] @ torch.linalg.inv(R4[i])  # i -> j
        X = torch.stack([(xs - K[i, 2, None, None]) / K[i, 0, None, None] * D, (ys - K[i, 3, None, None]) / K[i, 1, None, None] * D, D], 1)  # [N, 3, h, w]
        Xj = torch.einsum("nab,nbhw->nahw", T[:, :3, :3], X) + T[:, :3, 3, None, None]
        z = Xj[:, 2].clamp_min(1e-3)
        u = K[j, 0, None, None] * Xj[:, 0] / z + K[j, 2, None, None]
        vv = K[j, 1, None, None] * Xj[:, 1] / z + K[j, 3, None, None]
        grid = torch.stack([u / w * 2 - 1, vv / h * 2 - 1], -1)
        inside0 = (grid.abs() < 1).all(-1) & (Xj[:, 2] > 1e-2)
        grid = torch.where(inside0[..., None], grid, torch.zeros_like(grid)).clamp(-1, 1)  # no gradient from outside
        warped = F.grid_sample(src[j], grid, mode="bilinear", padding_mode="border", align_corners=False)
        inside = inside0 & ok_f[:, None, None]
        e = (warped - src).abs().mean(1)
        errs.append(torch.where(inside, e, torch.full_like(e, 10.0)))
    e = torch.stack(errs).min(0).values
    # pixels no neighbour sees get a constant penalty (above the typical error) instead of being dropped: otherwise the
    # loss is minimised by warping everything out of view (the first run diverged that way)
    return torch.where(e < 10.0, e, torch.full_like(e, 0.2)).mean()


def load_items(data, split, lam=0.1):
    items = []
    for p in sorted(glob.glob(os.path.join(data, f"{split}__*.npz"))):
        z = np.load(p)
        H, W = z["HW"]
        it = {"name": os.path.basename(p)[:-4], "A": operator(z["Hpp"], z["Bt"], lam), "files": list(z["files"]),
              "lgt": torch.from_numpy(z["lgt"]), "valid": torch.from_numpy(z["valid"])}
        it["hats"] = hats(int(H), int(W), tuple(z["lgt"].shape[-2:]))
        it["ds"] = it["name"].split("__")[1]
        if it["ds"] == "arkit":  # GT poses and intrinsics for the photometric loss
            m = np.load(os.path.join(os.path.dirname(it["files"][0]), "meta.npz"))
            E = [np.concatenate([cv2.Rodrigues(p_[:3].astype(np.float64))[0], p_[3:, None]], 1) for p_ in m["traj"]]
            it["meta"] = {"E": np.array(E, np.float32), "K": m["K"].astype(np.float32)}
        items.append(it)
    return items


def evaluate(api, items, cfg, dev):
    api.model.eval()
    r = {}
    with torch.no_grad():
        for it in items:
            ls, ld, rd = losses(api, it, cfg, dev, grad=False)
            r.setdefault(it["ds"], []).append((float(ls), float(ld), rd))
    return {ds: {"sens": float(np.mean([x[0] for x in v])), "sens_med": float(np.median([x[0] for x in v])), "depth": float(np.mean([x[1] for x in v])),
                 "rot_deg": float(np.mean([x[2] for x in v]))} for ds, v in r.items()}


def cmd_train(data, run, opts):
    cfg = {"w_sens": 1.0, "w_depth": 0.1, "steps": 3000, "lr": 1e-4, "seed": 0, "eval_every": 500, "w_sp": 0.0, "lam": 0.1,
           "robust": 10.0, "train": "held-out", "w_photo": 0.0}
    for k, v in opts.items():
        cfg[k] = type(cfg[k])(v)
    os.makedirs(run, exist_ok=True)
    os.chdir(rp.REPO_ROOT)
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    dev = torch.device("cuda")
    api = load_api()
    params = head_params(api)
    init = [p.detach().clone() for p in params]
    print(f"[ft] {sum(p.numel() for p in params) / 1e6:.2f}M trainable parameters", flush=True)
    train = [it for sp in cfg["train"].split(",") for it in load_items(data, sp, cfg["lam"])]
    val = load_items(data, "dev", cfg["lam"])
    if "held-out" not in cfg["train"]:  # the held-out benchmark scenes are a second validation set then
        val += [dict(it, ds="ho_" + it["ds"]) for it in load_items(data, "held-out", cfg["lam"])[::2]]
    print(f"[ft] {len(train)} train items, {len(val)} val items", flush=True)
    by_ds = {}
    for it in train:
        by_ds.setdefault(it["ds"], []).append(it)
    log = {"cfg": cfg, "evals": []}
    ev = {"step": 0, "val": evaluate(api, val, cfg, dev), "train": evaluate(api, train[:: max(3, len(train) // 60)], cfg, dev)}
    print(f"[ft] step 0 {json.dumps(ev)}", flush=True)
    log["evals"].append(ev)
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["steps"])
    t0 = time.time()
    acc = []
    for step in range(1, cfg["steps"] + 1):
        it = random.choice(by_ds[random.choice(list(by_ds))])
        ls, ld, rd = losses(api, it, cfg, dev)
        ls_r = cfg["robust"] * torch.log1p(ls / cfg["robust"]) if cfg["robust"] > 0 else ls  # outlier clips
        loss = cfg["w_sens"] * ls_r + cfg["w_depth"] * ld
        if cfg["w_photo"] > 0:
            loss = loss + cfg["w_photo"] * it.pop("_photo")
        if cfg["w_sp"] > 0:
            loss = loss + cfg["w_sp"] * sum(((p - q) ** 2).sum() for p, q in zip(params, init))
        opt.zero_grad()
        if not torch.isfinite(loss):
            print(f"[ft] step {step}: non-finite loss, skipped", flush=True)
            continue
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in params if p.grad is not None):
            print(f"[ft] step {step}: non-finite gradient, skipped", flush=True)
            opt.zero_grad()
            continue
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        acc.append((float(ls), float(ld), rd))
        if step % 50 == 0:
            a = np.mean(acc, 0)
            print(f"[ft] step {step} sens {a[0]:.4f} depth {a[1]:.5f} rot {a[2]:.3f} deg ({time.time() - t0:.0f}s)", flush=True)
            acc = []
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            ev = {"step": step, "val": evaluate(api, val, cfg, dev), "train": evaluate(api, train[:: max(3, len(train) // 60)], cfg, dev)}
            print(f"[ft] step {step} {json.dumps(ev)}", flush=True)
            log["evals"].append(ev)
            sd = {k: v for k, v in api.model.head.scratch.state_dict().items() if k.startswith(TRAIN_PREFIX)}
            torch.save(sd, os.path.join(run, "head.pt"))
            json.dump(log, open(os.path.join(run, "log.json"), "w"), indent=1)


def cmd_export(src, run, out_root, opts):
    split = opts.get("split", "dev")
    os.chdir(rp.REPO_ROOT)
    api = load_api()
    if run != "none":
        sd = torch.load(os.path.join(run, "head.pt"), map_location="cuda")
        missing = api.model.head.scratch.load_state_dict(sd, strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
    from depth_anything_3.utils.constants import EVAL_REF_VIEW_STRATEGY

    for ds in opts.get("datasets", ",".join(DEV)).split(","):
        for sd_ in scenes_of(src, ds, split):
            scene = os.path.relpath(sd_, os.path.join(src, "model_results", ds))
            ex = os.path.join(sd_, "unposed", "exports")
            meta = dict(np.load(os.path.join(ex, "gt_meta.npz"), allow_pickle=True))
            old = dict(np.load(os.path.join(ex, "mini_npz", "results.npz")))
            p = api.inference([str(f) for f in meta["image_files"]], ref_view_strategy=EVAL_REF_VIEW_STRATEGY)
            dcam = float(np.abs(np.asarray(p.extrinsics)[:, :3] - old["extrinsics"][:, :3]).max())
            o = os.path.join(out_root, "model_results", ds, scene, "unposed", "exports")
            os.makedirs(os.path.join(o, "mini_npz"), exist_ok=True)
            np.savez(os.path.join(o, "gt_meta.npz"), **meta)
            new = {**old, "depth": np.asarray(p.depth, np.float32), "conf": np.asarray(p.conf, np.float32)}
            np.savez(os.path.join(o, "mini_npz", "results.npz"), **new)
            g = try_gt(meta["image_files"], *new["depth"].shape[1:])
            from lpba_depth_oracle import abs_rel

            print(f"[export] {ds}/{scene}: camera change {dcam:.2e}, abs.rel. {abs_rel(old['depth'], g):.4f} -> {abs_rel(new['depth'], g):.4f}", flush=True)


def main():
    cmd = sys.argv[1]
    pos = [a for a in sys.argv[2:] if "=" not in a]
    opts = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    if cmd == "sens":
        cmd_sens(pos[0], pos[1], opts)
    elif cmd == "train":
        cmd_train(pos[0], pos[1], opts)
    elif cmd == "export":
        cmd_export(pos[0], pos[1], pos[2], opts)


if __name__ == "__main__":
    main()
