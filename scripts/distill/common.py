"""Shared pieces of the distillation experiments: data split, cached data, teachers, students,
target construction, losses and validation metrics."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WS = os.path.join(REPO_ROOT, "workspace", "distill")
CACHE_DIR = os.path.join(WS, "cache")
SCANNETPP_ROOT = os.path.join(REPO_ROOT, "workspace", "benchmark_dataset", "scannetpp")
IMG_H, IMG_W = 378, 504

# order of depth_anything_3.utils.constants.SCANNETPP_SCENES; even positions train, odd held out
ALL_SCENES = [
    "09c1414f1b", "1ada7a0617", "40aec5fffa", "3e8bba0176", "acd95847c5",
    "578511c8a9", "5f99900f09", "c4c04e6d6c", "f3d64c30f8", "7bc286c1b6",
    "c5439f4607", "286b55a2bf", "fb5a96b1a2", "7831862f02", "38d58a7a31",
    "bde1e479ad", "9071e139d9", "21d970d8de", "bcd2436daf", "cc5237fd77",
]
TRAIN_SCENES = ALL_SCENES[0::2]
VAL_SCENES = ALL_SCENES[1::2]

import functools
import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(REPO_ROOT, "scripts"))  # after scripts/distill (own baselines.py copy)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


# ----------------------------------------------------------------------------- data

_SCENE_CACHE = {}


EXTRA_CACHE_DIR = os.path.join(WS, "cache_extra")  # unlabeled 7-Scenes / ETH3D sequences (prep_extra.py)


def extra_scenes():
    return sorted(f[:-4] for f in os.listdir(EXTRA_CACHE_DIR)) if os.path.isdir(EXTRA_CACHE_DIR) else []


def load_scene(scene):
    if scene not in _SCENE_CACHE:
        path = os.path.join(CACHE_DIR, f"{scene}.npz")
        if not os.path.exists(path):
            path = os.path.join(EXTRA_CACHE_DIR, f"{scene}.npz")
        d = np.load(path)
        d = {k: d[k] for k in d.files}
        n = len(d["images"])
        d.setdefault("gt_depth", np.zeros((n, IMG_H, IMG_W), np.float16))  # unlabeled: no GT
        d.setdefault("extrinsics", np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)))
        d.setdefault("intrinsics", np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)))
        dvlt_path = os.path.join(DVLT_CACHE_DIR, f"{scene}.npy")
        if os.path.exists(dvlt_path):
            d["images_dvlt"] = np.load(dvlt_path, mmap_mode="r")
        _SCENE_CACHE[scene] = d
    return _SCENE_CACHE[scene]


def sample_views(scenes, num_views, rng):
    """Random scene, then num_views random frames of it (like the benchmark's random 100 of ~300).
    scenes can be a list, or a list of lists (pick a group uniformly, then a scene in it)."""
    scene = rng.choice(scenes)
    if isinstance(scene, list):
        scene = rng.choice(scene)
    d = load_scene(scene)
    idx = rng.sample(range(len(d["images"])), min(num_views, len(d["images"])))
    return batch_from_indices(d, idx)


DVLT_CACHE_DIR = os.path.join(WS, "cache_dvlt")  # DéjàView-exact preprocessing (prep_dvlt_cache.py)


def batch_from_indices(d, idx):
    dvlt_imgs = d.get("images_dvlt")
    return {
        # [1,S,3,H,W] float in [0,1] as DéjàView's own preprocessing makes them (fallback: cv2 cache)
        "images_dvlt": torch.from_numpy(np.ascontiguousarray(dvlt_imgs[idx]))[None]
        if dvlt_imgs is not None
        else torch.from_numpy(d["images"][idx]).permute(0, 3, 1, 2)[None].float() / 255.0,
        "images": torch.from_numpy(d["images"][idx]).permute(0, 3, 1, 2)[None],  # uint8 [1,S,3,H,W]
        "gt_depth": torch.from_numpy(d["gt_depth"][idx].astype(np.float32))[None],
        "gt_w2c": torch.from_numpy(d["extrinsics"][idx])[None],
        "gt_K": torch.from_numpy(d["intrinsics"][idx])[None],
    }


def val_subsets(num_views=32, seed=0, scenes=None):
    rng = random.Random(seed)
    out = []
    for scene in scenes or VAL_SCENES:
        n = len(load_scene(scene)["images"])
        out.append((scene, sorted(rng.sample(range(n), min(num_views, n)))))
    return out


def bench_subsets(scenes=None, max_frames=100):
    """The exact frames the DA3 evaluator uses (random.seed(42) shuffle, first max_frames, sorted);
    the cache has the evaluator's frame order (sorted iPhone names)."""
    out = []
    for scene in scenes or VAL_SCENES:
        n = len(load_scene(scene)["images"])
        if n <= max_frames:
            out.append((scene, list(range(n))))
            continue
        random.seed(42)
        idx = list(range(n))
        random.shuffle(idx)
        out.append((scene, sorted(idx[:max_frames])))
    return out


def to_unit(images_u8, device="cuda"):
    return images_u8.to(device, non_blocking=True).float() / 255.0


def imagenet_norm(x):
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


# ----------------------------------------------------------------------------- geometry

def to_h(m):
    """[...,3,4] -> [...,4,4]"""
    if m.shape[-2] == 4:
        return m
    bottom = torch.zeros(*m.shape[:-2], 1, 4, device=m.device, dtype=m.dtype)
    bottom[..., 3] = 1
    return torch.cat([m, bottom], -2)


def relative_to_first(w2c):
    """w2c [B,S,4,4] -> w2c expressed in the frame of camera 0."""
    c2w0 = torch.linalg.inv(w2c[:, :1])
    return w2c @ c2w0


def rays_from_cameras(c2w, K, h, w, img_h=IMG_H, img_w=IMG_W):
    """DA3 ray-map format: dir = R_c2w K^-1 [u+.5, v+.5, 1] on an h x w grid, origin = camera centre.
    c2w [B,S,4,4], K [B,S,3,3] at image resolution -> [B,S,h,w,6]."""
    Ks = K.clone()
    Ks[..., 0, :] *= w / img_w
    Ks[..., 1, :] *= h / img_h
    v, u = torch.meshgrid(
        torch.arange(h, device=K.device).float() + 0.5, torch.arange(w, device=K.device).float() + 0.5, indexing="ij"
    )
    pix = torch.stack([u, v, torch.ones_like(u)], -1)  # h w 3
    Kinv = torch.linalg.inv(Ks)  # B S 3 3
    dirs = torch.einsum("bsij,hwj->bshwi", c2w[..., :3, :3] @ Kinv, pix)
    origin = c2w[..., None, None, :3, 3].expand(*dirs.shape)
    return torch.cat([dirs, origin], -1)


# ----------------------------------------------------------------------------- models

def load_da3(name):
    from depth_anything_3.api import DepthAnything3

    return DepthAnything3.from_pretrained(name).model  # DepthAnything3Net


def da3_forward(net, x01, feat_grad=True):
    """Raw DA3 forward (ref view = first, as in the benchmark). x01: [B,S,3,H,W] in [0,1].
    Returns depth [B,S,H,W], ray [B,S,h,w,6], pose_enc [B,S,9], feats (4 x [B,S,N,C]), cam tokens."""
    H, W = x01.shape[-2:]
    x = imagenet_norm(x01)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feats, _ = net.backbone(x, cam_token=None, export_feat_layers=[], ref_view_strategy="first")
    with torch.autocast("cuda", enabled=False):
        feats_f = [(f[0].float(), f[1].float()) for f in feats]
        out = net.head(feats_f, H, W, patch_start_idx=0)
        pose_enc = net.cam_dec(feats_f[-1][1])
    return {
        "depth": out["depth"],
        "depth_conf": out["depth_conf"],
        "ray": out["ray"],
        "pose_enc": pose_enc,
        "feats": [f[0] for f in feats],
    }


def da3_cameras(pose_enc, H=IMG_H, W=IMG_W):
    """pose_enc -> (w2c [B,S,4,4], K [B,S,3,3]) as DepthAnything3Net does."""
    from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri

    c2w, K = pose_encoding_to_extri_intri(pose_enc, (H, W))
    return torch.linalg.inv(to_h(c2w)), K


def load_dvlt(steps=12):
    import baselines

    return baselines.DVLTBaseline("nvidia/dvlt", inference_steps=steps).model


def dvlt_forward(model, x01, steps, grad=False, z0=None, return_feats=False):
    """DéjàView forward with a K-step linspace schedule, optionally returning each step's tokens.
    z0 (encoder tokens incl. camera/register tokens) can be shared between teacher and student."""
    B, S, _, H, W = x01.shape
    if z0 is None:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z = model._encode_images(x01)
            reg = model.register_token.expand(B, S, -1, -1).reshape(B * S, model.num_register_tokens, -1)
            from dvlt.model.dvlt.model import _slice_expand_flatten

            cam = _slice_expand_flatten(model.camera_token, B, S)
            z0 = torch.cat([cam, reg, z], dim=1)
    rope_pos = model._get_rope_positions(B * S, H, W, x01.device)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    feats = []
    with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
        x = z0
        ts = torch.linspace(0.0, 1.0, steps).tolist()
        for i in range(steps):
            t_next = ts[i + 1] if i + 1 < steps else 1.0
            step = functools.partial(_untied_step, model, i) if hasattr(model, "untied_blocks") else model._interval_step
            if grad and model.training:
                x = torch.utils.checkpoint.checkpoint(step, x, ts[i], t_next, rope_pos, B, S, use_reentrant=False)
            else:
                x = step(x, ts[i], t_next, rope_pos, B, S)
            if return_feats:
                feats.append((t_next if i + 1 < steps else 1.0 + 1e-3, x))
        pred = model._decode(x, H, W, B, S, rope_pos)
    out = {
        "depth": pred["depth"][..., 0].float(),
        "rays": pred["rays"].float(),
        "pose_enc": pred.get("pose_enc"),
        "z0": z0,
    }
    if return_feats:
        out["feats"] = feats
    return out


def dvlt_cameras(rays, patch_size=14):
    """DéjàView rays -> (w2c [B,S,4,4], K) exactly as the benchmark adapter does."""
    from dvlt.common.rays import rays_to_pose

    H, W = rays.shape[2:4]
    with torch.autocast("cuda", enabled=False):
        c2w, K = rays_to_pose(rays.float(), torch.ones(rays.shape[:4], device=rays.device), H, W, patch_size)
    return torch.linalg.inv(to_h(c2w.float())), K.float()


# ----------------------------------------------------------------------------- targets / losses

def make_targets(depth, w2c, K, valid=None):
    """Common target format: everything relative to camera 0, scale-normalized by mean valid depth."""
    w2c = relative_to_first(to_h(w2c).float())
    if valid is None:
        valid = torch.isfinite(depth) & (depth > 0)
    scale = depth[valid].mean().clamp(min=1e-6)
    c2w = torch.linalg.inv(w2c)
    c2w = c2w.clone()
    c2w[..., :3, 3] /= scale
    return {"depth": depth / scale, "c2w": c2w, "K": K.float(), "valid": valid, "scale": scale}


def quat_from_c2w(c2w):
    from depth_anything_3.model.utils.transform import mat_to_quat

    return mat_to_quat(c2w[..., :3, :3])


def gradient_loss(pred, target, valid):
    """Mean abs difference of horizontal/vertical depth gradients (normalized depth), valid pixels only."""
    loss = 0
    for dim in (-1, -2):
        dp = pred.diff(dim=dim)
        dt = target.diff(dim=dim)
        m = valid.narrow(dim, 1, valid.shape[dim] - 1) & valid.narrow(dim, 0, valid.shape[dim] - 1)
        loss = loss + (dp - dt).abs()[m].mean()
    return loss


def da3_student_loss(pred, tgt, w):
    """pred from da3_forward (student), tgt from make_targets. Scale-invariant: the student's
    depth/translations are normalized by its own mean depth over the valid pixels."""
    valid = tgt["valid"]
    if w.get("log_depth"):
        # scale-invariant log loss; the implied student/teacher scale ratio also normalizes translations
        log_diff = torch.log(pred["depth"].clamp(min=1e-6)) - torch.log(tgt["depth"].clamp(min=1e-6))
        log_s = log_diff[valid].mean()
        s = torch.exp(log_s)
        d = pred["depth"] / s
        l_depth = (log_diff - log_s).abs()[valid].mean()
    else:
        s = pred["depth"][valid].mean().clamp(min=1e-6)
        d = pred["depth"] / s
        l_depth = (d - tgt["depth"]).abs()[valid].mean()
    logs = {}
    l_grad = gradient_loss(d, tgt["depth"], valid)
    # camera: pose_enc = [c2w translation, quat (xyzw), fov_h, fov_w]
    pe = pred["pose_enc"]
    t_tgt = tgt["c2w"][..., :3, 3]
    q_tgt = quat_from_c2w(tgt["c2w"])
    q_pred = F.normalize(pe[..., 3:7], dim=-1)
    q_sign = torch.sign((q_pred * q_tgt).sum(-1, keepdim=True)).detach()
    fov_h = 2 * torch.atan((IMG_H / 2) / tgt["K"][..., 1, 1])
    fov_w = 2 * torch.atan((IMG_W / 2) / tgt["K"][..., 0, 0])
    l_t = (pe[..., :3] / s - t_tgt).abs().mean()
    l_q = (q_pred - q_sign * q_tgt).abs().mean()
    l_f = (pe[..., 7:9] - torch.stack([fov_h, fov_w], -1)).abs().mean()
    l_cam = l_t + l_q + l_f
    # ray map (DA3's auxiliary head, trained jointly in DA3)
    h, w_ = pred["ray"].shape[2:4]
    ray_t = rays_from_cameras(tgt["c2w"], tgt["K"], h, w_)
    ray_p = pred["ray"]
    l_ray = (ray_p[..., :3] - ray_t[..., :3]).abs().mean() + (ray_p[..., 3:] / s - ray_t[..., 3:]).abs().mean()
    total = w.get("depth", 1) * l_depth + w.get("grad", 1) * l_grad + w.get("cam", 1) * l_cam + w.get("ray", 1) * l_ray
    logs.update(depth=l_depth.item(), grad=l_grad.item(), cam_t=l_t.item(), cam_q=l_q.item(), cam_fov=l_f.item(), ray=l_ray.item())
    return total, logs


class FeatProjector(nn.Module):
    """Per-layer linear maps from student to teacher feature width (feature distillation)."""

    def __init__(self, c_s, c_t, n=4):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(c_s, c_t) for _ in range(n)])

    def forward(self, feats_s, feats_t):
        loss = 0
        for p, fs, ft in zip(self.proj, feats_s, feats_t):
            ps = p(fs.float())
            loss = loss + (1 - F.cosine_similarity(ps, ft.float(), dim=-1)).mean() + F.smooth_l1_loss(
                F.layer_norm(ps, ps.shape[-1:]), F.layer_norm(ft.float(), ft.shape[-1:])
            )
        return loss / len(self.proj)


# ----------------------------------------------------------------------------- validation

def depth_metrics(pred, gt):
    """Per-sequence median scale alignment (one scale for all frames), AbsRel and delta<1.25."""
    valid = (gt > 0) & torch.isfinite(gt) & (gt < 20)
    p, g = pred[valid].float(), gt[valid].float()
    scale = torch.median(g) / torch.median(p)
    p = p * scale
    absrel = ((p - g).abs() / g).mean().item()
    d1 = (torch.maximum(p / g, g / p) < 1.25).float().mean().item()
    return absrel, d1


def pose_metrics(w2c_pred, w2c_gt):
    from depth_anything_3.bench.utils import compute_pose

    r = compute_pose(to_h(w2c_pred).double().cpu(), to_h(w2c_gt).double().cpu())
    return float(r.auc30), float(r.auc03)


def predict_fn_da3(net):
    def fn(x01):
        out = da3_forward(net, x01)
        w2c, K = da3_cameras(out["pose_enc"])
        return out["depth"], w2c

    return fn


def predict_fn_dvlt(model, steps):
    def fn(x01):
        out = dvlt_forward(model, x01, steps)
        w2c, _ = dvlt_cameras(out["rays"])
        return out["depth"], w2c

    return fn


@torch.no_grad()
def validate(predict, subsets, image_key="images"):
    """predict(x01 [1,S,3,H,W]) -> (depth [1,S,H,W], w2c [1,S,4,4]). Mean over val subsets.
    image_key "images_dvlt" feeds DéjàView-exact preprocessed images."""
    rows = []
    for scene, idx in subsets:
        b = batch_from_indices(load_scene(scene), idx)
        x = b[image_key].cuda().float() if image_key == "images_dvlt" else to_unit(b["images"])
        depth, w2c = predict(x)
        absrel, d1 = depth_metrics(depth[0], b["gt_depth"][0].cuda())
        auc30, auc03 = pose_metrics(w2c[0], b["gt_w2c"][0])
        rows.append(dict(scene=scene, absrel=absrel, d1=d1, auc30=auc30, auc03=auc03))
    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("absrel", "d1", "auc30", "auc03")}
    return mean, rows


def append_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def enable_da3_checkpointing(net):
    """Activation checkpointing for each backbone block and each DualDPT head chunk."""
    from torch.utils.checkpoint import checkpoint

    for blk in net.backbone.pretrained.blocks if hasattr(net.backbone, "pretrained") else net.backbone.blocks:
        orig = blk.forward
        blk.forward = lambda *a, _f=orig, **k: checkpoint(_f, *a, use_reentrant=False, **k)
    orig_head = net.head._forward_impl
    net.head._forward_impl = lambda *a, _f=orig_head, **k: checkpoint(_f, *a, use_reentrant=False, **k)


def enable_dvlt_checkpointing(model):
    """Activation checkpointing for DéjàView's ray/depth decoders (per decode chunk)."""
    from torch.utils.checkpoint import checkpoint

    for m in (model.ray_decoder, model.depth_decoder):
        orig = m.forward
        m.forward = lambda *a, _f=orig, **k: checkpoint(_f, *a, use_reentrant=False, **k)


def _untied_step(model, i, x, t_now, t_next, rope_pos, B, S):
    """DVLTModel._interval_step ("interval" time conditioning) with the i-th untied block."""
    t_pair = torch.tensor([[t_now, t_next]], device=x.device, dtype=torch.float32)
    return model.untied_blocks[i](x, t_pair.expand(B * S, -1), t_pair.expand(B, -1), rope_pos, B, S)


def untie_dvlt(model, K):
    """Replace the shared loop block by K independent copies (same compute at K steps, K x params).
    Also patches the inference solver so the benchmark adapter (model.forward) uses them."""
    import copy
    import types

    model.untied_blocks = nn.ModuleList([copy.deepcopy(model.recurrent_blocks[0]) for _ in range(K)])

    def solve(self, x, rope_pos, B, S):
        ts = torch.linspace(0.0, 1.0, K).tolist()
        for i in range(K):
            x = _untied_step(self, i, x, ts[i], ts[i + 1] if i + 1 < K else 1.0, rope_pos, B, S)
        return x

    model._solve_inference_linspace_k = types.MethodType(solve, model)
    return model
