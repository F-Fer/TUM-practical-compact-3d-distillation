"""Depth oracles and depth sources for PBA (reports/Learned refinement step between DA3 and DeepBA.md, §10 and §12):
how much does the pose accuracy of PBA depend on the depth, and on which part of its error?

    .venv/bin/python scripts/lpba_depth_oracle.py <mode> <src work dir> <dst work dir> [run_tto_pba overrides]

Runs scripts/run_tto_pba.py unchanged, except that each scene's predicted depth (DA3-SMALL's, from <src>) is replaced
before the refinement. Modes:

    da3            DA3's depth (the control: same code path)
    gt             the GT depth, scaled by one factor per scene to DA3's median depth (DA3's translations stay
                   consistent with it); pixels without GT get DA3's depth scaled per frame to the GT and the lowest
                   confidence, so they are never sampled (conf_quantile)
    lerp:L         log-depth interpolation DA3 -> GT by the fraction L (0 = DA3 scaled per scene, 1 = gt); pixels
                   without GT as in gt
    scale          DA3's depth with each frame's scale fitted to the GT (median ratio): only the per-frame scale error
                   is removed
    smooth:G       DA3's depth times a per-frame bilinear log-depth grid with G cells along the long side (corners at
                   the image corners, as PBA's grid) fitted to log(GT / DA3) by least squares: the per-frame scale and
                   the smooth part of the error are removed, the fine part stays
    model:<dir>    the depth of another model's predictions in <dir> (e.g. workspace/evaluation/DA3-LARGE-1.1), scaled
                   by one factor per scene to DA3-SMALL's median; cameras stay DA3-SMALL's (non-oracle)
    file:<dir>     the depth of results.npz files in <dir> (same layout as model:, e.g. a learned refiner's output)

GT depth exists for HiRoom (16-bit PNG, /65535 * 100 m), 7-Scenes (PNG mm; registered to the depth camera, shifted
4-12 px against the RGB images, knowledge/evaluation.md), ScanNet++ (render_depth PNG mm) and ETH3D (float32 raw, mask
masks_for_images == 1 invalid); not for DTU-64. The oracle modes need it (pass datasets=hiroom,7scenes,scannetpp,eth3d);
model: and file: run on any dataset. Per scene the log prints the depth error against the GT (abs. rel. after one scale
per scene) of DA3's, of the used and of PBA's output depth, and info gets depth_err_da3 / depth_err_used / depth_err_out
(nan without GT). With mode da3 this is plain run_tto_pba.py plus the depth diagnostic.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_tto_pba as rp  # noqa: E402


def gt_depth(image_file, H, W):
    """GT depth [H, W] in metres (0 = invalid) of a benchmark image, at the prediction's resolution."""
    f = str(image_file)
    if "/hiroom/" in f:
        d = cv2.imread(f.replace("/image/", "/depth/").rsplit(".", 1)[0] + ".png", -1).astype(np.float32) / 65535.0 * 100.0
    elif "/7scenes/" in f.lower():
        d = cv2.imread(f.replace(".color.png", ".depth.png"), -1).astype(np.float32)
        d[d == 65535] = 0
        d = d / 1000.0
    elif "/scannetpp/" in f:
        name = os.path.basename(f).rsplit(".", 1)[0]
        root = f.split("/merge_dslr_iphone/")[0] + "/merge_dslr_iphone"
        d = cv2.imread(os.path.join(root, "render_depth", name + ".png"), -1).astype(np.float32) / 1000.0
    elif "/eth3d/" in f:
        scene_root, name = f.split("/images/dslr_images/")
        raw = np.fromfile(os.path.join(scene_root, "ground_truth_depth", "dslr_images", name), dtype=np.float32)
        h = int(round(np.sqrt(raw.size * H / W)))
        d = raw.reshape(h, raw.size // h).copy()
        mask = cv2.imread(os.path.join(scene_root, "masks_for_images", "dslr_images", name.rsplit(".", 1)[0] + ".png"), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            d[mask == 1] = 0
    else:
        raise ValueError(f"no GT depth for {f}")
    d[~np.isfinite(d)] = 0
    return cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)


def try_gt(image_files, H, W):
    try:
        return np.stack([gt_depth(f, H, W) for f in image_files])
    except (ValueError, AttributeError, OSError, TypeError):
        return None


def abs_rel(depth, gt):
    if gt is None or (gt > 0).sum() < 100:
        return float("nan")
    v = gt > 0
    s = np.median(depth[v]) / np.median(gt[v])
    return float(np.mean(np.abs(depth[v] / s - gt[v]) / gt[v]))


def grid_shape(g, H, W):
    """Cells (gh, gw) of PBA's grid with g along the long side."""
    return (g, max(2, round(g * W / H))) if H > W else (max(2, round(g * H / W)), g)


def fit_smooth(r, valid, g, ridge=1e-3, max_px=40000, seed=0):
    """Bilinear grid [gh, gw] (corners at the image corners) fitted to r [H, W] on valid pixels; returns the dense field."""
    H, W = r.shape
    gh, gw = grid_shape(g, H, W)
    vs, us = np.nonzero(valid)
    if len(vs) > max_px:
        idx = np.random.default_rng(seed).choice(len(vs), max_px, replace=False)
        vs, us = vs[idx], us[idx]
    gu, gv = us * (gw - 1) / (W - 1), vs * (gh - 1) / (H - 1)
    u0, v0 = np.clip(np.floor(gu), 0, gw - 2).astype(int), np.clip(np.floor(gv), 0, gh - 2).astype(int)
    du, dv = gu - u0, gv - v0
    n = gh * gw
    A = np.zeros((len(vs), n))
    rows = np.arange(len(vs))
    for (a, b), w in (((0, 0), (1 - du) * (1 - dv)), ((0, 1), du * (1 - dv)), ((1, 0), (1 - du) * dv), ((1, 1), du * dv)):
        np.add.at(A, (rows, (v0 + a) * gw + u0 + b), w)
    x = np.linalg.solve(A.T @ A + ridge * len(vs) / n * np.eye(n), A.T @ r[vs, us])
    import torch
    import torch.nn.functional as F

    return F.interpolate(torch.from_numpy(x.reshape(1, 1, gh, gw)), size=(H, W), mode="bilinear", align_corners=True)[0, 0].numpy()


def other_depth(root, image_files):
    """Depth of the scene with these image files from another work dir with the same layout."""
    import glob

    key = str(image_files[0])
    if root not in _INDEX:
        _INDEX[root] = {}
        for meta in glob.glob(os.path.join(root, "model_results", "**", "unposed", "exports", "gt_meta.npz"), recursive=True):
            _INDEX[root][str(np.load(meta, allow_pickle=True)["image_files"][0])] = os.path.join(os.path.dirname(meta), "mini_npz", "results.npz")
    return np.load(_INDEX[root][key])["depth"].astype(np.float32)


_INDEX = {}


def make_depth(mode, da3, gt, image_files):
    """The depth that replaces DA3's, per mode (see the module docstring). gt may be None for model:/file:."""
    N, H, W = da3.shape
    if mode == "da3":
        return da3
    if mode.startswith(("model:", "file:")):
        other = other_depth(mode.split(":", 1)[1], image_files)
        return other * (np.median(da3) / np.median(other))
    valid = gt > 0
    s = np.median(da3[valid]) / np.median(gt[valid])
    gts = gt * s

    def fill(depth):  # pixels without GT: DA3 scaled per frame to the used depth
        for i in range(N):
            v = valid[i]
            k = np.median(depth[i][v] / da3[i][v]) if v.sum() > 100 else 1.0
            depth[i][~v] = da3[i][~v] * k
        return depth

    if mode == "gt":
        return fill(gts.copy())
    if mode.startswith("lerp:"):
        lam = float(mode.split(":")[1])
        out = np.where(valid, np.exp((1 - lam) * np.log(np.maximum(da3, 1e-6)) + lam * np.log(np.maximum(gts, 1e-6))), 0).astype(np.float32)
        return fill(out)
    if mode == "scale":
        out = da3.copy()
        for i in range(N):
            v = valid[i]
            if v.sum() > 100:
                out[i] *= np.median(gts[i][v] / da3[i][v])
        return out
    if mode.startswith("smooth:"):
        g = int(mode.split(":")[1])
        out = da3.copy()
        for i in range(N):
            v = valid[i]
            if v.sum() > 100:
                r = np.where(v, np.log(np.maximum(gts[i], 1e-6)) - np.log(np.maximum(da3[i], 1e-6)), 0)
                out[i] = da3[i] * np.exp(fit_smooth(r, v, g))
        return out
    raise ValueError(mode)


def main():
    mode, args = sys.argv[1], sys.argv[2:]
    refine = rp.refine_scene

    def patched(image_files, pred, cfg, device):
        da3 = pred["depth"].astype(np.float32)
        N, H, W = da3.shape
        gt = try_gt(image_files, H, W)
        needs_gt = not mode.startswith(("da3", "model:", "file:"))
        if needs_gt and (gt is None or (gt > 0).sum() < 100):
            raise RuntimeError(f"mode {mode} needs GT depth: {image_files[0]}")
        depth = make_depth(mode, da3, gt, image_files).astype(np.float32)
        e0 = abs_rel(da3, gt) if gt is not None else float("nan")
        e1 = abs_rel(depth, gt) if gt is not None else float("nan")
        print(f"[depth_oracle] {mode}: DA3 abs.rel. (scene scale) {e0:.4f} -> used {e1:.4f}")
        if gt is not None and mode not in ("da3",) and not mode.startswith(("model:", "file:")):
            conf = np.where(gt > 0, pred["conf"], pred["conf"].min() - 1).astype(np.float32)
        else:
            conf = pred["conf"]
        out, info, hist = refine(image_files, {**pred, "depth": depth, "conf": conf}, cfg, device)
        e2 = abs_rel(out["depth"], gt) if gt is not None else float("nan")
        print(f"[depth_oracle] depth after PBA {e2:.4f}")
        info["depth_err_da3"], info["depth_err_used"], info["depth_err_out"] = e0, e1, e2
        return out, info, hist

    rp.refine_scene = patched
    sys.argv = [rp.__file__] + args
    rp.main()


if __name__ == "__main__":
    main()
