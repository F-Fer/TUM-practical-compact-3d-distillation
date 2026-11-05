"""RoMa v2 as the tracker of run_tto_ba.py: dense matches from a few query frames to every other frame."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from vggt.utils.geometry import unproject_depth_map_to_point_map

_models = {}


def _roma(setting):
    if setting not in _models:
        from romav2 import RoMaV2

        model = RoMaV2()
        model.apply_setting(setting)
        _models[setting] = model
    return _models[setting]


def covisible(pts, ext, K, hw, q, k):
    """The k frames (other than q) that see the most of the query's 3D points, by DA3's cameras: set of indices."""
    pts = pts[:: max(len(pts) // 1024, 1)]
    cam = np.einsum("nij,pj->npi", ext[:, :3, :3], pts) + ext[:, None, :3, 3]
    uv = np.einsum("nij,npj->npi", K, cam)
    uv = uv[..., :2] / np.maximum(uv[..., 2:], 1e-6)
    ok = (cam[..., 2] > 0) & (uv[..., 0] >= 0) & (uv[..., 0] < hw[1]) & (uv[..., 1] >= 0) & (uv[..., 1] < hw[0])
    frac = ok.mean(1)
    frac[q] = -1
    return set(np.argsort(-frac)[:k].tolist())


def track_roma(image_files, pred, cfg, device):
    """Same return values as track() in run_tto_ba.py: tracks [N, P, 2] in pixels, vis [N, P], points3d, colors, L."""
    depth, conf, ext, K = pred["depth"], pred["conf"], pred["extrinsics"], pred["intrinsics"]
    N, H, W = depth.shape
    L = max(H, W)  # keeps the pixel coordinates in the frame BA expects

    imgs = np.stack([cv2.resize(cv2.imread(str(f))[..., ::-1], (W, H), interpolation=cv2.INTER_AREA) for f in image_files])
    images = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255
    points = unproject_depth_map_to_point_map(depth[..., None], ext, K)

    model = _roma(cfg["roma_setting"])
    queries = np.unique((np.linspace(0, N - 1, min(cfg["query_frames"], N)).round().astype(int) + cfg["query_offset"]) % N)
    all_tracks, all_vis, all_points, all_colors = [], [], [], []
    seen = np.zeros(N)  # visible tracks per frame
    todo, extra, used = list(queries), [], {int(q) for q in queries}
    while todo:
        q = todo.pop(0)
        # confident pixels away from the border
        valid = conf[q] > np.median(conf[q])
        m = int(0.02 * min(H, W))
        valid[:m], valid[-m:], valid[:, :m], valid[:, -m:] = False, False, False, False
        ys, xs = np.nonzero(valid)
        pick = np.random.choice(len(ys), min(cfg["max_query_pts"], len(ys)), replace=False)
        xy = np.stack([xs[pick], ys[pick]], 1).astype(np.float32)
        # RoMa's warps are in [-1, 1] with pixel centres at +0.5, DA3's pixel indices have them at integers
        grid = torch.from_numpy(((xy + 0.5) / [W, H] * 2 - 1).astype(np.float32)).to(device)[None, None]

        keep = covisible(points[q][ys[pick], xs[pick]], ext, K, (H, W), q, cfg["pairs"]) if cfg["pairs"] else None
        tracks = np.zeros((N, len(pick), 2), np.float32)
        vis = np.zeros((N, len(pick)), np.float32)
        img_q = images[q : q + 1].to(device)
        for j in range(N):
            if j == q:
                tracks[j], vis[j] = xy, 1
                continue
            if keep is not None and j not in keep:
                continue
            preds = model.match(img_q, images[j : j + 1].to(device))
            warp = preds["warp_AB"][0].permute(2, 0, 1)[None]
            overlap = preds["overlap_AB"][0].permute(2, 0, 1)[None]
            target = F.grid_sample(warp, grid, align_corners=False)[0, :, 0].T
            inside = (target.abs() <= 1).all(-1)
            tracks[j] = ((target + 1) / 2 * torch.tensor([W, H], device=device) - 0.5).cpu().numpy()
            vis[j] = (F.grid_sample(overlap, grid, align_corners=False)[0, 0, 0] * inside).cpu().numpy()

        all_tracks.append(tracks)
        all_vis.append(vis)
        all_points.append(points[q][ys[pick], xs[pick]])
        all_colors.append(imgs[q][ys[pick], xs[pick]])
        seen += (vis > cfg["vis_thresh"]).sum(1)
        # frames that no query sees well get no BA constraints, so they become queries themselves
        if not todo and len(extra) < cfg["extra_queries"]:
            weak = [j for j in np.argsort(seen) if j not in used and seen[j] < cfg["min_visible"]]
            if weak:
                extra.append(int(weak[0]))
                todo.append(int(weak[0]))
                used.add(int(weak[0]))
    return np.concatenate(all_tracks, 1), np.concatenate(all_vis, 1), np.concatenate(all_points), np.concatenate(all_colors), L
