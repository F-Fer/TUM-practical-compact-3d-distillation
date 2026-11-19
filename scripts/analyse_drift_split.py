"""Rotation vs translation split of DA3-SMALL's pose error (follow-up to results/failure_modes/DA3-SMALL/REPORT.md).

Reuses analyse_failure_modes.load_scene / pose_errors (same gauge, same smooth/jitter split, same units) and adds
per-component statistics plus oracle counterfactuals that remove only the smooth rotation or only the smooth
translation. Writes one row per scene to drift_split.json next to this script.
"""
import glob
import json
import multiprocessing
import os
import sys

import numpy as np

REPO = "/home/student/dev/TUM-compact-3d-finn"
sys.path.insert(0, os.path.join(REPO, "scripts"))
import analyse_failure_modes as A  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

MODEL = "DA3-SMALL"
GROSS_R, GROSS_T = 15.0, 0.25


def neighbour_cos(x, c_gt, max_dist=0.05):
    """Mean cosine similarity of residual vectors for camera pairs closer than max_dist scene depths (true)."""
    i, j = np.triu_indices(len(x), 1)
    d = np.linalg.norm(c_gt[i] - c_gt[j], axis=1)
    m = d < max_dist
    if m.sum() < 5:
        return np.nan, 0
    a, b = x[i[m]], x[j[m]]
    cos = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    return float(np.mean(cos)), int(m.sum())


def scene(args):
    dataset, sc = args
    import cv2, torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    data = A.load_scene(MODEL, dataset, sc)
    depth_scale = float(np.median(data["gt_depth"][data["valid"]]))
    p = A.pose_errors(data["pred_w2c"], data["gt_w2c"], depth_scale, data["bench_order"])
    r, t = p["r_res"], p["t_res"]
    rn, tn = np.linalg.norm(r, axis=1), np.linalg.norm(t, axis=1)
    gross_r, gross_t = rn > GROSS_R, tn > GROSS_T
    ok = ~(gross_r | gross_t)

    def energy_share(res, smooth, mask):
        num = (smooth[mask] ** 2).sum()
        den = (res[mask] ** 2).sum()
        return float(num / den) if den > 0 else np.nan

    # extra oracles: remove the smooth part of only one component
    pred_c2w, gt_c2w = np.linalg.inv(data["pred_w2c"]), np.linalg.inv(data["gt_w2c"])
    R_pred, c_pred = pred_c2w[:, :3, :3], pred_c2w[:, :3, 3]
    gauge = A.robust_rotation_alignment(R_pred, gt_c2w[:, :3, :3])
    sim3 = A.robust_scale_translation(c_pred, gt_c2w[:, :3, 3], gauge)
    c_al, R_al = A.apply_sim3(sim3, c_pred), gauge @ R_pred

    def rebuild(rr, tt):
        corr = Rotation.from_rotvec(np.radians(rr)) * Rotation.from_rotvec(np.radians(r)).inv()
        c2w = A.camera_pose(corr.as_matrix() @ R_al, c_al + (tt - t) * depth_scale)
        return A.auc_scores(c2w, gt_c2w, data["gt_w2c"], data["bench_order"])

    cf = dict(zip(p["cf_names"].tolist(), p["cf_auc"].tolist()))
    cf["no smooth rotation"] = rebuild(r - p["r_smooth"], t)
    cf["no smooth translation"] = rebuild(r, t - p["t_smooth"])

    rp, tp = p["rot_pairs"], p["trans_pairs"]
    iu = np.triu_indices(len(r), 1)
    rp, tp = rp[iu], tp[iu]
    near = p["pair_baseline"] < 0.1
    rc, nr = neighbour_cos(r, p["c_gt"])
    tc, _ = neighbour_cos(t, p["c_gt"])
    return dict(
        dataset=dataset, scene=sc, n=len(r),
        r_med=float(np.median(rn)), t_med=float(np.median(tn)),
        r_vn=float(p["r_vn"]), t_vn=float(p["t_vn"]),
        r_smooth_share=energy_share(r, p["r_smooth"], ok), t_smooth_share=energy_share(t, p["t_smooth"], ok),
        r_cos_near=rc, t_cos_near=tc, n_near=nr,
        gross_any=int((~ok).sum()), gross_r_only=int((gross_r & ~gross_t).sum()),
        gross_t_only=int((gross_t & ~gross_r).sum()), gross_both=int((gross_r & gross_t).sum()),
        pair_rot_med=float(np.median(rp)), pair_trans_med=float(np.median(tp)),
        pair_trans_binding=float(np.mean(tp > rp)),
        pair_rot_med_near=float(np.median(rp[near])) if near.any() else np.nan,
        pair_trans_med_near=float(np.median(tp[near])) if near.any() else np.nan,
        cf=cf,
    )


def main():
    tasks = []
    for d in A.DATASETS:
        root = os.path.join(REPO, "workspace/evaluation", MODEL, "model_results", d)
        for res in sorted(glob.glob(os.path.join(root, "**/unposed/exports/mini_npz/results.npz"), recursive=True)):
            tasks.append((d, os.path.relpath(res, root).split("/unposed/")[0]))
    rows = []
    with multiprocessing.get_context("fork").Pool(8) as pool:
        for k, row in enumerate(pool.imap_unordered(scene, tasks), 1):
            rows.append(row)
            print(f"[{k}/{len(tasks)}] {row['dataset']}/{row['scene']}", flush=True)
    out = os.path.join(REPO, "results/failure_modes", MODEL, "drift_split.json")
    json.dump(rows, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
