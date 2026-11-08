"""Which kind of pose error does a TTO run remove? Splits each run's camera errors as analyse_failure_modes.py does.

    uv run scripts/pose_change_decomp.py workspace/evaluation/DA3-SMALL workspace/tto/DA3-SMALL/<run> ... [--out x.json]

Per run and dataset, over the scenes all runs share: RMS rotation error (deg) and its smooth part and jitter (split along the
frame order by a rolling median), the same for the camera centres (in units of a GT scene scale), the share of grossly wrong
frames (> 15 deg or > 25 % of the scene scale) and the benchmark AUC@3. The scene scale is the median GT depth of
workspace/failure_modes/<model>/<dataset>/<scene>.npz where it exists, else the median distance of the GT camera centres from
their median (DTU-64 has no GT depth), so it is the same for every run.
"""

import argparse
import glob
import json
import os

import numpy as np

import analyse_failure_modes as afm

SETS = ("hiroom", "7scenes", "scannetpp", "dtu64", "eth3d")


def scenes_of(run, dataset):
    root = os.path.join(run, "model_results", dataset)
    return {os.path.relpath(p, root).split("/unposed/")[0]: p for p in glob.glob(os.path.join(root, "**/unposed/exports/mini_npz/results.npz"), recursive=True)}


def w2c(ext):
    T = np.tile(np.eye(4), (len(ext), 1, 1))
    T[:, :3] = ext[:, :3]
    return T


def scene_scale(dataset, scene, gt_w2c):
    fm = os.path.join(afm.REPO_ROOT, "workspace/failure_modes/DA3-SMALL", dataset, scene.replace("/", "_") + ".npz")
    if os.path.exists(fm):
        return float(np.load(fm)["median_gt_depth"])
    c = np.linalg.inv(gt_w2c)[:, :3, 3]
    return float(np.median(np.linalg.norm(c - np.median(c, 0), axis=1)))


def analyse(run, dataset, scene, path):
    gt = np.load(path.replace("mini_npz/results.npz", "gt_meta.npz"), allow_pickle=True)
    files = [str(f) for f in gt["image_files"]]
    order = np.argsort(files)  # frame order, as in analyse_failure_modes.load_scene
    pred = w2c(np.load(path)["extrinsics"])[order]
    gtw = w2c(gt["extrinsics"].astype(np.float64))[order]
    p = afm.pose_errors(pred, gtw, scene_scale(dataset, scene, gtw), np.argsort(order))
    rn, tn = np.linalg.norm(p["r_res"], axis=1), np.linalg.norm(p["t_res"], axis=1)
    gross = (rn > 15) | (tn > 0.25)
    return {
        "frames": len(rn),
        "r_rms": float(np.sqrt((rn**2).mean())), "r_med": float(np.median(rn)),
        "r_smooth_rms": float(np.sqrt((np.linalg.norm(p["r_smooth"], axis=1) ** 2).mean())),
        "r_jitter_rms": float(np.sqrt((np.linalg.norm(p["r_high"], axis=1) ** 2).mean())),
        "t_rms": float(np.sqrt((tn**2).mean())), "t_med": float(np.median(tn)),
        "t_smooth_rms": float(np.sqrt((np.linalg.norm(p["t_smooth"], axis=1) ** 2).mean())),
        "t_jitter_rms": float(np.sqrt((np.linalg.norm(p["t_high"], axis=1) ** 2).mean())),
        "gross": float(gross.mean()), "gross_frames": np.flatnonzero(gross).tolist(),
        "r_med_ok": float(np.median(rn[~gross])) if (~gross).any() else float("nan"),
        "t_med_ok": float(np.median(tn[~gross])) if (~gross).any() else float("nan"),
        "auc03": float(p["cf_auc"][0][0]), "auc03_rel": float(p["cf_auc"][0][2]),
        "auc03_no_smooth": float(p["cf_auc"][3][0]), "auc03_no_jitter": float(p["cf_auc"][2][0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = {}
    for d in SETS:
        common = set.intersection(*[set(scenes_of(r, d)) for r in a.runs])
        if not common:
            continue
        res[d] = {r: {s: analyse(r, d, s, scenes_of(r, d)[s]) for s in sorted(common)} for r in a.runs}
    keys = ("r_rms", "r_smooth_rms", "r_jitter_rms", "r_med_ok", "t_rms", "t_smooth_rms", "t_jitter_rms", "t_med_ok", "gross", "auc03", "auc03_rel")
    print("dataset scenes run | " + " ".join(keys))
    for d, runs in res.items():
        for r, sc in runs.items():
            vals = [np.nanmean([v[k] for v in sc.values()]) for k in keys]
            print(f"{d:9s} {len(sc):2d} {os.path.basename(os.path.normpath(r)):22s} | " + " ".join(f"{x:7.3f}" for x in vals))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
