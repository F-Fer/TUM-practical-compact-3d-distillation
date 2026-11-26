"""Which initial pose errors does each TTO method remove? Per frame pair, binned by DA3-SMALL's initial error.

    .venv/bin/python scripts/plot_error_basin.py   # -> results/pba/figures_depth/error_basin.png (+ table on stdout)

Pair error = max(relative rotation angle, relative translation angle) in degrees, computed with the DA3 evaluator's own
functions (bench/utils.py: align to the first camera, all pairs), i.e. the quantity AUC@3 integrates. Held-out scenes
(59) of the DA3-SMALL runs in workspace/tto/DA3-SMALL/sba_ho (PBA B′, NCC, RoMa + GPU BA); with --gmr also GMR (clean folds), written to error_basin_gmr.png.
Each scene gets the same total weight (pairs weighted by 1 / pairs in the scene), so 100-frame scenes don't dominate.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(REPO, "third_party", "Depth-Anything-3", "src"))
from depth_anything_3.bench.utils import align_to_first_camera, se3_to_relative_pose_error  # noqa: E402

INIT = "workspace/evaluation/DA3-SMALL"
H = "workspace/tto/DA3-SMALL/sba_ho"
METHODS = [("photometric BA (B′, 1 s)", f"{H}/pho25", "#86b6ef"), ("NCC photometric BA (15 s)", f"{H}/ncc800", "#2a78d6"),
           ("RoMa + sparse BA (36 s)", f"{H}/r8", "#eb6834")]
if "--gmr" in sys.argv:  # GMR (clean folds) as a fourth method
    METHODS.append(("GMR + sparse BA (20 s)", "workspace/tto/DA3-SMALL/gmr/t_cv_ho", "#1baf7a"))
DATASETS = ["hiroom", "7scenes", "scannetpp", "dtu64", "eth3d"]
BINS = [0, 3, 5, 10, 20, 45, 180]
OUT = os.path.join(REPO, "results", "pba", "figures_depth")


def hom(e):
    e = torch.from_numpy(np.asarray(e, dtype=np.float64))
    if e.shape[-2] == 3:
        e = torch.cat([e, torch.tensor([0, 0, 0, 1.0], dtype=e.dtype).expand(len(e), 1, 4)], 1)
    return e


def pair_err(pred, gt):
    r, t = se3_to_relative_pose_error(align_to_first_camera(hom(pred)), align_to_first_camera(hom(gt)), len(gt))
    return torch.maximum(r, t).numpy()


def npz(root, ds, scene):
    return os.path.join(REPO, root, "model_results", ds, scene, "unposed", "exports", "mini_npz", "results.npz")


def main():
    e0, w, ek = [], [], [[] for _ in METHODS]
    for ds in DATASETS:
        scenes = sorted(k for k in json.load(open(os.path.join(REPO, H, "pho25", "metric_results", f"{ds}_pba.json"))) if k != "mean")
        for sc in scenes:
            gt = np.load(os.path.join(REPO, INIT, "model_results", ds, sc, "unposed", "exports", "gt_meta.npz"), allow_pickle=True)["extrinsics"]
            if not all(os.path.exists(npz(r, ds, sc)) for _, r, _ in METHODS):
                print(f"missing {ds} {sc}", file=sys.stderr)
                continue
            a = pair_err(np.load(npz(INIT, ds, sc))["extrinsics"], gt)
            e0.append(a)
            w.append(np.full(len(a), 1.0 / len(a)))
            for k, (_, r, _) in enumerate(METHODS):
                ek[k].append(pair_err(np.load(npz(r, ds, sc))["extrinsics"], gt))
    e0, w = np.concatenate(e0), np.concatenate(w)
    ek = [np.concatenate(x) for x in ek]
    print(f"{len(w)} pairs, {w.sum():.0f} scenes")
    b = np.digitize(e0, BINS[1:-1])
    labels = [f"{lo}-{hi}°" if hi < 180 else f"> {lo}°" for lo, hi in zip(BINS[:-1], BINS[1:])]
    frac = np.zeros((len(METHODS), len(labels)))
    share = np.array([w[b == j].sum() for j in range(len(labels))]) / w.sum()
    for k in range(len(METHODS)):
        for j in range(len(labels)):
            m = b == j
            frac[k, j] = 100 * (w[m] * (ek[k][m] < 3)).sum() / w[m].sum()
    print("initial error   share  " + "  ".join(f"{n[:14]:>14s}" for n, _, _ in METHODS))
    for j, l in enumerate(labels):
        print(f"{l:14s} {100 * share[j]:5.1f}%  " + "  ".join(f"{frac[k, j]:14.1f}" for k in range(len(METHODS))))

    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor="#ffffff")
    x = np.arange(len(labels))
    bw = 0.8 / len(METHODS)
    for k, (name, _, c) in enumerate(METHODS):
        ax.bar(x + (k - (len(METHODS) - 1) / 2) * bw, frac[k], width=bw, color=c, edgecolor="#ffffff", linewidth=1.5, label=name, zorder=3)
    ax.set_xticks(x, [f"{l}\n({100 * s:.0f} % of pairs)" for l, s in zip(labels, share)], fontsize=11.5)
    ax.set_xlabel("pose error of DA3-SMALL's prediction (frame pair)")
    ax.set_ylabel("% of pairs below 3° after TTO")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=11.5, loc="upper right")
    ax.set_title("Which initial errors does each method remove? (DA3-SMALL, 59 held-out scenes)", loc="left", color=INK, fontsize=14, pad=12)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    name = "error_basin_gmr.png" if "--gmr" in sys.argv else "error_basin.png"
    fig.savefig(os.path.join(OUT, name), dpi=200)
    print(os.path.abspath(os.path.join(OUT, name)))


if __name__ == "__main__":
    main()
