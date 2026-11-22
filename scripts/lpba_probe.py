"""Check 2 (reports/Learned refinement step between DA3 and DeepBA.md, §9): is the error that PBA leaves in its weak
directions predictable from per-frame quantities that a learned step would see?

    .venv/bin/python scripts/lpba_probe.py <dir of npz from lpba_info.py> [<dir> ...] [--folds 5]

Target per frame i >= 1: y_i, the 6-vector (rotation in rad, translation in median depths; PBA's camera-frame
parameters) of the remaining error d projected onto the scene's weakest 10 % of pose directions. Features per frame
(40): the run's update from DA3 (6), the run's pose relative to the previous and next frame by index (12), the reduced
gradient of the objective (6, normalised per scene), log eigenvalues of the frame's 6x6 block of the Schur complement
(6), the frame's share of the weak basis (1), DA3 confidence relative to the scene (2), i / N and log N (2), the dataset
(5). Predictions are projected onto the scene's weak basis before scoring.

Models: zero (= the run), shrink (y = alpha * update, one scalar), ridge (linear), gbm (gradient-boosted trees, one per
output), all fitted on training scenes only (group k-fold over scenes, and leave-one-dataset-out). Scores on the test
scenes: explained = 1 - sum |P y_hat - y|^2 / sum |y|^2 (pooled over frames), and AUC@3 after applying P y_hat to the
run's poses (mean over datasets; "oracle" applies y itself = G0b's weak10).
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_anything_3.bench.utils import compute_pose  # noqa: E402
from depth_anything_3.utils.geometry import as_homogeneous  # noqa: E402
from lpba_weak_oracle import apply  # noqa: E402

SETS = ("hiroom", "7scenes", "scannetpp", "dtu64", "eth3d")


def load(path):
    z = dict(np.load(path, allow_pickle=True))
    N = len(z["d"])
    Vw = z["Vw"]
    y = (Vw @ (Vw.T @ z["d"][1:].reshape(-1))).reshape(N - 1, 6)
    ds = str(z["dataset"])
    i = np.arange(1, N)
    X = np.concatenate([z["upd"][1:], z["prev"][1:], z["nxt"][1:], z["grad"][1:], z["blocks"][1:], z["proj"][1:, None],
                        z["conf_mean"][1:, None], z["conf_med"][1:, None], (i / N)[:, None], np.full((N - 1, 1), np.log(N)),
                        np.tile(np.array([ds == s for s in SETS], float), (N - 1, 1))], 1)
    return {"name": os.path.basename(path)[:-4], "dataset": ds, "X": X, "y": y, "upd": z["upd"][1:], "Vw": Vw, "ext": z["ext"],
            "gt_ext": z["gt_ext"], "scale": float(z["scale"]), "N": N}


def project(s, yhat):
    return (s["Vw"] @ (s["Vw"].T @ yhat.reshape(-1))).reshape(-1, 6)


def auc(s, corr):
    full = np.zeros((s["N"], 6))
    full[1:] = corr
    e = apply(s["ext"], full, s["scale"])
    return float(compute_pose(torch.from_numpy(as_homogeneous(e)), torch.from_numpy(as_homogeneous(s["gt_ext"])))["auc03"])


def fit_predict(model, train, test):
    if model == "zero":
        return [np.zeros_like(s["y"]) for s in test]
    if model == "shrink":
        pu = [project(s, s["upd"]) for s in train]
        alpha = sum((p * s["y"]).sum() for p, s in zip(pu, train)) / max(sum((p**2).sum() for p in pu), 1e-30)
        return [alpha * s["upd"] for s in test]
    X = np.concatenate([s["X"] for s in train])
    Y = np.concatenate([s["y"] for s in train])
    sc = StandardScaler().fit(X)
    if model == "ridge":
        m = Ridge(alpha=10.0).fit(sc.transform(X), Y)
        return [m.predict(sc.transform(s["X"])) for s in test]
    if model == "gbm":
        ms = [HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0).fit(X, Y[:, k]) for k in range(6)]
        return [np.stack([m.predict(s["X"]) for m in ms], 1) for s in test]
    raise ValueError(model)


def report(title, scenes, preds):
    print(f"\n{title}")
    print(f"{'model':8s} | explained | AUC@3 x100: " + " ".join(f"{d[:5]:>5s}" for d in SETS) + " |  mean")
    models = list(preds)
    for m in models + ["oracle"]:
        corr = [s["y"] for s in scenes] if m == "oracle" else [project(s, p) for s, p in zip(scenes, preds[m])]
        num = sum(((c - s["y"]) ** 2).sum() for c, s in zip(corr, scenes))
        den = sum((s["y"] ** 2).sum() for s in scenes)
        a = [auc(s, c) for s, c in zip(scenes, corr)]
        per = [100 * np.mean([x for x, s in zip(a, scenes) if s["dataset"] == d]) if any(s["dataset"] == d for s in scenes) else np.nan for d in SETS]
        print(f"{m:8s} | {1 - num / den:9.3f} | {'':11s}" + " ".join(f"{v:5.1f}" for v in per) + f" | {np.nanmean(per):5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    scenes = [load(p) for d in a.dirs for p in sorted(glob.glob(os.path.join(d, "*.npz")))]
    print(f"{len(scenes)} scenes, {sum(s['N'] - 1 for s in scenes)} frames")
    models = ("zero", "shrink", "ridge", "gbm")

    # in-sample (train = test): an optimistic ceiling; if this explains little, nothing simple will
    report("in-sample (fit on all scenes, scored on the same)", scenes, {m: fit_predict(m, scenes, scenes) for m in models})

    groups = np.arange(len(scenes))
    preds = {m: [None] * len(scenes) for m in models}
    for tr, te in GroupKFold(a.folds).split(groups, groups=groups):
        for m in models:
            for i, p in zip(te, fit_predict(m, [scenes[i] for i in tr], [scenes[i] for i in te])):
                preds[m][i] = p
    report(f"{a.folds}-fold cross-validation over scenes", scenes, preds)

    preds = {m: [None] * len(scenes) for m in models}
    for d in SETS:
        te = [i for i, s in enumerate(scenes) if s["dataset"] == d]
        tr = [i for i, s in enumerate(scenes) if s["dataset"] != d]
        if not te:
            continue
        for m in models:
            for i, p in zip(te, fit_predict(m, [scenes[i] for i in tr], [scenes[i] for i in te])):
                preds[m][i] = p
    report("leave-one-dataset-out", scenes, preds)


if __name__ == "__main__":
    main()
