"""Run ba_lm on every dumped BA problem of a model and score the poses of every iteration against the ground truth.

Usage: ba_iteration_curves.py <model> [max_iters] [scene substring ...]     (run from the repository root; writes workspace/ba_iters/<model>_curves/)
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ba_lm as B  # noqa: E402
from depth_anything_3.bench.utils import compute_pose  # noqa: E402

model = sys.argv[1]
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 100
only = sys.argv[3:]
torch.set_num_threads(int(os.environ.get("THREADS", "12")))
out_dir = f"workspace/ba_iters/{model}_curves"
os.makedirs(out_dir, exist_ok=True)


def scene_of(name):
    ds, sc = name.split("__", 1)
    return ds, ("/".join(sc.split("_", 2)) if ds == "hiroom" else sc)


def auc(R, t, gt):
    R, t = R.cpu(), t.cpu()
    ext = torch.cat([torch.cat([R, t[:, :, None]], 2), torch.tensor([0, 0, 0, 1.0], dtype=R.dtype).expand(len(R), 1, 4)], 1).float()
    m = compute_pose(ext, torch.from_numpy(gt).float())
    return 100 * float(m["auc03"]), 100 * float(m["auc30"])


for path in sorted(glob.glob(f"workspace/ba_iters/{model}/*.npz"), key=os.path.getsize):  # small problems first
    name = os.path.basename(path)[:-4]
    out = f"{out_dir}/{name}.json"
    if os.path.exists(out) or (only and not any(o in name for o in only)):
        continue
    ds, sc = scene_of(name)
    print(f"start {name} {time.strftime('%H:%M:%S')}", flush=True)
    gtf = f"workspace/evaluation/{model}/model_results/{ds}/{sc}/unposed/exports/gt_meta.npz"
    gt = np.load(gtf, allow_pickle=True)["extrinsics"].astype(np.float64)
    d = np.load(path, allow_pickle=True)
    s = B.load(path)
    t0 = time.time()
    hist, _ = B.solve(s, max_iters=iters)
    rows = []
    for h in hist:  # the poses are scored at every iteration up to 30, then every 5th (scoring 100 frames is slow)
        scored = h["it"] <= 30 or h["it"] % 5 == 0
        a3, a30 = auc(h["R"], h["t"], gt) if scored else (None, None)
        rows.append(dict(it=h["it"], rms=h["rms"], auc3=a3, auc30=a30, accepted=h["accepted"], time=h["time"], f=float(h["f"][0])))
    p3, p30 = auc(torch.from_numpy(d["post_poses"][:, :, :3]).to(B.DT), torch.from_numpy(d["post_poses"][:, :, 3]).to(B.DT), gt)
    json.dump(dict(scene=name, frames=len(gt), obs=len(s["img"]), points=len(s["X"]), pycolmap=dict(auc3=p3, auc30=p30, f=float(d["post_cam_params"][0][0])), rows=rows), open(out, "w"))
    print(f"{name[:44]:44s} frames {len(gt):3d} obs {len(s['img']):6d}  {time.time() - t0:6.1f} s  AUC@3 it0 {rows[0]['auc3']:5.1f} it10 {rows[min(10, len(rows)-1)]['auc3']:5.1f} it{rows[-1]['it']} {rows[-1]['auc3']:5.1f} | pycolmap {p3:5.1f}", flush=True)
