"""Track BA (scripts/run_tto_ba.py) with pycolmap's bundle adjustment replaced by ba_lm, stopped after BA_MAX_ITERS iterations.

Same arguments as run_tto_ba.py. BA_MAX_ITERS=k (default 100), BA_MODULE names the module to wrap (default run_tto_ba),
BA_PRIOR=w anchors the poses to the initial ones (default 0 = none), BA_ITER_LOG=<file> receives one JSON line per BA call (iterations, seconds, cost before and after).
Only the Cauchy and the trivial loss, a shared or per-image focal length and the radial parameter are supported (as run_tto_ba.py uses them).
"""
import importlib
import json
import os
import sys
import time

import numpy as np
import pycolmap
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ba_lm as B  # noqa: E402

R = importlib.import_module(os.environ.get("BA_MODULE", "run_tto_ba"))
K = int(os.environ.get("BA_MAX_ITERS", "100"))
LOG = os.environ.get("BA_ITER_LOG", "")
PRIOR = float(os.environ.get("BA_PRIOR", "0"))  # weight of the prior on the initial poses (rotation per rad, translation per camera spread)


def state(rec):
    """Problem state: camera model and parameters, poses (cam_from_world 3x4), points and observations (image, point, u, v)."""
    img_ids = sorted(rec.images)
    pts = sorted(rec.points3D)
    img_index = {i: k for k, i in enumerate(img_ids)}
    pt_index = {p: k for k, p in enumerate(pts)}
    obs = []
    for p in pts:
        for el in rec.points3D[p].track.elements:
            xy = rec.images[el.image_id].points2D[el.point2D_idx].xy
            obs.append((img_index[el.image_id], pt_index[p], xy[0], xy[1]))
    cams = sorted(rec.cameras)
    return dict(
        img_ids=np.array(img_ids), pt_ids=np.array(pts),
        cam_of_img=np.array([rec.images[i].camera_id for i in img_ids]),
        poses=np.stack([rec.images[i].cam_from_world.matrix() for i in img_ids]),
        xyz=np.stack([rec.points3D[p].xyz for p in pts]),
        obs=np.array(obs, np.float64),
        cam_ids=np.array(cams),
        cam_params=np.array([rec.cameras[c].params for c in cams], np.float64),
        cam_model=np.array([str(rec.cameras[c].model) for c in cams]),
    )


def orthonormal(M):
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def bundle_adjustment(rec, opts):
    if not opts.refine_focal_length:
        print("warning: refine_focal=false is not supported by ba_lm, the focal length is refined", file=sys.stderr)
    st = state(rec)
    loss = opts.loss_function_type.name
    problem = {f"pre_{k}": v for k, v in st.items()}
    problem["loss"] = np.array([loss, float(opts.loss_function_scale) if loss == "CAUCHY" else 1e6], dtype=object)
    s = B.load(problem)
    t0 = time.perf_counter()
    hist, s2 = B.solve(s, max_iters=K, prior=(PRIOR, PRIOR))
    for k, i in enumerate(st["img_ids"]):
        rec.images[int(i)].cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(orthonormal(s2["R"][k].cpu().numpy())), s2["t"][k].cpu().numpy())
    for c, cid in enumerate(st["cam_ids"]):
        params = np.array(rec.cameras[int(cid)].params, np.float64)
        params[0] = float(s2["f"][c])
        if len(params) > 3:
            params[3] = float(s2["k"][c])
        rec.cameras[int(cid)].params = params
    X = s2["X"].cpu().numpy()
    for k, p in enumerate(st["pt_ids"]):
        rec.points3D[int(p)].xyz = X[k]
    if LOG:
        with open(LOG, "a") as f:
            f.write(json.dumps({"frames": len(st["img_ids"]), "points": len(st["pt_ids"]), "iterations": K, "prior": PRIOR, "seconds": time.perf_counter() - t0, "rms_start": hist[0]["rms"], "rms_end": hist[-1]["rms"]}) + "\n")


pycolmap.bundle_adjustment = bundle_adjustment

if __name__ == "__main__":
    R.main()
