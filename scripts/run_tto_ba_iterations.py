"""Track BA (scripts/run_tto_ba.py) that keeps the Ceres report and the state before and after every BA call.

Same arguments as run_tto_ba.py. BA_ITER_LOG=<file> receives one JSON line per BA call (frames, points, iterations, cost),
BA_DUMP_DIR=<dir> one npz per call with the problem before the BA (cameras, poses, points, observations) and pycolmap's
result, so that another solver can be started from the same problem. BA_MODULE names the module to wrap.
"""
import importlib
import json
import os
import re
import sys

import numpy as np
import pycolmap

R = importlib.import_module(os.environ.get("BA_MODULE", "run_tto_ba"))
LOG = os.environ.get("BA_ITER_LOG", "ba_iterations.jsonl")
DUMP = os.environ.get("BA_DUMP_DIR", "")
calls = 0


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
    cams = {c: (str(rec.cameras[c].model), np.array(rec.cameras[c].params, np.float64)) for c in rec.cameras}
    return dict(
        img_ids=np.array(img_ids),
        cam_of_img=np.array([rec.images[i].camera_id for i in img_ids]),
        poses=np.stack([rec.images[i].cam_from_world.matrix() for i in img_ids]),
        xyz=np.stack([rec.points3D[p].xyz for p in pts]),
        obs=np.array(obs, np.float64),
        cam_ids=np.array(sorted(cams)),
        cam_params=np.array([cams[c][1] for c in sorted(cams)]),
        cam_model=np.array([cams[c][0] for c in sorted(cams)]),
    )


def bundle_adjustment(rec, opts):
    global calls
    opts.print_summary = True
    print("BA_CALL", file=sys.stderr, flush=True)
    f = sys._getframe(1)  # the scene being processed is a local of the caller's loop
    while f is not None and "scene" not in f.f_locals:
        f = f.f_back
    name = f"{f.f_locals.get('dataset')}__{str(f.f_locals.get('scene')).replace('/', '_')}" if f is not None else f"call{calls:03d}"
    pre = state(rec) if DUMP else None
    original(rec, opts)
    if DUMP:
        os.makedirs(DUMP, exist_ok=True)
        post = state(rec)
        np.savez_compressed(f"{DUMP}/{name}.npz", **{f"pre_{k}": v for k, v in pre.items()}, **{f"post_{k}": v for k, v in post.items()},
                            loss=np.array([opts.loss_function_type.name, opts.loss_function_scale], dtype=object))
    with open(LOG, "a") as f:
        f.write(json.dumps({"call": calls, "scene": name, "frames": rec.num_reg_images(), "points": rec.num_points3D()}) + "\n")
    calls += 1


original = pycolmap.bundle_adjustment
try:  # make the solver report visible on stderr
    pycolmap.logging.minloglevel = 0
    pycolmap.logging.alsologtostderr = True
except Exception as e:
    print("logging switches not available:", e, file=sys.stderr)
pycolmap.bundle_adjustment = bundle_adjustment

if __name__ == "__main__":
    R.main()
