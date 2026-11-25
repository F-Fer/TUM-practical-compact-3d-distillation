"""Interpolate the ARKit poses of scripts/lpba_prep_arkit_pose.py windows at the frames' timestamps.

    .venv/bin/python scripts/gmr_arkit_interp.py <arkit dir> [--fold Training]

lpba_prep_arkit_pose.py stores, per frame, the nearest row of lowres_wide.traj. That trajectory is sampled at 10 Hz and
the vga_wide frames lie up to 33 ms from it (checked on 40958756: offsets -33 / +33 / 0 ms in turn; median rotation
3.9 deg per trajectory step), i.e. ~1.3 deg of rotation, ~10 px at 640 px: too wrong for a matching target. Here the
trajectory is fetched again and each frame's world-to-camera pose is interpolated (rotation by slerp, camera centre
linearly); meta.npz gets `traj` replaced (same layout: axis-angle + translation, world-to-camera) and the old values
kept as `traj_nearest`. Windows already converted (with `traj_nearest`) are skipped.
"""

import argparse
import concurrent.futures as cf
import glob
import os
import sys

import numpy as np
import requests
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "distill_steps"))
from prep_arkit import URL  # noqa: E402


def interp(T, ts):
    t = T[:, 0]
    R = Rotation.from_rotvec(T[:, 1:4])
    c = -np.einsum("nji,nj->ni", R.as_matrix(), T[:, 4:7])  # camera centres
    tq = np.clip(ts, t[0], t[-1])
    Rq = Slerp(t, R)(tq)
    cq = np.stack([np.interp(tq, t, c[:, k]) for k in range(3)], 1)
    tr = -np.einsum("nij,nj->ni", Rq.as_matrix(), cq)
    return np.concatenate([Rq.as_rotvec(), tr], 1), np.abs(ts - tq).max()


def fix(vid, dirs, fold):
    try:
        r = requests.get(f"{URL}/{fold}/{vid}/lowres_wide.traj", timeout=60)
        r.raise_for_status()
        T = np.array([[float(x) for x in l.split()] for l in r.text.strip().splitlines()])
    except Exception as e:  # noqa: BLE001
        return vid, f"failed {e!r}"
    msg = []
    for d in dirs:
        p = os.path.join(d, "meta.npz")
        m = dict(np.load(p))
        ts = m["ts"].astype(np.float64)
        new, clip = interp(T, ts)
        old = m["traj"]
        dang = np.degrees(np.linalg.norm((Rotation.from_rotvec(new[:, :3]) * Rotation.from_rotvec(old[:, :3]).inv()).as_rotvec(), axis=1))
        np.savez(p, **{**m, "traj": new, "traj_nearest": old})
        msg.append(f"{os.path.basename(d)}: change median {np.median(dang):.2f} max {dang.max():.2f} deg, clipped {1000 * clip:.0f} ms")
    return vid, "; ".join(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--fold", default="Training")
    a = ap.parse_args()
    by_vid = {}
    for d in sorted(glob.glob(os.path.join(a.root, "*_*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "meta.npz")) and "traj_nearest" not in np.load(os.path.join(d, "meta.npz")).files:
            by_vid.setdefault(os.path.basename(d).split("_")[0], []).append(d)
    with cf.ThreadPoolExecutor(8) as ex:
        for vid, msg in ex.map(lambda kv: fix(kv[0], kv[1], a.fold), by_vid.items()):
            print(f"{vid}: {msg}", flush=True)


if __name__ == "__main__":
    main()
