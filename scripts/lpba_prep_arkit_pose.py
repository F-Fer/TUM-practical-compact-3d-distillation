"""ARKitScenes clips with LiDAR depth and poses, for training DA3's depth head with GT (scripts/lpba_ft_head.py).

    .venv/bin/python scripts/lpba_prep_arkit_pose.py --meta workspace/distill/arkit_raw_metadata.csv \
        --out workspace/tto/DA3-SMALL/lpba/arkit --n 300 [--fold Training] [--windows 2] [--frames 24] [--stride 8]

Per video (raw fold, seeded order): `windows` windows of `frames` vga_wide frames (640 x 480) taken every `stride`-th
frame of those that have a LiDAR depth map with the same timestamp; per frame the RGB (JPEG q95), the depth (uint16 mm
png, 256 x 192, 0 where the confidence is < 2) and, per window, the intrinsics (vga) and the poses from
lowres_wide.traj (nearest timestamp; axis-angle + translation as in the file, converted by lpba_ft_head.py). HTTP range
reads of the remote zips (scripts/distill_steps/prep_arkit.py), no full download.
"""

import argparse
import concurrent.futures as cf
import io
import os
import random
import sys
import zipfile

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "distill_steps"))
from prep_arkit import URL, HTTPFile  # noqa: E402


def zopen(s, fold, vid, name):
    f = io.BufferedReader(HTTPFile(f"{URL}/{fold}/{vid}/{name}", s), buffer_size=1 << 18)
    return zipfile.ZipFile(f)


def fetch(vid, fold, out, windows, frames, stride, seed):
    done = os.path.join(out, f"{vid}_DONE")
    if os.path.exists(done):
        return vid, "exists"
    s = requests.Session()
    try:
        zr, zd, zc, zi = (zopen(s, fold, vid, n) for n in ("vga_wide.zip", "lowres_depth.zip", "confidence.zip", "vga_wide_intrinsics.zip"))
        traj = s.get(f"{URL}/{fold}/{vid}/lowres_wide.traj", timeout=60)
        traj.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return vid, f"skip ({e})"
    ts = lambda n: os.path.splitext(os.path.basename(n).rsplit("_", 1)[1])[0]  # noqa: E731
    rgb = {ts(n): n for n in zr.namelist() if n.endswith(".png")}
    dep = {ts(n): n for n in zd.namelist() if n.endswith(".png")}
    con = {ts(n): n for n in zc.namelist() if n.endswith(".png")}
    intr = {ts(n): n for n in zi.namelist() if n.endswith(".pincam")}
    # RGB frames with a depth map within 10 ms (the timestamps of the two streams don't always coincide)
    dts = np.array(sorted(float(t) for t in dep if t in con))
    dkey = {float(t): t for t in dep}
    pair = {}
    for t in rgb:
        if t not in intr or not len(dts):
            continue
        j = int(np.argmin(np.abs(dts - float(t))))
        if abs(dts[j] - float(t)) < 0.01:
            pair[t] = dkey[dts[j]]
    common = sorted(pair)
    span = (frames - 1) * stride + 1
    if len(common) < span + 10:
        return vid, f"skip ({len(common)} frames)"
    T = np.array([[float(x) for x in l.split()] for l in traj.text.strip().splitlines()])
    rng = random.Random(seed + int(vid))
    starts = sorted(rng.sample(range(0, len(common) - span), min(windows, len(common) - span)))
    n_ok = 0
    for w, st in enumerate(starts):
        sel = common[st : st + span : stride]
        d = os.path.join(out, f"{vid}_{w}")
        os.makedirs(d, exist_ok=True)
        poses, Ks = [], []
        for t in sel:
            Image.open(io.BytesIO(zr.read(rgb[t]))).convert("RGB").save(os.path.join(d, f"{t}.jpg"), quality=95)
            dm = np.array(Image.open(io.BytesIO(zd.read(dep[pair[t]]))))
            cm = np.array(Image.open(io.BytesIO(zc.read(con[pair[t]]))))
            dm[cm < 2] = 0
            Image.fromarray(dm.astype(np.uint16)).save(os.path.join(d, f"{t}_depth.png"))
            _, _, fx, fy, cx, cy = [float(x) for x in zi.read(intr[t]).decode().split()]
            Ks.append([fx, fy, cx, cy])
            j = int(np.argmin(np.abs(T[:, 0] - float(t))))
            poses.append(T[j, 1:7])
        np.savez(os.path.join(d, "meta.npz"), ts=np.array(sel), K=np.array(Ks), traj=np.array(poses))
        n_ok += 1
    open(done, "w").write(f"{n_ok}\n")
    return vid, f"{n_ok} windows of {len(common)} frames"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--fold", default="Training")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = [l.strip().split(",") for l in open(a.meta).readlines()[1:]]
    vids = sorted(r[0] for r in rows if r[3] == a.fold)
    random.Random(a.seed).shuffle(vids)
    os.makedirs(a.out, exist_ok=True)
    got, k = 0, 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        while got < a.n and k < len(vids):
            batch = vids[k : k + (a.n - got)]
            k += len(batch)
            for vid, msg in ex.map(lambda v: fetch(v, a.fold, a.out, a.windows, a.frames, a.stride, a.seed), batch):
                got += not msg.startswith("skip")
                print(f"[{got}/{a.n}] {vid}: {msg}", flush=True)


if __name__ == "__main__":
    main()
