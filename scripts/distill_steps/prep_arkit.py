"""Fetch a frame subset of ARKitScenes raw `vga_wide` videos without downloading whole zips (HTTP range reads of
the remote zip), re-encoded as JPEG q95, plus `lowres_wide.traj` (GT poses, unused by self-distillation).

  .venv/bin/python scripts/distill_steps/prep_arkit.py --n 150 --frames 240 --out workspace/distill_data/arkit_frames
Videos are drawn (seeded) from the Training fold of raw/metadata.csv; --fold Validation for held-out clips.
None of the DA3 benchmark datasets is ARKitScenes, so there is no overlap with the evaluation scenes.
"""
import argparse
import concurrent.futures as cf
import io
import os
import random
import zipfile

import requests
from PIL import Image

URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw"


class HTTPFile(io.RawIOBase):
    def __init__(self, url, s):
        self.url, self.s, self.pos = url, s, 0
        r = s.head(url, timeout=30)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        for _ in range(5):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException:
                continue
        else:
            raise IOError(self.url)
        self.pos += len(r.content)
        return r.content

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def fetch(vid, fold, out, frames):
    d = os.path.join(out, vid)
    if os.path.exists(os.path.join(d, "DONE")):
        return vid, "exists"
    s = requests.Session()
    try:
        f = io.BufferedReader(HTTPFile(f"{URL}/{fold}/{vid}/vga_wide.zip", s), buffer_size=1 << 20)
        z = zipfile.ZipFile(f)
    except Exception as e:  # noqa: BLE001
        return vid, f"skip ({e})"
    names = sorted(n for n in z.namelist() if n.endswith(".png"))
    if len(names) < 64:
        return vid, f"skip ({len(names)} frames)"
    idx = sorted(set(round(i * (len(names) - 1) / (frames - 1)) for i in range(min(frames, len(names)))))
    os.makedirs(d, exist_ok=True)
    for i in idx:
        n = names[i]
        img = Image.open(io.BytesIO(z.read(n))).convert("RGB")
        img.save(os.path.join(d, os.path.basename(n)[:-4] + ".jpg"), quality=95)
    r = s.get(f"{URL}/{fold}/{vid}/lowres_wide.traj", timeout=60)
    if r.ok:
        open(os.path.join(d, "lowres_wide.traj"), "wb").write(r.content)
    open(os.path.join(d, "DONE"), "w").write(f"{len(idx)} of {len(names)}\n")
    return vid, f"{len(idx)} of {len(names)} frames"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--fold", default="Training")
    ap.add_argument("--meta", required=True, help="raw/metadata.csv")
    ap.add_argument("--out", required=True)
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
            for vid, msg in ex.map(lambda v: fetch(v, a.fold, a.out, a.frames), batch):
                ok = not msg.startswith("skip")
                got += ok
                print(f"[{got}/{a.n}] {vid}: {msg}", flush=True)


if __name__ == "__main__":
    main()
