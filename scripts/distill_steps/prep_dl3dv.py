"""Fetch a frame subset of DL3DV-10K scenes (Hugging Face `DL3DV/DL3DV-ALL-960P`, gated: accept the terms and run
`.venv/bin/hf auth login` once) without downloading whole zips: HTTP range reads of the remote zip (as prep_arkit.py),
`frames` evenly spaced frames of `images_4/` (960x540 PNG) re-encoded as JPEG q95, plus `transforms.json`
(COLMAP intrinsics / poses, unused by self-distillation).

  .venv/bin/python scripts/distill_steps/prep_dl3dv.py --n 1000 --frames 120 --out workspace/distill_data/dl3dv_frames
Scenes are drawn (seeded) from the subsets given by --subsets. DL3DV is not part of the DA3 benchmark.
"""
import argparse
import concurrent.futures as cf
import io
import os
import random
import sys
import time
import zipfile

import requests
from huggingface_hub import HfApi, get_token
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_arkit import HTTPFile  # noqa: E402

REPO = "DL3DV/DL3DV-ALL-960P"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/main"


class SignedFile(HTTPFile):
    """HTTPFile on the CDN URL that huggingface.co redirects to (resolved once per scene, signed for ~1 h)."""

    def __init__(self, url, s):
        r = s.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.url, self.s, self.pos, self.size = r.url, s, 0, int(r.headers["Content-Length"])


def fetch(path, out, frames, token):
    vid = os.path.basename(path)[:-4]
    d = os.path.join(out, vid)
    if os.path.exists(os.path.join(d, "DONE")):
        return vid, "exists", 0
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    try:
        z = zipfile.ZipFile(io.BufferedReader(SignedFile(f"{URL}/{path}", s), buffer_size=1 << 20))
        names = sorted(n for n in z.namelist() if "/images_4/" in n and n.endswith(".png"))
        if len(names) < 48:
            return vid, f"skip ({len(names)} frames)", 0
        idx = sorted(set(round(i * (len(names) - 1) / (frames - 1)) for i in range(min(frames, len(names)))))
        os.makedirs(d, exist_ok=True)
        nb = 0
        for i in idx:
            b = z.read(names[i])
            nb += len(b)
            Image.open(io.BytesIO(b)).convert("RGB").save(os.path.join(d, os.path.basename(names[i])[:-4] + ".jpg"),
                                                          quality=95)
        tj = [n for n in z.namelist() if n.endswith("transforms.json")]
        if tj:
            open(os.path.join(d, "transforms.json"), "wb").write(z.read(tj[0]))
    except Exception as e:  # noqa: BLE001  (network errors skip the scene; no DONE, so a rerun retries it)
        return vid, f"skip ({type(e).__name__}: {e})", 0
    open(os.path.join(d, "DONE"), "w").write(f"{len(idx)} of {len(names)}\n")
    return vid, f"{len(idx)} of {len(names)} frames", nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--subsets", default="1K,2K", help="DL3DV batches to draw scenes from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    token = get_token()
    if not token:
        sys.exit("not logged in: run .venv/bin/hf auth login")
    api = HfApi(token=token)
    paths = sorted(f.path for sub in a.subsets.split(",")
                   for f in api.list_repo_tree(REPO, path_in_repo=sub, repo_type="dataset") if f.path.endswith(".zip"))
    random.Random(a.seed).shuffle(paths)
    os.makedirs(a.out, exist_ok=True)
    got, k, nb, t0 = 0, 0, 0, time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        while got < a.n and k < len(paths):
            batch = paths[k: k + (a.n - got)]
            k += len(batch)
            for vid, msg, b in ex.map(lambda p: fetch(p, a.out, a.frames, token), batch):
                got += not msg.startswith("skip")
                nb += b
                print(f"[{got}/{a.n}] {vid[:12]}: {msg} | {nb / 1e9:.1f} GB in {(time.time() - t0) / 60:.1f} min",
                      flush=True)


if __name__ == "__main__":
    main()
