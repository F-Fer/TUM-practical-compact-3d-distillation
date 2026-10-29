"""Cache extra *unlabeled* image sequences (no GT is stored or used) for the data-scaling test:
7-Scenes seq-01 of each scene (every 2nd frame) and the ETH3D DSLR images, center-cropped to 4:3
and resized to 504x378 like the ScanNet++ cache. -> workspace/distill/cache_extra/<name>.npz"""

import glob
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_H, IMG_W, REPO_ROOT  # noqa: E402

BENCH = os.path.join(REPO_ROOT, "workspace", "benchmark_dataset")
OUT = os.path.join(REPO_ROOT, "workspace", "distill", "cache_extra")


def crop_43(img):
    h, w = img.shape[:2]
    if w * 3 > h * 4:  # too wide
        nw = h * 4 // 3
        img = img[:, (w - nw) // 2:(w - nw) // 2 + nw]
    elif w * 3 < h * 4:
        nh = w * 3 // 4
        img = img[(h - nh) // 2:(h - nh) // 2 + nh]
    return cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)


def prep(job):
    name, files = job
    path = os.path.join(OUT, f"{name}.npz")
    if not os.path.exists(path):
        imgs = [crop_43(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)) for f in files]
        np.savez(path, images=np.stack(imgs))
    return name, len(files)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for d in sorted(glob.glob(os.path.join(BENCH, "7scenes/7Scenes/*/seq-01"))):
        jobs.append((f"7scenes_{d.split('/')[-2]}", sorted(glob.glob(os.path.join(d, "*.color.png")))[::2]))
    for d in sorted(glob.glob(os.path.join(BENCH, "eth3d/*/images/dslr_images"))):
        files = sorted(f for f in glob.glob(os.path.join(d, "*.JPG")))
        jobs.append((f"eth3d_{d.split('/')[-3]}", files))
    with Pool(8) as pool:
        for name, n in pool.imap_unordered(prep, jobs):
            print(name, n, flush=True)
