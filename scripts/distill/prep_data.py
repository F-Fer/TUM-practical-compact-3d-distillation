"""Cache the ScanNet++ benchmark iPhone frames at the models' input resolution (504x378).

Images are resized from the raw (distorted) 1920x1440 frames with cv2.INTER_AREA, like DA3's
input processor does for the benchmark (the evaluator passes the raw files to the model).
GT depth (render_depth, mm) and GT COLMAP poses are cached too: used only by the GT-supervised
baseline and for validation metrics.

    uv run scripts/distill/prep_data.py   -> workspace/distill/cache/<scene>.npz
"""

import os
import sys
from multiprocessing import Pool

import cv2
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE_DIR, IMG_H, IMG_W, REPO_ROOT, SCANNETPP_ROOT, ALL_SCENES  # noqa: E402


def prep(scene):
    from depth_anything_3.utils.read_write_model import read_model

    out_path = os.path.join(CACHE_DIR, f"{scene}.npz")
    if os.path.exists(out_path):
        return scene, "cached"
    root = os.path.join(SCANNETPP_ROOT, scene, "merge_dslr_iphone")
    cams, images, _ = read_model(os.path.join(root, "colmap/sparse_render_rgb"))
    imgs, depths, exts, ixts, names = [], [], [], [], []
    for image in sorted(images.values(), key=lambda im: im.name):
        if "iphone" not in image.name:
            continue
        path = os.path.join(root, "images", image.name)
        dpath = os.path.join(root, "render_depth", os.path.basename(image.name)[:-4] + ".png")
        if not os.path.exists(path):
            continue
        img = imageio.imread(path)
        h, w = img.shape[:2]
        imgs.append(cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA))
        if os.path.exists(dpath):
            d = imageio.imread(dpath).astype(np.float32) / 1000.0
            depths.append(cv2.resize(d, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST).astype(np.float16))
        else:
            depths.append(np.zeros((IMG_H, IMG_W), np.float16))
        ext = np.eye(4, dtype=np.float32)
        ext[:3, :3] = image.qvec2rotmat()
        ext[:3, 3] = image.tvec
        cam = cams[image.camera_id]
        K = np.eye(3, dtype=np.float32)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = cam.params[:4]
        K[0] *= IMG_W / w
        K[1] *= IMG_H / h
        exts.append(ext)
        ixts.append(K)
        names.append(image.name)
    np.savez(
        out_path,
        images=np.stack(imgs),
        gt_depth=np.stack(depths),
        extrinsics=np.stack(exts),
        intrinsics=np.stack(ixts),
        names=np.array(names),
    )
    return scene, len(imgs)


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with Pool(8) as pool:
        for scene, n in pool.imap_unordered(prep, ALL_SCENES):
            print(scene, n, flush=True)
