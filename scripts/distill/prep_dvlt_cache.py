"""Cache ScanNet++ frames exactly as DéjàView's preprocess_images produces them (torchvision
bilinear+antialias resize of the float image, center crop to multiples of 14, no uint8 rounding),
as float16 [N,3,378,504] in [0,1]. DéjàView at K < 8 is sensitive enough that the cv2 uint8
cache shifts its pose AUC@3 by ~10 points. -> workspace/distill/cache_dvlt/<scene>.npy"""

import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ALL_SCENES, REPO_ROOT, SCANNETPP_ROOT, WS, load_scene  # noqa: E402

OUT = os.path.join(WS, "cache_dvlt")


def prep(scene):
    import torch

    torch.set_num_threads(2)
    sys.path.insert(0, os.path.join(REPO_ROOT, "third_party", "dvlt", "src"))
    from dvlt.util.preprocess import preprocess_images

    path = os.path.join(OUT, f"{scene}.npy")
    if os.path.exists(path):
        return scene, "cached"
    names = load_scene(scene)["names"]
    root = os.path.join(SCANNETPP_ROOT, scene, "merge_dslr_iphone", "images")
    out = []
    for i in range(0, len(names), 16):
        imgs = [Image.open(os.path.join(root, n)) for n in names[i:i + 16]]
        out.append(preprocess_images(imgs, 504, 14, "cpu")["images"][0].half().numpy())
    np.save(path, np.concatenate(out))
    return scene, len(names)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with Pool(6) as pool:
        for scene, n in pool.imap_unordered(prep, ALL_SCENES):
            print(scene, n, flush=True)
