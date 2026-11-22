"""Depth oracle after checks 1-2 (reports/Learned refinement step between DA3 and DeepBA.md, §10): does PBA reach the
pose accuracy that the weak-direction oracle promises when the depth is right?

    .venv/bin/python scripts/lpba_depth_oracle.py <depth: da3 | gt> <src work dir> <dst work dir> [run_tto_pba overrides]

Runs scripts/run_tto_pba.py unchanged, except that each scene's predicted depth is replaced before the refinement:

    da3   DA3's depth (the control: same code path)
    gt    the dataset's GT depth at the prediction's resolution (nearest neighbour), scaled by one factor per scene to
          DA3's median depth (DA3's translations stay consistent with it); pixels without GT get DA3's depth scaled per
          frame to the GT and the lowest confidence, so they are never sampled (conf_quantile)

GT depth exists for HiRoom (16-bit PNG, /65535 * 100 m), 7-Scenes (PNG mm; registered to the depth camera, shifted
4-12 px against the RGB images, knowledge/evaluation.md), ScanNet++ (render_depth PNG mm) and ETH3D (float32 raw, mask
masks_for_images == 1 invalid); not for DTU-64, so pass datasets=hiroom,7scenes,scannetpp,eth3d. The per-scene log
also prints DA3's depth error against the GT (abs. rel. after the per-scene scale).
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_tto_pba as rp  # noqa: E402


def gt_depth(image_file, H, W):
    """GT depth [H, W] in metres (0 = invalid) of a benchmark image, at the prediction's resolution."""
    f = str(image_file)
    if "/hiroom/" in f:
        d = cv2.imread(f.replace("/image/", "/depth/").rsplit(".", 1)[0] + ".png", -1).astype(np.float32) / 65535.0 * 100.0
    elif "/7scenes/" in f.lower():
        d = cv2.imread(f.replace(".color.png", ".depth.png"), -1).astype(np.float32)
        d[d == 65535] = 0
        d = d / 1000.0
    elif "/scannetpp/" in f:
        name = os.path.basename(f).rsplit(".", 1)[0]
        root = f.split("/merge_dslr_iphone/")[0] + "/merge_dslr_iphone"
        d = cv2.imread(os.path.join(root, "render_depth", name + ".png"), -1).astype(np.float32) / 1000.0
    elif "/eth3d/" in f:
        scene_root, name = f.split("/images/dslr_images/")
        raw = np.fromfile(os.path.join(scene_root, "ground_truth_depth", "dslr_images", name), dtype=np.float32)
        h = int(round(np.sqrt(raw.size * H / W)))
        d = raw.reshape(h, raw.size // h).copy()
        mask = cv2.imread(os.path.join(scene_root, "masks_for_images", "dslr_images", name.rsplit(".", 1)[0] + ".png"), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            d[mask == 1] = 0
    else:
        raise ValueError(f"no GT depth for {f}")
    d[~np.isfinite(d)] = 0
    return cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)


def main():
    mode, args = sys.argv[1], sys.argv[2:]
    assert mode in ("da3", "gt"), mode
    refine = rp.refine_scene

    def patched(image_files, pred, cfg, device):
        da3 = pred["depth"].astype(np.float32)
        N, H, W = da3.shape
        gt = np.stack([gt_depth(f, H, W) for f in image_files])
        valid = gt > 0
        s = np.median(da3[valid]) / np.median(gt[valid])
        rel = np.abs(da3[valid] / s - gt[valid]) / gt[valid]
        print(f"[depth_oracle] valid {valid.mean():.2f} DA3 abs.rel. (scene scale) {rel.mean():.3f} median {np.median(rel):.3f}")
        if mode == "gt":
            depth = gt * s
            for i in range(N):  # invalid pixels: DA3's depth scaled per frame, never sampled
                v = valid[i]
                k = np.median(depth[i][v] / da3[i][v]) if v.sum() > 100 else 1.0
                depth[i][~v] = da3[i][~v] * k
            conf = np.where(valid, pred["conf"], pred["conf"].min() - 1).astype(np.float32)
            pred = {**pred, "depth": depth.astype(np.float32), "conf": conf}
        return refine(image_files, pred, cfg, device)

    rp.refine_scene = patched
    sys.argv = [rp.__file__] + args
    rp.main()


if __name__ == "__main__":
    main()
