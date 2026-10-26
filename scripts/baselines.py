"""Non-DA3 models (VGGT, Pi3, DéjàView) behind the interface the DA3 evaluator calls.

The evaluator only calls api.inference(image_files, [extrinsics, intrinsics], export_dir=...) and
scores exports/mini_npz/results.npz, of which it reads depth, extrinsics (world-to-camera, 3x4)
and intrinsics, all at the resolution of the depth maps (it rescales them to the original
images itself, separately in x and y). Each adapter runs its model with the model's own
preprocessing and writes that file in DA3's format.

None of these models takes camera poses as input. In posed mode (recon_posed) the evaluator
fuses the predicted depths with the GT poses after aligning the predicted poses to them, so the
GT extrinsics/intrinsics passed to inference() are ignored, as in the DA3 paper's baselines.

    load("facebook/VGGT-1B" | "yyfz233/Pi3" | "nvidia/dvlt") -> adapter
"""

import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Baseline:
    """Common inference()/export; subclasses implement preprocess() and forward()."""

    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    @torch.no_grad()
    def inference(self, image, extrinsics=None, intrinsics=None, export_dir=None, export_format="mini_npz", **kwargs):
        images = self.preprocess(list(image)).to(self.device)  # [1, S, 3, H, W] in [0, 1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.forward(images)
        out = {k: v.float().cpu().numpy() for k, v in out.items()}
        if export_dir is not None:
            if export_format != "mini_npz":
                print(f"[baselines] only mini_npz is exported, not {export_format}")
            path = os.path.join(export_dir, "exports", "mini_npz", "results.npz")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # same rounding as depth_anything_3.utils.export.npz.export_to_mini_npz
            np.savez_compressed(
                path,
                depth=np.round(out["depth"], 8),
                conf=np.round(out["conf"], 2),
                extrinsics=out["extrinsics"],
                intrinsics=out["intrinsics"],
            )
        return out


class VGGTBaseline(Baseline):
    """facebook/VGGT-1B. The aggregator runs with bf16 weights (heads in fp32, as upstream) and
    vggt_lowmem offloads its cached layers; together they make 100 frames fit in 10 GB."""

    def __init__(self, model_id):
        from vggt.models.vggt import VGGT

        import vggt_lowmem

        vggt_lowmem.install()
        self.model = VGGT.from_pretrained(model_id).eval()
        self.model.point_head = None  # world points aren't used; depth + cameras are
        self.model.aggregator.to(torch.bfloat16)

    def preprocess(self, image_files):
        from vggt.utils.load_fn import load_and_preprocess_images

        return load_and_preprocess_images(image_files)[None]  # width 518, crop mode (no crop for landscape)

    def forward(self, images):
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        pred = self.model(images)
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
        return dict(
            depth=pred["depth"][0, ..., 0],
            conf=pred["depth_conf"][0],
            extrinsics=extrinsics[0],
            intrinsics=intrinsics[0],
        )


class Pi3Baseline(Baseline):
    """yyfz233/Pi3. Encoder and decoder run with bf16 weights (heads in fp32, as upstream).
    Pi3 predicts no intrinsics; they are fitted to its local point maps with the repo's own
    least-squares helper (principal point at the image center)."""

    PIXEL_LIMIT = 255000  # default of pi3.utils.basic.load_images_as_tensor

    def __init__(self, model_id):
        from pi3.models.pi3 import Pi3

        self.model = Pi3.from_pretrained(model_id).eval()
        for module in (self.model.encoder, self.model.decoder):
            module.to(torch.bfloat16)

    def preprocess(self, image_files):
        # pi3.utils.basic.load_images_as_tensor only takes a directory, so its resize is repeated
        # here: one target size from the first image, <= PIXEL_LIMIT pixels, multiple of 14
        from torchvision.transforms.functional import to_tensor

        images = [Image.open(f).convert("RGB") for f in image_files]
        w_orig, h_orig = images[0].size
        scale = math.sqrt(self.PIXEL_LIMIT / (w_orig * h_orig))
        w_target, h_target = w_orig * scale, h_orig * scale
        k, m = round(w_target / 14), round(h_target / 14)
        while (k * 14) * (m * 14) > self.PIXEL_LIMIT:
            if k / m > w_target / h_target:
                k -= 1
            else:
                m -= 1
        size = (max(1, k) * 14, max(1, m) * 14)
        return torch.stack([to_tensor(img.resize(size, Image.Resampling.LANCZOS)) for img in images])[None]

    def forward(self, images):
        from pi3.utils.geometry import recover_intrinsic_from_rays_d

        pred = self.model(images)
        local_points = pred["local_points"][0]
        intrinsics = recover_intrinsic_from_rays_d(local_points, force_center_principal_point=True)
        return dict(
            depth=local_points[..., 2],
            conf=torch.sigmoid(pred["conf"][0, ..., 0]),
            extrinsics=torch.linalg.inv(pred["camera_poses"][0])[:, :3],
            intrinsics=intrinsics,
        )


class DVLTBaseline(Baseline):
    """nvidia/dvlt (DéjàView), K = 12 refinement steps (the checkpoint's default).

    Not installed as a package: its pyproject pins numpy 2.x (DA3 needs numpy < 2), so its src/
    is put on sys.path. The model is built from the checkpoint's config.json, with the decoder
    run over chunks of 8 frames instead of 128 (bit-identical outputs; 100 frames need 3.9 GiB
    instead of running out of memory)."""

    def __init__(self, model_id, inference_steps=12):
        sys.path.insert(0, os.path.join(REPO_ROOT, "third_party", "dvlt", "src"))
        from accelerate import PartialState
        from dvlt.model.dvlt.model import DVLT
        from huggingface_hub import hf_hub_download

        PartialState()  # dvlt logs through accelerate's logger, which needs an initialized state
        with open(hf_hub_download(model_id, "config.json")) as f:
            config = json.load(f)
        config.update(decode_chunk_size=8, inference_steps=inference_steps)
        wrapper = DVLT(**config)
        wrapper.load_pretrained(model_id, strict=True)
        self.model = wrapper.model.eval()
        self.patch_size = config["patch_size"]
        self.img_size = config["img_size"]

    def preprocess(self, image_files):
        from dvlt.util.preprocess import preprocess_images

        batch = preprocess_images([Image.open(f) for f in image_files], self.img_size, self.patch_size, "cpu")
        return batch["images"]  # longest side 504, center crop to multiples of 14

    def forward(self, images):
        from dvlt.common.rays import rays_to_pose

        H, W = images.shape[-2:]
        pred = self.model({"images": images})
        # as DVLT._postprocess_predictions: autocast off (@force_fp32), uniform weights
        # (default use_depth_conf_for_pose=False)
        with torch.autocast("cuda", enabled=False):
            c2w, intrinsics = rays_to_pose(pred["rays"], torch.ones_like(pred["depth_conf"]), H, W, self.patch_size)
        return dict(
            depth=pred["depth"][0, ..., 0],
            conf=pred["depth_conf"][0],
            extrinsics=torch.linalg.inv(c2w[0].float())[:, :3],
            intrinsics=intrinsics[0],
        )


MODELS = {
    "facebook/VGGT-1B": VGGTBaseline,
    "yyfz233/Pi3": Pi3Baseline,
    "nvidia/dvlt": DVLTBaseline,
}


def load(model_id):
    return MODELS[model_id](model_id)
