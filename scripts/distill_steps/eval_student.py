"""Benchmark a modified DéjàView through run_bench.py (DéjàView adapter, native 504-px preprocessing). Env options:

  DVLT_K=4          loop steps (linspace grid), default 12
  DVLT_CKPT=...pt   replace all weights by a train.py checkpoint
  DVLT_ENC=...pt    replace the image encoder by a train_encoder.py student (ViT-S + projection)
  DVLT_RES=336      run the network at this longest side (input resized from the 504-px frame with antialiasing),
                    then upsample depth / confidence back to the 504-px frame and rescale the intrinsics, so the
                    evaluator and run_tto_pba.py (which works at the prediction resolution) see 504-px predictions
  DVLT_ROPE=1       with DVLT_RES: stretch the RoPE patch positions to span the native grid (rope_scale.py)

  DVLT_K=4 DVLT_CKPT=.../final.pt .venv/bin/python scripts/distill_steps/eval_student.py nvidia/dvlt \
      "eval.datasets=[hiroom,eth3d,dtu64,7scenes,scannetpp]" "eval.modes=[pose]" workspace.work_dir=...
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "distill_steps"))
sys.path.insert(0, os.path.join(REPO, "third_party", "dvlt", "src"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import baselines  # noqa: E402

K = int(os.environ.get("DVLT_K", "12"))
CKPT = os.environ.get("DVLT_CKPT", "")
ENC = os.environ.get("DVLT_ENC", "")
RES = int(os.environ.get("DVLT_RES", "0"))
ROPE = os.environ.get("DVLT_ROPE", "") not in ("", "0")
if ROPE:
    import rope_scale

    rope_scale.install()
_init = baselines.DVLTBaseline.__init__
_forward = baselines.DVLTBaseline.forward


def init(self, model_id, inference_steps=None):
    _init(self, model_id, inference_steps=K)
    if CKPT:
        self.model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["model"], strict=True)
    if ENC:
        from train_encoder import load_student

        self.model.patch_embed_encoder = load_student(ENC)
    print(f"[eval_student] K={self.model.inference_steps} ckpt={CKPT or 'released'} enc={ENC or 'released'} "
          f"res={RES or 'native'} rope_scale={ROPE}", flush=True)


def forward(self, images):
    H, W = images.shape[-2:]
    if ROPE:
        rope_scale.SCALE[:] = [1.0, 1.0]
    if not RES or RES >= max(H, W):
        return _forward(self, images)
    from torchvision.transforms.functional import resize

    s = RES / max(H, W)
    h, w = round(H * s / 14) * 14, round(W * s / 14) * 14
    if ROPE:
        rope_scale.SCALE[:] = [(H // 14 - 1) / max(h // 14 - 1, 1), (W // 14 - 1) / max(w // 14 - 1, 1)]
    out = _forward(self, resize(images[0], [h, w], antialias=True)[None])
    up = lambda t: F.interpolate(t.float()[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
    Kmat = out["intrinsics"].float().clone()
    sx, sy = W / w, H / h
    Kmat[:, 0, 0] *= sx
    Kmat[:, 1, 1] *= sy
    Kmat[:, 0, 2] = (Kmat[:, 0, 2] + 0.5) * sx - 0.5
    Kmat[:, 1, 2] = (Kmat[:, 1, 2] + 0.5) * sy - 0.5
    return dict(depth=up(out["depth"]), conf=up(out["conf"]), extrinsics=out["extrinsics"], intrinsics=Kmat)


baselines.DVLTBaseline.__init__ = init
baselines.DVLTBaseline.forward = forward
import run_bench  # noqa: E402

run_bench.main()
