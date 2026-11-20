"""Run released DA3 models at a lower input resolution, optionally with RoPE positions stretched to the native grid.

DA3's backbone (DINOv2 + alternating frame/global attention) applies VGGT-style 2D RoPE from `rope_start` on: frame
attention gets integer patch coordinates (1..W/14, camera token 0), global attention gets `pos_nodiff` (all patches 1,
so no relative position). The learned DINOv2 position embedding is interpolated to the grid by upstream code.

install() patches (both only active when RES is set):
- `DepthAnything3.forward`: the 504-px batch from the API is resized (antialiased) to longest side RES (multiple of
  14), the model runs there, depth / conf (and any other HxW map) are upsampled bilinearly back to the 504-px frame and
  the intrinsics scaled (principal point stays at the centre), so the evaluator, the exports and run_tto_pba.py see 504-px predictions (as eval_student.py
  does for DéjàView). run_bench.py's timing hook wraps this, so forward_s includes resize and upsampling.
- with ROPE: `DinoVisionTransformer._prepare_rope` returns float positions idx * scale + 1 with
  scale = (native_n - 1) / (n - 1) per axis, and `RotaryPositionEmbedding2D.forward` evaluates float positions with
  the same bf16 einsum arithmetic as the cached integer table (bit-identical at scale 1, see da3_res_check.py).

  DA3_RES=336 DA3_ROPE=1 .venv/bin/python scripts/distill_steps/da3_res.py depth-anything/DA3-SMALL \
      "eval.datasets=[hiroom,eth3d,dtu64,7scenes,scannetpp]" "eval.modes=[pose]" workspace.work_dir=...
"""
import os
import sys

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = int(os.environ.get("DA3_RES", "0"))
ROPE = os.environ.get("DA3_ROPE", "") not in ("", "0")
FORCE_FLOAT = os.environ.get("DA3_ROPE_FLOAT", "") not in ("", "0")  # float RoPE path even at scale 1 (check)
SCALE = None  # [sy, sx] while a low-res forward runs with ROPE, else None (upstream integer path)
_installed = False


def install(res=None, rope=None):
    global _installed, RES, ROPE
    if res is not None:
        RES = res
    if rope is not None:
        ROPE = rope
    if _installed:
        return
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.model.dinov2.layers.rope import RotaryPositionEmbedding2D
    from depth_anything_3.model.dinov2.vision_transformer import DinoVisionTransformer

    orig_rope_fwd = RotaryPositionEmbedding2D.forward
    orig_prepare = DinoVisionTransformer._prepare_rope
    orig_forward = DepthAnything3.forward

    def rope_forward(self, tokens, positions):
        if not positions.is_floating_point():
            return orig_rope_fwd(self, tokens, positions)
        d = tokens.size(-1) // 2
        inv = 1.0 / (self.base_frequency ** (torch.arange(0, d, 2, device=tokens.device).float() / d))

        def rot(t, p):
            # same einsum as the cached table (a bf16 matmul under autocast), then token dtype, then cos / sin
            a = torch.einsum("i,j->ij", p.float().reshape(-1), inv).to(tokens.dtype)
            a = torch.cat((a, a), dim=-1).view(*p.shape, -1)
            return t * a.cos().to(tokens.dtype)[:, None] + self._rotate_features(t) * a.sin().to(tokens.dtype)[:, None]

        v, h = tokens.chunk(2, dim=-1)
        return torch.cat((rot(v, positions[..., 0]), rot(h, positions[..., 1])), dim=-1)

    def prepare_rope(self, B, S, H, W, device):
        scale = SCALE if SCALE is not None else ([1.0, 1.0] if FORCE_FLOAT else None)
        if scale is None or self.rope is None:
            return orig_prepare(self, B, S, H, W, device)
        gh, gw = H // self.patch_size, W // self.patch_size
        off = 1.0 if self.patch_start_idx > 0 else 0.0
        y = torch.arange(gh, device=device).float() * scale[0] + off
        x = torch.arange(gw, device=device).float() * scale[1] + off
        pos = torch.cartesian_prod(y, x).view(1, 1, gh * gw, 2).expand(B, S, -1, -1)
        nodiff = torch.full_like(pos, off)
        if self.patch_start_idx > 0:
            special = torch.zeros(B, S, self.patch_start_idx, 2, device=device)
            pos, nodiff = torch.cat([special, pos], dim=2), torch.cat([special, nodiff], dim=2)
        return pos.contiguous(), nodiff.contiguous()

    def forward(self, image, extrinsics=None, intrinsics=None, *args, **kwargs):
        global SCALE
        H, W = image.shape[-2:]
        if not RES or RES >= max(H, W):
            return orig_forward(self, image, extrinsics, intrinsics, *args, **kwargs)
        assert extrinsics is None, "low-res wrapper is for unposed (pose) mode only"
        from torchvision.transforms.functional import resize

        s = RES / max(H, W)
        h, w = round(H * s / 14) * 14, round(W * s / 14) * 14
        B, N = image.shape[:2]
        small = resize(image.flatten(0, 1), [h, w], antialias=True).view(B, N, 3, h, w)
        if ROPE:
            SCALE = [(H // 14 - 1) / max(h // 14 - 1, 1), (W // 14 - 1) / max(w // 14 - 1, 1)]
        try:
            out = orig_forward(self, small, None, None, *args, **kwargs)
        finally:
            SCALE = None
        for k in list(out.keys()):
            v = out[k]
            if torch.is_tensor(v) and v.dim() >= 3 and tuple(v.shape[-2:]) == (h, w) and k != "aux":
                lead = v.shape[:-2]
                out[k] = F.interpolate(v.float().reshape(-1, 1, h, w), size=(H, W), mode="bilinear",
                                       align_corners=False).view(*lead, H, W).to(v.dtype)
        if "intrinsics" in out and out["intrinsics"] is not None:
            Km = out["intrinsics"].clone()
            sx, sy = W / w, H / h
            Km[..., 0, 0] *= sx
            Km[..., 1, 1] *= sy
            # DA3's camera head puts the principal point at exactly (w/2, h/2), so scale it the same way (the
            # half-pixel formula of eval_student.py would move it 0.25 px off centre at 336 px, which TTO keeps)
            Km[..., 0, 2] *= sx
            Km[..., 1, 2] *= sy
            out["intrinsics"] = Km
        return out

    RotaryPositionEmbedding2D.forward = rope_forward
    DinoVisionTransformer._prepare_rope = prepare_rope
    DepthAnything3.forward = forward
    _installed = True


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    install()
    print(f"[da3_res] res={RES or 'native'} rope_scale={ROPE}", flush=True)
    import run_bench  # noqa: E402

    run_bench.main()
