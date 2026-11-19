"""Training-free RoPE position scaling for DéjàView at a lower input resolution.

DéjàView's frame attention (loop block and decoder blocks) uses 2D RoPE on integer patch coordinates
(1..W/14, special tokens 0, looked up with F.embedding). At a smaller input the same image content then gets
smaller position distances than in training. install(scale) makes the positions float and stretches the patch grid
so that it spans the range of the native grid: pos = idx * scale + 1. SCALE = [sy, sx] is read at call time, so a
caller can set it per forward. With SCALE = [1, 1] the float path reproduces the integer one (rope_check.py).
"""
import torch

SCALE = [1.0, 1.0]
_installed = False


def install():
    global _installed
    if _installed:
        return
    from dvlt.model.dvlt.model import DVLTModel
    from dvlt.model_components.layers.rope import RotaryPositionEmbedding2D

    _orig_forward = RotaryPositionEmbedding2D.forward

    def rope_forward(self, tokens, positions):
        if not positions.is_floating_point():
            return _orig_forward(self, tokens, positions)
        d = tokens.size(-1) // 2
        inv = 1.0 / (self.base_frequency ** (torch.arange(0, d, 2, device=tokens.device).float() / d))

        def rot(t, p):
            # Same arithmetic as the original table: the angles come from the same einsum, which under autocast is
            # a bf16 matmul (bf16 positions x bf16 inv_freq, bf16 result), then token dtype, cos/sin there. An fp32
            # product instead changes q/k by up to ~0.4 and the output depth by up to ~1 (see rope_dec_check.py).
            a = torch.einsum("i,j->ij", p.float().reshape(-1), inv).to(tokens.dtype)
            a = torch.cat((a, a), dim=-1).view(*p.shape, -1)
            return t * a.cos().to(tokens.dtype)[:, None] + self._rotate_features(t) * a.sin().to(tokens.dtype)[:, None]

        v, h = tokens.chunk(2, dim=-1)
        return torch.cat((rot(v, positions[..., 0]), rot(h, positions[..., 1])), dim=-1)

    def rope_positions(self, BS, H, W, device):
        gh, gw = H // self.patch_size, W // self.patch_size
        y = torch.arange(gh, device=device).float() * SCALE[0] + 1
        x = torch.arange(gw, device=device).float() * SCALE[1] + 1
        pos = torch.cartesian_prod(y, x).view(1, gh * gw, 2).expand(BS, -1, -1)
        special = torch.zeros(BS, self.patch_start_idx, 2, device=device)
        return torch.cat([special, pos], dim=1)

    RotaryPositionEmbedding2D.forward = rope_forward
    DVLTModel._get_rope_positions = rope_positions
    _installed = True
