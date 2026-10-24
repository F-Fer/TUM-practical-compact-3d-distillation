"""Lower the GPU memory of a DA3 forward pass without changing its outputs.

install() patches two methods of the (unmodified) third-party code:

- DinoVisionTransformer.get_intermediate_layers: the backbone normally keeps its 4 output
  layers on the GPU in fp32 ([1, S, N, 2048] for LARGE, ~1 GiB each at 100 frames) plus
  normalized copies of them. Here each layer is normalized on the GPU as soon as it is
  produced (same ops as the original, so bit-identical) and moved to CPU memory. Not pinned:
  PyTorch caches pinned blocks for the life of the process (~8 GiB after a LARGE benchmark
  inference pass), which then stay locked during TSDF fusion; pageable copies are as fast here.
  The camera tokens are small and stay on the GPU for cam_dec.
- DualDPT._forward_impl: moves the head's current frame chunk back to the GPU.

Peak memory of a LARGE-1.1 forward at the benchmark's largest input (100 frames, 378x504) drops
from OOM on the 10 GB 3080 to 5.5 GiB. A smaller head chunk_size (default 8, which da3.py doesn't
override) doesn't lower the 100-frame peak, which is in the backbone, and changes depth by ~2e-3,
so it is left alone.
"""

import torch
from depth_anything_3.model.dinov2.vision_transformer import (
    THRESH_FOR_REF_SELECTION,
    DinoVisionTransformer,
    reorder_by_reference,
    restore_original_order,
    select_reference_view,
)
from depth_anything_3.model.dualdpt import DualDPT
from depth_anything_3.utils.logger import logger




def _offload_layer(self, out_x):
    """get_intermediate_layers' post-processing of one output layer, written to CPU memory."""
    camera_token = out_x[:, :, 0].clone()
    start = 1 + self.num_register_tokens
    if out_x.shape[-1] == self.embed_dim:
        feat = self.norm(out_x)[..., start:, :].cpu()
    elif out_x.shape[-1] == self.embed_dim * 2:
        normed = self.norm(out_x[..., self.embed_dim :])
        shape = (*out_x.shape[:-2], out_x.shape[-2] - start, out_x.shape[-1])
        dtype = torch.promote_types(out_x.dtype, normed.dtype)  # what torch.cat would give
        feat = torch.empty(shape, dtype=dtype)
        feat[..., : self.embed_dim].copy_(out_x[..., start:, : self.embed_dim])
        feat[..., self.embed_dim :].copy_(normed[..., start:, :])
        del normed
    else:
        raise ValueError(f"Invalid output shape: {out_x.shape}")
    return feat, camera_token


def get_intermediate_layers(self, x, n=1, export_feat_layers=[], **kwargs):
    """Copy of _get_intermediate_layers_not_chunked + get_intermediate_layers, offloading outputs."""
    B, S, _, H, W = x.shape
    x = self.prepare_tokens_with_masks(x)
    output, total_block_len, aux_output = [], len(self.blocks), []
    blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
    pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)
    b_idx = None

    for i, blk in enumerate(self.blocks):
        if i < self.rope_start or self.rope is None:
            g_pos, l_pos = None, None
        else:
            g_pos = pos_nodiff
            l_pos = pos

        if self.alt_start != -1 and (i == self.alt_start - 1) and x.shape[1] >= THRESH_FOR_REF_SELECTION and kwargs.get("cam_token", None) is None:
            strategy = kwargs.get("ref_view_strategy", "saddle_balanced")
            logger.info(f"Selecting reference view using strategy: {strategy}")
            b_idx = select_reference_view(x, strategy=strategy)
            x = reorder_by_reference(x, b_idx)
            local_x = reorder_by_reference(local_x, b_idx)

        if self.alt_start != -1 and i == self.alt_start:
            if kwargs.get("cam_token", None) is not None:
                logger.info("Using camera conditions provided by the user")
                cam_token = kwargs.get("cam_token")
            else:
                ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                cam_token = torch.cat([ref_token, src_token], dim=1)
            x[:, :, 0] = cam_token

        if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
            x = self.process_attention(
                x, blk, "global", pos=g_pos, attn_mask=kwargs.get("attn_mask", None)
            )
        else:
            x = self.process_attention(x, blk, "local", pos=l_pos)
            local_x = x

        if i in blocks_to_take:
            out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
            if x.shape[1] >= THRESH_FOR_REF_SELECTION and self.alt_start != -1 and b_idx is not None:
                out_x = restore_original_order(out_x, b_idx)
            output.append(_offload_layer(self, out_x))
            del out_x
        if i in export_feat_layers:
            aux_output.append(x)

    aux_output = [self.norm(out)[..., 1 + self.num_register_tokens :, :] for out in aux_output]
    return tuple(output), aux_output


def install():
    DinoVisionTransformer.get_intermediate_layers = get_intermediate_layers

    orig_forward_impl = DualDPT._forward_impl

    def forward_impl(self, feats, *args, **kwargs):
        device = next(self.parameters()).device
        return orig_forward_impl(self, [f.to(device) for f in feats], *args, **kwargs)

    DualDPT._forward_impl = forward_impl
