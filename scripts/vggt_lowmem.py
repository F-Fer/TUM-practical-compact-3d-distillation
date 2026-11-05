"""Lower the GPU memory of a VGGT forward pass without changing its outputs.

install() wraps four methods of the (unmodified) third-party code:

- Aggregator._process_frame_attention / _process_global_attention: the aggregator keeps the
  frame and global tokens of its 4 cached layers (4, 11, 17, 23) on the GPU, concatenated to fp32
  [1, S, P, 2048] tensors (~0.8 GiB each at 100 frames, 378x518). Here the intermediates of the
  cached layers are moved to CPU memory as soon as a block produces them, so the concatenation
  happens on the CPU (an exact copy). Not pinned, for the reason given in da3_lowmem.py.
- CameraHead.forward: moves only the camera tokens of the last layer back to the GPU.
- DPTHead._forward_impl: moves the head's current frame chunk (8 frames) back to the GPU.

At 100 frames (378x518) VGGT-1B runs out of memory on the 10 GB 3080 without this, even with a
bf16 aggregator (3 cached layers alone take 2.4 GiB); the remaining peak is in the aggregator's
blocks, see baselines.py.
"""

import torch
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.models.aggregator import Aggregator


def _offloading(process):
    """Wrap _process_{frame,global}_attention to return cached layers' intermediates on the CPU."""

    def wrapper(self, tokens, B, S, P, C, idx, pos=None):
        tokens, idx, intermediates = process(self, tokens, B, S, P, C, idx, pos=pos)
        first = idx - len(intermediates)  # layer index of intermediates[0]
        intermediates = [
            t.cpu() if first + j in self.cached_layer_indices else t for j, t in enumerate(intermediates)
        ]
        return tokens, idx, intermediates

    return wrapper


def install():
    Aggregator._process_frame_attention = _offloading(Aggregator._process_frame_attention)
    Aggregator._process_global_attention = _offloading(Aggregator._process_global_attention)

    orig_camera_forward = CameraHead.forward

    def camera_forward(self, aggregated_tokens_list, *args, **kwargs):
        device = next(self.parameters()).device
        # the head only reads tokens[:, :, 0] of the last layer
        last = aggregated_tokens_list[-1][:, :, :1].to(device)
        return orig_camera_forward(self, [*aggregated_tokens_list[:-1], last], *args, **kwargs)

    CameraHead.forward = camera_forward

    orig_forward_impl = DPTHead._forward_impl

    def forward_impl(self, aggregated_tokens_list, images, patch_start_idx, frames_start_idx=None, frames_end_idx=None):
        if frames_start_idx is not None and frames_end_idx is not None:
            frames = slice(frames_start_idx, frames_end_idx)
            images = images[:, frames]
            aggregated_tokens_list = [t if t is None else t[:, frames] for t in aggregated_tokens_list]
        tokens = [t if t is None else t.to(images.device) for t in aggregated_tokens_list]
        return orig_forward_impl(self, tokens, images, patch_start_idx)

    DPTHead._forward_impl = forward_impl
