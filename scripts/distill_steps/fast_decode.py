"""Cheaper DéjàView decoding, training-free (replaces DVLTModel._decode; the third-party code is not patched on disk).

  depth_lr=True   the depth head's output stage (UV concat + 3x3 conv 66->32 + ReLU + 1x1 conv, fp32) runs on the last
                  upsampling stage's map (8x the patch grid, 216x288 at 378x504 input) instead of on that map
                  bilinearly upsampled to the full input size; only the 2 output logits (log-depth, conf) are
                  upsampled bilinearly. The released head upsamples the 64-channel features first.
  rays=False      skip the ray decoder's per-pixel output (linear + pixel shuffle, fp32): only its transformer
                  front-end runs, for the camera token that feeds the camera head. Poses then come from
                  `pose_enc` (camera head) instead of the RANSAC fit on the ray map (see `head_cameras`).

With depth_lr=False and rays=True the output equals the released `_decode` (checked in fast_decode_check.py).
"""
import torch
import torch.nn.functional as F

from dvlt.model.dvlt.heads import _concat_uv
from dvlt.model_components import activate_head
from dvlt.model_components.pose_encoding import pose_enc_to_extri_intri


def _front(dec, f, pos):
    y = dec.proj_in(f)
    for blk in dec.blocks:
        y = blk(y, pos=pos)
    return y


def _depth_lowres(dec, f, H, W, start, pos):
    B = f.shape[0]
    ph, pw = H // dec.patch_size, W // dec.patch_size
    x = dec.norm(_front(dec, f, pos))[:, start:]
    ar = W / H
    x = x.permute(0, 2, 1).reshape(B, -1, ph, pw)
    for block in dec.upsample_blocks:
        x = _concat_uv(x, ar)
        for layer in block:
            x = layer(x)
    x = _concat_uv(x, ar)
    with torch.amp.autocast("cuda", enabled=False):
        x = dec.output_block(x.float())
        return F.interpolate(x, (H, W), mode="bilinear", align_corners=False)


def decode(model, features, H, W, B, S, rope_pos, depth_lr=False, rays=True):
    BS = B * S
    chunk = model.decode_chunk_size or BS
    rays_c, depth_c, cam_c = [], [], []
    for s in range(0, BS, chunk):
        f, p = features[s:s + chunk], rope_pos[s:s + chunk]
        kw = dict(H=H, W=W, patch_start_idx=model.patch_start_idx, pos=p)
        if rays:
            r, rf = model.ray_decoder(f, **kw)
            rays_c.append(r)
        else:
            rf = _front(model.ray_decoder, f, p)
        cam_c.append(rf[:, 0])
        if depth_lr:
            depth_c.append(_depth_lowres(model.depth_decoder, f, H, W, model.patch_start_idx, p))
        else:
            depth_c.append(model.depth_decoder(f, **kw)[0])
    depth, conf = activate_head(torch.cat(depth_c), activation="exp_clamped", conf_activation="exp_plus_one")
    out = {"depth": depth.view(B, S, H, W, 1), "depth_conf": conf.view(B, S, H, W),
           "pose_enc": model.camera_head(torch.cat(cam_c), B, S)}
    if rays:
        r, _ = activate_head(torch.cat(rays_c), activation="identity", conf_activation=None)
        out["rays"] = r.view(B, S, H, W, 6)
        out["world_points"] = out["rays"][..., 3:6] + out["rays"][..., :3] * out["depth"]
    return out


def head_cameras(pose_enc, H, W):
    """pose_enc [S, 9] (camera-to-world T, quat xyzw, fov_h, fov_w; trained with CameraLoss's default c2w convention)
    -> world-to-camera [S, 3, 4] and intrinsics [S, 3, 3] (principal point at the image centre)."""
    c2w, K = pose_enc_to_extri_intri(pose_enc.float(), (H, W))
    R, t = c2w[:, :, :3], c2w[:, :, 3:]
    Rt = R.transpose(1, 2)
    return torch.cat([Rt, -Rt @ t], dim=2), K
