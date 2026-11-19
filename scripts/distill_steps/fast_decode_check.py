"""Checks and timings for fast_decode.py on one image folder (504 px, K = 16, bf16):
(1) fast_decode.decode(depth_lr=False, rays=True) == released _decode;
(2) depth_lr: depth / conf difference to the released head (abs-rel, after the same exp activation);
(3) camera head vs RANSAC ray fit: rotation / translation-direction / focal differences between the two;
(4) timings of the released decode + pose fit vs each variant, at 24 and 100 views.

  .venv/bin/python scripts/distill_steps/fast_decode_check.py --images <folder with >= 100 same-sized images>
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_dvlt import frames, timed  # noqa: E402
from train import load_dvlt  # noqa: E402

import fast_decode  # noqa: E402


def rot_deg(Ra, Rb):
    c = ((Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.rad2deg(torch.arccos(c.clamp(-1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="")
    args = ap.parse_args()
    dev = torch.device("cuda")
    from dvlt.common.rays import rays_to_pose
    from dvlt.model.dvlt.model import _slice_expand_flatten

    model = load_dvlt().to(dev).eval()
    model.inference_steps = 16
    model.decode_chunk_size = 8
    base = frames(100, args.images)
    for S in (24, 100):
        img = base[:, torch.linspace(0, 99, S).long()].to(dev)
        H, W = img.shape[-2:]
        rec = dict(views=S)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z0 = model._encode_images(img)
            reg = model.register_token.expand(1, S, -1, -1).reshape(S, model.num_register_tokens, -1)
            cam = _slice_expand_flatten(model.camera_token, 1, S)
            x = torch.cat([cam, reg, z0], dim=1)
            pos = model._get_rope_positions(S, H, W, dev)
            x = model._solve_inference(x, pos, 1, S)

            def fit(pred):
                with torch.autocast("cuda", enabled=False):
                    return rays_to_pose(pred["rays"].float(), torch.ones_like(pred["depth_conf"]), H, W, 14)

            t_ref, ref = timed(lambda: model._decode(x, H, W, 1, S, pos), 3)
            t_fit, (c2w, Kfit) = timed(lambda: fit(ref), 3)
            t_same, same = timed(lambda: fast_decode.decode(model, x, H, W, 1, S, pos), 3)
            t_lr, lr = timed(lambda: fast_decode.decode(model, x, H, W, 1, S, pos, depth_lr=True), 3)
            t_head, head = timed(lambda: fast_decode.decode(model, x, H, W, 1, S, pos, rays=False), 3)
            t_both, _ = timed(lambda: fast_decode.decode(model, x, H, W, 1, S, pos, depth_lr=True, rays=False), 3)
            t_hc, (w2c_h, Kh) = timed(lambda: fast_decode.head_cameras(head["pose_enc"][0], H, W), 3)
        rec.update(decode_released=t_ref, ray_fit=t_fit, decode_same=t_same, decode_depth_lr=t_lr,
                   decode_no_rays=t_head, decode_both=t_both, head_cameras=t_hc)
        for k in ("depth", "depth_conf", "rays"):
            rec[f"same_{k}_maxdiff"] = (ref[k].float() - same[k].float()).abs().max().item()
        d0, d1 = ref["depth"].float(), lr["depth"].float()
        rec["depth_lr_absrel_mean"] = ((d1 - d0).abs() / d0).mean().item()
        rec["depth_lr_absrel_p99"] = ((d1 - d0).abs() / d0).flatten().quantile(0.99).item() if S <= 24 else None
        rec["conf_lr_rel_mean"] = ((lr["depth_conf"] - ref["depth_conf"]).abs() / ref["depth_conf"]).mean().item()
        # camera head vs ray fit, relative poses to frame 0 (both are only defined up to a similarity)
        Rf, tf = c2w[0, :, :3, :3].float(), c2w[0, :, :3, 3].float()
        Rh = w2c_h[:, :, :3].transpose(1, 2)
        th = -(Rh @ w2c_h[:, :, 3:])[..., 0]
        rel_f = Rf[:1].transpose(1, 2) @ Rf
        rel_h = Rh[:1].transpose(1, 2) @ Rh
        rec["head_vs_fit_rel_rot_deg_mean"] = rot_deg(rel_f, rel_h)[1:].mean().item()
        rec["head_vs_fit_rel_rot_deg_max"] = rot_deg(rel_f, rel_h)[1:].max().item()
        df = torch.nn.functional.normalize((Rf[:1].transpose(1, 2) @ (tf - tf[:1])[..., None])[1:, :, 0], dim=-1)
        dh = torch.nn.functional.normalize((Rh[:1].transpose(1, 2) @ (th - th[:1])[..., None])[1:, :, 0], dim=-1)
        rec["head_vs_fit_tdir_deg_mean"] = torch.rad2deg(torch.arccos((df * dh).sum(-1).clamp(-1, 1))).mean().item()
        rec["focal_fit_mean"] = Kfit[0, :, 0, 0].mean().item()
        rec["focal_head_mean"] = Kh[:, 0, 0].mean().item()
        rec["head_vs_fit_scale_ratio"] = ((th - th.mean(0)).norm(dim=-1).mean() /
                                          (tf - tf.mean(0)).norm(dim=-1).mean()).item()
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
