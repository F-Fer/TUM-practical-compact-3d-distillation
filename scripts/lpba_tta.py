"""Test-time augmentation of DA3's depth for PBA: average the log depth of several forward passes (horizontal flip,
reversed frame order = another reference view), cameras from the plain pass. knowledge/ideas/learned-depth-basis.md.

    .venv/bin/python scripts/lpba_tta.py <src work dir> <out root> [split=dev] [datasets=hiroom,scannetpp,eth3d]
        [variants=flip,rev,revflip] [model=depth-anything/DA3-SMALL]

Writes <out root>/<name>/model_results/... per combination (base+flip, base+rev, all), each pass scaled to the plain
pass's median (one scale per scene) before averaging; prints abs. rel. vs GT of every combination and pass.
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpba_depth_oracle import abs_rel, try_gt  # noqa: E402
from depth_anything_3.bench.utils import compute_pose  # noqa: E402
from depth_anything_3.utils.geometry import as_homogeneous  # noqa: E402

DEV = {"hiroom": 10, "scannetpp": 3, "eth3d": 3, "7scenes": 2, "dtu64": 4}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rel_poses(E):
    """World-to-camera [N, 3, 4] -> 4x4 poses relative to frame 0 (E_i E_0^-1)."""
    E4 = np.concatenate([np.asarray(E, np.float64)[:, :3], np.tile([[[0, 0, 0, 1.0]]], (len(E), 1, 1))], 1)
    return E4 @ np.linalg.inv(E4[0])


def mean_poses(rs):
    """Average of relative poses: chordal mean of the rotations (SVD projection), mean of the translations."""
    R = np.mean([r[:, :3, :3] for r in rs], 0)
    U, _, Vt = np.linalg.svd(R)
    D = np.ones((len(R), 3))
    D[:, 2] = np.sign(np.linalg.det(U @ Vt))
    out = np.tile(np.eye(4), (len(R), 1, 1))
    out[:, :3, :3] = (U * D[:, None, :]) @ Vt
    out[:, :3, 3] = np.mean([r[:, :3, 3] for r in rs], 0)
    return out


def medoid_poses(rs):
    """Per frame, the pass whose relative pose is closest to the others' (sum of rotation angles + normalised
    translation distances): robust to a pass that puts a frame grossly wrong."""
    rs = np.stack(rs)  # [P, N, 4, 4]
    P, N = rs.shape[:2]
    out = rs[0].copy()
    scale = np.median(np.linalg.norm(rs[:, :, :3, 3], axis=-1)) + 1e-9
    for i in range(N):
        R = rs[:, i, :3, :3]
        t = rs[:, i, :3, 3]
        cos = (np.einsum("aij,bij->ab", R, R) - 1) / 2
        ang = np.arccos(np.clip(cos, -1, 1))
        dt = np.linalg.norm(t[:, None] - t[None], axis=-1) / scale
        out[i] = rs[int(np.argmin((ang + dt).sum(1))), i]
    return out


def main():
    src, root = sys.argv[1], sys.argv[2]
    opts = dict(a.split("=", 1) for a in sys.argv[3:])
    split = opts.get("split", "dev")
    os.chdir(REPO)
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.constants import EVAL_REF_VIEW_STRATEGY

    api = DepthAnything3.from_pretrained(opts.get("model", "depth-anything/DA3-SMALL")).to("cuda").eval()
    passes = opts.get("passes", "flip,rev,revflip").split(",")  # flip | rev | revflip | ref:<strategy>
    combos = {"tta_all": passes}
    if passes == ["flip", "rev", "revflip"]:
        combos.update({"tta_flip": ["flip"], "tta_rev": ["rev"]})
    singles = opts.get("singles", "0") == "1"  # also write every pass alone (its own depth and cameras)
    for ds in opts.get("datasets", "hiroom,scannetpp,eth3d").split(","):
        d = os.path.join(src, "model_results", ds)
        scenes = sorted(os.path.dirname(p) for p, _, _ in os.walk(d) if os.path.basename(p) == "unposed")
        scenes = scenes[: DEV[ds]] if split == "dev" else scenes[DEV[ds]:]
        for sd in scenes:
            scene = os.path.relpath(sd, d)
            ex = os.path.join(sd, "unposed", "exports")
            meta = dict(np.load(os.path.join(ex, "gt_meta.npz"), allow_pickle=True))
            pred = dict(np.load(os.path.join(ex, "mini_npz", "results.npz")))
            files = [str(f) for f in meta["image_files"]]
            base = pred["depth"].astype(np.float32)
            N, H, W = base.shape
            imgs = [Image.open(f).convert("RGB") for f in files]
            logs = {"base": np.log(np.maximum(base, 1e-6))}

            rels = {"base": rel_poses(pred["extrinsics"])}

            def run(ims, flip, rev, ref=EVAL_REF_VIEW_STRATEGY):
                ims = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in ims] if flip else list(ims)
                ims = ims[::-1] if rev else ims
                p = api.inference(ims, ref_view_strategy=ref)
                dep = np.asarray(p.depth, np.float32)
                E = np.asarray(p.extrinsics, np.float64)[:, :3]
                dep, E = (dep[::-1], E[::-1]) if rev else (dep, E)
                if flip:  # mirrored scene: un-mirror the cameras (S E S, S = diag(-1, 1, 1)) and the depth maps
                    dep = dep[:, :, ::-1]
                    S = np.diag([-1.0, 1.0, 1.0])
                    E = np.concatenate([S @ E[:, :, :3] @ S, (S @ E[:, :, 3:])], 2)
                k = np.median(base) / np.median(dep)
                E = E.copy()
                E[:, :, 3] *= k  # translations in the scale of the depth
                return np.log(np.maximum(dep * k, 1e-6)), rel_poses(E)

            for v in passes:
                if v.startswith("ref:"):
                    logs[v], rels[v] = run(imgs, False, False, v[4:])
                else:
                    logs[v], rels[v] = run(imgs, "flip" in v, v.startswith("rev"))
            gt = try_gt(files, H, W)
            gt_ext = torch.from_numpy(as_homogeneous(meta["extrinsics"]))
            E0 = np.concatenate([pred["extrinsics"][:, :3], np.tile([[[0, 0, 0, 1.0]]], (N, 1, 1))], 1)[0]

            def auc(rel):
                E = (rel @ E0)[:, :3]
                return 100 * float(compute_pose(torch.from_numpy(as_homogeneous(E.astype(np.float32))), gt_ext)["auc03"])

            cam = {v: auc(r) for v, r in rels.items()}
            cam["mean_all"] = auc(mean_poses([rels[v] for v in ["base"] + passes]))
            cam["medoid_all"] = auc(medoid_poses([rels[v] for v in ["base"] + passes]))
            print(f"[tta] {ds}/{scene}: init AUC@3 " + " ".join(f"{k_} {x:.1f}" for k_, x in cam.items()), flush=True)
            msg = [f"base {abs_rel(base, gt):.4f}"] + [f"{v} {abs_rel(np.exp(logs[v]), gt):.4f}" for v in passes]
            if singles:
                for v in passes:
                    nm = "single_" + v.replace(":", "_")
                    o = os.path.join(root, nm, "model_results", ds, scene, "unposed", "exports")
                    os.makedirs(os.path.join(o, "mini_npz"), exist_ok=True)
                    np.savez(os.path.join(o, "gt_meta.npz"), **meta)
                    np.savez(os.path.join(o, "mini_npz", "results.npz"), **{**pred, "depth": np.exp(logs[v]).astype(np.float32),
                                                                             "extrinsics": (rels[v] @ E0)[:, :3].astype(np.float32)})
            for name, vs in combos.items():
                dep = np.exp(np.mean([logs["base"]] + [logs[v] for v in vs], 0)).astype(np.float32)
                ext = (mean_poses([rels["base"]] + [rels[v] for v in vs]) @ E0)[:, :3].astype(np.float32)
                ext_m = (medoid_poses([rels["base"]] + [rels[v] for v in vs]) @ E0)[:, :3].astype(np.float32)
                dep_m = np.exp(np.median([logs["base"]] + [logs[v] for v in vs], 0)).astype(np.float32)
                for nm, out in ((name, {**pred, "depth": dep}), (name + "_cam", {**pred, "depth": dep, "extrinsics": ext}),
                                (name + "_med", {**pred, "depth": dep_m, "extrinsics": ext_m})):
                    o = os.path.join(root, nm, "model_results", ds, scene, "unposed", "exports")
                    os.makedirs(os.path.join(o, "mini_npz"), exist_ok=True)
                    np.savez(os.path.join(o, "gt_meta.npz"), **meta)
                    np.savez(os.path.join(o, "mini_npz", "results.npz"), **out)
                msg.append(f"{name} {abs_rel(dep, gt):.4f}")
            print(f"[tta] {ds}/{scene}: abs.rel. " + " ".join(msg), flush=True)


if __name__ == "__main__":
    main()
