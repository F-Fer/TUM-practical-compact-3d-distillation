"""First-order pose sensitivity of PBA to a depth error: which depth errors move PBA's optimum, and by how much?

    .venv/bin/python scripts/lpba_sens.py <src work dir> <out json> [datasets=hiroom,scannetpp,eth3d] [split=dev]
        [fields=da3,smooth4,smooth8,smooth16] [extra=<name>:<work dir with results.npz>,...] [samples=4096]

Per scene: state = GT poses and GT focal, depth = GT depth (scaled to DA3's median; pixels without GT get DA3's depth,
scaled per frame, and are never sampled). For each error field d_j (log depth of a candidate minus log GT, on GT
pixels; 0 elsewhere) PBA's Gauss-Newton system with the depth fixed gives the pose / focal shift its optimum makes
when the depth carries this error: dp_j = -Hpp^-1 sum_i Hgp[i, j] (the error enters as a 1-map basis with coefficient
1 in every frame; run_tto_pba.py depth_basis machinery). The shifted poses are scored with the evaluator's AUC
against the GT ("linearised AUC@3"); compare with the depth oracles of lpba_depth_oracle.py.
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_tto_pba as rp  # noqa: E402
from depth_anything_3.bench.utils import compute_pose  # noqa: E402
from depth_anything_3.utils.geometry import as_homogeneous  # noqa: E402
from lpba_depth_oracle import fit_smooth, other_depth, try_gt  # noqa: E402

DEV = {"hiroom": 10, "scannetpp": 3, "eth3d": 3}


def main():
    src, out_path = sys.argv[1], sys.argv[2]
    opts = dict(a.split("=", 1) for a in sys.argv[3:])
    datasets = opts.get("datasets", "hiroom,scannetpp,eth3d").split(",")
    fields = opts.get("fields", "da3,smooth4,smooth8,smooth16").split(",")
    extra = [e.split(":", 1) for e in opts.get("extra", "").split(",") if e]
    split = opts.get("split", "dev")
    os.chdir(rp.REPO_ROOT)
    dev = torch.device("cuda")
    extra_cfg = [x for x in opts.get("pba", "").split(";") if x]  # run_tto_pba.py overrides, ';'-separated
    cfg = rp.parse_overrides(["radial=false", f"lm_samples={opts.get('samples', 4096)}"] + extra_cfg)
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for ds in datasets:
        data_dir = os.path.join(src, "model_results", ds)
        scenes = sorted(os.path.dirname(p) for p, _, _ in os.walk(data_dir) if os.path.basename(p) == "unposed")
        scenes = scenes[: DEV[ds]] if split == "dev" else scenes[DEV[ds]:]
        for sd in scenes:
            scene = os.path.relpath(sd, data_dir)
            key = f"{ds}/{scene}"
            if key in results:
                continue
            ex = os.path.join(sd, "unposed", "exports")
            pred = dict(np.load(os.path.join(ex, "mini_npz", "results.npz")))
            gt = dict(np.load(os.path.join(ex, "gt_meta.npz"), allow_pickle=True))
            da3 = pred["depth"].astype(np.float32)
            N, H, W = da3.shape
            g = try_gt(gt["image_files"], H, W)
            if g is None:
                continue
            valid = g > 0
            s = np.median(da3[valid]) / np.median(g[valid])
            gts = g * s
            depth = gts.copy()
            for i in range(N):
                v = valid[i]
                k = np.median(depth[i][v] / da3[i][v]) if v.sum() > 100 else 1.0
                depth[i][~v] = da3[i][~v] * k
            lg = np.log(np.maximum(gts, 1e-6))
            cands = {}
            for f in fields:
                if f == "da3":
                    c = np.log(np.maximum(da3, 1e-6))
                elif f.startswith("smooth"):
                    G = int(f[6:])
                    c = np.log(np.maximum(da3, 1e-6)).copy()
                    for i in range(N):
                        if valid[i].sum() > 100:
                            r = np.where(valid[i], lg[i] - c[i], 0)
                            c[i] = c[i] + fit_smooth(r, valid[i], G)
                cands[f] = c
            for name, root in extra:
                o = other_depth(root, gt["image_files"])
                cands[name] = np.log(np.maximum(o * (np.median(da3) / np.median(o)), 1e-6))
            # error fields: per frame minus its median on GT pixels (the per-frame scale is free in PBA)
            B = []
            for c in cands.values():
                e = np.where(valid, c - lg, 0)
                for i in range(N):
                    if valid[i].sum() > 100:
                        e[i][valid[i]] -= np.median(e[i][valid[i]])
                B.append(e)
            basis = torch.from_numpy(np.stack(B, 1)).float()  # [N, F, H, W]
            r_ = int(opts.get("bres", 1))  # basis resolution H / bres (memory)
            if r_ > 1:
                basis = torch.nn.functional.interpolate(basis, size=(H // r_, W // r_), mode="area")
            basis = basis.to(dev)
            n_err = basis.shape[1]
            free_g = int(opts.get("free", 0))  # > 0: a free grid of this many cells (B'), eliminated by the Schur complement
            if free_g:
                gh, gw = (free_g, max(2, round(free_g * W / H))) if H > W else (max(2, round(free_g * H / W)), free_g)
                hat = torch.nn.functional.interpolate(torch.eye(gh * gw).reshape(gh * gw, 1, gh, gw), size=basis.shape[-2:], mode="bilinear", align_corners=True)[:, 0]
                basis = torch.cat([basis, hat[None].expand(N, -1, -1, -1).to(dev)], 1)
            # GT cameras in DA3's scale, GT focal at the prediction's resolution
            import cv2

            ext = gt["extrinsics"][:, :3].astype(np.float32).copy()
            ext[:, :, 3] *= s  # GT cameras in the scale of the scaled GT depth
            Kp = pred["intrinsics"].astype(np.float32).copy()
            w0 = cv2.imread(str(gt["image_files"][0])).shape[1]
            Kp[:, 0, 0] = gt["intrinsics"][:, 0, 0] * W / w0
            Kp[:, 1, 1] = gt["intrinsics"][:, 1, 1] * W / w0
            conf = np.where(valid, pred["conf"], pred["conf"].min() - 1).astype(np.float32)
            cfg_s = {**cfg, "focal_init": "pred"}
            images = rp.load_images(gt["image_files"], H, W).to(dev)
            m = rp.PBA(images, torch.from_numpy(depth).to(dev), torch.from_numpy(conf).to(dev), torch.from_numpy(ext).to(dev),
                       torch.from_numpy(Kp).to(dev), cfg_s, None, basis).to(dev)
            with torch.no_grad():
                R, t = m.poses()
                nbrs = rp.select_neighbors(m.depth0, R, t, m.intrinsics(), cfg["neighbors"], cfg["occ_thresh"], cfg["nbr_mode"], cfg["min_covis"])
                kern = rp.Kernel(m)
                kern.set_level(images.mean(1, keepdim=True), cfg["photo_loss"] != "ncc", 1.0)
                probs = rp.sampling_probs(images.mean(1, keepdim=True), m.valid)
                gen = torch.Generator(device=dev).manual_seed(0)
                pix = torch.multinomial(probs, cfg["lm_samples"], replacement=True, generator=gen)
                feat, quad = kern.make_pool(pix)
                k = nbrs.shape[1]
                chunk = max(1, int(1.6e7 / (k * cfg["lm_samples"] * (2 + basis.shape[1]))))
                Hpp, Hgp, Hgg, gp, gg, Hx, loss = kern.normal_eq(feat, quad, nbrs, chunk)
                Hs = Hpp + 1e-9 * Hpp.diagonal().mean() * torch.eye(len(Hpp), device=dev, dtype=Hpp.dtype)
                Kb = basis.shape[1]
                if free_g:  # eliminate the free grid coefficients (frames x cells) with their full coupling
                    Hfull = Hx.view(N, Kb, N, Kb).clone()
                    Hfull[range(N), :, range(N), :] += Hgg
                    fi = torch.arange(n_err, Kb, device=dev)
                    Hff = Hfull[:, fi][:, :, :, fi].reshape(N * len(fi), N * len(fi))
                    Hfp = Hgp[:, fi].reshape(N * len(fi), -1)  # [N nf, np]
                    Hff = Hff + 1e-9 * Hff.diagonal().mean() * torch.eye(len(Hff), device=dev, dtype=Hff.dtype)
                    X = torch.linalg.solve(Hff, Hfp)  # Hff^-1 Hfp
                    Hs = Hs - Hfp.T @ X
                gt_ext = torch.from_numpy(as_homogeneous(gt["extrinsics"]))
                rec = {}
                # the bias at the GT itself (gradient of the objective at the GT with the GT-like depth)
                for j, name in enumerate(["_gt_bias"] + list(cands)):
                    if j == 0:
                        rhs = gp
                    elif free_g:  # the error's coupling to the poses after the free grid adapted to it
                        hfe = Hfull[:, fi][:, :, :, j - 1].sum(-1).reshape(-1)  # [N nf]: free cells x error (all frames)
                        rhs = Hgp[:, j - 1].sum(0) - X.T @ hfe
                    else:
                        rhs = Hgp[:, j - 1].sum(0)
                    dp = -torch.linalg.solve(Hs, rhs)
                    n = 6 * (N - 1)
                    rot, trans = m.rot.data.clone(), m.trans.data.clone()
                    m.rot.data[1:] = dp[:n].view(N - 1, 6)[:, :3].float()
                    m.trans.data[1:] = dp[:n].view(N - 1, 6)[:, 3:].float()
                    R2, t2 = m.poses()
                    e_out = torch.cat([R2, t2[..., None] * m.scale], -1).cpu().numpy()
                    m.rot.data.copy_(rot)
                    m.trans.data.copy_(trans)
                    pose = compute_pose(torch.from_numpy(as_homogeneous(e_out)), gt_ext)
                    rec[name] = {"auc03": float(pose["auc03"]), "auc30": float(pose["auc30"]), "dlogf": float(dp[n]),
                                 "rot_deg": float(dp[:n].view(N - 1, 6)[:, :3].norm(dim=1).mean() * 180 / np.pi)}
                results[key] = rec
            print(f"[sens] {key}: " + " ".join(f"{k_} {v['auc03']:.3f}" for k_, v in rec.items()), flush=True)
            json.dump(results, open(out_path, "w"), indent=1)
            del m, kern, basis
            torch.cuda.empty_cache()
    names = sorted({k for r in results.values() for k in r})
    print("[sens] mean linearised AUC@3 per dataset:")
    for ds in datasets:
        rs = [r for k, r in results.items() if k.startswith(ds + "/")]
        print(f"  {ds:10s} " + " ".join(f"{n} {100 * np.mean([r[n]['auc03'] for r in rs if n in r]):.1f}" for n in names))


if __name__ == "__main__":
    main()
