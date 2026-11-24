"""Feature-span oracle: is the depth correction PBA needs already in the span of DA3's own depth-head features?

    .venv/bin/python scripts/lpba_basis.py <src work dir> <out root> [datasets=hiroom,7scenes,scannetpp,eth3d]
        [model=depth-anything/DA3-SMALL] [max_scenes=0]

Per scene: reruns DA3 (same preprocessing and reference-view strategy as the benchmark) with hooks on the main depth
head (DualDPT, activation exp, so log depth = w . phi + b):

    h32   the 32 post-ReLU channels before the last 1x1 conv (DA3's log depth is exactly linear in them)
    f64   the fused feature map entering output_conv2 (SMALL: 64 channels)

and fits, per frame, log(GT / DA3) on the pixels with GT by ridge least squares in the span of [1, basis] (basis = the
channels, or their per-scene PCA to K channels), as `smooth:G` in lpba_depth_oracle.py fits a bilinear grid. The
corrected depth DA3 * exp(fit) is written per variant to <out root>/<variant>/model_results/... (results.npz with the
depth only, plus gt_meta.npz), to be run with `lpba_depth_oracle.py ofile:<out root>/<variant> ...`. The smooth:G fits
are evaluated alongside for reference. The log prints per scene and variant the RMS log-depth error left after the fit
(on GT pixels, in-sample) and the number of coefficients per frame; summary in <out root>/fit_summary.json.
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpba_depth_oracle import fit_smooth, grid_shape, try_gt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANTS = ["h32", "h32p8", "h32p16", "f64", "f64p8", "f64p16", "f64p32", "s4+f64p16", "s8+f64p16", "s4+h32", "s8+f64"]
SMOOTH = [4, 8, 16]


def load_model(path):
    from depth_anything_3.api import DepthAnything3

    api = DepthAnything3.from_pretrained(path).to("cuda").eval()
    conv2 = api.model.head.scratch.output_conv2
    store = {"f": [], "h": []}
    conv2.register_forward_pre_hook(lambda m, inp: store["f"].append(inp[0].detach().to("cpu", torch.float16)))
    conv2[1].register_forward_hook(lambda m, inp, out: store["h"].append(out.detach().to("cpu", torch.float16)))
    return api, conv2, store


def features(api, conv2, store, image_files, ref_view_strategy):
    store["f"].clear()
    store["h"].clear()
    pred = api.inference([str(f) for f in image_files], ref_view_strategy=ref_view_strategy)
    f = torch.cat(store["f"])  # [N, 64, H, W]
    h = torch.cat(store["h"])  # [N, 32, H, W]
    last = conv2[2]
    with torch.no_grad():
        logd = torch.einsum("nchw,c->nhw", h.float(), last.weight[0, :, 0, 0].float().cpu()) + last.bias[0].float().cpu()
    return f, h, logd.numpy(), np.asarray(pred.depth)


def pca(feat, K, n=4096, seed=0):
    """Per-scene PCA of feat [N, C, H, W] to K channels: [N, K, H, W] float32."""
    N, C, H, W = feat.shape
    g = torch.Generator().manual_seed(seed)
    X = feat.permute(0, 2, 3, 1).reshape(-1, C)
    idx = torch.randint(0, X.shape[0], (min(n * N, X.shape[0]),), generator=g)
    S = X[idx].float()
    mu = S.mean(0)
    _, _, V = torch.linalg.svd(S - mu, full_matrices=False)
    P = V[:K].T  # [C, K]
    out = torch.einsum("nchw,ck->nkhw", feat.float() - mu[None, :, None, None], P)
    return out


def grid_basis(g, H, W):
    """The bilinear hat functions of PBA's grid with g cells along the long side: [gh * gw, H, W]."""
    import torch.nn.functional as F

    gh, gw = grid_shape(g, H, W)
    eye = torch.eye(gh * gw, dtype=torch.float32).reshape(gh * gw, 1, gh, gw)
    return F.interpolate(eye, size=(H, W), mode="bilinear", align_corners=True)[:, 0].numpy()


def fit_span(r, valid, B, ridge=1e-3, max_px=40000, seed=0):
    """Ridge LS fit of r [H, W] on valid pixels in the span of [1, B] (B [K, H, W]); returns the dense fit."""
    K = B.shape[0]
    vs, us = np.nonzero(valid)
    if len(vs) > max_px:
        idx = np.random.default_rng(seed).choice(len(vs), max_px, replace=False)
        vs, us = vs[idx], us[idx]
    Bf = B.reshape(K, -1)
    sd = Bf[:, valid.reshape(-1)].std(1) + 1e-6
    A = np.concatenate([np.ones((len(vs), 1)), (B[:, vs, us] / sd[:, None]).T], 1)
    reg = ridge * len(vs) / (K + 1) * np.eye(K + 1)
    reg[0, 0] = 0
    x = np.linalg.solve(A.T @ A + reg, A.T @ r[vs, us])
    return x[0] + np.tensordot(x[1:] / sd, B, axes=1)


def main():
    src, root = sys.argv[1], sys.argv[2]
    opts = dict(a.split("=", 1) for a in sys.argv[3:])
    datasets = opts.get("datasets", "hiroom,7scenes,scannetpp,eth3d").split(",")
    max_scenes = int(opts.get("max_scenes", 0))
    from depth_anything_3.utils.constants import EVAL_REF_VIEW_STRATEGY  # the benchmark's default ("first")

    os.chdir(REPO)
    api, conv2, store = load_model(opts.get("model", "depth-anything/DA3-SMALL"))
    summary = {}
    for ds in datasets:
        data_dir = os.path.join(src, "model_results", ds)
        scenes = sorted(os.path.dirname(p) for p, _, _ in os.walk(data_dir) if os.path.basename(p) == "unposed")
        if max_scenes:
            scenes = scenes[:max_scenes]
        for scene_dir in scenes:  # the order of run_tto_pba.py
            scene = os.path.relpath(scene_dir, data_dir)
            exports = os.path.join(scene_dir, "unposed", "exports")
            meta = dict(np.load(os.path.join(exports, "gt_meta.npz"), allow_pickle=True))
            da3 = np.load(os.path.join(exports, "mini_npz", "results.npz"))["depth"].astype(np.float32)
            N, H, W = da3.shape
            gt = try_gt(meta["image_files"], H, W)
            if gt is None or (gt > 0).sum() < 100:
                print(f"[basis] {ds}/{scene}: no GT, skipped", flush=True)
                continue
            f, h, logd, pdepth = features(api, conv2, store, meta["image_files"], EVAL_REF_VIEW_STRATEGY)
            # sanity: the hooked features must reproduce the saved depth frame by frame (same order, same resolution)
            assert f.shape[-2:] == (H, W), (f.shape, da3.shape)
            dev_rerun = np.abs(np.log(pdepth) - np.log(da3)).max()
            dev_lin = np.abs(logd - np.log(da3)).max()
            print(f"[basis] {ds}/{scene}: N={N} {H}x{W}; |log rerun - saved| max {dev_rerun:.2e}, |w.phi+b - log saved| max {dev_lin:.2e}", flush=True)
            if dev_lin > 0.05:
                raise RuntimeError(f"hooked features don't match the saved depth: {ds}/{scene}")

            valid = gt > 0
            s = np.median(da3[valid]) / np.median(gt[valid])
            gts = gt * s
            bases = {"h32": h, "f64": f}
            for v in VARIANTS:
                if "+" not in v and "p" in v[1:]:
                    base, K = v.split("p")
                    bases[v] = pca(bases[base], int(K))
            grids = {g: grid_basis(g, H, W) for g in (4, 8)}
            res = {}
            for v in VARIANTS + [f"smooth{g}" for g in SMOOTH] + ["none"]:
                out = da3.copy()
                err = []
                for i in range(N):
                    m = valid[i]
                    if m.sum() <= 100:
                        continue
                    r = np.where(m, np.log(np.maximum(gts[i], 1e-6)) - np.log(np.maximum(da3[i], 1e-6)), 0)
                    if v == "none":
                        fit = np.full_like(r, np.median(r[m]))
                    elif v.startswith("smooth"):
                        fit = fit_smooth(r, m, int(v[6:]))
                    elif "+" in v:  # joint fit: grid hat functions + feature channels (each scaled to unit variance)
                        g, b = v.split("+")
                        fit = fit_span(r, m, np.concatenate([grids[int(g[1:])][1:], bases[b][i].float().numpy()]))
                    else:
                        fit = fit_span(r, m, bases[v][i].float().numpy())
                    if not v.startswith("smooth"):  # features off GT pixels (sky, far) extrapolate: clip to the GT range
                        fit = np.clip(fit, fit[m].min(), fit[m].max())
                    out[i] = da3[i] * np.exp(fit)
                    err.append(np.sqrt(np.mean((r[m] - fit[m]) ** 2)))
                res[v] = float(np.mean(err))
                if v in VARIANTS:
                    d = os.path.join(root, v, "model_results", ds, scene, "unposed", "exports")
                    os.makedirs(os.path.join(d, "mini_npz"), exist_ok=True)
                    np.savez(os.path.join(d, "gt_meta.npz"), **meta)
                    np.savez(os.path.join(d, "mini_npz", "results.npz"), depth=out.astype(np.float32))
            summary[f"{ds}/{scene}"] = res
            print(f"[basis] {ds}/{scene} RMS log err left: " + " ".join(f"{k} {x:.4f}" for k, x in res.items()), flush=True)
            del f, h, bases
    keys = VARIANTS + [f"smooth{g}" for g in SMOOTH] + ["none"]
    by_ds = {}
    for k, r in summary.items():
        by_ds.setdefault(k.split("/")[0], []).append(r)
    agg = {ds: {v: float(np.mean([r[v] for r in rs])) for v in keys} for ds, rs in by_ds.items()}
    print("[basis] mean RMS log err left per dataset:")
    for ds, a in agg.items():
        print(f"  {ds:10s} " + " ".join(f"{k} {x:.4f}" for k, x in a.items()))
    ncoef = {"h32": 33, "h32p8": 9, "h32p16": 17, "f64": 65, "f64p8": 9, "f64p16": 17, "f64p32": 33, "s4+f64p16": 28, "s8+f64p16": 64, "s4+h32": 44, "s8+f64": 112}
    ncoef.update({f"smooth{g}": int(np.prod(grid_shape(g, 378, 504))) for g in SMOOTH})  # at 378 x 504
    with open(os.path.join(root, "fit_summary.json"), "w") as fh:
        json.dump({"per_scene": summary, "per_dataset": agg, "coefficients_per_frame": ncoef}, fh, indent=1)


if __name__ == "__main__":
    main()
