"""Check the analytic kernels of pba_analytic.py against autograd / forward-mode AD on one real scene, and time them.

Usage: .venv/bin/python scripts/bench_pba.py <dataset> <scene> [check|time|both] [key=value overrides of run_tto_pba.py]

    .venv/bin/python scripts/bench_pba.py eth3d courtyard both samples=1024 lm_samples=4096

Timing: median of repeated, alternating iterations in one process (so a busy node hits all variants alike). On a GPU,
each measurement ends with torch.cuda.synchronize().
"""

import os
import sys
import time

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_tto_pba as R  # noqa: E402
from pba_analytic import FlatAdam, Kernel  # noqa: E402

FAST = ["nbr_mode=spread", "neighbors=10", "lr_rot=3e-3", "lr_trans=3e-3", "lr_focal=3e-3", "samples=1024"]


def load(dataset, scene, cfg, device, model="DA3-SMALL"):
    exports = os.path.join(R.REPO_ROOT, "workspace/evaluation", model, "model_results", dataset, scene, "unposed", "exports")
    gt = dict(np.load(os.path.join(exports, "gt_meta.npz"), allow_pickle=True))
    pred = dict(np.load(os.path.join(exports, "mini_npz", "results.npz")))
    depth = torch.from_numpy(pred["depth"]).float().to(device)
    N, H, W = depth.shape
    os.chdir(R.REPO_ROOT)
    images = R.load_images(gt["image_files"], H, W).to(device)
    dt = torch.get_default_dtype()
    images, depth = images.to(dt), depth.to(dt)
    m = R.PBA(images, depth, torch.from_numpy(pred["conf"]).to(dt).to(device), torch.from_numpy(pred["extrinsics"]).to(dt).to(device),
              torch.from_numpy(pred["intrinsics"]).to(dt).to(device), cfg).to(device)
    with torch.no_grad():
        Rr, t = m.poses()
        nbrs, _ = R.select_neighbors(m.depth0, Rr, t, m.intrinsics(), cfg["neighbors"], cfg["occ_thresh"], cfg["nbr_mode"], cfg["min_covis"])
    return m, images, nbrs


def level_data(m, images, cfg, sigma, M):
    if cfg["features"] == "vgg":
        if "vgg" not in cfg:
            cfg["vgg"] = R.vgg_features(images, cfg["feat_dim"])
        fm = cfg["vgg"]
    else:
        fm = images.mean(1, keepdim=True) if cfg["gray"] else images
    img = R.gaussian_blur(fm, sigma)
    probs = R.sampling_probs(R.gaussian_blur(images, sigma), m.valid, cfg["grad_weight"])
    return img, img.permute(1, 0, 2, 3).reshape(img.shape[1], -1), torch.multinomial(probs, 16 * M, replacement=True)


def perturb(m, seed=0, rot=0.02, trans=0.02, depth=0.05):
    """A state away from the zero of all parameters, so that J_l, the brightness and the priors matter."""
    gen = torch.Generator(device=m.rot.device).manual_seed(seed)
    with torch.no_grad():
        m.rot.copy_(torch.randn(m.rot.shape, device=m.rot.device, generator=gen) * rot)
        m.trans.copy_(torch.randn(m.trans.shape, device=m.rot.device, generator=gen) * trans)
        m.grid.copy_(torch.randn(m.grid.shape, device=m.rot.device, generator=gen) * depth)
        m.logf.fill_(0.03)
        m.bright.copy_(torch.randn(m.bright.shape, device=m.rot.device, generator=gen) * 0.02)


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


def check_radial(m, images, nbrs, cfg):
    """Radial coefficient: analytic gradient and Gauss-Newton gradient against central differences of the same objective."""
    img, img_flat, pool = level_data(m, images, cfg, 2.0, cfg["samples"])
    K = Kernel(m)
    K.set_level(img, cfg["features"] == "rgb" and cfg["photo_loss"] != "ncc", 2.0)
    perturb(m)
    with torch.no_grad():
        m.kr.fill_(-0.08)
    idx = torch.randint(0, pool.shape[1], (m.N, cfg["samples"]), device=pool.device)
    feat, quad = K.make_pool(pool.gather(1, idx))
    params = [m.rot, m.trans, m.logf, m.grid, m.bright, m.kr]
    grads = [torch.zeros_like(p) for p in params]
    const = K.prep(nbrs)
    with torch.no_grad():
        K.gradient(feat, quad, nbrs, const, grads)
    Hpp, Hgp, Hgg, gp, gg, Hx, c = K.normal_eq(feat, quad, nbrs, 4, False)
    n_pose = 6 * (m.N - 1)
    # the mask is part of neither gradient (a flip is a jump), so the central differences keep the mask of the current state
    from pba_analytic import rho

    def frozen_cost(mask=None):
        st = K.state()
        w = K.warp(st, None, feat, quad, nbrs, const)
        r, _ = K.photo_res(st, w)
        if mask is None:
            return w.vis.float()
        den = mask.sum().clamp_min(1)
        return float(cfg["w_photo"] * (rho(r, K.dp, K.kind).mean(-1) * mask).sum() / den + cfg["w_geo"] * (rho(w.geo, K.dg, "huber") * mask).sum() / den) + K.priors()

    with torch.no_grad():
        mask = frozen_cost()
    for name, p, i in (("kr", m.kr, 0), ("logf", m.logf, 0), ("rot[1,0]", m.rot, (1, 0)), ("rot[3,2]", m.rot, (3, 2)), ("trans[2,1]", m.trans, (2, 1)), ("grid[1,5]", m.grid, (1, 2, 3)), ("bright[2,0]", m.bright, (2, 0))):
        h = 1e-6
        x0 = float(p.data[i])
        with torch.no_grad():
            p.data[i] = x0 + h
            fp = frozen_cost(mask)
            p.data[i] = x0 - h
            fm = frozen_cost(mask)
            p.data[i] = x0
        fd = (fp - fm) / (2 * h)
        g = grads[[id(q) for q in params].index(id(p))][i]
        print(f"  d/d{name:11s} central difference (frozen mask) {fd:+.6e} | gradient() {float(g):+.6e} | ratio {float(g) / fd:.5f}")
    print(f"  Gauss-Newton gradient: logf {float(gp[n_pose]):+.6e} kr {float(gp[n_pose + 1]):+.6e} vs gradient(): logf {float(grads[2][0]):+.6e} kr {float(grads[5][0]):+.6e}")
    print(f"  Gauss-Newton gradient of the poses: cos {cos(gp[:n_pose], torch.cat([grads[0][1:], grads[1][1:]], 1).reshape(-1)):.7f}")


def check_free(m, images, nbrs, cfg):
    """lm_free_depth: the blocks against central differences of the residuals (frozen mask and IRLS weights), the eliminated
    step against a dense solve of the whole system, and the cost of normal_eq against cost()."""
    from pba_analytic import rho_weight

    img, _, pool = level_data(m, images, cfg, 2.0, cfg["lm_samples"])
    K = Kernel(m)
    K.set_level(img, cfg["features"] == "rgb" and cfg["photo_loss"] != "ncc", 2.0)
    perturb(m)
    if K.radial:
        with torch.no_grad():
            m.kr.fill_(-0.08)
    P = cfg["lm_samples"] // K.S
    feat, quad = K.make_pool(pool.gather(1, torch.randint(0, pool.shape[1], (m.N, P), device=pool.device)))
    if K.ncc:
        feat, quad = feat.flatten(1, 2), quad.flatten(1, 2)
    K.z = torch.randn(m.N, P, device=pool.device) * 0.03
    const = K.prep(nbrs)
    with torch.no_grad():
        Hpp, _, _, gp, _, _, c = K.normal_eq(feat, quad, nbrs, 4, False)
    print(f"  loss of normal_eq {c:.10f}, cost() {K.cost(feat, quad, nbrs, 4):.10f}")
    h, b, chunks, size = K.fd
    Vfull = torch.cat([c_[1] for c_ in chunks])  # [N, P, L]
    ilfull = torch.cat([c_[2] for c_ in chunks])

    def res():
        st = K.state()
        w = K.warp(st, None, feat, quad, nbrs, const)
        m_ = w.vis.float()
        if K.ncc:
            b_, a_, aa, bb, _, _, mp = K.ncc_terms(w, m_)
            return b_ / torch.sqrt(bb + 1e-8)[..., None] - a_ / torch.sqrt(aa + 1e-8)[..., None], w.geo, mp, m_
        return K.photo_res(st, w)[0], w.geo, m_, m_

    with torch.no_grad():
        r0, g0, m0, mg0 = res()
        n_vis = float(mg0.sum())
        if K.ncc:
            ncc_scale = n_vis / float(m0.sum())
            wp = (m0 * cfg["w_photo"] * ncc_scale)[..., None].expand_as(r0)
        else:
            wp = m0[..., None] * rho_weight(r0, K.dp, K.kind) * (cfg["w_photo"] / r0.shape[-1])
        wg = mg0 * rho_weight(g0, K.dg, "huber") * K.w_geo_lm

        def diff(setter, eps=1e-6):
            setter(eps)
            a = res()[:2]
            setter(-2 * eps)
            b_ = res()[:2]
            setter(eps)
            return (a[0] - b_[0]) / (2 * eps), (a[1] - b_[1]) / (2 * eps)

        for i_frame, p_ in ((1, 0), (2, 3)):
            def zset(e, i=i_frame, p=p_):
                K.z[i, p] += e
            jz = diff(zset)
            hn = float(((jz[0] ** 2 * wp).sum() + (jz[1] ** 2 * wg).sum()) / n_vis) + K.w_z / n_vis
            print(f"  sample ({i_frame},{p_}): h central difference {hn:.6e} | normal_eq {float(h[i_frame, p_]):.6e}")
            nb = nbrs[i_frame]
            L = Vfull.shape[-1]
            tests = [("rot source", m.rot, (i_frame, 1), 1), ("trans source", m.trans, (i_frame, 2), 5), ("rot target 0", m.rot, (int(nb[0]), 0), 6),
                     ("trans target 2", m.trans, (int(nb[2]), 1), 6 + 12 + 4)]
            if K.has_focal:
                tests.append(("logf", m.logf, 0, 6 + 6 * nbrs.shape[1]))
            if K.radial:
                tests.append(("kr", m.kr, 0, L - 1))
            for name, q, ix, loc in tests:
                def pset(e, q=q, ix=ix):
                    q.data[ix] += e
                je = diff(pset)
                vn = float(((jz[0] * je[0] * wp).sum() + (jz[1] * je[1] * wg).sum()) / n_vis)
                print(f"    V[{name:14s}] central difference {vn:+.6e} | normal_eq {float(Vfull[i_frame, p_, loc]):+.6e}")
    # the eliminated step against a dense solve of the whole system
    lam = 1e-3
    Dp = Hpp.diagonal().clamp_min(1e-30)
    S = Hpp + torch.diag_embed(lam * Dp)
    dp_, dz = K.solve_free(S, gp, lam)
    n_p, T = len(gp), len(gp) + K.z.numel()
    H = torch.zeros(T, T, dtype=torch.float64, device=gp.device)
    H[:n_p, :n_p] = S
    for i in range(m.N):
        for pp in range(P):
            row = n_p + i * P + pp
            keep = ilfull[i] < n_p
            vals = torch.zeros(n_p, dtype=torch.float64, device=gp.device)
            vals.index_add_(0, ilfull[i][keep], Vfull[i, pp][keep])
            H[row, :n_p] = vals
            H[:n_p, row] = vals
            H[row, row] = h[i, pp] * (1 + lam)
    d = torch.linalg.solve(H, -torch.cat([gp, b.reshape(-1)]))
    print(f"  step, eliminated against dense: poses cos {cos(dp_, d[:n_p]):.10f} max rel {rel(dp_, d[:n_p]):.2e}; depths cos {cos(dz.reshape(-1), d[n_p:]):.10f} max rel {rel(dz.reshape(-1), d[n_p:]):.2e}")
    z_ = K.z.clone()
    c0 = K.cost(feat, quad, nbrs, 4)
    lam_ = 1e-3
    for it in range(8):
        lam_, a, b_, ok = K.lm_step(feat, quad, nbrs, 4, lam_, "dense", it=it)
        print(f"  LM {it}: cost {a:.8f} -> {b_:.8f} accepted {ok} lambda {lam_:.1e} |z| {float(K.z.abs().mean()):.4f}")
    K.z = None
    return z_, c0


def check_patch(m, images, nbrs, cfg):
    """pba_patch: the blocks of the geometric objective against central differences, the step against a dense solve, the LM loop."""
    from pba_analytic import rho_weight
    from pba_patch import PatchKernel

    perturb(m)
    if cfg["radial"]:
        with torch.no_grad():
            m.kr.fill_(-0.08)
    K = PatchKernel(m, nbrs, cfg["radial"], images)
    K.set_image(R.gaussian_blur(images.mean(1, keepdim=True), 2.0), 2.0)
    K.z = torch.randn_like(K.z) * 0.02
    K.align(cfg["patch_lk"])
    print(f"  {K.P} patches per frame, {K.S} pixels each, {K.stats}")
    Hpp, _, _, gp, _, _, c = K.normal_eq(None, None, nbrs, 1, False)
    print(f"  loss of normal_eq {c:.10f}, cost() {K.cost(None, None, nbrs, 1):.10f}")
    h, b, chunks, size = K.fd
    _, Vfull, ilfull = chunks[0]
    g0 = K.geo_warp(K.state())
    m0 = g0["mask"].to(g0["z"].dtype)
    n_vis = float(m0.sum())
    wr = m0 * rho_weight(K.maha(g0), K.delta, "huber")
    W11, W12, W22 = (K.info * wr[..., None]).unbind(-1)
    wg = m0 * rho_weight(g0["geo"], K.dg, "huber") * K.w_geo_lm

    def diff(setter, eps=1e-6):
        setter(eps)
        a = K.geo_warp(K.state())
        setter(-2 * eps)
        b_ = K.geo_warp(K.state())
        setter(eps)
        return [(a[x] - b_[x]) / (2 * eps) for x in ("eu", "ev", "geo")]

    def quad_form(a, b_):
        return float((a[0] * (W11 * b_[0] + W12 * b_[1]) + a[1] * (W12 * b_[0] + W22 * b_[1]) + a[2] * wg * b_[2]).sum()) / n_vis

    vis = m0.sum(1)
    picks = [(int(i), int(p)) for i, p in (vis > 0).nonzero()[[0, len((vis > 0).nonzero()) // 2]]]
    for i_frame, p_ in picks:
        def zset(e, i=i_frame, p=p_):
            K.z[i, p] += e
        jz = diff(zset)
        print(f"  patch ({i_frame},{p_}): h central difference {quad_form(jz, jz) + K.w_z / n_vis:.6e} | normal_eq {float(h[i_frame, p_]):.6e}")
        nb = nbrs[i_frame]
        L = Vfull.shape[-1]
        tests = [("rot source", m.rot, (i_frame, 1), 1), ("trans source", m.trans, (i_frame, 2), 5), ("rot target 0", m.rot, (int(nb[0]), 0), 6), ("trans target 2", m.trans, (int(nb[2]), 1), 6 + 12 + 4)]
        if K.has_focal:
            tests.append(("logf", m.logf, 0, 6 + 6 * nbrs.shape[1]))
        if K.radial:
            tests.append(("kr", m.kr, 0, L - 1))
        for name, q, ix, loc in tests:
            def pset(e, q=q, ix=ix):
                q.data[ix] += e
            je = diff(pset)
            print(f"    V[{name:14s}] central difference {quad_form(jz, je):+.6e} | normal_eq {float(Vfull[i_frame, p_, loc]):+.6e}")
    n_pose = 6 * (m.N - 1)
    for name, q, ix, idx in (("rot[1,0]", m.rot, (1, 0), 0), ("trans[2,1]", m.trans, (2, 1), 6 + 4), ("logf", m.logf, 0, n_pose)) + ((("kr", m.kr, 0, n_pose + 1),) if K.radial else ()):
        e = 1e-6
        q.data[ix] += e
        fp = K.cost(None, None, nbrs, 1)
        q.data[ix] -= 2 * e
        fm = K.cost(None, None, nbrs, 1)
        q.data[ix] += e
        print(f"  d/d{name:11s} central difference {(fp - fm) / (2 * e):+.6e} | normal_eq {float(gp[idx]):+.6e}")
    i_, p_ = picks[0]
    K.z[i_, p_] += 1e-6
    fp = K.cost(None, None, nbrs, 1)
    K.z[i_, p_] -= 2e-6
    fm = K.cost(None, None, nbrs, 1)
    K.z[i_, p_] += 1e-6
    print(f"  d/dz          central difference {(fp - fm) / 2e-6:+.6e} | normal_eq {float(b[i_, p_]):+.6e}")
    lam = 1e-3
    S = Hpp + torch.diag_embed(lam * Hpp.diagonal().clamp_min(1e-30))
    dp_, dz = K.solve_free(S, gp, lam)
    n_p, P = len(gp), K.z.shape[1]
    T = n_p + K.z.numel()
    H = torch.zeros(T, T, dtype=torch.float64, device=gp.device)
    H[:n_p, :n_p] = S
    for i in range(m.N):
        keep = ilfull[i] < n_p
        for pp in range(P):
            row = n_p + i * P + pp
            vals = torch.zeros(n_p, dtype=torch.float64, device=gp.device)
            vals.index_add_(0, ilfull[i][keep], Vfull[i, pp][keep])
            H[row, :n_p] = vals
            H[:n_p, row] = vals
            H[row, row] = h[i, pp] * (1 + lam)
    d = torch.linalg.solve(H, -torch.cat([gp, b.reshape(-1)]))
    print(f"  step, eliminated against dense: poses cos {cos(dp_, d[:n_p]):.10f} max rel {rel(dp_, d[:n_p]):.2e}; depths cos {cos(dz.reshape(-1), d[n_p:]):.10f} max rel {rel(dz.reshape(-1), d[n_p:]):.2e}")
    lam_ = 1e-3
    for it in range(8):
        lam_, a, b_, ok = K.lm_step(None, None, nbrs, 1, lam_, "dense", it=it)
        print(f"  LM {it}: cost {a:.8f} -> {b_:.8f} accepted {ok} lambda {lam_:.1e} |z| {float(K.z.abs().mean()):.4f}")


def op_sections(m, images, nbrs, cfg):
    """Tensor operations (not views) per section of an analytic Adam iteration."""
    img, _, pool = level_data(m, images, cfg, 2.0, cfg["samples"])
    K = Kernel(m)
    K.set_level(img, cfg["features"] == "rgb", 2.0)
    fp, qp = K.make_pool(pool)
    const = K.prep(nbrs)
    lrs = [cfg["lr_rot"], cfg["lr_trans"], cfg["lr_focal"], cfg["lr_depth"], cfg["lr_bright"], cfg["lr_radial"]]
    fa = FlatAdam([m.rot, m.trans, m.logf, m.grid, m.bright, m.kr], lrs)

    def nonview(fn):
        with ProfileOps() as c:
            out = fn()
        return out, sum(v[0] for k, v in c.t.items() if k not in VIEWS), sorted(((k, v[0]) for k, v in c.t.items() if k not in VIEWS), key=lambda x: -x[1])[:10]

    with torch.no_grad():
        idx = torch.randint(0, pool.shape[1], (m.N, cfg["samples"]), device=pool.device)
        (f, q), n1, h1 = nonview(lambda: K.take(fp, qp, idx))
        st, n2, h2 = nonview(K.state)
        w, n3, h3 = nonview(lambda: K.warp(st, None, f, q, nbrs, const))
        _, n4, h4 = nonview(lambda: K.photo_res(st, w))
        _, n5, h5 = nonview(lambda: K.gradient(f, q, nbrs, const, fa.grads))
        _, n6, h6 = nonview(lambda: fa.step(1.0))
    print(f"non-view ops: take {n1} | state {n2} | warp {n3} | photo_res {n4} | gradient total {n5} (with state+warp+photo_res) | adam step {n6} | whole iteration {n1 + n5 + n6}")
    for name, h in (("state", h2), ("warp", h3), ("photo_res", h4), ("gradient", h5)):
        print(f"  {name}: " + " ".join(f"{k} {v}" for k, v in h))


def check_chord(m, images, nbrs, cfg):
    """The gradient of a chord step has to be the Gauss-Newton system's right-hand side, in the same order."""
    img, img_flat, pool = level_data(m, images, cfg, 1.0, cfg["lm_samples"] // 4)
    K = Kernel(m)
    K.set_level(img, True)
    perturb(m)
    feat, quad = K.make_pool(pool[:, : cfg["lm_samples"]])
    with torch.no_grad():
        lam, c0, c1, ok = K.lm_step(feat, quad, nbrs, 8, 1e-3, cfg["lm_solver"], it=0)
        print(f"  full step: cost {c0:.6f} -> {c1:.6f} accepted {ok}, factor kept: {K.factor is not None}")
        Hpp, Hgp, Hgg, gp, gg, Hx, c = K.normal_eq(feat, quad, nbrs, 8, False)
        grads = [torch.zeros_like(p.data) for p in (m.rot, m.trans, m.logf, m.grid, m.bright, m.kr)]
        K.gradient(feat, quad, nbrs, K.prep(nbrs), grads)
        g = torch.cat([torch.cat([grads[0][1:], grads[1][1:]], 1).reshape(-1), grads[2].reshape(-1)] + ([grads[5].reshape(-1)] if K.radial else []))
        print(f"  right-hand side of the poses, focal{', radial' if K.radial else ''}: cos {cos(g.double(), gp):.8f} max rel {rel(g.double(), gp):.2e}; cells: cos {cos(grads[3].reshape(-1).double(), gg.reshape(-1)):.8f}")
        L, sc = K.factor
        n = len(gp)
        rhs = torch.cat([g.double(), grads[3].reshape(-1).double()])
        full = torch.cat([gp, gg.reshape(-1)])
        d_c = -torch.cholesky_solve((rhs / sc).float()[:, None], L)[:, 0].double() / sc if sc is not None else -torch.cholesky_solve(rhs[:, None], L)[:, 0]
        print(f"  chord direction: g.d {float(rhs @ d_c):+.4e} (must be < 0), |d| pose {float(d_c[:n].norm()):.4e} cells {float(d_c[n:].norm()):.4e}, frobenius of L {float(L.norm()):.3e}, sc {'none' if sc is None else f'{float(sc.min()):.2e}..{float(sc.max()):.2e}'}")
        for scale in (1.0, 0.5, 0.25, 0.1, 0.03):
            state = [x.data.clone() for x in (m.rot, m.trans, m.logf, m.grid, m.kr)]
            K.apply(d_c[:n] * scale, d_c[n:].view_as(m.grid.data) * scale)
            print(f"    step x{scale}: cost {K.cost(feat, quad, nbrs, 8):.6f} (from {K.cost.__self__.priors() * 0 + c:.6f})")
            for x, q in zip((m.rot, m.trans, m.logf, m.grid, m.kr), state):
                x.data.copy_(q)
        for i in range(1, 4):
            r = K.chord_step(feat, quad, nbrs, 8, lam)
            print(f"  chord step {i}: " + (f"cost {r[1]:.6f} -> {r[2]:.6f}" if r else "rejected"))


def check(m, images, nbrs, cfg):
    img, img_flat, pool = level_data(m, images, cfg, 2.0, cfg["samples"])
    K = Kernel(m)
    rgb = cfg["features"] == "rgb" and cfg["photo_loss"] != "ncc"
    S = 9 if cfg["photo_loss"] == "ncc" else 1
    K.set_level(img, rgb, 2.0)
    perturb(m)
    # --- gradient
    idx = torch.randint(0, pool.shape[1], (m.N, cfg["samples"] // S), device=pool.device)
    pix = pool.gather(1, idx)
    u, v = (pix % m.W).float(), (pix // m.W).float()
    for p in m.parameters():
        p.grad = None
    loss, parts = m.loss(u, v, nbrs, img_flat, 2.0)
    loss.backward()
    grads = [torch.zeros_like(p) for p in (m.rot, m.trans, m.logf, m.grid, m.bright)]
    feat, quad = K.make_pool(pix)
    if S > 1:
        feat, quad = feat.flatten(1, 2), quad.flatten(1, 2)
    with torch.no_grad():
        total, p2 = K.gradient(feat, quad, nbrs, K.prep(nbrs), grads, want_loss=True)
    print(f"loss autograd {float(loss):.8f} analytic {total:.8f} | photo {float(parts['photo']):.8f} {p2['photo']:.8f} geo {float(parts['geo']):.8f} {p2['geo']:.8f}")
    for name, g, p in zip(("rot", "trans", "logf", "grid", "bright"), grads, (m.rot, m.trans, m.logf, m.grid, m.bright)):
        print(f"  grad {name:7s} cos {cos(g, p.grad):.7f} max rel diff {rel(g, p.grad):.2e} norm {float(p.grad.norm()):.3e}")
    # --- Gauss-Newton system
    M = cfg["lm_samples"] or cfg["samples"]
    img, img_flat, pool = level_data(m, images, cfg, 0.0, M // 4)
    K.set_level(img, rgb, 1.0)
    pix = pool[:, :M // S]
    u, v = (pix % m.W).float(), (pix // m.W).float()
    layout = R.lm_layout(m)
    U, V = m.patch_samples(u, v, 1.0) if S > 1 else (u, v)
    Hm, g = R.lm_normal_equations(m, U, V, nbrs, img_flat, S, layout, 1 if S > 1 else 2)
    feat, quad = K.make_pool(pix)
    if S > 1:
        feat, quad = feat.flatten(1, 2), quad.flatten(1, 2)
    with torch.no_grad():
        Hpp, Hgp, Hgg, gp, gg, Hx, c = K.normal_eq(feat, quad, nbrs, 1 if S > 1 else 2, True)
    with torch.no_grad():
        c0 = float(m.loss(u, v, nbrs, img_flat, 1.0)[0])
    n, T = len(gp), Hm.shape[0]
    H = torch.zeros_like(Hm)
    H[:n, :n] = Hpp
    H[n:, :n] = Hgp.reshape(-1, n)
    H[:n, n:] = Hgp.reshape(-1, n).T
    H[n:, n:] = Hx + torch.block_diag(*Hgg)
    print(f"LM system: {T} unknowns, cost analytic {c:.8f} vs model.loss {c0:.8f}")
    print(f"  H pose block rel {rel(H[:n, :n], Hm[:n, :n]):.2e} | grid-pose {rel(H[n:, :n], Hm[n:, :n]):.2e} | grid-grid {rel(H[n:, n:], Hm[n:, n:]):.2e} | whole {rel(H, Hm):.2e}")
    print(f"  g rel pose {rel(gp, g[:n]):.2e} grid {rel(gg.reshape(-1), g[n:]):.2e} cos {cos(torch.cat([gp, gg.reshape(-1)]), g):.8f}")
    # the part of the grid-grid block that lm_solver=schur leaves out
    off = (Hx - Hx * 0)
    block = torch.block_diag(*Hgg)
    print(f"  grid-grid: frobenius norm of the coupling of different frames {float(off.norm()):.3e} vs the frame blocks {float(block.norm()):.3e}")
    # same step with both solvers of a dense system
    lam = 1e-3
    diag = Hm.diagonal().clamp_min(1e-12 * float(Hm.diagonal().max()) + 1e-30)
    L, _ = torch.linalg.cholesky_ex(Hm + torch.diag(lam * diag))
    d_old = -torch.cholesky_solve(g[:, None], L)[:, 0]
    Dp, Dg = Hpp.diagonal().clamp_min(1e-30), Hgg.diagonal(dim1=1, dim2=2).clamp_min(1e-30)
    dp1, dg1 = K.solve(Hpp, Hgp, Hgg, gp, gg, Hx, Dp, Dg, lam, "dense")
    d1 = torch.cat([dp1, dg1.reshape(-1)])
    dp2, dg2 = K.solve(Hpp, Hgp, Hgg, gp, gg, None, Dp, Dg, lam, "schur")
    d2 = torch.cat([dp2, dg2.reshape(-1)])
    for tol in (1e-1, 1e-2, 1e-3):
        K.cfg["lm_cg_tol"], K.cfg["lm_cg_iters"] = tol, 100
        K.cg_iters = []
        dp3, dg3 = K.solve(Hpp, Hgp, Hgg, gp, gg, Hx, Dp, Dg, lam, "pcg")
        d3 = torch.cat([dp3, dg3.reshape(-1)])
        print(f"  LM step: pcg to {tol:.0e}: {K.cg_iters[-1]} iterations, cos {cos(d3, d1):.6f} pose {cos(dp3, dp1):.6f} cells {cos(dg3, dg1):.6f}")
    print(f"  LM step: analytic dense vs jvp: cos {cos(d1, d_old):.7f} rel {rel(d1, d_old):.2e} | schur (no cross-frame cells) vs jvp: cos {cos(d2, d_old):.5f}, pose part cos {cos(d2[:n], d_old[:n]):.5f}, cells cos {cos(d2[n:], d_old[n:]):.5f}")
    # descent of the true loss along each step
    state = [x.data.clone() for x in (m.rot, m.trans, m.logf, m.grid)]
    for name, d in (("jvp dense", d_old), ("analytic dense", d1), ("schur", d2)):
        K.apply(d[:n], d[n:].view_as(m.grid.data))
        with torch.no_grad():
            cc = float(m.loss(u, v, nbrs, img_flat, 1.0)[0])
        print(f"    cost after the step ({name}): {cc:.6f} (from {c0:.6f})")
        for x, s in zip((m.rot, m.trans, m.logf, m.grid), state):
            x.data.copy_(s)


class CountOps(TorchDispatchMode):
    """Number of tensor operations (what a GPU pays per iteration whatever the number of pixels)."""

    def __init__(self):
        super().__init__()
        self.n = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.n += 1
        return func(*args, **(kwargs or {}))


def count_ops(fn):
    with CountOps() as c:
        fn()
    return c.n


class ProfileOps(TorchDispatchMode):
    """Calls and seconds per tensor operation."""

    def __init__(self):
        super().__init__()
        self.t = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        t0 = time.perf_counter()
        out = func(*args, **(kwargs or {}))
        e = self.t.setdefault(func.__name__.split(".")[0], [0, 0.0])
        e[0] += 1
        e[1] += time.perf_counter() - t0
        return out


def op_table(name, fn, reps=5, top=22):
    with ProfileOps() as c:
        for _ in range(reps):
            fn()
    rows = sorted(c.t.items(), key=lambda kv: -kv[1][1])
    n = sum(v[0] for v in c.t.values()) / reps
    views = sum(v[0] for k, v in c.t.items() if k in VIEWS) / reps
    print(f"-- {name}: {n:.0f} ops per call, of which {views:.0f} views (no GPU work); total {sum(v[1] for v in c.t.values()) / reps * 1e3:.2f} ms")
    print("   " + "  ".join(f"{k} {v[0] / reps:.0f}x {v[1] / reps * 1e3:.2f}ms" for k, v in rows[:top]))


VIEWS = {"view", "reshape", "expand", "unsqueeze", "squeeze", "transpose", "permute", "slice", "select", "index_select_view", "unbind", "narrow", "alias", "detach", "movedim", "split_with_sizes", "as_strided"}


def med_time(fns, reps, warm=3):
    """Alternate the functions, median seconds per call of each."""
    times = {k: [] for k in fns}
    for i in range(reps + warm):
        for k, fn in fns.items():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.perf_counter()
            fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if i >= warm:
                times[k].append(time.perf_counter() - t)
    return {k: float(np.median(v)) for k, v in times.items()}


def loop_check(m, images, nbrs, cfg, iters=30, lm=3):
    """Whole loop (levels of Adam, then LM on the same random numbers) with both implementations. In float64 (PBA_DOUBLE=1)
    the trajectories must stay together, in float32 they separate through rounding (Adam normalises the steps)."""
    lrs = [cfg["lr_rot"], cfg["lr_trans"], cfg["lr_focal"], cfg["lr_depth"], cfg["lr_bright"]]
    params = [m.rot, m.trans, m.logf, m.grid, m.bright]
    M, ML = cfg["samples"], cfg["lm_samples"]
    dev = m.rot.device
    results = []
    for run in ("old", "new", "new_jitter"):
        impl = run[:3]
        torch.manual_seed(1)
        with torch.no_grad():
            for p in params:
                p.zero_()
            if run == "new_jitter":  # control: how far does a tiny difference carry on its own?
                gen = torch.Generator(device=dev).manual_seed(5)
                m.rot.add_(1e-9 * torch.randn(m.rot.shape, device=dev, generator=gen, dtype=m.rot.dtype))
        K = Kernel(m)
        fa = FlatAdam(params, lrs) if impl == "new" else None
        layout = R.lm_layout(m)
        for sigma in (4.0, 1.0):
            img, img_flat, pool = level_data(m, images, cfg, sigma, M)
            K.set_level(img, True)
            if impl == "new":
                fa.reset()
                fp, qp = K.make_pool(pool)
                rows = K.frames[:, None]
            else:
                opt = torch.optim.Adam([{"params": [p], "lr": lr} for p, lr in zip(params, lrs)])
                sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: 0.1 + 0.9 * 0.5 * (1 + torch.pi * 0 + np.cos(np.pi * it / iters)))
            for it in range(iters):
                idx = torch.randint(0, pool.shape[1], (m.N, M), device=dev)
                if impl == "new":
                    with torch.no_grad():
                        K.gradient(fp[rows, idx], qp[rows, idx], nbrs, K.prep(nbrs), fa.grads)
                        fa.step(0.1 + 0.45 * (1 + float(np.cos(np.pi * it / iters))))
                else:
                    pix = pool.gather(1, idx)
                    loss, _ = m.loss((pix % m.W).float(), (pix // m.W).float(), nbrs, img_flat)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    sched.step()
        img, img_flat, pool = level_data(m, images, cfg, 0.0, ML // 4)
        K.set_level(img, True)
        lam = cfg["lm_lambda"]
        for _ in range(lm):
            pix = pool.gather(1, torch.randint(0, pool.shape[1], (m.N, ML), device=dev))
            if impl == "new":
                with torch.no_grad():
                    f, q = K.make_pool(pix)
                    lam, c0, c1, ok = K.lm_step(f, q, nbrs, 8, lam, "dense")
            else:
                lam, c0, c1, ok = R.lm_step(m, (pix % m.W).float(), (pix // m.W).float(), nbrs, img_flat, 1.0, 1, layout, 2, lam)
            print(f"  {run} LM cost {c0:.8f} -> {c1:.8f} lambda {lam:.2e}")
        results.append([p.data.clone() for p in params])
    for name, a, b, c in zip(("rot", "trans", "logf", "grid", "bright"), *results):
        print(f"  final {name:7s} max |old - new| {float((a - b).abs().max()):.3e} | control max |new - new with 1e-9 jitter| {float((b - c).abs().max()):.3e} | (max |old| {float(a.abs().max()):.3e})")
    # one Adam step on the same gradient
    torch.manual_seed(3)
    x = [torch.nn.Parameter(torch.randn(5, 3, device=dev)), torch.nn.Parameter(torch.randn(7, device=dev))]
    y = [torch.nn.Parameter(q.data.clone()) for q in x]
    o = torch.optim.Adam([{"params": [x[0]], "lr": 1e-2}, {"params": [x[1]], "lr": 3e-3}])
    fa = FlatAdam(y, [1e-2, 3e-3])
    for _ in range(5):
        g = [torch.randn_like(q) for q in x]
        for q, gg in zip(x, g):
            q.grad = gg.clone()
        o.step()
        for q, gg in zip(fa.grads, g):
            q.copy_(gg)
        fa.step(1.0)
    print(f"  FlatAdam vs torch Adam after 5 steps: max diff {max(float((a - b).abs().max()) for a, b in zip(x, y)):.2e}")
    with torch.no_grad():
        for p in params:
            p.zero_()


def adam_lm_state(m, images, nbrs, cfg, lm_iters, device):
    """Run the fast hybrid: Adam 4 x 50 at cfg["samples"], then lm_iters LM iterations at cfg["lm_samples"] (dense solver)."""
    lrs = [cfg["lr_rot"], cfg["lr_trans"], cfg["lr_focal"], cfg["lr_depth"], cfg["lr_bright"]]
    params = [m.rot, m.trans, m.logf, m.grid, m.bright]
    K = Kernel(m)
    fa = FlatAdam(params, lrs)
    M = cfg["samples"]
    for sigma in (8.0, 4.0, 2.0, 1.0):
        img, _, pool = level_data(m, images, cfg, sigma, M)
        K.set_level(img, True)
        fa.reset()
        fp, qp = K.make_pool(pool)
        rows, const = K.frames[:, None], K.prep(nbrs)
        for it in range(cfg["iters"]):
            idx = torch.randint(0, pool.shape[1], (m.N, M), device=device)
            with torch.no_grad():
                K.gradient(fp[rows, idx], qp[rows, idx], nbrs, const, fa.grads)
                fa.step(0.1 + 0.45 * (1 + float(np.cos(np.pi * it / cfg["iters"]))))
    ML = cfg["lm_samples"]
    img, img_flat, pool = level_data(m, images, cfg, 1.0, ML // 4)
    K.set_level(img, True)
    lam = cfg["lm_lambda"]
    chunk = max(1, int((1.6e7 if device.type == "cuda" else 4e6) / (nbrs.shape[1] * ML)))
    for _ in range(lm_iters):
        with torch.no_grad():
            f, q = K.make_pool(pool.gather(1, torch.randint(0, pool.shape[1], (m.N, ML), device=device)))
            lam, c0, c1, ok = K.lm_step(f, q, nbrs, chunk, lam, "dense")
    return K, img, lam


def noise_check(m, images, nbrs, cfg, device, sizes=(1024, 4096, 16384, 65536), lm_iters=(0, 5, 20), pairs=3):
    """How much of a Gauss-Newton step is signal? At converged states, the steps of pairs of independent pixel samples.
    Pose part only (rotation + translation, the part the metric sees). Also the spectrum of the pose system with the depth
    grid eliminated (dense, largest M) and how much of the step falls into its weakest directions."""
    params = [m.rot, m.trans, m.logf, m.grid, m.bright]
    done = 0
    torch.manual_seed(0)
    # continue one optimisation through the checkpoints
    for target in lm_iters:
        if target == lm_iters[0]:
            with torch.no_grad():
                for p in params:
                    p.zero_()
            K, img, lam = adam_lm_state(m, images, nbrs, cfg, target, device)
            ml = cfg["lm_samples"]
            sigma_img = img
        else:
            _, _, pool = level_data(m, images, cfg, 1.0, cfg["lm_samples"] // 4)
            K.set_level(sigma_img, True)
            for _ in range(target - done):
                with torch.no_grad():
                    f, q = K.make_pool(pool.gather(1, torch.randint(0, pool.shape[1], (m.N, ml), device=device)))
                    lam, c0, c1, ok = K.lm_step(f, q, nbrs, max(1, int(4e6 / (nbrs.shape[1] * ml))), lam, "dense")
        done = target
        K.set_level(sigma_img, True)
        print(f"after {target} LM iterations (lambda {lam:.1e}):")
        rows = []
        n_pose = 6 * (m.N - 1)
        for M in sizes:
            probs = R.sampling_probs(R.gaussian_blur(images, 1.0), m.valid, cfg["grad_weight"])
            for damping in (1e-3, 1e-7):
                cs, norms = [], []
                for _ in range(pairs):
                    steps = []
                    for _ in range(2):
                        pix = torch.multinomial(probs, M, replacement=True)
                        with torch.no_grad():
                            f, q = K.make_pool(pix)
                            Hpp, Hgp, Hgg, gp, gg, Hx, c = K.normal_eq(f, q, nbrs, max(1, int(4e6 / (nbrs.shape[1] * M))), True)
                            Dp, Dg = Hpp.diagonal().clamp_min(1e-30), Hgg.diagonal(dim1=1, dim2=2).clamp_min(1e-30)
                            dp_, dg_ = K.solve(Hpp, Hgp, Hgg, gp, gg, Hx, Dp, Dg, damping, "dense")
                        steps.append(dp_[:n_pose])
                    cs.append(cos(steps[0], steps[1]))
                    norms.append(float((steps[0].norm() + steps[1].norm()) / 2))
                rows.append(f"    M={M:6d} lambda={damping:.0e}: step agreement cos {np.mean(cs):+.3f} (+- {np.std(cs):.3f}), step norm {np.mean(norms):.4f}")
        print("\n".join(rows))
        # spectrum of the system at the largest M, depth eliminated exactly (Hx included through the dense inverse)
        with torch.no_grad():
            f, q = K.make_pool(torch.multinomial(probs, sizes[-1], replacement=True))
            Hpp, Hgp, Hgg, gp, gg, Hx, c = K.normal_eq(f, q, nbrs, max(1, int(4e6 / (nbrs.shape[1] * sizes[-1]))), True)
        np_ = len(gp)
        Ggf = Hx + torch.block_diag(*Hgg)
        Gpf = Hgp.reshape(-1, np_)
        S = Hpp - Gpf.T @ torch.linalg.solve(Ggf, Gpf)
        ev, U = torch.linalg.eigh(S[:n_pose, :n_pose])
        q_ = [0.0, 0.01, 0.1, 0.5, 0.9, 1.0]
        print("    pose spectrum quantiles " + " ".join(f"q{qq:.2f}={float(ev[int(qq * (len(ev) - 1))]):.2e}" for qq in q_) + f" (condition {float(ev[-1] / ev[0].clamp_min(1e-30)):.1e})")
        # share of the GN pose step in the weakest 10 % directions
        rhs = gp[:n_pose] - (Gpf.T @ torch.linalg.solve(Ggf, gg.reshape(-1)))[:n_pose]
        y = (U.T @ (-rhs)) / ev.clamp_min(1e-12 * float(ev[-1]))
        n10 = max(1, len(ev) // 10)
        print(f"    GN pose step: share of its squared norm in the weakest 10% directions: {float((y[:n10] ** 2).sum() / (y**2).sum()):.3f}, in the weakest 1%: {float((y[: max(1, len(ev) // 100)] ** 2).sum() / (y**2).sum()):.3f}")


def setup_timing(m, images, nbrs, cfg, reps=8):
    """Per-level setup, per step."""
    M = cfg["samples"]
    grey = images.mean(1, keepdim=True)
    K = Kernel(m)

    def blur8():
        return R.gaussian_blur(grey, 8.0)

    def blur1():
        return R.gaussian_blur(grey, 1.0)

    img = R.gaussian_blur(grey, 1.0)
    probs = R.sampling_probs(img, m.valid, cfg["grad_weight"])
    K.set_level(img, True)
    pool = torch.multinomial(probs, 16 * M, replacement=True)
    t = med_time({
        "blur_rgb8": lambda: R.gaussian_blur(images, 8.0), "blur8": blur8, "blur1": blur1,
        "probs": lambda: R.sampling_probs(img, m.valid, cfg["grad_weight"]),
        "multinomial": lambda: torch.multinomial(probs, 16 * M, replacement=True),
        "set_level": lambda: K.set_level(img, True),
        "make_pool": lambda: K.make_pool(pool),
        "covis": lambda: R.select_neighbors(m.depth0, *m.poses(), m.intrinsics(), cfg["neighbors"], cfg["occ_thresh"], cfg["nbr_mode"], cfg["min_covis"]),
    }, reps, warm=1)
    print(f"setup, N={m.N} M={M}: " + " ".join(f"{k} {v * 1e3:.2f}ms" for k, v in t.items()))


def timing(m, images, nbrs, cfg, reps=30):
    with torch.no_grad():
        setup_timing(m, images, nbrs, cfg)
    S = 9 if cfg["photo_loss"] == "ncc" else 1
    rgb = cfg["features"] == "rgb" and S == 1
    M = cfg["samples"] // S
    img, img_flat, pool = level_data(m, images, cfg, 2.0, M)
    dev = pool.device
    K = Kernel(m)
    K.set_level(img, rgb, 2.0)
    feat_pool, quad_pool = K.make_pool(pool)
    rows = K.frames[:, None]
    perturb(m, rot=0.005, trans=0.005, depth=0.01)
    params = [m.rot, m.trans, m.logf, m.grid, m.bright]
    lrs = [cfg["lr_rot"], cfg["lr_trans"], cfg["lr_focal"], cfg["lr_depth"], cfg["lr_bright"]]
    # the steps of both variants use their own copies, the parameters are restored after each call
    opt = torch.optim.Adam([{"params": [p], "lr": lr} for p, lr in zip(params, lrs)])
    saved = [p.data.clone() for p in params]

    def restore():
        for p, s in zip(params, saved):
            p.data.copy_(s)

    def adam_old():
        idx = torch.randint(0, pool.shape[1], (m.N, M), device=dev)
        pix = pool.gather(1, idx)
        u, v = (pix % m.W).float(), (pix // m.W).float()
        loss, _ = m.loss(u, v, nbrs, img_flat, 2.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    def grad_old():
        idx = torch.randint(0, pool.shape[1], (m.N, M), device=dev)
        pix = pool.gather(1, idx)
        u, v = (pix % m.W).float(), (pix // m.W).float()
        loss, _ = m.loss(u, v, nbrs, img_flat, 2.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()

    tres = med_time({"old": adam_old, "old_grad_only": grad_old}, reps)
    ops_old = count_ops(adam_old)
    op_table("autograd Adam iteration", adam_old)
    restore()
    fa = FlatAdam(params, lrs)
    const = K.prep(nbrs)

    def adam_new():
        idx = torch.randint(0, pool.shape[1], (m.N, M), device=dev)
        with torch.no_grad():
            K.gradient(*K.take(feat_pool, quad_pool, idx), nbrs, const, fa.grads)
            fa.step(1.0)

    def grad_new():
        idx = torch.randint(0, pool.shape[1], (m.N, M), device=dev)
        with torch.no_grad():
            K.gradient(*K.take(feat_pool, quad_pool, idx), nbrs, const, fa.grads)

    tres.update(med_time({"analytic": adam_new, "analytic_grad_only": grad_new}, reps))
    op_table("analytic Adam iteration", adam_new)
    print(f"tensor operations per Adam iteration: old {ops_old} | analytic {count_ops(adam_new)}")
    print(f"Adam iteration, N={m.N} k={nbrs.shape[1]} M={M}: old {1e3 * tres['old']:.2f} ms (gradient {1e3 * tres['old_grad_only']:.2f}) | analytic {1e3 * tres['analytic']:.2f} ms (gradient {1e3 * tres['analytic_grad_only']:.2f}) | x{tres['old'] / tres['analytic']:.2f}")

    # --- LM parts
    ML = (cfg["lm_samples"] or M) // S
    img, img_flat, pool = level_data(m, images, cfg, 0.0, ML // 4)
    K.set_level(img, rgb, 1.0)
    pix = pool[:, :ML]
    u, v = (pix % m.W).float(), (pix // m.W).float()
    U, V = m.patch_samples(u, v, 1.0) if S > 1 else (u, v)
    layout = R.lm_layout(m)
    k = nbrs.shape[1]
    D = S * img_flat.shape[0] + S
    chunk_old = max(1, int(1.5e8 / (k * ML * D * 21)))
    chunk_new = max(1, int((1.6e7 if dev.type == "cuda" else 4e6) / (k * ML * D)))
    feat, quad = K.make_pool(pix)
    if S > 1:
        feat, quad = feat.flatten(1, 2), quad.flatten(1, 2)
    Hm, g = R.lm_normal_equations(m, U, V, nbrs, img_flat, S, layout, chunk_old)
    diag = Hm.diagonal().clamp_min(1e-30)
    with torch.no_grad():
        Hpp, Hgp, Hgg, gp, gg, Hx, c = K.normal_eq(feat, quad, nbrs, chunk_new, False)
    Dp, Dg = Hpp.diagonal().clamp_min(1e-30), Hgg.diagonal(dim1=1, dim2=2).clamp_min(1e-30)
    r = reps // 3

    def cost_old():
        with torch.no_grad():
            m.loss(u, v, nbrs, img_flat, 1.0)

    def cost_new():
        with torch.no_grad():
            K.cost(feat, quad, nbrs, chunk_new)

    def ne_new():
        with torch.no_grad():
            K.normal_eq(feat, quad, nbrs, chunk_new, False)

    def ne_new_x():
        with torch.no_grad():
            K.normal_eq(feat, quad, nbrs, chunk_new, True)

    def solve_old():
        torch.cholesky_solve(g[:, None], torch.linalg.cholesky_ex(Hm + torch.diag(1e-3 * diag))[0])

    def solve_schur():
        K.solve(Hpp, Hgp, Hgg, gp, gg, None, Dp, Dg, 1e-3, "schur")

    with torch.no_grad():
        HppX, HgpX, HggX, gpX, ggX, HxX, _ = K.normal_eq(feat, quad, nbrs, chunk_new, True)
    K.cfg["lm_cg_tol"], K.cfg["lm_cg_iters"] = 1e-2, 50

    def solve_pcg():
        K.solve(HppX, HgpX, HggX, gpX, ggX, HxX, Dp, Dg, 1e-3, "pcg")

    def solve_d64():
        K.solve(HppX, HgpX, HggX, gpX, ggX, HxX, Dp, Dg, 1e-3, "dense")

    def solve_d32():
        K.solve(HppX, HgpX, HggX, gpX, ggX, HxX, Dp, Dg, 1e-3, "dense32")

    print(f"tensor operations per LM normal equations: jvp {count_ops(lambda: R.lm_normal_equations(m, U, V, nbrs, img_flat, S, layout, chunk_old))} | analytic {count_ops(ne_new)}")
    t = med_time({
        "ne_old": lambda: R.lm_normal_equations(m, U, V, nbrs, img_flat, S, layout, chunk_old),
        "ne_new": ne_new, "ne_new_x": ne_new_x, "cost_old": cost_old, "cost_new": cost_new, "solve_old": solve_old, "solve_schur": solve_schur, "solve_pcg": solve_pcg, "solve_d64": solve_d64, "solve_d32": solve_d32,
    }, r, warm=1)
    print(f"LM, N={m.N} M={ML}, {Hm.shape[0]} unknowns (chunks old {chunk_old}, new {chunk_new} frames):")
    print(f"  normal equations: jvp {t['ne_old']:.4f} s | analytic {t['ne_new']:.4f} s (with cross-frame cells {t['ne_new_x']:.4f}) | x{t['ne_old'] / t['ne_new']:.1f}")
    print(f"  cost: old {t['cost_old']:.4f} s | new {t['cost_new']:.4f} s")
    print(f"  solve: dense {t['solve_old']:.4f} s | schur {t['solve_schur']:.4f} s | x{t['solve_old'] / t['solve_schur']:.1f} | pcg to 1e-2 {t['solve_pcg']:.4f} s ({K.cg_iters[-1] if K.cg_iters else 'preconditioner not positive definite'})")
    print(f"  solve with assembly of the dense matrix: float64 {t['solve_d64']:.4f} s | float32 {t['solve_d32']:.4f} s")
    old = 3 * t["cost_old"] + t["ne_old"] + t["solve_old"]
    new = 2 * t["cost_new"] + t["ne_new"] + t["solve_schur"]
    print(f"  one iteration (one solve, three / two cost evaluations): old {old:.4f} s | new {new:.4f} s | x{old / new:.1f}")


def main():
    args = sys.argv[1:]
    dataset, scene = args[0], args[1]
    mode = args[2] if len(args) > 2 and args[2] in ("check", "time", "both", "loop", "noise", "radial", "chord", "ops", "free", "patch") else "both"
    rest = [a for a in args[2:] if "=" in a]
    if os.environ.get("PBA_DOUBLE"):
        torch.set_default_dtype(torch.float64)
    cfg = R.parse_overrides(FAST + rest)
    cfg["lm_samples"] = cfg["lm_samples"] or 4096
    if mode == "noise":
        cfg["iters"] = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    m, images, nbrs = load(dataset, scene, cfg, device)
    print(f"== {dataset} {scene} N={m.N} {m.H}x{m.W} device={device} threads={torch.get_num_threads()} gray={cfg['gray']}")
    if mode == "ops":
        op_sections(m, images, nbrs, cfg)
    if mode == "radial":
        check_radial(m, images, nbrs, cfg)
    if mode == "free":
        check_free(m, images, nbrs, cfg)
    if mode == "patch":
        check_patch(m, images, nbrs, cfg)
    if mode == "chord":
        check_chord(m, images, nbrs, cfg)
    if mode == "loop":
        loop_check(m, images, nbrs, cfg)
    if mode == "noise":
        noise_check(m, images, nbrs, cfg, device)
    if mode in ("check", "both"):
        check(m, images, nbrs, cfg)
        with torch.no_grad():
            for p in m.parameters():
                p.zero_()
    if mode in ("time", "both"):
        timing(m, images, nbrs, cfg)


if __name__ == "__main__":
    main()
