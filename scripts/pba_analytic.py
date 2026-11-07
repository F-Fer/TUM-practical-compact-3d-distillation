"""Hand-derived gradients and Gauss-Newton systems of the photometric BA in run_tto_pba.py (impl=analytic, lm_impl=analytic).

Same objective as PBA.loss, for per-pixel kernels and NCC, on grey, colour or feature channels. A warp of source pixel p
of frame i into neighbour j is written in the camera of the source:

    X_c = (x, y, d) with d = d0(p) exp(g_i(p)), x = (u - cx) / f * d, y = (v - cy) / f * d
    X_t = R_ji X_c + t_ji,  R_ji = R_j R_i^T,  t_ji = t_j - R_ji t_i

and the derivatives follow from the 3-vector dL/dX_t of each warp. With the pose parametrisation of PBA.poses
(R = exp(w) R0, t = exp(w) t0 + tau) and phi = J_l(w) dw, the left tangent:

    target frame:  dX_t / dphi_j . a = (X_t - tau_j) x a      dX_t / dtau_j . a = a
    source frame:  dX_t / dphi_i . a = (X_c - tau_i) x (-R_ij a)     dX_t / dtau_i . a = -R_ij a

so the pose gradient of a whole frame pair needs only sum(dL/dX_t) and sum(X_t x dL/dX_t).

Why it is faster than autograd: on an RTX 5090 an Adam iteration of the autograd version takes 4-7 ms and depends little
on the number of frames, so most of it is the cost of ~740 tensor operations (forward, backward, optimiser), not the
pixels. Here it takes 266 operations and 1-1.4 ms. For LM the Jacobian comes from one pass instead of 15 forward-mode
passes and the pose block of a frame pair is a matrix product; the dense solve in single precision (lm_solver=dense32)
is 6 times faster than in double. scripts/bench_pba.py checks all of it against autograd and forward-mode AD in float64.

lm_solver=schur and pcg (eliminate the depth grid, without or with iterating on the coupling of the frames' cells) and
lm_refactor > 1 (reuse a factorisation) are faster still per iteration but lose ScanNet++, see results/pba/REPORT3.md.

lm_free_depth gives every LM sample (with NCC every patch) its own log depth z with a prior, instead of the grid. A warp of
sample p then depends on z_p, the poses of its two frames, the focal and k only, so the block of the z is diagonal and the
Schur complement is exact, see solve_free. The target's depth is held fixed, which is what removes the coupling.
"""

import math
import sys

import torch


def so3_exp_jl(w):
    """Rodrigues as run_tto_pba.so3_exp, and the left Jacobian J_l(w) = I + b [w]x + c [w]x^2. w [..., 3].
    No branches for small angles (they cost tensor operations in every iteration): in float32, 1 - cos and 1 - sin(t)/t
    lose their relative precision below ~1e-3 rad, but they are then multiplied by [w]x^2 ~ t^2."""
    th2 = (w * w).sum(-1, keepdim=True)[..., None] + 1e-12
    th = torch.sqrt(th2)
    a = torch.sin(th) / th
    b = (1 - torch.cos(th)) / th2
    c = (1 - a) / th2
    x, y, z = w.unbind(-1)
    o = torch.zeros_like(x)
    k = torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(*w.shape[:-1], 3, 3)
    k2 = k @ k
    eye = torch.eye(3, device=w.device, dtype=w.dtype)
    return eye + a * k + b * k2, eye + b * k + c * k2


def rho_prime(r, c, kind):
    """rho'(r) of run_tto_pba.robust."""
    if kind == "huber":
        return (r / c).clamp(-1, 1)
    x2 = (r / c) ** 2
    if kind == "l2":
        return r / c
    if kind == "cauchy":
        return r / (c * (1 + x2))
    if kind == "gm":
        return r / (c * (1 + x2) ** 2)
    if kind == "tukey":
        return r * (1 - x2).clamp_min(0) ** 2 / c
    raise ValueError(f"unknown kernel {kind}")


def rho(r, c, kind):
    if kind == "huber":
        a = r.abs()
        return torch.where(a < c, 0.5 * r * r / c, a - 0.5 * c)
    x2 = (r / c) ** 2
    if kind == "l2":
        return 0.5 * c * x2
    if kind == "cauchy":
        return 0.5 * c * torch.log1p(x2)
    if kind == "gm":
        return 0.5 * c * x2 / (1 + x2)
    if kind == "tukey":
        return c / 6 * (1 - (1 - x2.clamp(max=1)) ** 3)
    raise ValueError(f"unknown kernel {kind}")


def rho_weight(r, c, kind):
    """IRLS weight rho'(r) / r."""
    if kind == "huber":
        return torch.where(r.abs() < c, 1 / c, 1 / r.abs().clamp_min(1e-12))
    x2 = (r / c) ** 2
    if kind == "l2":
        return torch.full_like(r, 1 / c)
    if kind == "cauchy":
        return 1 / (c * (1 + x2))
    if kind == "gm":
        return 1 / (c * (1 + x2) ** 2)
    if kind == "tukey":
        return (1 - x2).clamp_min(0) ** 2 / c
    raise ValueError(f"unknown kernel {kind}")


class FlatAdam:
    """Adam on one vector holding all parameters (the parameters become views of it), with a learning rate per entry.

    Same update as torch.optim.Adam with one group per parameter, in 6 instead of ~50 tensor operations."""

    def __init__(self, params, lrs, betas=(0.9, 0.999), eps=1e-8):
        self.params, self.b1, self.b2, self.eps = params, betas[0], betas[1], eps
        self.x = torch.cat([p.data.reshape(-1) for p in params])
        self.g = torch.zeros_like(self.x)
        self.grads, o = [], 0
        for p in params:
            n = p.numel()
            p.data = self.x[o : o + n].view(p.shape)
            self.grads.append(self.g[o : o + n].view(p.shape))
            o += n
        self.lr = torch.cat([torch.full((p.numel(),), float(lr), device=self.x.device, dtype=self.x.dtype) for p, lr in zip(params, lrs)])
        self.reset()

    def reset(self):
        self.m, self.v, self.t = torch.zeros_like(self.x), torch.zeros_like(self.x), 0

    def step(self, scale=1.0):
        self.t += 1
        self.m.lerp_(self.g, 1 - self.b1)
        self.v.mul_(self.b2).addcmul_(self.g, self.g, value=1 - self.b2)
        denom = (self.v.sqrt() / math.sqrt(1 - self.b2**self.t)).add_(self.eps)
        self.x.sub_(self.lr * (self.m / denom) * (scale / (1 - self.b1**self.t)))


class Warps:
    """Forward quantities of a batch of warps, see Kernel.warp."""

    __slots__ = ("rad", "A", "E", "Xc", "Xt", "ds", "s", "Rji", "tji", "src", "nbrs", "iz", "ux", "vy", "I", "Iu", "Iv", "lu", "lv", "geo", "vis", "base_g", "wt", "fxt", "fyt")


class Kernel:
    """The objective of PBA.loss at one pyramid level, its gradient and its Gauss-Newton blocks. All in torch.no_grad."""

    def __init__(self, model, cfg=None):
        cfg = cfg or model.cfg
        if cfg["focal"] == "per_frame" or cfg["depth_grid_fine"] > 0 or cfg["depth_param"] != "grid" or cfg["photo_agg"] != "mean" or cfg["w_match"] > 0:
            sys.exit("impl=analytic / lm_impl=analytic support focal=shared|none, depth_grid_fine=0, depth_param=grid, photo_agg=mean and w_match=0")
        self.m, self.cfg = model, cfg
        self.N, self.H, self.W = model.N, model.H, model.W
        self.gh, self.gw = model.grid.shape[1:]
        self.nc = self.gh * self.gw
        self.sx, self.sy = (self.gw - 1) / (self.W - 1), (self.gh - 1) / (self.H - 1)
        self.has_grid, self.has_focal, self.has_pose = cfg["depth_grid"] > 0, cfg["focal"] == "shared", cfg["poses"]
        self.kind, self.dp, self.dg = cfg["photo_loss"], cfg["huber_photo"], cfg["huber_geo"]
        self.ncc, self.S = cfg["photo_loss"] == "ncc", 9 if cfg["photo_loss"] == "ncc" else 1
        self.radial = cfg["radial"]  # one shared coefficient of the division model: x_undistorted = x_distorted / (1 + k r_distorted^2)
        self.quad = torch.tensor([0, 1, self.gw, self.gw + 1], device=model.R0.device)
        self.frames = torch.arange(self.N, device=model.R0.device)
        self.fx0, self.fy0, self.cx, self.cy = model.K0[:, 0, 0], model.K0[:, 1, 1], model.K0[:, 0, 2], model.K0[:, 1, 2]
        dev, dt = model.R0.device, model.R0.dtype
        self.f0 = torch.stack([self.fx0, self.fy0], -1)
        self.c0 = torch.stack([self.cx, self.cy], -1)
        self.uv_lo = torch.zeros(2, device=dev, dtype=dt)
        self.uv_hi = torch.tensor([self.W - 1.0, self.H - 1.0], device=dev, dtype=dt)  # last pixel
        self.uv_base = torch.tensor([self.W - 2.0, self.H - 2.0], device=dev, dtype=dt)  # last pixel a bilinear cell can start at
        self.tex_quad = torch.tensor([0, 1, self.W, self.W + 1], device=dev)
        self.grid_scale = torch.tensor([self.sx, self.sy], device=dev, dtype=dt)
        self.grid_base = torch.tensor([self.gw - 2.0, self.gh - 2.0], device=dev, dtype=dt)
        self.use_bright = False
        self.cg_iters, self.schur_failed = [], 0
        self.factor, self.grad_buf, self.chords, self.factorisations = None, None, 0, 0
        self.w_geo_lm = cfg["w_geo"] if cfg["lm_w_geo"] < 0 else cfg["lm_w_geo"]  # of the LM levels only
        self.z, self.w_z, self.fd = None, cfg["w_free_depth"], None  # lm_free_depth: log depth per sample [N, P], its prior, the blocks of normal_eq

    def prep(self, nbrs):
        """What warp needs of the neighbours nbrs [n, k] and does not change during a level."""
        return self.f0[nbrs][:, :, None], self.c0[nbrs][:, :, None], nbrs[:, :, None] * (self.H * self.W), nbrs[:, :, None] * self.nc

    # ---------------------------------------------------------------- data of one level
    def set_level(self, img, feature_is_rgb=True, stride=1.0):
        """img [N, C, H, W]. The texture holds the channels and the depth, so that one lookup fetches both."""
        self.C, self.stride = img.shape[1], stride
        self.tex = torch.cat([img.permute(0, 2, 3, 1).reshape(-1, self.C), self.m.depth0.reshape(-1, 1)], 1)
        self.use_bright = bool(feature_is_rgb and self.cfg["bright"])
        self.bright_on = self.m.free[:, 0] * float(self.use_bright)

    def make_pool(self, pix):
        """Candidate pixels pix [N, P] (long) -> per-sample constants feat [N, P, D] (image coordinates at the initial focal
        length x, y, depth, 4 bilinear grid weights, C channels) and grid quads [N, P], where D = 7 + C.
        With NCC: of the 3x3 patches around them, feat [N, P, 9, D + 1] and quads [N, P, 9], the source patch zero-mean,
        its squared norm in the last channel."""
        W, H = self.W, self.H
        u, v = (pix % W).float(), (pix // W).float()
        if self.ncc:
            off = torch.tensor([-1.0, 0.0, 1.0], device=u.device) * self.stride
            du_, dv_ = off[None, :].expand(3, 3).reshape(-1), off[:, None].expand(3, 3).reshape(-1)
            u, v = (u[..., None] + du_).clamp(0, W - 1), (v[..., None] + dv_).clamp(0, H - 1)
        frame = self.frames[:, None, None] if self.ncc else self.frames[:, None]
        u0, v0 = u.long(), v.long()
        flat = self.tex[frame * (H * W) + v0 * W + u0]
        if self.ncc and self.stride != int(self.stride):  # not on the pixel grid: bilinear, as bilinear_gather
            u1, v1 = u.floor().clamp_max(W - 2), v.floor().clamp_max(H - 2)
            a, b = (u - u1)[..., None], (v - v1)[..., None]
            base = frame * (H * W) + (v1 * W + u1).long()
            top = self.tex[base] * (1 - a) + self.tex[base + 1] * a
            bot = self.tex[base + W] * (1 - a) + self.tex[base + W + 1] * a
            ch = (top * (1 - b) + bot * b)[..., :-1]
            flat = torch.cat([ch, flat[..., -1:]], -1)
        gu, gv = u * self.sx, v * self.sy
        gu0, gv0 = gu.floor().clamp(0, self.gw - 2), gv.floor().clamp(0, self.gh - 2)
        du, dv = gu - gu0, gv - gv0
        sh = (self.N, 1, 1) if self.ncc else (self.N, 1)  # image coordinates at the initial focal length
        xb, yb = (u - self.cx.view(sh)) / self.fx0.view(sh), (v - self.cy.view(sh)) / self.fy0.view(sh)
        head = torch.stack([xb, yb, flat[..., -1], (1 - du) * (1 - dv), du * (1 - dv), (1 - du) * dv, du * dv], -1)
        ch = flat[..., :-1]
        if self.ncc:
            ch = ch - ch.mean((-2, -1), keepdim=True)
            aa = (ch * ch).sum((-2, -1), keepdim=True).expand(*ch.shape[:-1], 1)
            ch = torch.cat([ch, aa], -1)
        return torch.cat([head, ch], -1), gv0.long() * self.gw + gu0.long()

    def take(self, feat_pool, quad_pool, idx):
        """The candidates idx [N, P] of a pool as samples [N, M, D] (with NCC, M = 9 P)."""
        rows = self.frames[:, None]
        f, q = feat_pool[rows, idx], quad_pool[rows, idx]
        return (f.flatten(1, 2), q.flatten(1, 2)) if self.ncc else (f, q)

    # ---------------------------------------------------------------- state
    def state(self):
        m = self.m
        fr = m.free
        if self.has_pose:
            dR, Jl = so3_exp_jl(m.rot.data * fr)
            tau = m.trans.data * fr
            R = dR @ m.R0
            t = (dR @ m.t0[..., None])[..., 0] + tau
        else:
            R, Jl, tau, t = m.R0, None, torch.zeros_like(m.t0), m.t0
        s = torch.exp(m.logf.data) if self.has_focal else 1.0
        return R, t, Jl, tau, s, m.bright.data * self.bright_on[:, None] if self.use_bright else None

    # ---------------------------------------------------------------- forward
    def warp(self, st, src, feat, quad, nbrs, const):
        """Source frames src [n] (None: all frames) with samples feat [n, M, D], quad [n, M] into their neighbours nbrs
        [n, k], const = prep(nbrs). u and v are handled as one vector and the 4 corners of a bilinear lookup as one index:
        every tensor operation costs ~4 us on a GPU, whatever its size."""
        R, t, _, _, s_f, _ = st
        W = self.W
        w = Warps()
        f_t, c_t, off, goff = const
        fxy = f_t * s_f
        ws = feat[..., 3:7]
        grid = self.m.grid.data.view(self.N, -1)
        if self.has_grid:
            rows = self.frames[:, None, None] if src is None else src[:, None, None]
            ds = feat[..., 2] * torch.exp((grid[rows, quad[..., None] + self.quad] * ws).sum(-1))
        else:
            ds = feat[..., 2]
        if self.z is not None:
            z = self.z if src is None else self.z[src]
            ds = ds * torch.exp(z.repeat_interleave(self.S, 1) if self.ncc else z)
        Rs, ts = (R, t) if src is None else (R[src], t[src])
        xy = feat[..., :2] / s_f
        if self.radial:
            k_r = self.m.kr.data
            rd2 = (xy * xy).sum(-1)
            D = 1 + k_r * rd2
            xy = xy / D[..., None]
            w.A, w.E = (1 - k_r * rd2) / D, rd2 / D  # d log x_u / d(-log f) and d log x_u / d(-k), source side
        ds1 = ds[..., None]
        Xc = torch.cat([xy * ds1, ds1], -1)
        Rji = R[nbrs] @ Rs[:, None].transpose(-1, -2)
        tji = t[nbrs] - (Rji @ ts[:, None, :, None])[..., 0]
        Xt = Xc[:, None] @ Rji.transpose(-1, -2) + tji[:, :, None]
        z = Xt[..., 2]
        zc = z.clamp_min(1e-4)
        iz = 1 / zc
        Xn = Xt[..., :2] * iz[..., None]
        if self.radial:  # r_d = s r_u with s = 2 / (1 + sqrt(1 - 4 k r_u^2))
            xn, yn = Xn.unbind(-1)
            r2 = (Xn * Xn).sum(-1)
            q = torch.sqrt((1 - 4 * k_r * r2).clamp_min(1e-4))
            sc = 2 / (1 + q)
            w.rad = (xn, yn, r2, sc, k_r * sc * sc / q, q)  # the 5th is d s / d r_u^2
            uvx = fxy * sc[..., None] * Xn
        else:
            uvx = fxy * Xn
        uv = uvx + c_t
        uvc = uv.clamp(self.uv_lo, self.uv_hi)
        inside = (uv == uvc).all(-1) & (z > 1e-4)
        # texture: channels + depth, bilinear, with its derivatives in u and v
        uv0 = torch.minimum(uvc.floor(), self.uv_base)
        duv = uvc - uv0
        du, dv = duv[..., :1], duv[..., 1:]
        base = off + (uv0[..., 1] * W + uv0[..., 0]).long()
        a, b, c, d = self.tex[base[..., None] + self.tex_quad].unbind(-2)
        ab, cd = b - a, d - c
        top, bot = a + ab * du, c + cd * du
        Tv = bot - top
        val, Tu = top + Tv * dv, ab + (cd - ab) * dv
        # depth correction of the target at the warped pixel, with its derivatives
        if self.has_grid:
            guv = uvc * self.grid_scale
            guv0 = torch.minimum(guv.floor(), self.grid_base)
            gdu, gdv = (guv - guv0).unbind(-1)
            base_g = goff + (guv0[..., 1] * self.gw + guv0[..., 0]).long()
            a, b, c, d = grid.reshape(-1)[base_g[..., None] + self.quad].unbind(-1)
            ab, cd = b - a, d - c
            top, bot = a + ab * gdu, c + cd * gdu
            gv_ = bot - top
            g_t, g_u = top + gv_ * gdv, (ab + (cd - ab) * gdv) * self.sx
            g_v = gv_ * self.sy
            e_g = torch.exp(g_t)
            w.base_g, w.wt = base_g, (gdu, gdv)
        else:
            e_g, g_u, g_v = 1.0, 0.0, 0.0
            w.base_g, w.wt = None, None
        dt = val[..., -1] * e_g
        w.vis = inside & (z < dt * (1 + self.cfg["occ_thresh"]))
        idt = 1 / dt.clamp_min(1e-4)
        w.geo = torch.log(zc * idt)
        idD = e_g * idt
        w.lu, w.lv = Tu[..., -1] * idD + g_u, Tv[..., -1] * idD + g_v  # d log(depth of the target) / d(u, v)
        w.I, w.Iu, w.Iv = val[..., :-1], Tu[..., :-1], Tv[..., :-1]
        w.ux, w.vy = uvx.unbind(-1)
        w.fxt, w.fyt = fxy.unbind(-1)
        w.Xc, w.Xt, w.ds, w.s, w.Rji, w.tji, w.src, w.nbrs, w.iz = Xc, Xt, ds, feat[..., 7:], Rji, tji, src, nbrs, iz
        return w

    def photo_res(self, st, w):
        """Photometric residual [n, k, M, C] after the affine brightness, and exp(a_j) [n, k, 1, 1] (1.0 without brightness)."""
        br = st[-1]
        if br is None:
            return w.I - w.s[:, None], 1.0
        a, b = br.unbind(-1)
        if self.cfg["bright_rel"]:  # only the source's contrast counts: the residual can't vanish by shrinking all frames' contrast
            a_s, b_s = (a, b) if w.src is None else (a[w.src], b[w.src])
            eat = torch.exp(a[w.nbrs] - a_s[:, None])[:, :, None, None]
            return w.I * eat + (b[w.nbrs] - b_s[:, None])[:, :, None, None] - w.s[:, None], eat
        ea = torch.exp(a)
        eat = ea[w.nbrs][:, :, None, None]
        eas, bs = (ea, b) if w.src is None else (ea[w.src], b[w.src])
        return w.I * eat + b[w.nbrs][:, :, None, None] - (w.s[:, None] * eas[:, None, None, None] + bs[:, None, None, None]), eat

    def ncc_terms(self, w, m):
        """Per-patch NCC of the warped patches: zero-mean target and source patch [n, k, P, D], D = 9 C, squared norm of
        the source patch [n, 1, P], of the target [n, k, P], NCC [n, k, P] and the mask of fully visible patches."""
        n, k, M, C = *w.I.shape[:3], w.I.shape[3]
        P = M // 9
        b = w.I.reshape(n, k, P, 9 * C)
        b = b - b.mean(-1, keepdim=True)
        a = w.s[:, None, ..., :-1].reshape(n, 1, P, 9 * C)
        aa = w.s[:, None, ::9, -1]
        bb = (b * b).sum(-1)
        D = torch.sqrt(aa * bb + 1e-8)
        return b, a, aa, bb, D, (a * b).sum(-1) / D, m.reshape(n, k, P, 9).amin(-1)

    def priors(self):
        """Value of the non-data terms of PBA.loss."""
        m, cfg = self.m, self.cfg
        v = 0.0
        if self.has_grid:
            v = v + cfg["w_prior"] * float((m.grid.data**2).mean())
        if cfg["w_focal"] > 0:
            v = v + cfg["w_focal"] * float((m.logf.data**2).mean())
        if cfg["w_pose"] > 0:
            v = v + cfg["w_pose"] * float(((m.rot.data * m.free) ** 2 + (m.trans.data * m.free) ** 2).mean())
        if self.radial:
            v = v + cfg["w_radial"] * float((m.kr.data**2).sum())
        return v

    def z_prior(self, n):
        """0.5 w_z z^2 per sample, in the units of a warp (the data terms are divided by the visible warps n)."""
        return 0.5 * self.w_z * float((self.z**2).sum()) / n if self.z is not None else 0.0

    def cost(self, feat, quad, nbrs, chunk):
        """PBA.loss for samples feat [N, M, D]."""
        cfg = self.cfg
        st = self.state()
        sp, sg, n, np_ = 0.0, 0.0, 0.0, 0.0
        for i in range(0, self.N, chunk):
            src = self.frames[i : i + chunk]
            nb = nbrs[src]
            w = self.warp(st, src, feat[src], quad[src], nb, self.prep(nb))
            m = w.vis.float()
            if self.ncc:
                *_, ncc, mp = self.ncc_terms(w, m)
                sp, np_ = sp + ((1 - ncc) * mp).sum(), np_ + mp.sum()
            else:
                r, _ = self.photo_res(st, w)
                sp = sp + (rho(r, self.dp, self.kind).mean(-1) * m).sum()
            sg, n = sg + (rho(w.geo, self.dg, "huber") * m).sum(), n + m.sum()
        n = max(float(n), 1.0)
        return cfg["w_photo"] * float(sp) / (max(float(np_), 1.0) if self.ncc else n) + self.w_geo_lm * float(sg) / n + self.priors() + self.z_prior(n)

    # ---------------------------------------------------------------- gradient for Adam
    def gradient(self, feat, quad, nbrs, const, grads, want_loss=False):
        """Fill grads (rot, trans, logf, grid, bright) with the gradient of PBA.loss. Returns (total, parts) if want_loss.
        No float() in here: it would make every iteration wait for the GPU."""
        cfg, m_, N = self.cfg, self.m, self.N
        st = self.state()
        _, _, Jl, tau, _, br = st
        w = self.warp(st, None, feat, quad, nbrs, const)
        m = w.vis.float()
        den = m.sum().clamp_min(1)
        c = self.C
        dLg = m * rho_prime(w.geo, self.dg, "huber") * (cfg["w_geo"] / den)  # [N, k, M]
        if self.ncc:  # dL / d(warped value), L = (1 - NCC) over the fully visible patches
            b_, a_, aa, bb, D, ncc, mp = self.ncc_terms(w, m)
            den_p = mp.sum().clamp_min(1)
            dLt = ((a_ - (ncc * aa / D)[..., None] * b_) * (-cfg["w_photo"] * mp / (D * den_p))[..., None]).reshape(*w.I.shape)
            r = dLr = None
        else:
            r, eat = self.photo_res(st, w)
            dLr = m[..., None] * rho_prime(r, self.dp, self.kind) * (cfg["w_photo"] / (c * den))  # [N, k, M, C]
            dLt = dLr * eat if br is not None else dLr
        Gu = (dLt * w.Iu).sum(-1) - dLg * w.lu
        Gv = (dLt * w.Iv).sum(-1) - dLg * w.lv
        T = Gu * w.ux + Gv * w.vy
        if self.radial:
            xn, yn, r2, sc, dsc, q = w.rad
            Pu, Pv = Gu * w.fxt, Gv * w.fyt  # dL / d(u, v) in normalised image coordinates
            cross_ = 2 * dsc * (Pu * xn + Pv * yn)  # from d s / d r^2: d L / d(x_n, y_n) = (Pu, Pv) s + cross_ (x_n, y_n)
            gX = torch.stack([(Pu * sc + cross_ * xn) * w.iz, (Pv * sc + cross_ * yn) * w.iz, (dLg - T - cross_ * r2) * w.iz], -1)
        else:
            gX = torch.stack([Gu * w.fxt * w.iz, Gv * w.fyt * w.iz, (dLg - T) * w.iz], -1)  # dL / dX_t
        rot, trans, logf, grid, bright = grads[:5]
        S1 = gX.sum(2)
        Xr = w.Xt - w.tji[:, :, None]  # R_ji X_c
        if self.has_pose:
            S2 = torch.linalg.cross(w.Xt, gX, dim=-1).sum(2)
            Rij = w.Rji.transpose(-1, -2)
            S1s = (Rij @ S1[..., None])[..., 0]  # R_ij S1, in the source camera
            S2s = (Rij @ (S2 - torch.linalg.cross(w.tji, S1, dim=-1))[..., None])[..., 0]
            S1ss = S1s.sum(1)
            trans.copy_(-S1ss)
            rot.copy_(torch.linalg.cross(tau, S1ss, dim=-1) - S2s.sum(1))
            j = nbrs.reshape(-1)
            trans.index_add_(0, j, S1.reshape(-1, 3))
            rot.index_add_(0, j, (S2 - torch.linalg.cross(tau[nbrs], S1, dim=-1)).reshape(-1, 3))
            rot.copy_((Jl.transpose(-1, -2) @ rot[..., None])[..., 0] * m_.free)
            trans.mul_(m_.free)
            if cfg["w_pose"] > 0:
                f = 2 * cfg["w_pose"] / (3 * N)
                rot.add_(m_.rot.data * m_.free * f)
                trans.add_(m_.trans.data * m_.free * f)
        else:
            rot.zero_(), trans.zero_()
        dLs = (gX * Xr).sum(-1)  # dL / dlog(source depth) per warp
        if self.has_grid:
            ws, gflat = feat[..., 3:7], grid.view(N, -1)
            gflat.zero_()
            gflat.scatter_add_(1, (quad[..., None] + self.quad).reshape(N, -1), (dLs.sum(1)[..., None] * ws).reshape(N, -1))
            gdu, gdv = w.wt
            a, b = dLg * (gdu - 1), -dLg * gdu  # -dLg (1 - gdu), -dLg gdu
            ga, gb, gw_ = a * gdv, b * gdv, self.gw
            base = w.base_g
            grid.reshape(-1).index_add_(0, torch.cat([base, base + 1, base + gw_, base + gw_ + 1], -1).reshape(-1),
                                        torch.cat([a - ga, b - gb, ga, gb], -1).reshape(-1))
            grid.add_(m_.grid.data, alpha=2 * cfg["w_prior"] / m_.grid.numel())
        else:
            grid.zero_()
        if self.has_focal or self.radial:
            r3 = w.Rji[..., 2][:, :, None]
            src_part = -dLs + w.ds[:, None] * (gX * r3).sum(-1)  # d L / d(-log x_u) of the source's (x, y)
        if self.has_focal:
            logf.fill_((T + (src_part * w.A[:, None] if self.radial else src_part)).sum())
            if cfg["w_focal"] > 0:
                logf.add_(m_.logf.data, alpha=2 * cfg["w_focal"] / m_.logf.numel())
        else:
            logf.zero_()
        if self.radial:
            grads[5].fill_((T * r2 * sc / q + src_part * w.E[:, None]).sum())
            grads[5].add_(m_.kr.data, alpha=2 * cfg["w_radial"])
        # brightness: r = I_j exp(a_j) + b_j - (I_i exp(a_i) + b_i)
        if br is not None:
            s_all = dLr.sum(-1).sum(2)  # [N, k]
            ga_, gb_ = bright[:, 0], bright[:, 1]
            ja = nbrs.reshape(-1)
            if cfg["bright_rel"]:  # r = I_j exp(a_j - a_i) + b_j - b_i - I_i
                e_ = (dLr * w.I).sum((-1, -2)) * eat[:, :, 0, 0]
                ga_.copy_(-e_.sum(1))
                ga_.index_add_(0, ja, e_.reshape(-1))
            else:
                ea = torch.exp(br[:, 0])
                ga_.copy_(-(dLr * w.s[:, None]).sum((-1, -2)).sum(1) * ea)
                ga_.index_add_(0, ja, ((dLr * w.I).sum((-1, -2)) * ea[nbrs]).reshape(-1))
            gb_.copy_(-s_all.sum(1))
            gb_.index_add_(0, ja, s_all.reshape(-1))
            bright.mul_(self.bright_on[:, None])
        else:
            bright.zero_()
        if not want_loss:
            return None
        lp = float(((1 - ncc) * mp).sum() / den_p) if self.ncc else float((rho(r, self.dp, self.kind).mean(-1) * m).sum() / den)
        lg = float((rho(w.geo, self.dg, "huber") * m).sum() / den)
        total = cfg["w_photo"] * lp + cfg["w_geo"] * lg + self.priors()
        lprior = float((m_.grid.data**2).mean()) if self.has_grid else 0.0
        return total, {"photo": lp, "geo": lg, "prior": lprior, "visible": float(m.mean())}

    # ---------------------------------------------------------------- Gauss-Newton
    def normal_eq(self, feat, quad, nbrs, chunk, cross_frames):
        """Gauss-Newton blocks of the IRLS-weighted objective at the current parameters and samples feat [N, M, D].

        Unknowns: p = poses of frames 1..N-1 (rot, trans), the log focal; g = the grid cells. Returns the blocks
        Hpp [np, np], Hgp [N, nc, np], Hgg [N, nc, nc] (the coupling of a frame's own cells), gp [np], gg [N, nc], the
        coupling of the cells of different frames Hx [N nc, N nc] (or None without cross_frames) and the loss.
        """
        cfg, N, nc = self.cfg, self.N, self.nc
        dev, st = self.frames.device, self.state()
        Jl, tau = st[2], st[3]
        n_pose = 6 * (N - 1) if self.has_pose else 0
        nq = 14 if self.radial else 13  # columns per warp: 2 x 6 poses, log focal, (radial coefficient)
        np_ = n_pose + 1 + int(self.radial)
        dummy = np_  # index of everything that is fixed (frame 0, or the poses or the focal when they are not optimised)
        pose_idx = torch.full((N, 6), dummy, dtype=torch.long, device=dev)
        if self.has_pose:
            pose_idx[1:] = torch.arange(n_pose, device=dev).view(N - 1, 6)
        focal_idx = n_pose if self.has_focal else dummy
        k_idx = n_pose + 1
        Hpp = torch.zeros((np_ + 1) ** 2, device=dev)
        gp = torch.zeros(np_ + 1, device=dev)
        has_grid = self.has_grid and self.z is None  # with free depths the grid stays in the warp, fixed
        Hgp = torch.zeros(N * nc * (np_ + 1), device=dev) if has_grid else None
        Hgg = torch.zeros(N, nc, nc, device=dev) if has_grid else None
        gg = torch.zeros(N, nc, device=dev) if has_grid else None
        Hx = torch.zeros(N * nc, N * nc, device=dev) if has_grid and cross_frames else None
        sp, sg, n_vis, np_sum = 0.0, 0.0, 0.0, 0.0
        ncc_scale = 1.0
        if self.z is not None:
            hz, bz, fd_chunks = torch.zeros_like(self.z), torch.zeros_like(self.z), []
        if self.ncc:  # the rows are normalised by the visible samples at the end, NCC's by the visible patches
            nv, npatch = 0.0, 0.0
            for i in range(0, N, chunk):
                src = self.frames[i : i + chunk]
                nb = nbrs[src]
                w = self.warp(st, src, feat[src], quad[src], nb, self.prep(nb))
                m = w.vis.float()
                nv, npatch = nv + float(m.sum()), npatch + float(m.reshape(*m.shape[:2], -1, 9).amin(-1).sum())
            ncc_scale = max(nv, 1.0) / max(npatch, 1.0)
        for i in range(0, N, chunk):
            src = self.frames[i : i + chunk]
            nb = nbrs[src]
            n, k = len(src), nb.shape[1]
            w = self.warp(st, src, feat[src], quad[src], nb, self.prep(nb))
            M, C = w.I.shape[2], w.I.shape[3]
            m = w.vis.float()
            sg += float((rho(w.geo, self.dg, "huber") * m).sum())
            n_vis += float(m.sum())
            iz = w.iz
            if self.radial:
                xn, yn, r2, sc, dsc, q = w.rad
                fi, gi = w.fxt * iz, w.fyt * iz
                a_, xy = sc + 2 * r2 * dsc, 2 * xn * yn * dsc
                P1 = torch.stack([fi * (sc + 2 * xn * xn * dsc), fi * xy, -fi * xn * a_], -1)
                P2 = torch.stack([gi * xy, gi * (sc + 2 * yn * yn * dsc), -gi * yn * a_], -1)
                k_t = r2 * sc / q  # d u / dk = u_x k_t (and v)
            else:
                P1 = torch.stack([w.fxt * iz, torch.zeros_like(iz), -w.ux * iz], -1)  # d u / dX_t
                P2 = torch.stack([torch.zeros_like(iz), w.fyt * iz, -w.vy * iz], -1)  # d v / dX_t
            Xr = w.Xt - w.tji[:, :, None]
            Rji = w.Rji
            r3 = Rji[..., 2][:, :, None]
            Xs = (w.Xc - tau[src][:, None])[:, None]
            Xs_t = w.Xt - tau[nb][:, :, None]
            ds_ = w.ds[:, None]

            def rows(gX, T):
                """Columns rot_s, trans_s, rot_t, trans_t (in tangent coordinates), logf and dlog(source depth) of warps with
                X-space gradient gX [n, k, M, 3]."""
                ts = -(gX @ Rji)
                dds = (gX * Xr).sum(-1)
                sp_ = -dds + ds_ * (gX * r3).sum(-1)
                cols = [torch.linalg.cross(Xs, ts, dim=-1), ts, torch.linalg.cross(Xs_t, gX, dim=-1), gX]
                if self.radial:
                    cols += [(T + sp_ * w.A[:, None])[..., None], (T * k_t + sp_ * w.E[:, None])[..., None]]
                else:
                    cols += [(T + sp_)[..., None]]
                return torch.cat(cols, -1), dds

            # geometric row
            gXg = -w.lu[..., None] * P1 - w.lv[..., None] * P2
            gXg[..., 2] += iz
            Jg, dg_ = rows(gXg, -(w.lu * w.ux + w.lv * w.vy))
            wg = m * rho_weight(w.geo, self.dg, "huber") * self.w_geo_lm
            if self.ncc:
                # photometric rows: difference of the normalised patches, half its squared norm is 1 - NCC
                b_, a_, aa, bb, D_, ncc, mp = self.ncc_terms(w, m)
                sp += float(((1 - ncc) * mp).sum())
                np_sum += float(mp.sum())
                Ju, du_ = rows(P1, w.ux)
                Jv, dv_ = rows(P2, w.vy)
                Jb = w.Iu[..., None] * torch.cat([Ju, du_[..., None]], -1)[..., None, :] + w.Iv[..., None] * torch.cat([Jv, dv_[..., None]], -1)[..., None, :]
                P_ = M // 9
                D = 9 * C
                Jb = Jb.reshape(n, k, P_, D, nq + 1)
                Jb = Jb - Jb.mean(-2, keepdim=True)
                nb_ = torch.sqrt(bb + 1e-8)
                bh = b_ / nb_[..., None]
                Jb = (Jb - bh[..., None] * (bh[..., None, :] @ Jb)) / nb_[..., None, None]
                r = bh - a_ / torch.sqrt(aa + 1e-8)[..., None]  # [n, k, P, D]
                Jp, dp_ = Jb[..., :nq], Jb[..., nq]
                wp = (mp * cfg["w_photo"] * ncc_scale)[..., None].expand_as(r)
            else:
                r, eat = self.photo_res(st, w)
                sp += float((rho(r, self.dp, self.kind).mean(-1) * m).sum())
                # X-space gradient of the photometric rows [n, k, M, C, 3]: per channel
                E = eat[..., 0] if torch.is_tensor(eat) else eat
                Tp = (w.Iu * w.ux[..., None] + w.Iv * w.vy[..., None])
                if C == 1:
                    Jp, dp_ = rows((w.Iu[..., 0, None] * P1 + w.Iv[..., 0, None] * P2) * (E[..., None] if torch.is_tensor(E) else E), Tp[..., 0] * E)
                    Jp, dp_ = Jp[..., None, :], dp_[..., None]
                else:
                    Ju, du_ = rows(P1, w.ux)
                    Jv, dv_ = rows(P2, w.vy)
                    Ea = E[..., None, None] if torch.is_tensor(E) else E
                    Jp = (w.Iu[..., None] * Ju[..., None, :] + w.Iv[..., None] * Jv[..., None, :]) * Ea
                    dp_ = (w.Iu * du_[..., None] + w.Iv * dv_[..., None]) * (E[..., None] if torch.is_tensor(E) else E)
                wp = m[..., None] * rho_weight(r, self.dp, self.kind) * (cfg["w_photo"] / C)
            if not self.has_focal:
                Jg[..., 12] = 0
                Jp[..., 12] = 0
            # pose and focal: one 13 x 13 block per frame pair
            Ap = Jp.reshape(n, k, -1, nq)
            wa = wp.reshape(n, k, -1)
            ra = r.reshape(n, k, -1)
            H13 = (Ap * wa[..., None]).transpose(-1, -2) @ Ap + (Jg * wg[..., None]).transpose(-1, -2) @ Jg
            g13 = (Ap * (wa * ra)[..., None]).sum(2) + (Jg * (wg * w.geo)[..., None]).sum(2)
            # to the coordinates of the parameters (dphi = J_l(rot) drot)
            T13 = torch.eye(nq, device=dev, dtype=H13.dtype).repeat(n, k, 1, 1)
            if self.has_pose:
                T13[..., 0:3, 0:3] = Jl[src][:, None].transpose(-1, -2).expand(n, k, 3, 3)
                T13[..., 6:9, 6:9] = Jl[nb].transpose(-1, -2)
            H13 = T13 @ H13 @ T13.transpose(-1, -2)
            g13 = (T13 @ g13[..., None])[..., 0]
            ids = [pose_idx[src][:, None].expand(n, k, 6), pose_idx[nb], torch.full((n, k, 1), focal_idx, device=dev)]
            if self.radial:
                ids.append(torch.full((n, k, 1), k_idx, device=dev))
            ids = torch.cat(ids, -1)
            Hpp.index_add_(0, (ids[..., None] * (np_ + 1) + ids[..., None, :]).reshape(-1), H13.reshape(-1))
            gp.index_add_(0, ids.reshape(-1), g13.reshape(-1))
            if self.z is not None:
                # per sample: its diagonal entry, gradient, and coupling with the poses of its source and target frames, the focal and k
                dpw = dp_ * wp
                gw_ = dg_ * wg
                if self.ncc:
                    Vs = (gw_[..., None] * Jg).reshape(n, k, M // 9, 9, nq).sum(3)
                    hg_, bg_ = (gw_ * dg_).reshape(n, k, M // 9, 9).sum(-1), (gw_ * w.geo).reshape(n, k, M // 9, 9).sum(-1)
                else:
                    Vs, hg_, bg_ = gw_[..., None] * Jg, gw_ * dg_, gw_ * w.geo
                V = ((dpw[..., None] * Jp).sum(3) + Vs) @ T13.transpose(-1, -2)  # [n, k, P, nq], in the coordinates of the parameters
                hz[src] += ((dpw * dp_).sum(-1) + hg_).sum(1)
                bz[src] += ((dpw * r).sum(-1) + bg_).sum(1)
                Vl = torch.cat([V[..., :6].sum(1), V[..., 6:12].transpose(1, 2).flatten(2), V[..., 12:].sum(1)], -1)  # [n, P, L], L = 6 + 6 k + 1 or 2
                il = torch.cat([ids[:, 0, :6], ids[:, :, 6:12].flatten(1), ids[:, 0, 12:]], -1)  # [n, L]
                fd_chunks.append((src, Vl.double(), il))
            if not has_grid:
                continue
            # depth grid: dense bilinear weights [.., M, nc] of the source and the warped pixels
            ws = feat[src][..., 3:7]
            Ws = torch.zeros(n, M, nc, device=dev).scatter_(2, quad[src][..., None] + self.quad, ws)
            gdu, gdv = w.wt
            ct = w.base_g[..., None] - nb[:, :, None, None] * nc + self.quad  # cell within the target's grid
            wt = torch.stack([(1 - gdu) * (1 - gdv), gdu * (1 - gdv), (1 - gdu) * gdv, gdu * gdv], -1)
            Wt = torch.zeros(n, k, M, nc, device=dev).scatter_(3, ct, wt)
            # source role: the row's derivative in log depth is dp_ / dg_. The photometric rows of NCC belong to a patch
            # and use the mean weights of its samples (as the forward-mode version: the weights of the patch centre)
            dsp = (dp_ * wp)  # [n, k, P, Dp]
            Vp = (dsp[..., None] * Jp).sum(3)  # [n, k, P, 13]
            Vg = (dg_ * wg)[..., None] * Jg  # [n, k, M, 13]
            gsp = (dsp * r).sum(-1).sum(1)  # [n, P]
            gsg = (dg_ * wg * w.geo).sum(1)  # [n, M]
            h2p = (dsp * dp_).sum(-1).sum(1)
            h2g = (dg_ * dg_ * wg).sum(1)
            if self.ncc:
                Wp = Ws.reshape(n, -1, 9, nc).mean(2)  # [n, P, nc]
                Hs = Wp[:, None].transpose(-1, -2) @ Vp + Ws[:, None].transpose(-1, -2) @ Vg
                gg[src] += (Wp.transpose(-1, -2) @ gsp[..., None])[..., 0] + (Ws.transpose(-1, -2) @ gsg[..., None])[..., 0]
                Hgg[src] += (Wp * h2p[..., None]).transpose(-1, -2) @ Wp + (Ws * h2g[..., None]).transpose(-1, -2) @ Ws
            else:
                Hs = Ws[:, None].transpose(-1, -2) @ (Vp + Vg)  # [n, k, nc, 13]
                gg[src] += (Ws.transpose(-1, -2) @ (gsp + gsg)[..., None])[..., 0]
                Hgg[src] += (Ws * (h2p + h2g)[..., None]).transpose(-1, -2) @ Ws
            # target role: only the geometric row depends on the target's cells, with derivative -1
            Ht = Wt.transpose(-1, -2) @ (-wg[..., None] * Jg)  # [n, k, nc, 13]
            gt = Wt.transpose(-1, -2) @ (-wg * w.geo)[..., None]  # [n, k, nc, 1]
            gg.index_add_(0, nb.reshape(-1), gt[..., 0].reshape(-1, nc))
            # 16 entries per warp, by frame of the target (a matrix product would touch nc^2 per warp)
            fr = nb[:, :, None, None]
            flat = (fr * nc + ct)[..., None] * nc + ct[..., None, :]  # frame nc^2 + cell nc + cell
            Hgg.view(-1).index_add_(0, flat.reshape(-1), (wt[..., None] * wt[..., None, :] * wg[..., None, None]).reshape(-1))
            Tt = T13.transpose(-1, -2)
            Hs, Ht = Hs @ Tt, Ht @ Tt
            size = nc * (np_ + 1)
            cell = torch.arange(nc, device=dev)[None, None, :, None] * (np_ + 1)
            Hgp.index_add_(0, (src[:, None, None, None] * size + cell + ids[:, :, None]).reshape(-1), Hs.reshape(-1))
            Hgp.index_add_(0, (nb[:, :, None, None] * size + cell + ids[:, :, None]).reshape(-1), Ht.reshape(-1))
            if Hx is not None:
                # 16 entries per warp: cell of the source frame x cell of the target frame
                cs_ = src[:, None, None, None] * nc + (quad[src][..., None] + self.quad)[:, None]  # [n, 1, M, 4]
                ct_ = fr * nc + ct  # [n, k, M, 4]
                flat = cs_[..., None] * (N * nc) + ct_[..., None, :]
                vals = ws[:, None, :, :, None] * wt[..., None, :] * (-dg_ * wg)[..., None, None]
                Hx.view(-1).index_add_(0, flat.reshape(-1), vals.reshape(-1))
        n_vis = max(n_vis, 1.0)
        Hpp = Hpp.view(np_ + 1, np_ + 1)[:np_, :np_].double() / n_vis
        gp = gp[:np_].double() / n_vis
        if self.has_pose and cfg["w_pose"] > 0:
            f = 2 * cfg["w_pose"] / (3 * N)
            Hpp[range(n_pose), range(n_pose)] += f
            gp[:n_pose] += f * torch.cat([self.m.rot.data[1:], self.m.trans.data[1:]], 1).reshape(-1).double()
        if self.radial:  # keeps the coefficient near zero where the data do not determine it
            Hpp[k_idx, k_idx] += 2 * cfg["w_radial"]
            gp[k_idx] += 2 * cfg["w_radial"] * float(self.m.kr.data[0])
        if self.has_focal and cfg["w_focal"] > 0:
            Hpp[focal_idx, focal_idx] += 2 * cfg["w_focal"]
            gp[focal_idx] += 2 * cfg["w_focal"] * float(self.m.logf.data[0])
        if has_grid:
            Hgp = Hgp.view(N, nc, np_ + 1)[..., :np_].double() / n_vis
            Hgg = Hgg.double() / n_vis
            gg = gg.double() / n_vis
            f = 2 * cfg["w_prior"] / self.m.grid.numel()
            Hgg[:, range(nc), range(nc)] += f
            gg += f * self.m.grid.data.view(N, -1).double()
            if Hx is not None:
                Hx = (Hx + Hx.T).double() / n_vis
        loss = cfg["w_photo"] * sp / (max(np_sum, 1.0) if self.ncc else n_vis) + self.w_geo_lm * sg / n_vis + self.priors() + self.z_prior(n_vis)
        if self.z is not None:
            self.fd = ((hz.double() + self.w_z) / n_vis, (bz.double() + self.w_z * self.z.double()) / n_vis, [(s_, V_ / n_vis, i_) for s_, V_, i_ in fd_chunks], np_ + 1)
        return Hpp, Hgp, Hgg, gp, gg, Hx, loss

    def apply(self, delta_p, delta_g, sign=1.0):
        m = self.m
        d = delta_p.float() * sign
        if self.has_pose:
            n = 6 * (self.N - 1)
            q = d[:n].view(self.N - 1, 6)
            m.rot.data[1:] += q[:, :3]
            m.trans.data[1:] += q[:, 3:]
            if self.has_focal:
                m.logf.data += d[n]
            if self.radial:
                m.kr.data += d[n + 1]
        else:
            if self.has_focal:
                m.logf.data += d[0]
            if self.radial:
                m.kr.data += d[1]
        if self.radial:
            m.kr.data.clamp_(max=self.cfg["radial_max"])
        if delta_g is not None:
            if self.z is not None:
                self.z += delta_g.float() * sign
            else:
                m.grid.data += delta_g.float().view_as(m.grid.data) * sign

    def chord_step(self, feat, quad, nbrs, chunk, lam):
        """LM iteration with the factorisation of an earlier iteration: only the gradient is new. Returns None if it does not
        lower the cost of these samples (the caller then does a full iteration)."""
        L, sc = self.factor
        if self.grad_buf is None:
            self.grad_buf = [torch.zeros_like(p.data) for p in (self.m.rot, self.m.trans, self.m.logf, self.m.grid, self.m.bright, self.m.kr)]
        cost0 = self.cost(feat, quad, nbrs, chunk)
        self.gradient(feat, quad, nbrs, self.prep(nbrs), self.grad_buf)
        rot, trans, logf, grid, _, kr = self.grad_buf
        parts = []
        if self.has_pose:
            parts.append(torch.cat([rot[1:], trans[1:]], 1).reshape(-1))
        parts.append(logf.reshape(-1) if self.has_focal else torch.zeros(1, device=rot.device, dtype=rot.dtype))
        if self.radial:
            parts.append(kr.reshape(-1))
        n = sum(p.numel() for p in parts)
        if self.has_grid:
            parts.append(grid.reshape(-1))
        rhs = torch.cat(parts).double()
        if sc is not None:
            d = -torch.cholesky_solve((rhs / sc).float()[:, None], L)[:, 0].double() / sc
        else:
            d = -torch.cholesky_solve(rhs[:, None], L)[:, 0]
        dp_, dg_ = d[:n], d[n:].view(self.N, self.nc) if self.has_grid else None
        params = (self.m.rot, self.m.trans, self.m.logf, self.m.grid, self.m.kr)
        state = [x.data.clone() for x in params]
        # the matrix is from an earlier point, and a full step there may have been doubled: try shorter steps, keep the first that helps
        for scale in (1.0, 0.5, 0.25):
            self.apply(dp_ * scale, dg_ * scale if dg_ is not None else None)
            cost1 = self.cost(feat, quad, nbrs, chunk)
            if cost1 < cost0:
                return lam, cost0, cost1, True
            for x, s_ in zip(params, state):
                x.data.copy_(s_)
        return None

    def lm_step(self, feat, quad, nbrs, chunk, lam, solver, max_tries=6, it=0):
        """One Levenberg-Marquardt iteration on fixed samples, with the accept/reject rule of run_tto_pba.lm_step."""
        cfg = self.cfg
        if it == 0:
            self.factor = None
        elif cfg["lm_refactor"] > 1 and it % cfg["lm_refactor"] != 0 and self.factor is not None and solver in ("dense", "dense32"):
            res = self.chord_step(feat, quad, nbrs, chunk, lam)
            if res is not None:
                self.chords += 1
                return res
        self.factorisations += 1
        Hpp, Hgp, Hgg, gp, gg, Hx, cost0 = self.normal_eq(feat, quad, nbrs, chunk, solver in ("dense", "dense32", "pcg"))
        solves_grid = self.has_grid and self.z is None
        if solves_grid:
            dmax = max(float(Hpp.diagonal().max()), float(Hgg.diagonal(dim1=1, dim2=2).max()))
        else:
            dmax = float(Hpp.diagonal().max())
        floor = 1e-12 * dmax + 1e-30
        Dp = Hpp.diagonal().clamp_min(floor)
        Dg = Hgg.diagonal(dim1=1, dim2=2).clamp_min(floor) if solves_grid else None
        if cfg["lm_mu"] > 0:
            mean = (float(Dp.sum()) + (float(Dg.sum()) if solves_grid else 0.0)) / (len(Dp) + (Dg.numel() if solves_grid else 0))
            Hpp = Hpp + torch.diag_embed(torch.full_like(Dp, cfg["lm_mu"] * mean))
            if solves_grid:
                Hgg = Hgg + torch.diag_embed(torch.full_like(Dg, cfg["lm_mu"] * mean))
        params = (self.m.rot, self.m.trans, self.m.logf, self.m.grid, self.m.kr)
        state = [x.data.clone() for x in params]
        z0 = self.z.clone() if self.z is not None else None
        for _ in range(max_tries):
            step = self.solve(Hpp, Hgp, Hgg, gp, gg, Hx, Dp, Dg, lam, solver)
            if step is not None:
                dp_, dg_ = step
                self.apply(dp_, dg_)
                cost1 = self.cost(feat, quad, nbrs, chunk)
                if cost1 < cost0:
                    self.apply(dp_, dg_)
                    cost2 = self.cost(feat, quad, nbrs, chunk)
                    lam_min = cfg["lm_lambda_min"]
                    if cost2 < cost1:
                        return max(lam / 3, lam_min), cost0, cost2, True
                    self.apply(dp_, dg_, -1.0)
                    return max(lam / 3, lam_min), cost0, cost1, True
                for x, s_ in zip(params, state):
                    x.data.copy_(s_)
                if z0 is not None:
                    self.z.copy_(z0)
            lam *= 4
        return lam, cost0, cost0, False

    def solve_free(self, S, gp, lam):
        """Exact Schur complement of the free depths: [[S, V^T], [V, diag(h)]] (dp, dz) = -(gp, b), with h damped like the rest.
        Each source frame contributes one matrix of 6 + 6 k + 1 or 2 rows, whatever the number of samples."""
        h, b, chunks, size = self.fd
        hl = h * (1 + lam)
        C = torch.zeros(size * size, device=gp.device, dtype=gp.dtype)
        rhs = torch.zeros(size, device=gp.device, dtype=gp.dtype)
        for src, V, il in chunks:
            Vh = V / hl[src][..., None]
            C.index_add_(0, (il[..., None] * size + il[:, None, :]).reshape(-1), (V.transpose(-1, -2) @ Vh).reshape(-1))
            rhs.index_add_(0, il.reshape(-1), (Vh * b[src][..., None]).sum(1).reshape(-1))
        L, info = torch.linalg.cholesky_ex(S - C.view(size, size)[:-1, :-1])
        if int(info) != 0:
            return None
        dp_ = -torch.cholesky_solve((gp - rhs[:-1])[:, None], L)[:, 0]
        x = torch.cat([dp_, torch.zeros(1, device=gp.device, dtype=gp.dtype)])  # the last entry is the fixed frame's
        dz = -b / hl
        for src, V, il in chunks:
            dz[src] -= (V / hl[src][..., None] * x[il][:, None]).sum(-1)
        return dp_, dz

    def solve(self, Hpp, Hgp, Hgg, gp, gg, Hx, Dp, Dg, lam, solver):
        """(H + lam diag(H)) delta = -g, None if not PD. dense: Cholesky of everything. schur: eliminate the cells, without the
        coupling Hx of the cells of different frames (a different step). pcg: conjugate gradients on the exact system, with schur as preconditioner."""
        S = Hpp + torch.diag_embed(lam * Dp)
        if self.z is not None:
            return self.solve_free(S, gp, lam)
        if not self.has_grid:
            L, info = torch.linalg.cholesky_ex(S)
            return (-torch.cholesky_solve(gp[:, None], L)[:, 0], None) if int(info) == 0 else None
        A = Hgg + torch.diag_embed(lam * Dg)
        if solver == "frames":  # only the cells, each frame on its own (the others held fixed): pba_patch.depth_stage
            L, info = torch.linalg.cholesky_ex(A)
            if bool((info != 0).any()):
                return None
            return torch.zeros_like(gp), -torch.cholesky_solve(gg[..., None], L)[..., 0]
        if solver in ("dense", "dense32"):
            N, nc = self.N, self.nc
            n = len(gp)
            T = n + N * nc
            H = torch.zeros(T, T, device=gp.device, dtype=gp.dtype)
            H[:n, :n] = S
            Hgpf = Hgp.reshape(N * nc, n)
            H[n:, :n] = Hgpf
            H[:n, n:] = Hgpf.T
            if Hx is not None:
                H[n:, n:] = Hx
            H[n:, n:] += torch.block_diag(*A)
            rhs = torch.cat([gp, gg.reshape(-1)])
            if solver == "dense32":  # single precision after Jacobi scaling: about half the time of float64 on a GPU
                sc = H.diagonal().clamp_min(1e-30).sqrt()
                L, info = torch.linalg.cholesky_ex((H / sc[:, None] / sc[None, :]).float())
                if int(info) != 0:
                    return None
                self.factor = (L, sc)
                d = -torch.cholesky_solve((rhs / sc).float()[:, None], L)[:, 0].double() / sc
                return d[:n], d[n:].view(N, nc)
            L, info = torch.linalg.cholesky_ex(H)
            if int(info) != 0:
                return None
            self.factor = (L, None)
            d = -torch.cholesky_solve(rhs[:, None], L)[:, 0]
            return d[:n], d[n:].view(N, nc)
        L, info = torch.linalg.cholesky_ex(A)
        if bool((info != 0).any()):
            return None
        Y = torch.cholesky_solve(Hgp, L)  # [N, nc, np]
        Hpg = Hgp.transpose(-1, -2)
        S0, S = S, S - (Hpg @ Y).sum(0)
        Lp, info = torch.linalg.cholesky_ex(S)
        if int(info) != 0:
            self.schur_failed += 1
            return None

        def schur(rp, rg):
            """The system without the coupling of different frames: right-hand side rp [np], rg [N, nc]."""
            z = torch.cholesky_solve(rg[..., None], L)  # [N, nc, 1]
            dp_ = torch.cholesky_solve((rp - (Hpg @ z)[..., 0].sum(0))[:, None], Lp)[:, 0]
            return dp_, z[..., 0] - Y @ dp_

        if solver == "schur":
            dp_, dg_ = schur(gp, gg)
            return -dp_, -dg_
        # solver == "pcg": conjugate gradients on the whole system, with the above as the preconditioner
        N, nc = self.N, self.nc
        Hgpf = Hgp.reshape(N * nc, -1)
        Hxf = Hx

        def apply(xp, xg):
            f = xg.reshape(-1)
            yg = Hgpf @ xp + (A @ xg[..., None])[..., 0].reshape(-1) + Hxf @ f
            return S0 @ xp + Hgpf.T @ f, yg.view(N, nc)

        def dot(a, b):
            return (a[0] * b[0]).sum() + (a[1] * b[1]).sum()

        b = (-gp, -gg)
        x = (torch.zeros_like(gp), torch.zeros_like(gg))
        r = b
        z = schur(*r)
        d = z
        rz = dot(r, z)
        b2 = float(dot(b, b))
        tol = self.cfg["lm_cg_tol"] ** 2 * b2
        n_it = 0
        for n_it in range(1, self.cfg["lm_cg_iters"] + 1):
            Hd = apply(*d)
            alpha = rz / dot(d, Hd)
            x = (x[0] + alpha * d[0], x[1] + alpha * d[1])
            r = (r[0] - alpha * Hd[0], r[1] - alpha * Hd[1])
            if float(dot(r, r)) < tol:
                break
            z = schur(*r)
            rz2 = dot(r, z)
            beta = rz2 / rz
            rz = rz2
            d = (z[0] + beta * d[0], z[1] + beta * d[1])
        self.cg_iters.append(n_it)
        return x
