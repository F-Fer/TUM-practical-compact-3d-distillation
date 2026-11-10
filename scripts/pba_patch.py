"""Split photometric BA in two: align small patches once, then solve the geometry without looking at the images.

    align     warp each patch of each frame into each neighbour with the current parameters and run Lucas-Kanade on it alone.
              A warp yields a target pixel t, a 2 x 2 matrix W (an edge only constrains the direction across it) and an NCC score.
    geometry  LM on sum rho(sqrt(e^T W e)), e = projection - t, with one free log depth z per patch and a prior on it, and a
              depth-consistency row against the target's depth at t. No lookups: the objective is fixed, so LM can settle.
    repeat    from the new poses, coarse to fine over the blur levels of `patch`.

A warp's Jacobian has entries only for its two frames, the focal, k and its patch's z, so the z are eliminated exactly by the
Schur complement of Kernel.solve_free, and the system stays at the size of the poses. It is the structure of DROID-SLAM and
DPVO, without the network and started from DA3's prediction.

The photometric meaning of W: around the aligned position, the patch's sum of squares is about e^T H e with H = sum J J^T of the
zero-mean image gradients, so patch_info=photo uses W = H / S, and sqrt(e^T W e) is in intensities, with the scale of huber_photo.
patch_info=shape keeps only the shape of H, in pixels, with the Huber scale patch_delta: it doesn't weight patches by contrast.

patch_depth_iters refines the depth grid afterwards with the poses fixed, a frame at a time (Kernel.solve, solver=frames).
"""

import math

import torch
import torch.nn.functional as F

from pba_analytic import Kernel, rho, rho_weight


class PatchKernel(Kernel):
    """Kernel's parameters, LM loop and elimination (lm_step, apply, solve_free), with the geometric objective in normal_eq and cost."""

    def __init__(self, model, nbrs, radial, images):
        super().__init__(model)
        cfg = self.cfg
        self.nbrs, self.radial = nbrs, radial
        self.has_grid = False  # the grid is part of the source and target depth here, and isn't an unknown
        self.k = nbrs.shape[1]
        self.info_mode, self.delta = cfg["patch_info"], cfg["huber_photo"] if cfg["patch_info"] == "photo" else cfg["patch_delta"]
        self.r = cfg["patch_radius"]
        off = torch.arange(-self.r, self.r + 1, device=nbrs.device, dtype=model.R0.dtype)
        self.off_u, self.off_v = off[None, :].expand(len(off), -1).reshape(-1), off[:, None].expand(-1, len(off)).reshape(-1)
        self.S = len(self.off_u)
        self.ncc = False  # Kernel's rows aren't used, but warp and cost test it
        self.dense = model.dense_depth().detach()
        self.dense_flat = self.dense.reshape(-1)
        self.select(images, cfg["patch_count"])
        d = self.lookup_depth(self.frames[:, None], self.pu, self.pv)
        self.dsrc = d  # [N, P] source depth of the patch centres, grid included
        self.xb = torch.stack([(self.pu - self.cx[:, None]) / self.fx0[:, None], (self.pv - self.cy[:, None]) / self.fy0[:, None]], -1)
        self.z = torch.zeros_like(d)
        self.w_z = cfg["w_free_depth"]
        self.tgt, self.info, self.valid, self.Dt = None, None, None, None
        self.stats = {}

    # ---------------------------------------------------------------- patches
    def select(self, images, P):
        """P patches per frame: the steepest confident pixel of each cell of a grid, so that they cover the image; the steepest cells if there are more."""
        grey = images.mean(1, keepdim=True)
        k = torch.exp(-((torch.arange(-3, 4, device=grey.device, dtype=grey.dtype)) ** 2) / 2)
        k = k / k.sum()
        g = F.conv2d(F.pad(grey, (3, 3, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
        g = F.conv2d(F.pad(g, (0, 0, 3, 3), mode="replicate"), k.view(1, 1, -1, 1))
        gx = F.pad(g[..., :, 2:] - g[..., :, :-2], (1, 1, 0, 0))
        gy = F.pad(g[..., 2:, :] - g[..., :-2, :], (0, 0, 1, 1))
        score = torch.sqrt(gx**2 + gy**2) * self.m.valid[:, None].to(g.dtype)
        cell = max(2, int(math.sqrt(self.H * self.W / max(P, 1))))
        val, idx = F.max_pool2d(score, cell, stride=cell, return_indices=True)
        val, idx = val.flatten(1), idx.flatten(1)
        P_ = min(P, val.shape[1])
        top, order = val.topk(P_, dim=1)
        pix = idx.gather(1, order)
        self.pu, self.pv = (pix % self.W).to(g.dtype), (pix // self.W).to(g.dtype)
        self.pmask = top > 0
        self.P = P_

    def pattern(self, stride):
        """Pattern pixels of the patches at this stride: normalised coordinates at the initial focal [N, P, S, 2], depth [N, P, S] and pixels."""
        u = (self.pu[..., None] + self.off_u * stride).clamp(0, self.W - 1)
        v = (self.pv[..., None] + self.off_v * stride).clamp(0, self.H - 1)
        xy = torch.stack([(u - self.cx[:, None, None]) / self.fx0[:, None, None], (v - self.cy[:, None, None]) / self.fy0[:, None, None]], -1)
        return u, v, xy, self.lookup_depth(self.frames[:, None, None], u, v)

    # ---------------------------------------------------------------- lookups
    def lookup_depth(self, frame, u, v):
        W, H = self.W, self.H
        u0, v0 = u.floor().clamp(0, W - 2), v.floor().clamp(0, H - 2)
        du, dv = u - u0, v - v0
        base = frame * (H * W) + v0.long() * W + u0.long()
        d = self.dense_flat
        return (d[base] * (1 - du) + d[base + 1] * du) * (1 - dv) + (d[base + W] * (1 - du) + d[base + W + 1] * du) * dv

    def set_image(self, img, stride):
        """img [N, 1, H, W], blurred: its values and central-difference gradients, and the source patches of this level."""
        gu = F.pad(img[..., :, 2:] - img[..., :, :-2], (1, 1, 0, 0)) * 0.5
        gv = F.pad(img[..., 2:, :] - img[..., :-2, :], (0, 0, 1, 1)) * 0.5
        self.ptex = torch.cat([img, gu, gv], 1).permute(0, 2, 3, 1).reshape(-1, 3)
        self.stride = stride
        self.u, self.v, self.pxy, self.pd = self.pattern(stride)
        a = self.lookup(self.frames[:, None, None], self.u, self.v)[..., 0]
        a = a - a.mean(-1, keepdim=True)
        self.a, self.aa = a, (a * a).sum(-1)

    def lookup(self, frame, u, v):
        """Value and gradient [..., 3] at (u, v) in the frames `frame`, bilinear, (u, v) inside the image."""
        W, H = self.W, self.H
        u0, v0 = u.detach().floor().clamp(0, W - 2), v.detach().floor().clamp(0, H - 2)
        du, dv = (u - u0)[..., None], (v - v0)[..., None]
        base = frame * (H * W) + v0.long() * W + u0.long()
        a, b, c, d = self.ptex[base[..., None] + self.tex_quad].unbind(-2)
        return (a + (b - a) * du) * (1 - dv) + (c + (d - c) * du) * dv

    # ---------------------------------------------------------------- forward
    def project(self, st, xy0, ds):
        """Source points at normalised coordinates xy0 [N, P, ..., 2], depth ds [N, P, ...] -> uv and z in the neighbours [N, k, P, ..., ]."""
        R, t, _, _, s_f, _ = st
        nbrs = self.nbrs
        xy = xy0 / s_f
        if self.radial:
            k_r = self.m.kr.data
            D = 1 + k_r * (xy * xy).sum(-1)
            xy = xy / D[..., None]
        extra = xy0.dim() - 3
        Xc = torch.cat([xy * ds[..., None], ds[..., None]], -1)
        Rji = R[nbrs] @ R[:, None].transpose(-1, -2)
        tji = t[nbrs] - (Rji @ t[:, None, :, None])[..., 0]
        rs = (*Rji.shape[:2], *([1] * (1 + extra)), 3, 3)
        ts = (*tji.shape[:2], *([1] * (1 + extra)), 3)
        Xt = (Rji.view(rs) @ Xc[:, None][..., None])[..., 0] + tji.view(ts)
        z = Xt[..., 2]
        iz = 1 / z.clamp_min(1e-4)
        Xn = Xt[..., :2] * iz[..., None]
        if self.radial:
            q = torch.sqrt((1 - 4 * k_r * (Xn * Xn).sum(-1)).clamp_min(1e-4))
            Xn = Xn * (2 / (1 + q))[..., None]
        fs = (*nbrs.shape, *([1] * (1 + extra)), 2)
        return self.f0[nbrs].view(fs) * s_f * Xn + self.c0[nbrs].view(fs), z

    # ---------------------------------------------------------------- alignment
    def align(self, iters):
        """Lucas-Kanade per warp from the current parameters: sets tgt, info, valid and Dt."""
        cfg = self.cfg
        st = self.state()
        uv, z = self.project(st, self.pxy, self.pd * torch.exp(self.z)[..., None])  # [N, k, P, S, 2], [N, k, P, S]
        lo, hi = self.uv_lo, self.uv_hi
        frame = self.nbrs[:, :, None, None]
        d = torch.zeros(*uv.shape[:3], 2, device=uv.device, dtype=uv.dtype)
        a, aa = self.a[:, None], self.aa[:, None]
        n = self.S
        for it in range(iters + 1):
            q = uv + d[:, :, :, None]
            inside = ((q >= lo) & (q <= hi)).all(-1) & (z > 1e-4)
            val = self.lookup(frame, q.clamp(lo, hi)[..., 0], q.clamp(lo, hi)[..., 1])
            b, gu, gv = val.unbind(-1)
            b = b - b.mean(-1, keepdim=True)
            bb = (b * b).sum(-1)
            gain = torch.sqrt(aa / bb.clamp_min(1e-10)).clamp(0.2, 5.0)
            gu = (gu - gu.mean(-1, keepdim=True)) * gain[..., None]
            gv = (gv - gv.mean(-1, keepdim=True)) * gain[..., None]
            h11, h12, h22 = (gu * gu).sum(-1), (gu * gv).sum(-1), (gv * gv).sum(-1)
            if it == iters:
                break
            r = b * gain[..., None] - a
            g1, g2 = (gu * r).sum(-1), (gv * r).sum(-1)
            e = 1e-3 * (h11 + h22) + 1e-10
            det = (h11 + e) * (h22 + e) - h12 * h12
            step = torch.stack([-((h22 + e) * g1 - h12 * g2) / det, -((h11 + e) * g2 - h12 * g1) / det], -1)
            # at most a stride per step: a linear model of the image isn't valid further than this
            step = step * (self.stride / step.norm(dim=-1, keepdim=True).clamp_min(self.stride))
            d = d + step
        ncc = (a * b).sum(-1) / torch.sqrt(aa * bb + 1e-10)
        mask = inside.all(-1) & self.pmask[:, None] & (ncc > cfg["patch_ncc"]) & (d.norm(dim=-1) <= cfg["patch_max_disp"] * self.stride)
        # target of the patch centre: the middle of the pattern
        c = n // 2
        uvc = uv[:, :, :, c]
        self.tgt = uvc + d
        tu, tv = self.tgt.clamp(lo, hi).unbind(-1)
        self.Dt = self.lookup_depth(self.nbrs[:, :, None], tu, tv)
        mask = mask & (z[:, :, :, c] < self.Dt * (1 + cfg["occ_thresh"]))
        if self.info_mode == "photo":
            self.info = torch.stack([h11, h12, h22], -1) / n
        else:
            tr = (h11 + h22).clamp_min(1e-12)
            e = cfg["patch_iso"]
            sc = 2 / (tr * (1 + 2 * e))
            self.info = torch.stack([(h11 + e * tr) * sc, h12 * sc, (h22 + e * tr) * sc], -1) / self.stride**2
        self.valid = mask
        vis = float(mask.float().mean())
        self.stats = {"valid": vis, "disp": float((d.norm(dim=-1) * mask).sum() / mask.sum().clamp_min(1)), "ncc": float((ncc * mask).sum() / mask.sum().clamp_min(1))}

    # ---------------------------------------------------------------- objective
    def geo_warp(self, st):
        """Patch centres in the neighbours, with what the Jacobian needs. As Kernel.warp, without lookups."""
        R, t, _, _, s_f, _ = st
        nbrs = self.nbrs
        fxy = self.f0[nbrs][:, :, None] * s_f
        xy = self.xb / s_f
        out = {}
        if self.radial:
            k_r = self.m.kr.data
            rd2 = (xy * xy).sum(-1)
            D = 1 + k_r * rd2
            xy = xy / D[..., None]
            out["A"], out["E"] = (1 - k_r * rd2) / D, rd2 / D
        ds = self.dsrc * torch.exp(self.z)
        Xc = torch.cat([xy * ds[..., None], ds[..., None]], -1)
        Rji = R[nbrs] @ R[:, None].transpose(-1, -2)
        tji = t[nbrs] - (Rji @ t[:, None, :, None])[..., 0]
        Xt = Xc[:, None] @ Rji.transpose(-1, -2) + tji[:, :, None]
        z = Xt[..., 2]
        zc = z.clamp_min(1e-4)
        iz = 1 / zc
        Xn = Xt[..., :2] * iz[..., None]
        if self.radial:
            xn, yn = Xn.unbind(-1)
            r2 = (Xn * Xn).sum(-1)
            q = torch.sqrt((1 - 4 * k_r * r2).clamp_min(1e-4))
            sc = 2 / (1 + q)
            out["rad"] = (xn, yn, r2, sc, k_r * sc * sc / q, q)
            uvx = fxy * sc[..., None] * Xn
        else:
            uvx = fxy * Xn
        e = uvx + self.c0[nbrs][:, :, None] - self.tgt
        out.update(Xc=Xc, Xt=Xt, iz=iz, z=z, zc=zc, Rji=Rji, tji=tji, ds=ds, ux=uvx[..., 0], vy=uvx[..., 1], fxt=fxy[..., 0], fyt=fxy[..., 1], eu=e[..., 0], ev=e[..., 1])
        out["geo"] = torch.log(zc / self.Dt.clamp_min(1e-4))
        out["mask"] = self.valid & (z > 1e-4)
        return out

    def maha(self, g):
        W11, W12, W22 = self.info.unbind(-1)
        return torch.sqrt(W11 * g["eu"] ** 2 + 2 * W12 * g["eu"] * g["ev"] + W22 * g["ev"] ** 2 + 1e-20)

    def cost(self, feat, quad, nbrs, chunk):
        g = self.geo_warp(self.state())
        m = g["mask"].to(g["z"].dtype)
        n = max(float(m.sum()), 1.0)
        return float((rho(self.maha(g), self.delta, "huber") * m).sum()) / n + self.w_geo_lm * float((rho(g["geo"], self.dg, "huber") * m).sum()) / n + self.priors() + self.z_prior(n)

    def normal_eq(self, feat, quad, nbrs, chunk, cross_frames):
        """Blocks of the geometric objective. Unknowns as Kernel.normal_eq without the grid: the z go to self.fd for solve_free."""
        cfg, N, k = self.cfg, self.N, self.k
        dev, st = self.frames.device, self.state()
        Jl, tau = st[2], st[3]
        n_pose = 6 * (N - 1)
        nq = 14 if self.radial else 13
        np_ = n_pose + 1 + int(self.radial)
        dummy = np_
        pose_idx = torch.full((N, 6), dummy, dtype=torch.long, device=dev)
        pose_idx[1:] = torch.arange(n_pose, device=dev).view(N - 1, 6)
        focal_idx = n_pose if self.has_focal else dummy
        k_idx = n_pose + 1
        nbrs = self.nbrs
        g = self.geo_warp(st)
        m = g["mask"].to(g["z"].dtype)
        n_vis = max(float(m.sum()), 1.0)
        iz, Rji = g["iz"], g["Rji"]
        if self.radial:
            xn, yn, r2, sc, dsc, q = g["rad"]
            fi, gi = g["fxt"] * iz, g["fyt"] * iz
            a_, xy = sc + 2 * r2 * dsc, 2 * xn * yn * dsc
            P1 = torch.stack([fi * (sc + 2 * xn * xn * dsc), fi * xy, -fi * xn * a_], -1)
            P2 = torch.stack([gi * xy, gi * (sc + 2 * yn * yn * dsc), -gi * yn * a_], -1)
            k_t = r2 * sc / q
        else:
            P1 = torch.stack([g["fxt"] * iz, torch.zeros_like(iz), -g["ux"] * iz], -1)
            P2 = torch.stack([torch.zeros_like(iz), g["fyt"] * iz, -g["vy"] * iz], -1)
        Xr = g["Xt"] - g["tji"][:, :, None]
        r3 = Rji[..., 2][:, :, None]
        Xs = (g["Xc"] - tau[:, None])[:, None]
        Xs_t = g["Xt"] - tau[nbrs][:, :, None]
        ds_ = g["ds"][:, None]

        def rows(gX, T):
            ts = -(gX @ Rji)
            dds = (gX * Xr).sum(-1)
            sp_ = -dds + ds_ * (gX * r3).sum(-1)
            cols = [torch.linalg.cross(Xs, ts, dim=-1), ts, torch.linalg.cross(Xs_t, gX, dim=-1), gX]
            if self.radial:
                cols += [(T + sp_ * g["A"][:, None])[..., None], (T * k_t + sp_ * g["E"][:, None])[..., None]]
            else:
                cols += [(T + sp_)[..., None]]
            return torch.cat(cols, -1), dds

        Ju, du = rows(P1, g["ux"])
        Jv, dv = rows(P2, g["vy"])
        gXg = torch.zeros_like(P1)
        gXg[..., 2] = iz
        Jg, dg = rows(gXg, torch.zeros_like(iz))
        if not self.has_focal:
            Ju[..., 12], Jv[..., 12], Jg[..., 12] = 0, 0, 0
        wr = m * rho_weight(self.maha(g), self.delta, "huber")
        W11, W12, W22 = (self.info * wr[..., None]).unbind(-1)
        eu, ev, geo = g["eu"], g["ev"], g["geo"]
        wg = m * rho_weight(geo, self.dg, "huber") * self.w_geo_lm
        wJu = W11[..., None] * Ju + W12[..., None] * Jv
        wJv = W12[..., None] * Ju + W22[..., None] * Jv
        wJg = wg[..., None] * Jg
        H13 = Ju.transpose(-1, -2) @ wJu + Jv.transpose(-1, -2) @ wJv + Jg.transpose(-1, -2) @ wJg
        g13 = (wJu * eu[..., None] + wJv * ev[..., None] + wJg * geo[..., None]).sum(2)
        T13 = torch.eye(nq, device=dev, dtype=H13.dtype).repeat(N, k, 1, 1)
        T13[..., 0:3, 0:3] = Jl[:, None].transpose(-1, -2).expand(N, k, 3, 3)
        T13[..., 6:9, 6:9] = Jl[nbrs].transpose(-1, -2)
        H13 = T13 @ H13 @ T13.transpose(-1, -2)
        g13 = (T13 @ g13[..., None])[..., 0]
        ids = [pose_idx[:, None].expand(N, k, 6), pose_idx[nbrs], torch.full((N, k, 1), focal_idx, device=dev)]
        if self.radial:
            ids.append(torch.full((N, k, 1), k_idx, device=dev))
        ids = torch.cat(ids, -1)
        Hpp = torch.zeros((np_ + 1) ** 2, device=dev, dtype=H13.dtype)
        gp = torch.zeros(np_ + 1, device=dev, dtype=H13.dtype)
        Hpp.index_add_(0, (ids[..., None] * (np_ + 1) + ids[..., None, :]).reshape(-1), H13.reshape(-1))
        gp.index_add_(0, ids.reshape(-1), g13.reshape(-1))
        # the z: diagonal entry, gradient and coupling per patch
        hz = (du * (W11 * du + W12 * dv) + dv * (W12 * du + W22 * dv) + dg * dg * wg).sum(1)
        bz = (du * (W11 * eu + W12 * ev) + dv * (W12 * eu + W22 * ev) + dg * wg * geo).sum(1)
        V = (du[..., None] * wJu + dv[..., None] * wJv + dg[..., None] * wJg) @ T13.transpose(-1, -2)
        Vl = torch.cat([V[..., :6].sum(1), V[..., 6:12].transpose(1, 2).flatten(2), V[..., 12:].sum(1)], -1)
        il = torch.cat([ids[:, 0, :6], ids[:, :, 6:12].flatten(1), ids[:, 0, 12:]], -1)
        Hpp = Hpp.view(np_ + 1, np_ + 1)[:np_, :np_].double() / n_vis
        gp = gp[:np_].double() / n_vis
        if self.radial:
            Hpp[k_idx, k_idx] += 2 * cfg["w_radial"]
            gp[k_idx] += 2 * cfg["w_radial"] * float(self.m.kr.data[0])
        if self.has_focal and cfg["w_focal"] > 0:
            Hpp[focal_idx, focal_idx] += 2 * cfg["w_focal"]
            gp[focal_idx] += 2 * cfg["w_focal"] * float(self.m.logf.data[0])
        self.fd = ((hz.double() + self.w_z) / n_vis, (bz.double() + self.w_z * self.z.double()) / n_vis, [(self.frames, Vl.double() / n_vis, il)], np_ + 1)
        loss = float((rho(self.maha(g), self.delta, "huber") * m).sum()) / n_vis + self.w_geo_lm * float((rho(geo, self.dg, "huber") * m).sum()) / n_vis + self.priors() + self.z_prior(n_vis)
        return Hpp, None, None, gp, None, None, loss


def parse_schedule(spec):
    """"8:2,4:2,1:3" -> [(blur sigma, alignments), ...]"""
    out = []
    for s in spec.split(","):
        a, _, b = s.partition(":")
        out.append((float(a), int(b or 1)))
    return out


def run_patch(model, nbrs, images, blur, cfg, hist, info, tracker, radial, t_start, device, sampling_probs):
    """The alignments and geometric solves of `patch`, then the depth stage. Returns the state of the radial coefficient."""
    K = PatchKernel(model, nbrs, radial, images)
    lam0 = max(cfg["lm_lambda"], cfg["lm_lambda_min"])
    level_id = 1000
    n_inner = n_align = 0
    gate_at = cfg["patch_gate_after"]
    for li, (sigma, n_out) in enumerate(parse_schedule(cfg["patch"])):
        K.set_image(blur("_grey", sigma), cfg["patch_stride"] * max(1.0, sigma))
        for o in range(n_out):
            K.align(cfg["patch_lk"])
            n_align += 1
            lam = lam0
            for it in range(cfg["patch_inner"]):
                lam, c0, c1, ok = K.lm_step(None, None, nbrs, 1, lam, "dense", it=it)
                n_inner += 1
                hist.append({"level": level_id + li, "outer": o, "it": it, "loss": c0, "loss_after": c1, "lambda": lam, "accepted": ok, "t": _now(t_start, device), **K.stats})
                if tracker:
                    tracker.it += 1
                    tracker.record(level_id + li, force=True)
                if not ok or (c0 - c1) / max(c0, 1e-12) < cfg["patch_tol"]:
                    break
        if K.radial and gate_at == li:  # no Adam before it: decide on k from this level's solve
            info["radial_k_gate"] = float(model.kr)
            if abs(info["radial_k_gate"]) < cfg["radial_gate"]:
                with torch.no_grad():
                    model.kr.zero_()
                K.radial = False
                info["radial_gated"] = 1
    info.update(patch_valid=K.stats.get("valid", 0.0), patch_disp=K.stats.get("disp", 0.0), patch_ncc_mean=K.stats.get("ncc", 0.0), patch_inner=n_inner, patch_align=n_align, patch_z_abs=float(K.z.abs().mean()))
    radial = K.radial
    if cfg["patch_depth_iters"] > 0:
        depth_stage(model, nbrs, blur, cfg, hist, radial, t_start, device, sampling_probs)
    return radial


def _now(t_start, device):
    import time

    return time.perf_counter() - t_start


def depth_stage(model, nbrs, blur, cfg, hist, radial, t_start, device, sampling_probs):
    """Fit the depth grid with the poses, focal and k fixed: photometric and geometric rows of Kernel, the cells of each frame alone."""
    with torch.no_grad():
        R, t = model.poses()
        model.K0.copy_(model.intrinsics())
        model.R0.copy_(R)
        model.t0.copy_(t)
        model.rot.zero_(), model.trans.zero_(), model.logf.zero_()
    K = Kernel(model)
    K.radial, K.has_pose, K.has_focal = radial, False, False
    sigma = cfg["patch_depth_sigma"]
    img = blur("_grey", sigma)
    K.set_level(img, True, cfg["patch_stride"] * max(1.0, sigma))
    probs = sampling_probs(img, model.valid, cfg["grad_weight"], cfg["strat_cells"])
    M = cfg["lm_samples"] or cfg["samples"]
    pool = torch.multinomial(probs, min(4, cfg["pool_factor"]) * M, replacement=True)
    N, k = model.N, nbrs.shape[1]
    D = 2
    chunk = cfg["lm_chunk"] or max(1, int((1.6e7 if device.type == "cuda" else 4e6) / (k * M * D)))
    lam = max(cfg["lm_lambda"], cfg["lm_lambda_min"])
    gen_idx = lambda: pool.gather(1, torch.randint(0, pool.shape[1], (N, M), device=device))  # noqa: E731
    for it in range(cfg["patch_depth_iters"]):
        feat, quad = K.make_pool(gen_idx())
        lam, c0, c1, ok = K.lm_step(feat, quad, nbrs, chunk, lam, "frames", it=it)
        hist.append({"level": 2000, "it": it, "loss": c0, "loss_after": c1, "lambda": lam, "accepted": ok, "t": _now(t_start, device)})
