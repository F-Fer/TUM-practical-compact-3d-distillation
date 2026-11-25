"""Local windows instead of one global pass: run DA3 on overlapping windows of consecutive frames, compare the accuracy
of the within-window relative poses / depth with the full pass, stitch the windows (Sim(3) chain over the shared
frames, or each window aligned to the full pass) and write the result as a new initialisation for PBA.
knowledge/ideas/local-windows.md.

    .venv/bin/python scripts/lpba_win.py <src work dir> <out root> [split=dev] [datasets=scannetpp,7scenes]
        [win=16] [stride=8] [ref=middle] [model=depth-anything/DA3-SMALL] [min_frames=40] [max_scenes=0]

Writes <out root>/{chain,anchor,anchor_dep}/model_results/... :
    chain       poses + depth from the windows only: window k+1 is put into window k's frame by the Sim(3) that maps
                the shared frames' cameras (scale from the depth ratio); a frame's pose / log depth is the mean over
                its windows
    anchor      each window is aligned to the full pass (robust Sim(3) on its cameras), then averaged: local structure
                from the windows, global placement from the full pass (costs the full pass too)
    anchor_dep  the full pass's cameras, depth averaged over full pass + windows (depth only)
"""

import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpba_depth_oracle import abs_rel, try_gt  # noqa: E402

DEV = {"hiroom": 10, "scannetpp": 3, "eth3d": 3, "7scenes": 2, "dtu64": 4}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def h44(E):
    E = np.asarray(E, np.float64)
    if E.shape[-2] == 4:
        return E
    return np.concatenate([E[:, :3], np.tile([[[0, 0, 0, 1.0]]], (len(E), 1, 1))], 1)


def proj_rot(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def pair_err(E, G, pairs):
    """Relative-pose errors (rotation deg, translation direction deg) of world-to-camera poses E vs G over pairs."""
    i, j = pairs[:, 0], pairs[:, 1]
    Tp = E[j] @ np.linalg.inv(E[i])
    Tg = G[j] @ np.linalg.inv(G[i])
    Rd = np.einsum("nij,nkj->nik", Tp[:, :3, :3], Tg[:, :3, :3])
    rot = np.degrees(np.arccos(np.clip((np.trace(Rd, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    a, b = Tp[:, :3, 3], Tg[:, :3, 3]
    cos = np.abs((a * b).sum(1)) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    tr = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return rot, tr


def auc(rot, tr, thr):
    e = np.maximum(rot, tr)
    h = np.histogram(e, bins=np.arange(thr + 1))[0] / max(len(e), 1)
    return 100 * float(np.mean(np.cumsum(h)))


def sim3_from_cams(Ea, Eb, s):
    """Transform A (4x4, world b -> world a, with scale s on b's world) such that Ea_i ~ Eb_i(scaled) A^-1, from the
    shared cameras: per camera A_i = Ea_i^-1 Eb_i(s); rotations by chordal mean, translations by the mean."""
    Eb = Eb.copy()
    Eb[:, :3, 3] *= s
    A = np.linalg.inv(Ea) @ Eb
    out = np.eye(4)
    out[:3, :3] = proj_rot(A[:, :3, :3].mean(0))
    out[:3, 3] = np.median(A[:, :3, 3], 0)
    return out


def robust_sim3_from_cams(Ea, Eb, s):
    """As sim3_from_cams, but the medoid of the per-camera estimates (robust to grossly wrong cameras in a)."""
    Eb = Eb.copy()
    Eb[:, :3, 3] *= s
    A = np.linalg.inv(Ea) @ Eb
    R, t = A[:, :3, :3], A[:, :3, 3]
    ang = np.arccos(np.clip((np.einsum("aij,bij->ab", R, R) - 1) / 2, -1, 1))
    sc = np.median(np.linalg.norm(Eb[:, :3, 3] - Eb[:, :3, 3].mean(0), axis=1)) + 1e-9
    dt = np.linalg.norm(t[:, None] - t[None], axis=-1) / sc
    score = np.median(ang + dt, 1)
    m = int(np.argmin(score))
    inl = (ang[m] + dt[m]) < max(2 * score[m], 0.02)
    out = np.eye(4)
    out[:3, :3] = proj_rot(R[inl].mean(0))
    out[:3, 3] = t[inl].mean(0)
    return out, inl


def mean_pose(Ts):
    out = np.eye(4)
    out[:3, :3] = proj_rot(np.mean([T[:3, :3] for T in Ts], 0))
    out[:3, 3] = np.mean([T[:3, 3] for T in Ts], 0)
    return out


def windows(N, win, stride):
    if N <= win:
        return [list(range(N))]
    starts = list(range(0, N - win + 1, stride))
    if starts[-1] != N - win:
        starts.append(N - win)
    return [list(range(s, s + win)) for s in starts]


def main():
    src, root = sys.argv[1], sys.argv[2]
    opts = dict(a.split("=", 1) for a in sys.argv[3:])
    split = opts.get("split", "dev")
    win, stride = int(opts.get("win", 16)), int(opts.get("stride", 8))
    ref = opts.get("ref", "middle")
    min_frames = int(opts.get("min_frames", 40))
    max_scenes = int(opts.get("max_scenes", 0))
    os.chdir(REPO)
    from depth_anything_3.api import DepthAnything3

    api = DepthAnything3.from_pretrained(opts.get("model", "depth-anything/DA3-SMALL")).to("cuda").eval()
    tot = {}
    for ds in opts.get("datasets", "scannetpp,7scenes").split(","):
        d = os.path.join(src, "model_results", ds)
        scenes = sorted(os.path.dirname(p) for p, _, _ in os.walk(d) if os.path.basename(p) == "unposed")
        scenes = scenes[: DEV[ds]] if split == "dev" else scenes[DEV[ds]:] if split == "held-out" else scenes
        if max_scenes:
            scenes = scenes[:max_scenes]
        for sd in scenes:
            scene = os.path.relpath(sd, d)
            ex = os.path.join(sd, "unposed", "exports")
            meta = dict(np.load(os.path.join(ex, "gt_meta.npz"), allow_pickle=True))
            pred = dict(np.load(os.path.join(ex, "mini_npz", "results.npz")))
            files = [str(f) for f in meta["image_files"]]
            base = pred["depth"].astype(np.float32)
            N, H, W = base.shape
            G = h44(meta["extrinsics"])
            E0 = h44(pred["extrinsics"])

            def write(name, depth, ext, intr=None):
                o = os.path.join(root, name, "model_results", ds, scene, "unposed", "exports")
                os.makedirs(os.path.join(o, "mini_npz"), exist_ok=True)
                np.savez(os.path.join(o, "gt_meta.npz"), **meta)
                out = {**pred, "depth": depth.astype(np.float32), "extrinsics": ext[:, :3].astype(np.float32)}
                if intr is not None:
                    out["intrinsics"] = intr.astype(np.float32)
                np.savez(os.path.join(o, "mini_npz", "results.npz"), **out)

            if N < min_frames:
                for name in ("chain", "anchor", "anchor_dep", "anchor1", "anchor1_dep0"):
                    write(name, base, E0)
                continue
            imgs = [Image.open(f).convert("RGB") for f in files]
            wins = windows(N, win, stride)
            t0 = time.time()
            outs = []
            for w in wins:
                p = api.inference([imgs[i] for i in w], ref_view_strategy=ref)
                outs.append((np.asarray(p.depth, np.float32), h44(np.asarray(p.extrinsics, np.float64)), np.asarray(p.intrinsics, np.float64)))
            torch.cuda.synchronize()
            dt = time.time() - t0
            gt = try_gt(files, H, W)

            # ---- local accuracy: pairs inside a window, full pass vs window pass
            stats = {"full": [], "win": []}
            for w, (dep, E, _) in zip(wins, outs):
                w = np.asarray(w)
                ii, jj = np.triu_indices(len(w), 1)
                pl = np.stack([ii, jj], 1)
                stats["win"].append(pair_err(E, G[w], pl))
                stats["full"].append(pair_err(E0[w], G[w], pl))
            msg = []
            for k, v in stats.items():
                rot = np.concatenate([x[0] for x in v])
                tr = np.concatenate([x[1] for x in v])
                msg.append(f"{k}: rot med {np.median(rot):.2f} tr med {np.median(tr):.2f} A@3 {auc(rot, tr, 3):.1f} A@30 {auc(rot, tr, 30):.1f}")
                tot.setdefault(ds, {}).setdefault("loc_" + k, []).append((auc(rot, tr, 3), auc(rot, tr, 30), np.median(rot), np.median(tr)))
            print(f"[win] {ds}/{scene} N={N} {len(wins)} windows {dt:.2f}s | local pairs " + " | ".join(msg), flush=True)

            # ---- per-window scale to the full pass's depth, and log-depth sums
            logs = [np.log(np.maximum(base, 1e-6))]
            cnt = np.ones(N)
            sum_log = logs[0].copy()
            scales = []
            for w, (dep, E, _) in zip(wins, outs):
                s = float(np.median(base[w]) / np.median(dep))
                scales.append(s)
                sum_log[w] += np.log(np.maximum(dep * s, 1e-6))
                cnt[w] += 1
            dep_avg = np.exp(sum_log / cnt[:, None, None])
            dep_win = np.exp((sum_log - logs[0]) / np.maximum(cnt - 1, 1)[:, None, None])

            # ---- anchor: each window aligned to the full pass (robust), poses averaged over windows
            poses = [[] for _ in range(N)]
            n_out = 0
            for w, (dep, E, _), s in zip(wins, outs, scales):
                A, inl = robust_sim3_from_cams(E0[w], E, s)
                n_out += int((~inl).sum())
                Es = E.copy()
                Es[:, :3, 3] *= s
                for a, i in enumerate(w):
                    poses[i].append(Es[a] @ np.linalg.inv(A))
            E_anchor = np.stack([mean_pose(p) for p in poses])
            # anchor1: each frame from the one window in which it is most central (no averaging of two alignments)
            best = {}
            for k_, w in enumerate(wins):
                for a, i in enumerate(w):
                    c = abs(a - (len(w) - 1) / 2)
                    if i not in best or c < best[i][0]:
                        best[i] = (c, k_, a)
            E_anchor1 = np.stack([poses[i][[k2 for k2, w in enumerate(wins) if i in w].index(best[i][1])] for i in range(N)])
            dep_one = np.stack([outs[best[i][1]][0][best[i][2]] * scales[best[i][1]] for i in range(N)])

            # ---- chain: window k+1 into window k's world over the shared frames; scale from the shared depth
            world = [np.eye(4)]  # A_k: world of window k -> world of window 0
            cs = [1.0]
            for k in range(1, len(wins)):
                sh = sorted(set(wins[k - 1]) & set(wins[k]))
                ia = [wins[k - 1].index(i) for i in sh]
                ib = [wins[k].index(i) for i in sh]
                s = float(np.median(outs[k - 1][0][ia]) / np.median(outs[k][0][ib]))
                Ea = outs[k - 1][1][ia].copy()
                Ea[:, :3, 3] *= cs[-1]
                A = sim3_from_cams(Ea, outs[k][1][ib], cs[-1] * s)
                world.append(world[-1] @ A)
                cs.append(cs[-1] * s)
            poses = [[] for _ in range(N)]
            sum_log_c = np.zeros_like(base)
            for w, (dep, E, _), A, s in zip(wins, outs, world, cs):
                Es = E.copy()
                Es[:, :3, 3] *= s
                for a, i in enumerate(w):
                    poses[i].append(Es[a] @ np.linalg.inv(A))
                sum_log_c[w] += np.log(np.maximum(dep * s, 1e-6))
            E_chain = np.stack([mean_pose(p) for p in poses])
            dep_chain = np.exp(sum_log_c / (cnt - 1)[:, None, None])
            # bring the chain into the full pass's gauge and scale (frame 0's pose, median depth), so PBA's defaults fit
            k = float(np.median(base) / np.median(dep_chain))
            E_chain[:, :3, 3] *= k
            dep_chain *= k
            E_chain = E_chain @ np.linalg.inv(E_chain[0]) @ E0[0]
            E_anchor = E_anchor @ np.linalg.inv(E_anchor[0]) @ E0[0]
            E_anchor1 = E_anchor1 @ np.linalg.inv(E_anchor1[0]) @ E0[0]

            ii, jj = np.triu_indices(N, 1)
            pl = np.stack([ii, jj], 1)
            res = {}
            for name, E in (("full", E0), ("anchor", E_anchor), ("anchor1", E_anchor1), ("chain", E_chain)):
                rot, tr = pair_err(E, G, pl)
                res[name] = (auc(rot, tr, 3), auc(rot, tr, 30))
                tot.setdefault(ds, {}).setdefault("glob_" + name, []).append(res[name])
            de = {"full": abs_rel(base, gt), "avg": abs_rel(dep_avg, gt), "win": abs_rel(dep_win, gt), "chain": abs_rel(dep_chain, gt)}
            tot.setdefault(ds, {}).setdefault("dep", []).append([de[k_] for k_ in ("full", "avg", "win", "chain")])
            print(f"[win]   global A@3/A@30 " + " ".join(f"{k_} {a:.1f}/{b:.1f}" for k_, (a, b) in res.items()) +
                  f" | anchor outliers {n_out} | depth abs.rel. " + " ".join(f"{k_} {100 * v:.2f}" for k_, v in de.items()), flush=True)
            write("chain", dep_chain, E_chain)
            write("anchor", dep_avg, E_anchor)
            write("anchor_dep", dep_avg, E0)
            write("anchor1", dep_one, E_anchor1)
            write("anchor1_dep0", base, E_anchor1)
    for ds, t in tot.items():
        print(f"[win] == {ds}: " + " | ".join(f"{k} " + "/".join(f"{x:.2f}" for x in np.nanmean(np.asarray(v, float), 0)) for k, v in t.items()))


if __name__ == "__main__":
    main()
