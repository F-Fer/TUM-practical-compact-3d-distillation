"""Targets for "TTO as teacher": released DéjàView (504 px, K_t = 16, ray-fit extrinsics + camera-head intrinsics,
the best combination after TTO) on random ARKitScenes training clips, then photometric TTO (`run_tto_pba.py`
refine_scene, iters = 25). Per clip it stores the frames, the teacher cameras, the refined cameras and the refined
log-depth correction (pooled to the patch grid). ARKit GT poses (lowres_wide.traj, interpolated to the frame
timestamps, slerp) are used only to report how much TTO improves the targets (AUC@3 / AUC@30 before -> after); they never
enter the targets.

  .venv/bin/python scripts/distill_steps/gen_tto_targets.py --n 1500 --out workspace/distill_data/tto_targets
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "third_party", "dvlt", "src"))
from train import load_dvlt  # noqa: E402
from train_res import DATA, Folders, run  # noqa: E402


def read_traj(path):
    """ARKit lowres_wide.traj -> timestamps [N], world-to-camera [N, 4, 4]."""
    import cv2

    ts, T = [], []
    for line in open(path):
        v = [float(x) for x in line.split()]
        if len(v) != 7:
            continue
        M = np.eye(4)
        M[:3, :3] = cv2.Rodrigues(np.asarray(v[1:4]))[0]
        M[:3, 3] = v[4:7]
        ts.append(v[0]), T.append(M)
    return np.asarray(ts), np.stack(T)


def gt_w2c(traj, stamps):
    """Poses interpolated to the frame timestamps (slerp / linear on camera-to-world; traj at ~10 Hz).
    None if a frame lies outside the trajectory or in a gap > 0.25 s."""
    from scipy.spatial.transform import Rotation, Slerp

    ts, T = traj
    c2w = np.linalg.inv(T)
    out = []
    for s in stamps:
        j = int(np.searchsorted(ts, s))
        if j == 0 or j >= len(ts) or ts[j] - ts[j - 1] > 0.25:
            return None
        a = (s - ts[j - 1]) / (ts[j] - ts[j - 1])
        R = Slerp([0, 1], Rotation.from_matrix(c2w[[j - 1, j], :3, :3]))(a).as_matrix()
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = R, (1 - a) * c2w[j - 1, :3, 3] + a * c2w[j, :3, 3]
        out.append(np.linalg.inv(M))
    return np.stack(out)


def rel_errors(w2c_a, w2c_b):
    """Pairwise relative rotation error and translation-direction error (degrees) between two camera sets."""
    def rel(w2c):
        R, t = w2c[:, :3, :3], w2c[:, :3, 3]
        Rij = np.einsum("iab,jcb->ijac", R, R)  # R_i R_j^T
        tij = t[:, None] - np.einsum("ijab,jb->ija", Rij, t)
        return Rij, tij
    Ra, ta = rel(w2c_a)
    Rb, tb = rel(w2c_b)
    iu = np.triu_indices(len(w2c_a), 1)
    Ra, Rb, ta, tb = Ra[iu], Rb[iu], ta[iu], tb[iu]
    c = (np.einsum("nab,nab->n", Ra, Rb) - 1) / 2
    er = np.degrees(np.arccos(np.clip(c, -1, 1)))
    na, nb = np.linalg.norm(ta, axis=1), np.linalg.norm(tb, axis=1)
    ct = np.einsum("na,na->n", ta, tb) / np.maximum(na * nb, 1e-12)
    et = np.degrees(np.arccos(np.clip(ct, -1, 1)))
    return np.maximum(er, et)


def auc(err, thr):
    th = np.linspace(0, thr, 101)[1:]
    return float(np.mean([(err < t).mean() for t in th]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_views", type=int, default=16)
    ap.add_argument("--max_views", type=int, default=32)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import run_tto_pba
    from dvlt.common.rays import rays_to_pose

    import fast_decode

    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    cfg = run_tto_pba.parse_overrides([f"iters={a.iters}"])
    dirs = sorted(d for d in glob.glob(os.path.join(DATA, "train", "*")) if os.path.exists(f"{d}/DONE"))
    ds = Folders(dirs, a.start + a.n, a.min_views, a.max_views, a.seed, return_files=True)
    model = load_dvlt().to(dev).eval()
    trajs = {}
    stats = []
    t0 = time.time()
    for i in range(a.start, a.start + a.n):
        path = os.path.join(a.out, f"{i:05d}.npz")
        if os.path.exists(path):
            continue
        images, vid, files = ds[i]
        S, _, H, W = images.shape
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = run(model, images[None].to(dev), 16)
        with torch.autocast("cuda", enabled=False):
            c2w, _ = rays_to_pose(pred["rays"].float(), torch.ones_like(pred["depth_conf"]), H, W, 14)
        w2c = torch.linalg.inv(c2w[0].float())[:, :3]
        _, Kh = fast_decode.head_cameras(pred["pose_enc"][0], H, W)
        p = dict(depth=pred["depth"][0, ..., 0].float().cpu().numpy(), conf=pred["depth_conf"][0].float().cpu().numpy(),
                 extrinsics=w2c.cpu().numpy(), intrinsics=Kh.cpu().numpy())
        out, info, _ = run_tto_pba.refine_scene(files, p, cfg, dev)
        lr = np.log(np.maximum(out["depth"], 1e-6) / np.maximum(p["depth"], 1e-6))
        lr = F.adaptive_avg_pool2d(torch.from_numpy(lr)[:, None], (H // 14, W // 14))[:, 0].numpy()
        rec = dict(i=i, video=vid, views=S)
        if vid not in trajs:
            tp = os.path.join(DATA, "train", vid, "lowres_wide.traj")
            trajs[vid] = read_traj(tp) if os.path.exists(tp) else None
        gt = None
        if trajs[vid] is not None:
            stamps = [float(os.path.basename(f)[:-4].split("_")[-1]) for f in files]
            gt = gt_w2c(trajs[vid], stamps)
        if gt is not None:
            eb, ea = rel_errors(p["extrinsics"], gt), rel_errors(out["extrinsics"], gt)
            rec.update(auc3_before=auc(eb, 3), auc3_after=auc(ea, 3), auc30_before=auc(eb, 30), auc30_after=auc(ea, 30))
        rec.update(photo_before=info["photo_before"], photo_after=info["photo_after"], opt_s=info["opt_s"])
        np.savez(path, files=np.asarray([os.path.relpath(f, DATA) for f in files]), ext_teacher=p["extrinsics"],
                 K_teacher=p["intrinsics"], ext_ref=out["extrinsics"], K_ref=out["intrinsics"],
                 logdepth_ref=lr.astype(np.float32), gt_w2c=gt if gt is not None else np.zeros(0),
                 **{k: np.float32(v) for k, v in rec.items() if isinstance(v, float)})
        stats.append(rec)
        g = [s for s in stats if "auc3_before" in s]
        if g:
            m = lambda k: 100 * np.mean([s[k] for s in g])
            print(f"[{i}] {vid} S={S} auc3 {100 * rec.get('auc3_before', np.nan):.1f}->{100 * rec.get('auc3_after', np.nan):.1f} | "
                  f"running mean ({len(g)}): auc3 {m('auc3_before'):.1f}->{m('auc3_after'):.1f}, "
                  f"auc30 {m('auc30_before'):.1f}->{m('auc30_after'):.1f} | {(time.time() - t0) / len(stats):.2f} s/clip",
                  flush=True)


if __name__ == "__main__":
    main()
