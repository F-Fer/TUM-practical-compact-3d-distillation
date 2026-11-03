"""Write the final metrics of a PBA run into its wandb run (called at the end of pba_run.sh, after scoring).

Usage: .venv/bin/python scripts/wandb_summary.py <reference run dir> <run dir>
       (no-op unless WANDB_PROJECT is set; WANDB_RUN_ID / WANDB_RESUME=allow select the run to resume)

Summary keys, one value per run for the runs table and scatter plots across runs (the numbers of plot_compute.py):
    final/s_per_scene   DA3 forward + PBA time per scene
    final/auc03         pose AUC@3 (%), mean of the 5 pose datasets
    final/f1            recon F1 (%), mean of HiRoom, 7-Scenes, ScanNet++
    final/f1_all        recon F1 (%), mean of HiRoom, 7-Scenes, ScanNet++, ETH3D (runs with PBA_ETH3D_RECON=1)
    final/<dataset>/auc03, final/<dataset>/auc30, final/<dataset>/f1, final/dtu/chamfer_mm (per dataset)
    ref/...             the same for the reference run (the unrefined predictions, init_<subset>)
    gain/auc03, gain/f1 final minus ref
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_ablation import forward_times  # noqa: E402
from plot_compute import POSE, RECON, RECON_ALL, REPO_ROOT, load, point  # noqa: E402


def metrics(run, forward):
    out = {}
    p = point(run, forward)
    if p:
        out.update(s_per_scene=p[0], auc03=p[1], f1=p[2])
    if p4 := point(run, forward, RECON_ALL):
        out["f1_all"] = p4[2]
    for d in POSE:
        r = load(run, f"{d}_pba")
        for k in ("auc03", "auc30"):
            if r:
                out[f"{d}/{k}"] = 100 * sum(v[k] for v in r.values()) / len(r)
    for d in RECON_ALL:
        r = load(run, f"{d}_recon_unposed")
        if r:
            out[f"{d}/f1"] = 100 * sum(v["fscore"] for v in r.values()) / len(r)
    r = load(run, "dtu_recon_unposed")
    if r:
        out["dtu/chamfer_mm"] = sum(v["overall"] for v in r.values()) / len(r)
    return out


def main():
    if not os.environ.get("WANDB_PROJECT") or os.environ.get("WANDB_MODE") == "disabled":
        return
    import wandb

    ref, run_dir = sys.argv[1], sys.argv[2]
    os.chdir(REPO_ROOT)
    model = os.path.basename(os.path.dirname(os.path.normpath(run_dir)))
    forward = forward_times(model)
    final, base = metrics(run_dir, forward), metrics(ref, forward) if os.path.isdir(ref) else {}
    summary = {f"final/{k}": v for k, v in final.items()} | {f"ref/{k}": v for k, v in base.items()}
    summary |= {f"gain/{k}": final[k] - base[k] for k in ("auc03", "f1", "f1_all") if k in final and k in base}
    try:
        run = wandb.init(resume="allow")
    except Exception as e:  # noqa: BLE001
        print(f"[wandb_summary] wandb init failed: {e!r}")
        return
    run.summary.update(summary)
    run.finish()
    print("[wandb_summary] " + json.dumps({k: round(v, 2) for k, v in summary.items() if k.count("/") == 1}))


if __name__ == "__main__":
    main()
