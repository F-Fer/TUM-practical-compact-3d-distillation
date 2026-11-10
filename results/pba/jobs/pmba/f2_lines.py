#!/usr/bin/env python3
"""Job lines for the cheaper feature-metric line of the compute plot (2026-11-09), all four models, dev subset.

Usage: f2_lines.py gpu|cpu. Same code snapshot (f1) and protocol as f1_lines.py: the GPU runs don't score, the CPU lines
score recon F1 and delete the predictions (score_clean.sh).
  fmh: VGG16 features (8 PCA channels) with the per-pixel Huber loss at 1024 px (no brightness parameters), LM on the
       same residual with the stop rule, shared focal, radial gate 0.01. Session 4's x1_vgg arm with the shared focal.
"""
import sys

SNAP = "f1"
MODELS = ["DA3-SMALL", "DA3-BASE", "DA3-LARGE-1.1", "DA3-GIANT-1.1"]
COMMON = "nbr_mode=spread neighbors=10 lr_rot=3e-3 lr_trans=3e-3 lr_focal=3e-3 levels=8,4,2,1,1 lm_levels=4 impl=analytic lm_impl=analytic lm_solver=dense32 radial=true radial_max=0 radial_gate=0.01 focal_init=median"
H = f"{COMMON} features=vgg photo_loss=huber vgg_fast=true feat_dim=8 samples=1024 lm_iters=15 lm_samples=4096 lm_stop=2e-3"
TS = (25, 50, 200, 400, 800)

runs = sorted(((1 + t / 100, m, f"fmh_it{t}", f"{H} iters={t}") for t in TS for m in MODELS), key=lambda r: -r[0])
for _, m, name, ov in runs:
    if sys.argv[1] == "gpu":
        print(f"MODEL={m} WANDB_MODE=disabled bash $J/run.sh {SNAP} {name} dev 0 {ov} > $J/runlogs/{name}_{m}.log 2>&1")
    else:
        print(f"MODEL={m} bash $J/score_clean.sh {name} > $J/runlogs/score_{name}_{m}.log 2>&1")
