#!/usr/bin/env python3
"""Job lines for the compute plot with the session-5 configurations (2026-11-09), all four models, dev subset.

Usage: f1_lines.py gpu|cpu. gpu: the PBA runs (no scoring), longest first so that three GPU workers finish together.
cpu: recon F1 of every run, then its predictions are deleted (score_clean.sh).
Families (run names <family>_it<T>, T = Adam iterations per level):
  pho: REPORT5's B' = Recommended + bright_rel + focal_init=median (Adam on grey Huber, LM on Huber with the stop rule)
  ncc: REPORT5's N + focal_init=median (Adam on grey Huber, then 40 LM iterations on NCC of 3x3 grey patches)
  fm:  REPORT3's feature-metric configuration (VGG16, 8 PCA channels, NCC, LM stop rule) with radial gate 0.01 and
       focal_init=median
"""
import sys

SNAP = "f1"
MODELS = ["DA3-SMALL", "DA3-BASE", "DA3-LARGE-1.1", "DA3-GIANT-1.1"]
COMMON = "nbr_mode=spread neighbors=10 lr_rot=3e-3 lr_trans=3e-3 lr_focal=3e-3 levels=8,4,2,1,1 lm_levels=4 impl=analytic lm_impl=analytic lm_solver=dense32 radial=true radial_max=0 radial_gate=0.01 focal_init=median"
B = f"{COMMON} samples=1024 lm_iters=15 lm_samples=4096 lm_stop=2e-3 bright_rel=true"
N = f"{COMMON} samples=1024 lm_samples=4096 lm_photo_loss=ncc bright_rel=true lm_iters=40"
V = f"{COMMON} features=vgg photo_loss=ncc vgg_fast=true feat_dim=8 samples=4096 lm_iters=15 lm_stop=2e-3"
FAMILIES = {"fm": (V, (25, 50, 200, 400, 800)), "ncc": (N, (50, 200, 800)), "pho": (B, (25, 50, 200, 400, 800))}
# rough GPU minutes per dev run (session-3/4 logs; fm above T = 50 is a guess), for the longest-first order
COST = {"fm": 1.0, "ncc": 0.8, "pho": 0.5}

runs = [(COST[f] * (1 + t / 100), m, f"{f}_it{t}", f"{ov} iters={t}") for f, (ov, ts) in FAMILIES.items() for t in ts for m in MODELS]
runs.sort(key=lambda r: -r[0])
for _, m, name, ov in runs:
    if sys.argv[1] == "gpu":
        print(f"MODEL={m} WANDB_MODE=disabled bash $J/run.sh {SNAP} {name} dev 0 {ov} > $J/runlogs/{name}_{m}.log 2>&1")
    else:
        print(f"MODEL={m} bash $J/score_clean.sh {name} > $J/runlogs/score_{name}_{m}.log 2>&1")
