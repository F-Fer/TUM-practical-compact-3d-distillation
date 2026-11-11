# Photometric / feature-metric BA: the method and the current configurations

Status: **adopted** (the TTO we build on) · Area: PBA · Updated 2026-11-10

`scripts/run_tto_pba.py` (one file since 2026-11-10) refines saved DA3 predictions per scene and
writes them back in DA3's `results.npz` format, so the benchmark scores them unchanged. Reports: `results/pba/REPORT.md`
(session 1) to `REPORT7.md`.

## What it optimises

- **Unknowns:** a pose update per frame (frame 0 fixed), one focal scale shared by all frames (since session 5 the
  per-frame focals are first replaced by their median, [focal](focal-and-intrinsics.md)), a log-depth correction per
  frame on a coarse grid (8 cells along the long side, [depth](depth-parametrisation.md)), per-frame affine brightness
  (grey/RGB Huber only, [brightness](brightness-model.md)), optionally one radial lens coefficient ([radial](radial-lens-model.md)).
- **Pairs:** covisibility from the initial prediction; 10 targets per frame spread over the covisibility ranking
  (`nbr_mode=spread`, [neighbours](neighbours-and-pairs.md)).
- **Residuals:** per iteration, M source pixels per frame (gradient-weighted from confident pixels,
  [sampling](pixel-sampling.md)) are warped into each target: photometric (Huber on grey, or NCC of 3x3 patches,
  on grey or VGG features, [loss](loss-kernel-and-colour.md), [ncc](ncc.md), [vgg](feature-metric-vgg.md)), geometric
  (Huber on log-depth consistency), occlusion test, depth prior.
- **Schedule:** Adam on 4 blur levels (σ = 8, 4, 2, 1 px) with T iterations each, then one LM level on the sharp
  images (the "hybrid", [optimizer](optimizer-adam-lm-hybrid.md)) with a stop rule on held-out pixels
  ([lm-stop-rule](lm-stop-rule.md)).

## Current configurations

The defaults of `scripts/run_tto_pba.py` are the photometric configuration B′ (grey Huber with relative brightness,
1,024 px, 10 spread neighbours, 8-cell grid, shared focal from the median, gated radial coefficient, Adam 4 x T with
T = 50, LM 15 at 4,096 px with the stop rule); the others are a few overrides. The refactored script gives bit-identical
results to the code that made the runs (checked on the CPU, 2026-11-10).

```sh
S=workspace/evaluation/<model>; D=workspace/tto/<model>
.venv/bin/python scripts/run_tto_pba.py $S $D/pho subset=dev          # photometric B′ (iters=25 is as good on dev)
.venv/bin/python scripts/run_tto_pba.py $S $D/fmh subset=dev features=vgg   # VGG + Huber: best F1 on LARGE/GIANT, +0.3-0.9 s
.venv/bin/python scripts/run_tto_pba.py $S $D/ncc subset=dev iters=800 lm_photo_loss=ncc lm_iters=40 lm_stop=0   # best pose SMALL/BASE (held-out SMALL 57.2)
.venv/bin/python scripts/run_tto_pba.py $S $D/fm subset=dev features=vgg photo_loss=ncc samples=4096 lm_samples=0 iters=800   # VGG + NCC, 3.3x fmh's time
.venv/bin/python scripts/eval_tto.py $S $D/pho hiroom,7scenes,scannetpp,eth3d recon_unposed --workers 1   # recon F1 (CPU job)
```

Track-BA polish (Antonin): `init_from=<track-BA run> poses=false focal=none` (depth only) or `init_from=<run>` (full);
since the refactor these run on the current defaults, not the old autograd path.

Results of each: [matched-compute](matched-compute.md). Older recommended configurations (v3, the fast hybrid,
REPORT3's Recommended) are superseded by these and only matter to read old runs.

## History of the configuration

| session | change | effect | file |
|---|---|---|---|
| 1 (10-30) | v3: VGG + NCC, spread k = 10, Adam 4x200, focal + 8-cell grid | SMALL full HiRoom 9.4 → 37.1 at 18 s | REPORT1 |
| 2 (10-31) | + LM polish after short Adam (hybrid) | = 3-10x more Adam for ~1 s | [optimizer](optimizer-adam-lm-hybrid.md) |
| 2 | grey instead of RGB (default since 11-02) | same quality, 6 % faster | [loss](loss-kernel-and-colour.md) |
| 3 (11-05) | analytic kernels, float32 solve | 7.7x faster | [speed](speed.md) |
| 3 | LM stop rule, gated radial coefficient | held-out 34.0 → 37.1 | [stop](lm-stop-rule.md), [radial](radial-lens-model.md) |
| 4 (11-07) | relative brightness | long runs work; held-out +2.1 | [brightness](brightness-model.md) |
| 4 | LM on NCC after Adam | held-out +9.6 at 4.8 s | [ncc](ncc.md) |
| 5 (11-08) | shared focal (`focal_init=median`) | held-out +4.7 (T = 50), +9.9 (NCC) | [focal](focal-and-intrinsics.md) |
| 7 (11-10) | VGG + Huber at 1,024 px | feature-metric 3.3x cheaper | [vgg](feature-metric-vgg.md) |

## What the code no longer has

The 2026-11-10 refactor kept only the analytic path and the options above (plus `w_pose`, `strat_cells`, `init_gt`,
`init_from`, `poses`, `focal`, `depth_grid`, `neighbors`/`nbr_mode`, the learning rates and loss scales). Removed, with
the idea file that says why: the autograd/jvp path and L-BFGS / LM-only optimisers ([optimizer](optimizer-adam-lm-hybrid.md),
[speed](speed.md)); absolute brightness, RGB, other robust kernels, min over neighbours ([brightness](brightness-model.md),
[loss](loss-kernel-and-colour.md)); DINO levels and token matches ([dino](dino-features.md)); depth head, fine grid, free
depth ([depth](depth-parametrisation.md)); patch BA ([patch-ba](patch-ba.md)); pose basis, Laplacian prior
([pose prior](pose-prior-and-flat-directions.md)); focal prior, principal point ([focal](focal-and-intrinsics.md));
`camera=radial`, `res_scale`, `sample_chunk` ([radial](radial-lens-model.md), [resolution](resolution-and-pyramid.md));
relocalisation, gross-frame oracle ([gross failures](gross-failures-and-reference-view.md)); depth-edge masks, fitted
priors, `nbr_refresh`, CUDA graph, the radial gate by held-out cost, per-iteration tracking, W&B logging. All of it is at
commit `1c1a921` (`git show 1c1a921:scripts/run_tto_pba.py`, `pba_analytic.py`, `bench_pba.py` for the float64 checks).

- Peak GPU memory: photometric 5.2-5.3 GiB, VGG configurations 10.4-10.8 GiB (100 frames).
- Peak GPU memory: photometric 5.2-5.3 GiB, VGG configurations 10.4-10.8 GiB (100 frames).
- PBA can't fix DA3's gross pose failures (7-9 % of frames on ScanNet++/ETH3D stay wrong; [gross failures](gross-failures-and-reference-view.md)).
