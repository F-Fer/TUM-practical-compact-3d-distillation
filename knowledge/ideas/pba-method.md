# Photometric / feature-metric BA: the method and the current configurations

Status: **adopted** (the TTO we build on) · Area: PBA · Updated 2026-11-10

`scripts/run_tto_pba.py` (+ analytic kernels `scripts/pba_analytic.py`) refines saved DA3 predictions per scene and
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

## Current configurations (session-5 setup, `results/pba/jobs/pmba/f1_lines.py`, `f2_lines.py`)

```sh
COMMON="nbr_mode=spread neighbors=10 lr_rot=3e-3 lr_trans=3e-3 lr_focal=3e-3 levels=8,4,2,1,1 lm_levels=4 impl=analytic lm_impl=analytic lm_solver=dense32 radial=true radial_max=0 radial_gate=0.01 focal_init=median"
# photometric B′ (cheapest; T = 25 is enough): grey Huber, relative brightness
pho="$COMMON samples=1024 lm_iters=15 lm_samples=4096 lm_stop=2e-3 bright_rel=true iters=25"
# photometric + 40 LM iterations on NCC of 3x3 grey patches: best pose on SMALL/BASE (held-out SMALL 57.2 at T = 800)
ncc="$COMMON samples=1024 lm_samples=4096 lm_photo_loss=ncc bright_rel=true lm_iters=40 iters=800"
# feature-metric VGG + Huber at 1,024 px: best F1 on LARGE/GIANT, ~0.3-0.9 s more than pho
fmh="$COMMON features=vgg photo_loss=huber vgg_fast=true feat_dim=8 samples=1024 lm_iters=15 lm_samples=4096 lm_stop=2e-3 iters=50"
# feature-metric VGG + NCC at 4,096 px: best pose on LARGE at T = 800 but 3.3x fmh's time
fm="$COMMON features=vgg photo_loss=ncc vgg_fast=true feat_dim=8 samples=4096 lm_iters=15 lm_stop=2e-3 iters=800"
.venv/bin/python scripts/run_tto_pba.py workspace/evaluation/<model> workspace/tto/<model>/<run> $pho datasets=hiroom max_scenes=10
```

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

## Requirements and limits

- Analytic path (`impl=analytic lm_impl=analytic`) needed by `bright_rel`, `lm_stop`, `lm_solver=dense32`,
  `lm_photo_loss`, `radial`, `pp`. The old autograd/jvp path is needed by `camera=radial`, `res_scale`,
  `sample_chunk` (REPORT6) and is 3-6x slower.
- Peak GPU memory: photometric 5.2-5.3 GiB, VGG configurations 10.4-10.8 GiB (100 frames).
- PBA can't fix DA3's gross pose failures (7-9 % of frames on ScanNet++/ETH3D stay wrong; [gross failures](gross-failures-and-reference-view.md)).
