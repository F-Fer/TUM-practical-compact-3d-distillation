# Make PBA optimise the depth: a shared scene-level depth correction instead of per-frame grids

Status: **open (first prototype negative)** · Area: TTO method / depth · Updated 2026-11-23

**Answer (dev, SMALL, seed 0):** none of the variants beats B′ (recon F1 48.8 ± ~1.5 seed noise). Longer B′ (T = 25 → 100 → 400) improves the depth error only slightly (HiRoom 4.85 → 4.63 %, ScanNet++ 6.02 → 5.63 %) and F1 48.8 → 48.8 → 50.5 (within seed noise); the autograd world-space field (16³, 8³) is worse than the per-frame grid in the same autograd path at every T and *raises* the ScanNet++ depth error (6.0 → 6.9-8.8 %). Freezing the grid in the coarse levels (`grid_levels=3,4`) ties B′ with slightly lower depth error; `w_prior` 0.1-30 and the coupling penalty (`grid_proj`) don't help. A per-pixel plane sweep (NCC) on fixed cameras improves the depth only with GT cameras; on B′'s cameras it worsens it, and alternating sweep ↔ pose polish drifts (AUC 42.8 → 36.7). At B′'s pose accuracy the poses are the limit: the lever is an independent pose signal (tracks, better initial cameras), not the depth parametrisation. Ceilings: GT-quality depth is worth up to +26 F1 on HiRoom and +16 on ETH3D. Proposal: replace (or constrain) the per-frame 8-cell log-depth grids of `run_tto_pba.py` by a
depth correction that all frames share, so that a frame can no longer absorb its own pose error by bending its own
depth. Judge every variant by **reconstruction F1** (TSDF, 4 sets; DTU-49 Chamfer) and by the **depth error against GT**
before / after PBA, with pose AUC as a secondary metric.

## Why

- PBA's pose is depth-limited ([depth-bottleneck](depth-bottleneck.md)): GT depth fixed lifts B′ 38.2 → 49.1 AUC@3
  (HiRoom / ScanNet++ / ETH3D); a GT-fitted smooth 16-cell correction of DA3's depth gets 46.6 of GT's 49.4.
- But PBA doesn't improve the depth itself: B′ moves DA3's depth error only 6.3 % → 5.4-6.0 %; alternating a 32-cell
  depth stage with a pose stage leaves it unchanged; small learned refiners and depth distillation don't transfer
  ([depth-bottleneck](depth-bottleneck.md)).
- The reason is the parametrisation: every frame has its own grid, tied to the others only softly by the geometric term.
  A smooth change of frame i's grid can compensate a change of frame i's pose, so the photometric objective is flat
  along these directions (condition number ~1e5; fixing the grids stiffens the weak pose directions ×29, ETH3D ×16,
  HiRoom ×52; [learned-refinement-step](learned-refinement-step.md) check 1). With GT depth, the free grid even costs
  3.2 AUC@3.
- Classical BA avoids this because the structure is shared: a 3D point is one unknown that every view must agree on.

If PBA could optimise the depth properly, the TTO budget would buy better depth and better poses together: more
iterations, finer depth corrections and higher resolution would scale the result, which the current method can't.

## Proposal

**Scene-level depth field.** One correction field in world coordinates, applied to every frame's back-projected points:
a coarse 3D grid (trilinear or cubic B-spline, ~8³-16³ cells over the scene's bounding box) of log-depth scale along the
viewing ray (or a 3D displacement), plus one scale per frame. A pose error of frame i can then only be absorbed by
bending a region that the other frames see too, which their residuals contradict. Parameters: a few thousand, fewer
than 100 frames × 48 cells.

Open design choices: field in world vs reference-camera coordinates; scale vs displacement; bounding box and resolution
(robust to far points / sky, ETH3D); how it interacts with the per-frame scale and the focal; regularisation (smoothness,
prior towards zero).

**Prototype path.** Autograd first (the autograd PBA path is at commit `1c1a921`), on the mini set (13 scenes) or a few
dev scenes; analytic Jacobians in `Kernel` only once it works.

## Cheaper variants to try first (no new parametrisation)

1. **Coupling-aware depth update:** in the LM solve, remove from each grid step the component that a pose change
   explains, i.e. project the grid update orthogonal to the coupling block `Hgp` (forbids exactly the absorption).
2. **Staged freedoms:** poses + focal only (grid frozen) in the coarse Adam levels, the grid released only at the end
   with a stronger prior. Related evidence: one pose polish with the depth fixed after B′ gave +0.9 AUC@3 (81 scenes,
   one seed, [learned-refinement-step](learned-refinement-step.md) §11).
3. **Re-tune `w_prior`** on the current configuration: never done since session 2 (README open question 7); it is the
   baseline the other variants must beat.

## Evaluation

- **Primary:** recon F1, `recon_unposed`, 4 sets (HiRoom, 7-Scenes, ScanNet++, ETH3D) on dev, ≥ 3 seeds; DTU-49 Chamfer.
  ETH3D dev recon takes ~80 s and ≤ 15 GB RAM; on tum1 use one fusion worker.
- **Fast diagnostic:** depth abs. rel. vs GT before / after PBA (HiRoom, ScanNet++, ETH3D; one scale per scene, as
  `scripts/lpba_depth_oracle.py` / `lpba_stage.py` print it), seconds per scene.
- **Secondary:** AUC@3 / AUC@30.
- **Step 1, the ceilings:** dev recon F1 of B′, GT depth fixed, the smooth-16 oracle and LARGE depth + SMALL cameras
  (`lpba_depth_oracle.py` with `save=true`, then `scripts/eval_tto.py`). They say how much F1 better depth can buy, i.e.
  the target for this idea.

## Evidence

| date | setting (model, set, seeds, baseline) | change | result | source |
|---|---|---|---|---|
| 11-23 | SMALL, dev 22 scenes, seed 0, B′ `iters=25` (F1 48.8, AUC@3 43.2; seeds 0-2: F1 47.2-50.3, AUC 42.1-45.3) | step 1 ceilings (`lpba_depth_oracle.py`): GT depth fixed / GT-fitted smooth 16 cells / LARGE-1.1's depth | F1 HiRoom / 7-Sc / ScanNet++ / ETH3D (mean): B′ 56.9 / 50.8 / 36.3 / 51.1 (48.8); GT 82.8 / 37.6 / 32.5 / 67.0 (55.0); smooth16 66.1 / 22.9 / 38.1 / 71.4 (49.6); LARGE 64.9 / 56.3 / 32.7 / 66.3 (55.0, AUC 48.3); the 7-Sc / ScanNet++ GT oracles only cover GT pixels, unreliable for F1 | `sdf/bp_*`, `gtfix_s0`, `smooth16_s0`, `large_s0` |
| 11-23 | same | step 2: `w_prior` 0.1 / 0.3 / 3 / 10 / 30 (default 1) | F1 48.1 / 47.6 / 46.6 / 47.5 / 46.0; AUC 46.4 / 42.2 / 40.6 / 38.8 / 32.9; depth err roughly flat (ScanNet++ 7.1 % at 0.1) | `sdf/wp*` |
| 11-23 | same | step 2: grid frozen except in levels 4 / 3,4 / 0-3 (`grid_levels`) | F1 48.4 / **49.0** / 40.2; AUC 42.8 / 43.2 / 30.8; depth err H/S/E 4.62 / 5.97 / 4.14 % (gl34) vs 4.85 / 6.02 / 4.32 | `sdf/gl*` |
| 11-23 | same | step 2: LM penalty on grid steps in range(Hgp) (`grid_proj`, w 1000 / 10) | F1 41.1 / 42.2, AUC 30.7 / 31.8 (focal drifts) | `sdf/gp*` |
| 11-23 | same | iterations: B′ `iters` 25 / 100 / 400 | F1 48.8 / 48.8 / 50.5; AUC 43.2 / 46.3 / 45.8; depth err HiRoom 4.85 / 4.79 / 4.63, ScanNet++ 6.02 / 5.89 / 5.63, ETH3D 4.32 / 4.26 / 4.23 % | `sdf/bp_wrap`, `bp_T*` |
| 11-23 | same, autograd Adam (`lpba_field.py`, pinhole) + analytic pose polish with depth fixed (`_pol`) | step 3: world-space trilinear field 16³ (and 8³) + per-frame scale vs per-frame 8-cell grid, `iters` 25 / 100 / 400 | F1 (pol) grid 39.3 / 46.4 / 48.6, field16 39.7 / 44.0 / 46.7, field8 41.3 (T 25); AUC (pol) grid 30.6 / 36.8 / 44.3, field16 30.0 / 35.4 / 39.6; depth err ScanNet++ grid 6.69 / 6.34 / 5.69, field16 8.81 / 8.28 / 6.89 % (DA3 6.05) | `sdf/f_*` |
| 11-23 | SMALL, dev 22 scenes, seed 0, state = B′ `iters=25` (F1 49.9, AUC@3 42.8) | option 1: per-pixel plane sweep on fixed cameras (`scripts/lpba_mvs.py`, NCC 7×7, ±15 % in 33 steps, 10 neighbours, confidence-weighted correction blurred σ 32 px, 3 rounds); GT cameras vs B′ cameras; then 3 cycles sweep ↔ pose polish (depth fixed) | GT cameras: depth err H/S/E 4.84/6.02/4.31 → 3.94/5.36/4.24 %, F1 57.0 (incl. GT cameras, control GT cams + B′ depth not run). B′ cameras: depth err rises (cycle 1-3: ScanNet++ 6.36 / 6.78 / 7.24 %), F1 45.3 / 42.3 / 43.3, AUC@3 after polish 40.8 / 38.7 / 36.7: the alternation drifts. The stereo signal exists only with good cameras | `mvs/*`, `results/pba/runs/DA3-SMALL/mvs_*` |

## Open / don't repeat

- Already negative: alternation of a 32-cell depth stage with poses fixed (depth error unchanged); a learned depth
  update from residual maps; depth distillation LARGE → SMALL at tum1 scale ([depth-bottleneck](depth-bottleneck.md)).
- Free per-pixel depth and depth-head fine-tuning lost from DA3's start ([depth-parametrisation](depth-parametrisation.md)):
  a shared field is different (few parameters, shared across frames), but check that file's failure modes before.

Related: [depth-bottleneck](depth-bottleneck.md), [learned-refinement-step](learned-refinement-step.md),
[pose-prior-and-flat-directions](pose-prior-and-flat-directions.md), [depth-prior](depth-prior.md),
[depth-parametrisation](depth-parametrisation.md)

Code: removed in the merge cleanup of 2026-11-26 (closed experiment); the last version is at commit `107ea16`, e.g. `git show 107ea16:scripts/<name>.py`: `lpba_field.py`, `lpba_mvs.py`, `lpba_stage.py`, and the `run_tto_pba.py` options `grid_levels`, `grid_proj`, `grid_proj_w`.
