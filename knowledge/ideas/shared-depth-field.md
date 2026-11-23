# Make PBA optimise the depth: a shared scene-level depth correction instead of per-frame grids

Status: **open (proposal, nothing run)** · Area: TTO method / depth · Updated 2026-11-23

**Answer:** none yet. Proposal: replace (or constrain) the per-frame 8-cell log-depth grids of `run_tto_pba.py` by a
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

## Open / don't repeat

- Already negative: alternation of a 32-cell depth stage with poses fixed (depth error unchanged); a learned depth
  update from residual maps; depth distillation LARGE → SMALL at tum1 scale ([depth-bottleneck](depth-bottleneck.md)).
- Free per-pixel depth and depth-head fine-tuning lost from DA3's start ([depth-parametrisation](depth-parametrisation.md)):
  a shared field is different (few parameters, shared across frames), but check that file's failure modes before.

Related: [depth-bottleneck](depth-bottleneck.md), [learned-refinement-step](learned-refinement-step.md),
[pose-prior-and-flat-directions](pose-prior-and-flat-directions.md), [depth-prior](depth-prior.md),
[depth-parametrisation](depth-parametrisation.md)
