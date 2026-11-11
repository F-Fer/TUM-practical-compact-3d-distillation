# Correspondences inside or before the photometric BA

Status: **rejected:** DINO token matches, RoMa matches as a depth-lifted or epipolar term inside PBA, chaining sparse
BA → PBA (session 1) · **works for SMALL in a different form:** track BA first, then a depth-only polish
([track-ba](track-ba.md)) · **open:** a wide-baseline term targeted at the weak pose directions · Area: PBA + matching ·
Updated 2026-11-10

**Answer:** every attempt to add matches to the photometric objective lowered AUC@3. Starting PBA from a sparse BA's
poses was worse than starting from DA3 when the sparse BA had damaged good poses.

## Evidence

| date | setting | change | result | source |
|---|---|---|---|---|
| 10-30 | SMALL/LARGE dev | PBA started from `vggt-ba` sparse BA (`init_from=...-ba`) | no gain: HiRoom worse (sparse BA had damaged those poses), DTU 72.3 → 69.6. Without rescaling the translations to DA3's scale: degenerate solution, AUC@30 = 10 | REPORT1 |
| 10-31 | single scene | DA3 token mutual-NN matches as a Huber reprojection term | A@30 0.687 → 0.52-0.61 (0.77 without); matches mostly wrong | REPORT2 §3, [dino](dino-features.md) |
| ≤ 11-05 | dev (Antonin) | RoMa matches as a depth-lifted or epipolar (Sampson) term inside PBA (wrapper around `run_tto_pba.py`, idea from BA-T arXiv 2606.03287; not kept) | lowers AUC@3 | CLAUDE.md "Negative results" |
| ≤ 11-05 | dev (Antonin) | RoMa tracks + radial BA, then depth-only polish (`init_from=<ba run> poses=false focal=none`) / full polish | depth-only keeps the BA's poses; full polish raises F1 but moves the poses again (undid the radial ETH3D gain and DTU-64 100) | [track-ba](track-ba.md) |
| ≤ 11-05 | dev (Antonin) | per-scene gate between track BA and PBA; initial reprojection residual as a predictor of where BA helps | gate doesn't beat the per-model choice; residual doesn't predict | CLAUDE.md "Negative results" |

## Open

A wide-baseline term chosen for the weak modes (REPORT5 §11): ScanNet++ and ETH3D poses are where tracks help SMALL
most, and DA3's remaining error is in directions the photometric term barely constrains
([pose prior](pose-prior-and-flat-directions.md)). Track BA was tested before the shared focal and relative brightness.

Related: [track-ba](track-ba.md), [dino-features](dino-features.md).
