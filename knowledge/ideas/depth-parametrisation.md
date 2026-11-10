# Depth parametrisation in PBA

Status: **adopted:** 8-cell log-depth grid per frame · **rejected / not better:** 4, 6, 32-cell (fine) grids, free
per-sample depth from DA3's start, fine-tuning DA3's depth head (SMALL) · **open:** 12-16 cells (memory), depth head
on LARGE (better recon, 8x time) · Area: PBA · Updated 2026-11-10

**Answer:** some depth freedom is essential (frozen depth collapses), a coarse grid is the right amount from DA3's
start: finer or free depth can explain away pose errors. 16 cells holds the poses better in the GT-start test but the
dense LM system doesn't fit; 12 cells gains where it fits.

## Evidence

| date | setting | change | result | source |
|---|---|---|---|---|
| 10-30 | SMALL dev, v1 | + 8-cell grid on top of focal | HiRoom F1 34.6 → 42.0 (+ pose) | REPORT1 |
| 10-30 | SMALL dev | 16-cell grid | 23.7 vs 24.0: no better | REPORT1 |
| 10-31 | SMALL dev, Adam 4x200 | + 32-cell fine grid from level 2 | neutral (HiRoom/ScanNet++ F1 +1.5/+2.4, 7-Scenes F1 −3.4) | REPORT2 §4 |
| 10-31 | SMALL dev, Adam 4x50 | DA3 depth head fine-tuned per scene (`depth_param=head`, Test3R-like, cameras free) instead of grid | same pose, worse recon (7-Scenes F1 −5.6, DTU CD worse), 2x time | REPORT2 §4 |
| 10-31 | LARGE dev, Adam 4x50 | same | better recon: 7-Scenes F1 59.9 → 63.6, ScanNet++ 61.4 → 66.4, DTU-64 A@3 69.6 → 72.6; HiRoom/ETH3D −0.6/−2.6; 38.6 vs 4.7 s (8x). Not tuned (lr 1e-5, 8 frames/step) | REPORT2 §4 |
| 11-05 | SMALL, LM from GT poses | frozen / 4 / 8 / 16 cells | HiRoom 39.3 / 60.6 / 66.5 / 70.4 at LM it. 5 (24.6 / 40.3 / 51.9 / 56.0 at 20); DTU 68.3 / 82.8 / 84.8 / 88.5 | REPORT3 §3 |
| 11-05 | SMALL subset | 4-cell grid in the pipeline | 24.7 vs 26.5-26.9 (worse); 6-cell unclear | REPORT3 §4 |
| 11-07 | SMALL, LM from GT, NCC | free per-sample depth (`lm_free_depth`, exact Schur) | passes the 16-cell bar at 1/14 of its time (0.17 vs 2.42 s): HiRoom 91.8/78.9, DTU 99.6/97.4; but NCC with frozen depth is already 88.3/64.6: most of the gain is NCC | REPORT4 §4 |
| 11-07 | SMALL dev, Adam 4x50 + 40 LM on NCC | free depth vs grid | **29.6 vs 33.9**: from DA3's start free depth explains away pose errors (converges by iteration 15) | REPORT4 §4, §6 |
| 11-08 | SMALL dev, 3 seeds | 12-cell grid (`depth_grid=12`) | gains on the 3 sets that fit in 8 GB (DTU +1.9, ETH3D +5.1; 39.9 vs 37.7 on those sets); + fitted prior 40.3; 100-frame sets OOM with the dense LM | REPORT5 §4 |

## Notes

- Reading of the GT-start test (REPORT3): poses compensate depth errors the grid can't express. REPORT4 later showed
  that with NCC most of that bias disappears, so "depth rigidity" was mainly a per-pixel-loss effect
  ([objective-bias](objective-bias-gt-start.md)).
- The geometric row pulls free depths back towards DA3's errors (no geometric row: HiRoom 86.0 at it. 20, REPORT4 §4).
- Eliminating the grid from the LM system (Schur without cross-frame coupling) loses ScanNet++ ([speed](speed.md)).

## Open

- 12-16 cells with a block-wise or Adam-only depth solve (dense LM too large at 100 frames on 8 GB; the 5090 has 32 GB).
- Network-space depth on LARGE with cheaper variants (LoRA/bias-only, fewer refreshes, last level only).

Related: [depth-prior](depth-prior.md), [da3-reconditioning](da3-reconditioning.md), [patch-ba](patch-ba.md).
