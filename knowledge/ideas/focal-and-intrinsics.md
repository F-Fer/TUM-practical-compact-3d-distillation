# Focal length and other intrinsics

Status: **adopted:** shared focal refined by PBA, initialised as the median of DA3's per-frame focals
(`focal_init=median`) · **rejected:** global focal prior, free principal point · **open:** DTU focal drift · Area: PBA,
camera model · Updated 2026-11-10

**Answer:** DA3's focal is the largest single camera error on HiRoom (median 19 % too long) and ETH3D (7 %), and its
per-frame focals differ by 0.5-1 % within a scene although every benchmark scene has one camera. Refining one shared
focal was the biggest factor of session 1; replacing the per-frame focals by their median first was the largest single
gain since session 3, and it is free.

## Evidence

| date | setting | change | result | source |
|---|---|---|---|---|
| 10-30 | SMALL dev, v1 | + focal refinement (poses only → poses + focal → + grid) | HiRoom F1 25.2 → 34.6 → 42.0; focal error 10-17 % → 1-3 % on HiRoom/ETH3D | REPORT1 |
| 10-30 | SMALL / LARGE DTU-49 | focal prior `w_focal=10` | DTU CD 4.30 → 3.35 (SMALL), 2.59 → 2.28 (LARGE), but HiRoom AUC@3 34.2 → 21.2: needs large corrections there | REPORT1 |
| 10-30 | LARGE DTU | no grid + `w_focal=10` | DTU CD 0.99 mm (DA3 alone 1.71), SMALL breaks (CD 7.30) | REPORT1 |
| 11-07 | DA3-SMALL, all scenes (oracle) | focal/GT ratio | HiRoom median 1.19 (p10-p90 1.02-1.27), 7-Scenes 0.90, ETH3D 1.07 (up to 1.32), ScanNet++/DTU 0.99-1.00; fixing the focal alone +18 F1 on HiRoom (own back-projection F1) | failure-mode report |
| 11-08 | SMALL dev, 3 seeds | **`focal_init=median`** (B50 → B′) | 33.7 → **44.7** (HiRoom 34.7 → 43.9, DTU 55.1 → 86.2), F1 47.2 → 50.8 | REPORT5 §2 |
| 11-08 | SMALL held-out | same | 39.3 → **44.0** (HiRoom +9.3, DTU +12.9; ScanNet++ 0: its focal is right and consistent) | REPORT5 §2 |
| 11-08 | SMALL held-out, T = 800 | same | 41.5 → 48.6 | REPORT5 §2 |
| 11-08 | SMALL held-out, NCC config | same | 47.3 → **57.2**, AUC@30 86.7 → 88.4, F1 51.3 → 57.0 | REPORT5 §2 |
| 11-08 | BASE / LARGE held-out, B50 → B′ | same | 50.2 → 57.4, 66.8 → 69.7 (gain shrinks as DA3's focal gets more consistent: per-frame log-std 0.4-1.1 % BASE, 0.2-0.6 % LARGE); with NCC 57.6 → 68.3, 71.3 → 74.8 | REPORT5 §7 |
| 11-08 | SMALL dev | shared principal-point offset (`pp=true`) | 33.9 vs 33.7; moves ≤ 0.3 px although DTU's GT principal point is 7/6 px off centre: the data don't determine it (first order it's a rotation) | REPORT5 §4 |

## Mechanism

A frame whose focal is 1 % off puts pixels up to 2.5 px off at 250 px from the centre; the poses compensate with small
rotations/translations. The shared focal lowers the typical error of the good frames (DTU median rotation 0.54° → 0.46°,
HiRoom 0.61° → 0.49°), not the smooth error of ScanNet++ (REPORT5 §1).

## Open

- **DTU focal still drifts** under PBA (mean error 1.5 % → 7 %, with or without the median init): focal-depth
  trade-off on narrow-FOV object scenes. Needs observability-aware regularisation (e.g. from the Hessian w.r.t. log
  focal), not a global prior (REPORT1, REPORT5 §11).
- 7-Scenes "focal error" of 8-10 % never decreases: GT intrinsics problem ([evaluation](../evaluation.md)).

Related: [radial-lens-model](radial-lens-model.md), [depth-prior](depth-prior.md), [da3-failure-modes](da3-failure-modes.md).
