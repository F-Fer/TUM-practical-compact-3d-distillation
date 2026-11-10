# Model in the loop: re-run DA3 on the refined cameras

Status: **rejected as a default** (not a consistent gain after the hybrid; helps DTU Chamfer) · Area: TTO pipeline ·
Updated 2026-11-10

`scripts/run_tto_recond.py`: feed PBA's refined cameras into DA3's posed mode (intrinsics converted back to the original
image size), keep the refined cameras and take DA3's new depth. One extra forward pass.

## Evidence (REPORT2 §4, §6)

| setting | result |
|---|---|
| SMALL dev, after Adam 4x200 | F1 HiRoom +2.8 (47.4 → 50.2), ScanNet++ +2.1; pose unchanged by construction |
| control: DA3 re-run on its own (unrefined) cameras | neutral to slightly worse (HiRoom F1 23.9 → 21.9) |
| + a second short PBA round | +1 F1 HiRoom on dev |
| full benchmark, after the hybrid (Adam 4x200 + LM 5) | F1 down 0.4-6 (SMALL HiRoom −5.9), DTU Chamfer better (SMALL 3.66 → 3.51, BASE 3.09 → 2.58, LARGE 1.73 → 1.71) |

**Reading:** once PBA refines depth and cameras jointly, its depth is already better than what DA3 produces for the same
cameras, except on DTU.

Related: [depth-parametrisation](depth-parametrisation.md) (network-space depth), [matched-compute](matched-compute.md).
