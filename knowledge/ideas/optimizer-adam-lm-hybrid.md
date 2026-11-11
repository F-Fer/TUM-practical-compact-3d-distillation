# Optimiser: Adam, LM, L-BFGS and the Adam + LM hybrid

Status: **adopted** (Adam on blurred levels, then LM on the sharp level) · Area: PBA · Updated 2026-11-10

**Answer:** Adam has the wide basin, LM converges fast but only from close. A short Adam run followed by a few LM
iterations at full resolution is worth 3-10x more Adam iterations for about 1 s. LM alone, LM pyramids, L-BFGS and
"no Adam" all lose.

## Evidence

| date | setting | change | result | source |
|---|---|---|---|---|
| 10-30 | SMALL dev, DTU | L-BFGS vs Adam | no better per second (DTU 43.6 in 25 s vs 58.8 in 21 s), 12.6 GiB | REPORT1 |
| 10-30 | SMALL dev, v1 | Adam learning rate 3x (`lr_rot/lr_trans/lr_focal=3e-3`) or 400 → 800 iterations | HiRoom 24 → 34, DTU 41 → 59; 10x overshoots (focal error 17 %). 3e-3 is used since | REPORT1 |
| 10-31 | SMALL dev, v3 | LM alone 4x10 vs Adam 4x200 | ScanNet++ 26.9 vs 17.9 (global solve propagates), ETH3D A@30 79.6 vs 70.2; worse on 7-Scenes/ETH3D A@3 | REPORT2 §2 |
| 10-31 | one HiRoom scene | why LM crawls | actual decrease ~2x the Gauss-Newton prediction: J^T J overestimates curvature of large-residual NCC; steps too short far from the optimum | REPORT2 §2 |
| 10-31 | SMALL dev | **Adam 4x50 + LM 5 (6.0 s)** vs Adam 4x200 (16.7 s) | same level on HiRoom, ScanNet++, DTU; Adam 4x200 + LM 5 (19.5 s) ≈ Adam 4x800 (~65 s) | REPORT2 §2 |
| 10-31 | BASE / LARGE dev | same | BASE hybrid beats 4x200 on HiRoom (53.3 vs 50.8); LARGE beats 4x800 on ScanNet++ (71.4 vs 68.0) | REPORT2 §2 |
| 10-31 | full benchmark | V3 Adam 4x200 + LM 5 vs session 1's v3 (same 18 s) | SMALL HiRoom 37.1 → 45.7, ScanNet++ 28.6 → 38.5, DTU CD 4.21 → 3.66; BASE HiRoom 48.8 → 57.4, ScanNet++ 49.1 → 55.4; LARGE (`w_prior=100`) HiRoom 79.8 → 82.8, F1 82.8 → 84.4; ETH3D the exception (SMALL −0.9, BASE −3.7, LARGE −8 without pose prior). RGB fast hybrid (Adam 4x50 + LM 5) 1.6 s: SMALL HiRoom 36.7, LARGE best DTU CD 1.59 | REPORT2 §6 |
| 10-31 | all models | catch | 7-Scenes and ETH3D get worse with LM (LARGE ETH 52.4 → 28.4) → flat directions / lens, see [pose prior](pose-prior-and-flat-directions.md), [radial](radial-lens-model.md) | REPORT2 §2 |
| 10-31 | SMALL, rotations perturbed 5° / 15° | basin | Adam on the VGG pyramid recovers from 5° (HiRoom 30.7 vs 34.2); LM alone diverges from 15° (0.0); Adam 4x50 + LM 5 poor from 15° (1.3/53.4) | REPORT2 §5 |
| 11-02 | SMALL dev, RGB fast, 22 runs `abl_*` | ablation | LM 5 is the decisive component: Adam 4x10 + LM 5 (1.1 s) = Adam 4x200 (3.7 s); LM needs an Adam warm start (LM alone 19.8); LM 2 < 5 ≈ 10. Level pyramid and the split of iterations across levels don't matter at 200 iterations; 1 level x 800 beats 4x200 at equal time (28.0/47.2 vs 27.2/44.7) | memory note, `results/pba/figures2/ablation_small.png` |
| 11-02 | LARGE dev | Adam 4x25 + LM 5 | costs pose (53.2 vs Adam 4x50 56.7), all of it ETH3D courtyard (65.9 → 20.6, below init) | memory note |
| 11-05 | SMALL subset/dev | LM pyramid without Adam | 24.7 (DTU 45.8); lower initial damping `lm_lambda` 1e-4/1e-5: 22.0/21.0 vs 23.1 | REPORT3 §4 |
| 11-05 | SMALL dev | LM 3 instead of 5 | 28.6/81.3/44.6 vs 28.6/82.0/47.2 (F1 −2.6) | REPORT3 §4 |
| 11-07 | SMALL dev | no Adam: LM on NCC at 4 blur levels | 29.5 (grid), 23.1 (free depth) vs 33.9 with Adam 4x50 first | REPORT4 §6 |

## Mechanism

Gauss-Newton needs small residuals; Adam's normalised steps don't care about curvature. LM's damping decays to 1e-7
and takes large steps in near-null directions, so converged LM drifts along flat directions faster than Adam
([pose prior](pose-prior-and-flat-directions.md)). Order that falls out (REPORT2 §5): coarse features for large errors
(only with poor inits, [dino](dino-features.md)) → blurred Adam → LM at full resolution.

## Don't repeat

L-BFGS; LM without an Adam warm start; reusing the factorisation (chord method) or eliminating the grid
([speed](speed.md)); fewer than 5 LM iterations without the stop rule.

Related: [lm-stop-rule](lm-stop-rule.md), [iterations-over-t](iterations-over-t.md), [ncc](ncc.md) (LM on NCC).
