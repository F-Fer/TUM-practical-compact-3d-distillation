# Speed of the photometric and feature-metric BA

Status: **adopted:** analytic kernels, float32 Cholesky, fast VGG extractor, 8 channels, VGG + Huber at 1,024 px ·
**rejected:** grid elimination / PCG, chord method, CUDA graph (no gain at T = 50) · Area: PBA engineering · Updated 2026-11-10

**Answer:** the photometric BA went from 18 s (v3) to 0.2-0.3 s per scene at T = 25, the feature-metric from 18 s to
0.5-0.7 s. Most of it came from removing per-operation overhead (hand-derived gradients and Jacobians), then from
objectives that need fewer iterations ([brightness](brightness-model.md), [focal](focal-and-intrinsics.md)), then from a
cheaper feature-metric loss (REPORT3, REPORT7).

## Evidence

| date | change | effect | source |
|---|---|---|---|
| 10-31 | Adam 4x50 + LM 5 instead of long Adam | = 3-10x more Adam iterations for ~1 s | REPORT2 |
| 11-05 | analytic Adam gradient (740 → 266 tensor ops, `FlatAdam`, no `float()` sync) | 1.30 → 0.68 s per scene | REPORT3 §2 |
| 11-05 | analytic LM Jacobians instead of forward-mode jvp | → 0.40 s; peak memory 21 → 5 GiB; normal equations 140 → 22 ms at 100 frames | REPORT3 §1-2 |
| 11-05 | Adam 4x25, LM at 1,024 px | → 0.23 s | REPORT3 §2 |
| 11-05 | float32 Cholesky after Jacobi scaling (38 → 6 ms at 100 frames) + trimming | → **0.17 s** (7.7x at held-out 34.0 → 32.6); checked in float64 against the old code: loss to 8 digits, gradient cosine 1.0000000 | REPORT3 §1-2 |
| 11-05 | VGG + NCC: analytic, `vgg_fast`, 8 channels | 4.98 → 1.17 s (4.3x, AUC@3 31.4 → 31.5) | REPORT3 §5 |
| 11-08 | `cuda_graph=true` (3070) | same quality; HiRoom T = 800 −32 %, 100-frame scenes 0 % (compute-bound on the 3070), T = 50 slower (1.38 → 1.61 s); 5090 unmeasured | REPORT5 §6 |
| 11-10 | VGG + Huber 1,024 px instead of VGG + NCC 4,096 px | Adam step 11.0 → 2.2 ms at 100 frames; TTO 15.7 → 4.7 s at T = 800 (changes the result: [vgg](feature-metric-vgg.md)) | REPORT7 |

## Timing reference (RTX 5090; REPORT3 §1, REPORT7 §1)

| | 12 frames | 100 frames |
|---|---|---|
| Adam iteration, autograd (grey, 1,024 px) | 3.9 ms | 6.8 ms |
| Adam iteration, analytic, grey Huber 1,024 px | 1.0-1.16 ms | 1.24-1.35 ms |
| Adam iteration, analytic, VGG + Huber 1,024 px | 1.07 ms | 2.22 ms |
| Adam iteration, analytic, grey NCC 4,096 px | | 5.5 ms |
| Adam iteration, analytic, VGG + NCC 4,096 px | 1.26 ms | 11.0 ms |
| LM normal equations, jvp / analytic (4,096 px) | 23 / 3.0 ms | 140 / 22 ms |
| dense Cholesky, float64 / float32 | 2.6 / 0.6 ms | 38 / 6 ms |
| one cost evaluation | 0.8 ms | 3.2 ms |

Photometric Adam is launch-overhead bound (~266 ops at 4-5 µs); 8x the pixels add only 30 %. Feature-metric VGG + NCC
reads 32x the data and is compute bound on scenes with many frames. Feature extraction 0.06-0.36 s; LM with the stop
rule < 1 s. Image decoding (~1.8 s per scene) isn't counted anywhere.

## Rejected (REPORT3 §4)

- **Eliminating the depth grid** (Schur without cross-frame cell coupling): dev 27.1 vs 29.4, ScanNet++ 18.4 → 8.9, can
  be indefinite; PCG with it as preconditioner (5-10 iterations, tol 1e-3) loses the same way (ScanNet++ 17 → 4.3).
  Float32 made the dense solve as cheap anyway.
- **Chord method** (`lm_refactor`, reuse the factorisation): 24.4 / 25.4 (ScanNet++ 7.7 / 8.6).
- More pixels or 2x resolution for accuracy ([sampling](pixel-sampling.md), [resolution](resolution-and-pyramid.md)).

## Open

`camera=radial`, `res_scale` and `sample_chunk` (REPORT6) were only on the autograd/jvp path (3-6x slower), removed in the 2026-11-10 refactor; the code is at commit `1c1a921`.

Related: [feature-metric-vgg](feature-metric-vgg.md), [pba-method](pba-method.md).
