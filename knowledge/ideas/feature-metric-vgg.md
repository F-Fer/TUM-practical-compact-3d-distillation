# Feature-metric BA on VGG16 features

Status: **adopted, two variants:** VGG + Huber at 1,024 px (`fmh`, cheap, best F1 on LARGE/GIANT) and VGG + NCC at
4,096 px (`fm`, best pose on LARGE at long T, 3.3x the time) · Area: PBA loss / speed · Updated 2026-11-10

Features: VGG16 relu1_2 + relu2_2, upsampled, PCA-reduced per scene (8 channels now, 16 before session 3),
the fast extractor (always on since 2026-11-10) keeps the network between scenes, runs 8 frames per pass and does the PCA by eigendecomposition
(0.06-0.36 s per scene). No brightness parameters.

## Evidence

| date | setting | change | result | source |
|---|---|---|---|---|
| 10-30 | SMALL dev | VGG features (Huber) / VGG + NCC / + spread k = 10 | 24.7 / 24.3 vs 24.0; with spread neighbours DTU 41 → 48.5 (59 with more iterations): wide baselines only help with a viewpoint-robust cost | REPORT1 |
| 10-30 | full benchmark | v3 (VGG + NCC, 4x200, 18 s) | SMALL HiRoom 9.4 → 37.1, DTU CD 5.64 → 4.21; LARGE w_prior = 100 HiRoom F1 69.5 → 82.8 | REPORT1 |
| 10-30 | dev compute plot | cheap RGB Huber vs VGG + NCC at equal time | RGB on the Pareto front except DTU | REPORT1 |
| 11-05 | SMALL dev, 4x50 + LM 5 | speed ladder | 4.98 s (16 ch) → 2.21 analytic → 1.73 fast extractor → **1.17 s with 8 channels**, AUC@3 31.4 → 31.5 | REPORT3 §5 |
| 11-05 | SMALL dev | 8 ch / 4 ch at 1,024 px, 4x25 | 29.2 (0.37 s) / 28.6 (0.33 s) | REPORT3 §5 |
| 11-06 | SMALL held-out | VGG + NCC 8 ch, stop rule + gated radial | **38.5 / 85.1 at 1.54 s**, the most accurate of session 3 (DTU 66.5 vs 60.8 photometric) | REPORT3 §6 |
| 11-07 | SMALL dev T = 800, 3 seeds | VGG 8 ch + Huber (no brightness) | 38.0 (4.45 s) vs grey Huber + relative brightness 39.0, grey NCC 37.6 (7.68 s) | REPORT4 §1 |
| 11-10 | dev, all 4 models, shared focal, T = 25-800 | VGG + Huber 1,024 px (`fmh`) vs VGG + NCC 4,096 px (`fm`) | TTO 4.7 vs 15.7 s at T = 800 (3.3x), 0.5-0.7 vs 0.9-1.1 s at T = 25. Pose 1.7-3.5 lower at T = 800 (ETH3D/DTU), level with photometric. F1 equal or higher on BASE/LARGE/GIANT: LARGE 72.0 at T = 400, GIANT 77.0 at T = 50 (best F1 of the plot) | REPORT7 |

## Cost (per Adam iteration, 100 frames, RTX 5090; REPORT7 §1)

grey Huber 1,024 px 1.24 ms · VGG + Huber 1,024 px 2.22 ms · grey NCC 4,096 px 5.5 ms · VGG + NCC 4,096 px 11.0 ms. At
12 frames all are near the launch-overhead floor (~1.1-1.3 ms). NCC at half VGG + Huber's data costs 2.5x as much. Peak
GPU memory 10.4 GiB (fmh) / 10.8 GiB (fm) vs 5.3 GiB photometric: neither VGG variant fits the 8 GB local GPUs at 100
frames.

## Open

- Separate loss from sample count on ETH3D: VGG + Huber with `samples=4096`, and VGG + Huber Adam followed by NCC LM.
- No held-out test of either VGG variant with the shared focal.

Related: [ncc](ncc.md), [speed](speed.md), [dino-features](dino-features.md).
