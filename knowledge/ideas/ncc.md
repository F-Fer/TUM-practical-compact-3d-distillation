# NCC (normalized cross-correlation of 3x3 patches) as the photometric residual

Status: **adopted** as an LM stage after photometric Adam (`lm_photo_loss=ncc`, best pose on SMALL/BASE) and inside
the VGG + NCC feature-metric configuration · **open:** a stop rule for the NCC LM; why it helps ETH3D (loss or the
4,096 samples) · Area: PBA loss · Updated 2026-11-10

NCC subtracts each patch's mean, divides by its standard deviation and correlates (1 = identical up to gain and
offset). It needs no brightness parameters and is invariant to local exposure changes, but works on patches, so it
needs many samples (4,096 per frame = 455 patches) and costs more per sample than Huber.

**Answer:** with per-pixel grey Huber the objective's optimum is biased; NCC removes most of the bias
([GT-start](objective-bias-gt-start.md)). As a long LM stage after Adam on Huber it gives the best pose found:
held-out SMALL 46.7 → 57.2 with the shared focal. It is consistently worse on 7-Scenes, and on LARGE/GIANT it lowers F1.

## Evidence

| date | setting | change | result | source |
|---|---|---|---|---|
| 10-30 | SMALL dev | + NCC (RGB) / VGG + NCC | 23.9 / 24.3 vs 24.0; VGG + NCC + spread neighbours: DTU 41 → 48.5 | REPORT1 |
| 11-07 | SMALL dev T = 800, no brightness | NCC 4,096 px in Adam | 29.3 / 34.9 / 37.6 at T = 50 / 200 / 800 (7.68 s at 800); grey Huber with relative brightness 39.0 at half the cost | REPORT4 §1 |
| 11-07 | SMALL, GT start | NCC vs Huber | NCC with frozen depth (88.3/64.6 HiRoom) beats Huber with 16 cells | REPORT4 §4 |
| 11-07 | SMALL dev | Adam 4xT (Huber, relative brightness) + n LM on NCC | T = 50 + 40/80/160: 33.9/36.3/38.4; T = 200 + 40: 39.2 (1.72 s); T = 800 + 40: **44.0** (3 seeds, 4.56 s), + 160: 46.6 (6.87 s); Huber LM with stop rule at T = 800: 39.0 | REPORT4 §7 |
| 11-07 | SMALL held-out | same | T = 200 + 40: 44.2 (1.93 s); **T = 800 + 40: 46.7** (4.78 s) = +9.6 over Recommended (ScanNet++ +15.2, ETH3D +17.3, HiRoom +9.1, DTU +6.9); 7-Scenes the one loser | REPORT4 §7 |
| 11-08 | held-out, + shared focal | NCC config | SMALL 57.2 / F1 57.0, BASE 68.3, LARGE 74.8 | REPORT5 §2, §7 |
| 11-10 | dev, all models (figures5, `ncc_it<T>`) | NCC config with shared focal, T = 50/200/800 | SMALL 43.8/50.7/55.6, BASE 59.2/61.4/64.8, LARGE 71.3/72.9/71.9, GIANT 74.7/74.0/74.8; F1 (4 sets) LARGE 65.5-66.9 and GIANT 68-70, below photometric (70-76) | runs.csv |
| 11-10 | dev, VGG features, T = 800 | VGG + NCC (4,096 px) vs VGG + Huber (1,024 px) | +1.7 to +3.5 AUC@3, all from ETH3D (+8.5 to +12.5) and DTU-64 (+2.6 to +5.6); level elsewhere | REPORT7 §4 |

## Notes

- An LM iteration on NCC costs ~18 ms per scene (SMALL dev); NCC Adam at 100 frames 5.5 ms per iteration vs 1.24 ms grey
  Huber ([speed](speed.md)).
- ETH3D keeps gaining with more NCC LM iterations (47.5 → 63.5 from 20 to 160) while 7-Scenes doesn't: a stop rule per
  scene is the obvious next step (REPORT4 §11).
- Guess, not tested: ETH3D's exposure varies between images and NCC is invariant to local gain/offset; neither Huber
  configuration has a brightness model on VGG features.

Related: [feature-metric-vgg](feature-metric-vgg.md), [loss](loss-kernel-and-colour.md), [matched-compute](matched-compute.md).
