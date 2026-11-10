# How DA3 fails: structure of its pose and depth errors

Status: **analysed for DA3-SMALL** (all 90 scenes) · **open:** the same on larger models · Area: diagnosis · Updated 2026-11-10

Report: `results/failure_modes/DA3-SMALL/REPORT.md` (2026-11-07). Scripts: `analyse_failure_modes.py`,
`plot_failure_modes.py`, `reconstruction_errors.py` (all take the model name).

**Answer:** pose error is mostly smooth and shared between nearby cameras, plus gross failures in runs on ScanNet++ and
ETH3D. Depth error has little pixel noise: a smooth low-frequency part plus outliers at edges and failed surfaces. For
reconstruction the **cameras (incl. intrinsics) cost more F1 than the depth** on every dataset.

## Poses

| dataset | median rotation | median translation (% depth) | gross frames | their share of rot. error² | best oracle AUC@3 gain |
|---|---|---|---|---|---|
| HiRoom | 1.35° | 3.8 % | 1.2 % | 54 % | no jitter: 9.5 → 24.1 |
| 7-Scenes | 1.85° | 3.1 % | 0 % | 0 % | no smooth part: 14.5 → 52.6 |
| ScanNet++ | 2.47° | 8.2 % | 13.6 % | 94 % | no smooth part: 9.0 → 19.5 |
| DTU | 1.56° | 2.2 % | 0 % | 0 % | no smooth part: 23.4 → 59.9 |
| ETH3D | 1.68° | 7.4 % | 17.8 % | 96 % | no smooth part: 8.6 → 21.7 |

- Error similarity ~0.9 for cameras 0.01-0.02 scene depths apart, 0.45 at 0.1, 0-0.2 at 0.2. HiRoom's "jitter" is a
  sampling effect (its cameras are ≥ 0.2 depths apart).
- Removing outliers adds 1-3 AUC@3 but 5-7 AUC@30 on ScanNet++/ETH3D. AUC@3 is low because the median pair is already
  near 3°; translation direction of close pairs (< 0.1 depths) is off by 20° (7-Scenes) / 34° (ScanNet++).
- Gross failures and the reference view: [gross-failures-and-reference-view](gross-failures-and-reference-view.md).

## Depth

| dataset | median abs. error | pixels > 10 % off | fine-noise σ | within 3 %: scale → + offset → + smooth → + outliers removed |
|---|---|---|---|---|
| HiRoom | 2.7 % | 8.8 % | 0.55 % | 53 → 65 → 91 → 98 |
| 7-Scenes | 3.3 % | 15.9 % | 1.38 % | 47 → 52 → 76 → 88 |
| ScanNet++ | 3.4 % | 14.9 % | 0.90 % | 46 → 58 → 83 → 93 |
| DTU | 1.1 % | 2.0 % | 0.30 % | 86 → 90 → 97 → 100 |
| ETH3D | 3.7 % | 16.1 % | 0.91 % | 43 → 54 → 86 → 93 |

Outliers cluster at depth edges (46-61 % of pixels within 1 px of a GT edge) and on failed surfaces; confidence ranks
them with AUROC 0.66-0.73. No peak at the 14 px ViT patch. Pose and depth errors per frame are barely correlated
(Spearman ≤ 0.21); pose and depth scales agree on average but vary per scene (p10-p90 0.91-1.15 on ETH3D).

## Cameras or depth for reconstruction (own back-projection F1, not the benchmark's TSDF F1)

Fixing cameras gains more than fixing depth on all datasets (HiRoom +43.1 vs +14.4, 7-Scenes +9.8 vs +7.7, ScanNet++
+20.6 vs +11.2, DTU +34.4 vs +26.6, ETH3D +26.8 vs +13.8; 67 of 90 scenes). HiRoom: the focal (median 19 % too long;
fixing it alone +18 F1). DTU: fixing only the depth-pose scale mismatch adds 22 F1 (5 mm threshold = 0.8 % of depth;
scan9's pose scale is 8 % below its depth scale, the benchmark's worst DTU scene). Inside depth, removing the smooth
error gives 93-96 %, outliers only 1-3 points. Errors compound (HiRoom 828769: depth 79, poses 52, intrinsics 43,
all together 2).

## What it led to

Shared focal ([focal](focal-and-intrinsics.md), adopted), relocalisation (rejected), pose basis and Laplacian prior
(rejected), fitted depth prior and depth-edge masks (neutral): REPORT5. Untested: several reference views, aligning
depth and pose scales (+22 F1 DTU oracle).

Related: [evaluation](../evaluation.md) (frame-order and 7-Scenes pitfalls found here).
