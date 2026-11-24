# Test-time augmentation of DA3's depth and cameras before PBA

Status: **open (SMALL positive: 7 passes +5.1 dev / +2.9 held-out AUC@3 on 5 sets, HiRoom +11; BASE neutral on dev; not a matched-compute win; self-distilling the ensemble into one pass fails, on ARKit and in-domain)** · Area: TTO method / init · Updated 2026-11-24

**Answer (SMALL, B′ `iters=25`):** averaging DA3-SMALL's log depth and its cameras (relative to frame 0) over 4
forward passes (plain, horizontal flip, reversed frame order = another reference view, both) before B′ lifts AUC@3
on the 5 pose sets from 42.9 to **46.4 on dev** (3 seeds) and from 43.8 to **45.5 on held-out** (59 scenes, 2 seeds;
HiRoom 52.2 → 59.7, ScanNet++ 34.7 → 37.9, 7-Scenes ±0, DTU −1.1, ETH3D −1.8). It lowers DA3's depth error (dev abs.
rel. H / S / E 5.37 / 6.04 / 5.24 → 4.80 / 5.19 / 4.95 %) and the initial pose error except on DTU (the flipped /
reversed passes' cameras are worse there). The reference view matters: the reversed order alone changes a scene's
depth error by up to a third, in both directions. Cost: 3 extra SMALL forward passes (~+1.2 s at the 5090's 0.40 s per
pass), more than BASE's forward, and BASE + B′ is far better (dev 59.2, held-out 57.4): an accuracy result that says
DA3's outputs carry pass-to-pass variance, not a compute win.

## Method

`scripts/lpba_tta.py <src> <out root> [split] [datasets]`: per scene the 3 extra passes, depth scaled to the plain
pass's median and averaged in log; cameras: per pass world-to-camera poses relative to frame 0 (the flipped pass
un-mirrored, S E S with S = diag(-1, 1, 1); translations scaled with the depth), rotations averaged by SVD projection,
translations averaged; written as `tta_{flip,rev,all}` (depth only) and `*_cam` (depth + cameras).

## Evidence

| date | setting (model, set, seeds, baseline) | change | result | source |
|---|---|---|---|---|
| 11-24 | SMALL, dev H (10) / E (3), B′ T = 25, 3 seeds (baseline 44.7 ±1.0 / 31.2 ±3.0) | depth TTA: flip / rev / all 4 | H 46.6 / 46.6 / 47.6, E 33.9 / 19.1 / 30.9; depth fixed (`depth_grid=0`, s0): H 26.6 → 27.1 / 25.4 / 28.5, E 14.8 → 12.6 / 11.7 / 12.8 | `lpba/tta/r_tta_*` |
| 11-24 | same | depth + cameras TTA (flip / all 4) | H 45.9 / **48.7 ±1.6**, E 31.8 / **33.0 ±0.8**; depth error after PBA H 4.88 → 4.31 % | `lpba/tta/r_tta_*_cam_*` |
| 11-24 | SMALL, held-out H (20) / E (8), 2 seeds (baseline 53.2 ±1.0 / 32.5 ±0.5) | depth TTA all 4 / flip | 55.2 ±0.5 / 35.6 ±0.8; flip 54.1 / (E seeds 32.9, 55.8) | `lpba/tta/r_ho_*` |
| 11-24 | SMALL, **full dev (22 scenes, 5 sets), 3 seeds**, run_tto_pba.py `subset=dev iters=25` directly (baseline `lpba_pho_it25_s0-2`: 42.9, H / 7-Sc / S / DTU / E 44.6 / 36.8 / 24.6 / 86.9 / 21.4) | depth + cameras TTA, all 4 passes | **46.4** (46.2 / 46.2 / 46.7): 46.4 / 35.8 / 33.0 / 83.9 / 32.7 (ETH3D dev is chaotic, see evaluation.md) | `lpba/tta/full_dev_s*` |
| 11-24 | SMALL, **full held-out (59 scenes)**, 2 seeds (baseline `pho_it25_ho_s0` + `tta/base_ho_s1`: 43.8; 52.2 / 22.5 / 34.7 / 78.4 / 31.3) | same | **45.5** (45.7 / 45.2): HiRoom **59.7**, 7-Sc 22.8, ScanNet++ **37.9**, DTU 77.3, ETH3D 29.5; init AUC@3 H 8.9 → 11.4, S 9.5 → 12.2, DTU 31.0 → 25.4 (the flipped / reversed passes' cameras are worse on DTU) | `lpba/tta/full_ho_s*` |
| 11-24 | SMALL, dev (3 seeds) / held-out (2 seeds), HiRoom + DTU + ETH3D (the 12-cell grid runs out of memory on the 100-frame sets on the 3080) | 2x2: grid 8 / 12 cells x TTA (depth + cameras, 4 passes) | 3-set mean dev 51.0 / 53.8 (g12) / 54.3 (TTA) / **57.7** (both); held-out 53.9 / 54.9 / 55.5 / **56.2** (HiRoom 52.2 / 53.2 / 59.7 / 60.2, DTU 78.4 / 80.3 / 77.3 / 79.2, ETH3D 31.3 / 31.3 / 29.5 / 29.3): the two gains add | `lpba/tta/g12_*` |
| 11-24 | **DA3-BASE**, full dev, 3 seeds, B′ T = 25 (baseline 59.8: H 67.4 / 7-Sc 36.9 / S 46.3 / DTU 91.5 / E 56.9) | TTA depth + cameras, 4 passes | 60.4: HiRoom **60.0 (−7.4)**, 7-Sc 37.6, ScanNet++ 50.9 (+4.6), DTU 89.1 (−2.4), ETH3D 64.2 (+7.3): roughly neutral overall, and the per-dataset signs differ from SMALL's (HiRoom gained most there) | `lpba/tta/bB_*` |
| 11-24 | SMALL, full dev, 3 seeds (baseline 42.9 with the benchmark's reference strategy `first`) | other reference strategies of DA3 (`ref_view_strategy`): a single pass with `saddle_balanced` (DA3's API default); an ensemble of first + middle + saddle_balanced + saddle_sim_range (depth + cameras averaged, no flips) | single saddle_balanced **35.5** (H 38.4, S 13.2, E 8.8: much worse; its own cameras, the plain pass's confidence); **ensemble 47.3** (48.3 / 34.8 / 29.1 / 84.6 / 39.9), flip / reverse TTA 46.4: the averaging helps, not a better single reference. Init AUC@3 per scene: single strategies scatter around `first`, the mean is above it in most scenes | `lpba/tta/ref_*` |
| 11-24 | SMALL, full held-out (59), 2 seeds (baseline 43.8) | 4-reference ensemble (depth + cameras) | **44.9** (44.9 / 44.9): H 60.3, 7-Sc 20.6, S 39.3, DTU 77.4, E 27.0 (flip / reverse TTA 45.5). Both ensembles: HiRoom +8, ScanNet++ +3-5, ETH3D / 7-Scenes / DTU −1 to −4; depth-only TTA helped ETH3D held-out (32.5 → 35.6), so the losses come from the averaged cameras | `lpba/tta/refho_s*` |
| 11-24 | SMALL, full dev (3 seeds) / held-out (2 seeds) | **7 passes** (flip, reversed, both, references middle / saddle_balanced / saddle_sim_range): mean fusion vs per-frame medoid cameras + per-pixel median depth | dev **48.0** (mean; H 48.7, S 32.6, E 39.1) / 46.7 (medoid); held-out **46.7** both: mean H 63.3, 7-Sc 23.0, S 42.7, DTU 77.2, E 27.1; medoid 63.6 / 22.1 / 40.5 / 75.5 / 31.6 (keeps ETH3D, loses DTU). Init AUC@3 dev 14.5 → 18.4 (mean) / 15.8 (medoid). Cost 7 SMALL passes (~2.8 s on the 5090) > LARGE's forward (1.6 s; LARGE + B′ held-out 69.7) | `lpba/tta/p7*` |
| 11-24 | SMALL student: main depth branch + camera decoder (4.4M) fine-tuned to reproduce the 7-pass ensemble (log depth + relative poses) in one plain pass, on 600 ARKit windows without labels, 3k steps, lr 5e-5 (`scripts/lpba_selfdistill.py`) | B′ from the student's own depth + cameras | ARKit validation: depth −13 %, rotation −20 % towards the ensemble; benchmark: depth abs. rel. dev 4.61 → 4.63 %, init AUC@3 dev 16.2 → 12.2, held-out 13.6 → 9.5; **B′ dev 42.9 → 35.9, held-out 43.8 → 38.5** (DTU 78.4 → 61.3): the ARKit-tuned camera decoder doesn't transfer | `lpba/sd/r_*` |
| 11-24 | same student, **in-domain**: ensemble targets on 131 frame subsets (≤ 24 frames) of the 59 held-out benchmark scenes (no labels), 2k steps → dev | B′ from the student | in-domain validation: depth −18 %, rotation −28 %, translation −33 % towards the ensemble; **dev B′ 41.2** (H 38.2, 7-Sc 37.2, S 25.6, DTU 80.5, E 24.6) vs 42.9 plain and 48.0 for the ensemble itself: the ensemble's gain doesn't survive distillation at this scale | `lpba/sd/r2_*` |

## Open / don't repeat

- Matched compute: not competitive (7 SMALL passes cost more than LARGE's forward). It shows that DA3-SMALL's
  cameras and depth carry pass-to-pass variance (reference view, mirroring) that averaging removes; HiRoom and
  ScanNet++ gain most, ETH3D / DTU lose a little from the averaged cameras.
- Distilling the ensemble into one pass (`lpba_selfdistill.py`) didn't work at tum1 scale (ARKit: 35.9 / 38.5 dev /
  held-out, in-domain 41.2 dev); a variant that keeps DA3's camera decoder frozen and distils only the depth, or
  much more data, is untested.
- Untested: more passes than 7, random frame orders / subsets, confidence-weighted fusion, TTA for LARGE.

Related: [depth-bottleneck](depth-bottleneck.md), [learned-depth-basis](learned-depth-basis.md),
[matched-compute](matched-compute.md)
