# Depth as the bottleneck of PBA's pose: which depth error matters, and where better depth can come from

Status: **open** (diagnosis measured; non-oracle sources measured; learned refiners and depth distillation at tum1
scale negative) ·
Area: TTO method / depth · Updated 2026-11-26

**Answer:** on DA3-SMALL + B′ (`iters=25`), the pose that PBA reaches is limited by the depth it is given
([learned-refinement-step](learned-refinement-step.md), depth oracle: GT depth fixed 38.2 → 49.1 on HiRoom / ScanNet++ /
ETH3D). What matters is the depth's shape at the scale of ~16 grid cells: a GT-fitted 16-cell smooth correction of
DA3's depth gets 46.6 of GT's 49.4 (depth fixed, 3 sets), 8 cells 42.9, 4 cells 37.6, per-frame scale alone 24.8 (= no
correction, 23.3). With the depth fixed the AUC is roughly linear in the remaining log-depth error (λ = 0 / 0.5 / 1:
23.3 / 37.6 / 49.4), so a depth has to be more than halfway to the GT to beat B′'s free grid (37.9). The multi-view
terms don't find that correction themselves (alternation: depth error unchanged). **DA3-LARGE-1.1's depth with
DA3-SMALL's cameras + B′: 52.5 on all 81 scenes vs 43.6** (HiRoom 64.1, ScanNet++ 40.9, ETH3D 42.8, DTU 87.0, 7-Scenes
27.5). Small learned refiners trained on 200 ARKitScenes videos (LiDAR depth) don't improve DA3's or B′'s depth on
held-out videos, with or without multi-view residual maps as input, and more data (50 → 200 videos) helps only a little.
**2×2 on all 81 scenes** (B′, seed 0; cameras × depth): SMALL/SMALL 43.6, SMALL/BASE 50.7, SMALL/LARGE 52.5,
LARGE/SMALL 58.3, LARGE/BASE 64.0, LARGE/LARGE 69.8. Of the SMALL → LARGE gap after TTO (+26.2), the depth is +8.9, the
cameras +14.7, the interaction +2.6; BASE's depth gives 80 % of LARGE's depth gain with SMALL's cameras (+7.1 of +8.9)
but only half with LARGE's (+5.7 of +11.5): the better the cameras, the more the finer depth of LARGE pays off. **Distilling LARGE's depth into SMALL** (head + last
6 backbone blocks, 4k steps on the 3080) doesn't transfer to unseen scenes yet: trained on ARKit + outdoor clips, dev
depth error and AUC unchanged within noise; trained on the 59 held-out benchmark scenes (teacher = LARGE's predictions,
no GT), dev 43.3 vs 42.9 ± 1.3 with SMALL's cameras. Fine-tuning the backbone breaks SMALL's own cameras (dev 26.9) even
with a camera-token loss, so the distilled depth must be combined with the original cameras.

## Evidence

| date | setting (model, set, seeds, baseline) | change | result | source |
|---|---|---|---|---|
| 11-22 | SMALL, HiRoom / 7-Scenes / ScanNet++ / ETH3D (68 scenes), seed 0, B′ `iters=25`, `depth_grid=0` unless noted, 3080 | depth oracles (`scripts/lpba_depth_oracle.py`): DA3 scaled per scene (`lerp:0`), per-frame scale from GT (`scale`), GT-fitted smooth correction with 4 / 8 / 16 cells (`smooth:G`), halfway in log depth (`lerp:0.5`), GT | AUC@3 HiRoom / ScanNet++ / ETH3D (7-Scenes): lerp:0 28.7 / 21.2 / 20.0 (19.9); scale 29.7 / 24.4 / 20.3 (21.6); smooth:4 43.7 / 31.3 / 37.9 (17.3); smooth:8 51.6 / 33.1 / 44.0 (16.4); smooth:16 55.8 / 35.3 / 48.7 (18.2); lerp:0.5 47.5 / 32.2 / 33.2 (21.4); GT 62.6 / 39.6 / 45.9 (20.0); reference B′ with DA3 depth and free grid 50.3 / 33.8 / 29.6 (26.9) | `lpba_pa_*`, `lpba_dor_*` |
| 11-22 | SMALL, all 81 pose scenes, seed 0 | DA3-LARGE-1.1's depth (one scale per scene to SMALL's median), SMALL's cameras; fixed / B′ with free grid | fixed 46.0 (HiRoom 59.3, 7-Scenes 28.3, ScanNet++ 37.7, DTU 69.8, ETH3D 34.9); **free grid 52.5** (64.1, 27.5, 40.9, 87.0, 42.8) vs SMALL's depth 43.6 (49.8, 26.5, 33.2, 81.2, 27.3) | `lpba_pa_large_fix_s0`, `lpba_pa_large_s0` |
| 11-22 | ARKitScenes, 200 training videos × 2 windows of 12-40 frames (LiDAR depth, conf ≥ 2), 16 → 96 val windows, 3080 | DA3-SMALL depth error vs LiDAR | abs. rel. 9.7-11.3 % (one scale per window); mean log error 0.112 → 0.069 with a per-frame scale, 0.035 / 0.021 / 0.011 with a fitted 4 / 8 / 16-cell smooth correction (16 val windows); B′ makes it slightly worse (11.3 → 11.8 %) | `lpba_arkit_da3.py` (removed 2026-11-23, never committed; the description here is all that is left), `lpba_prep_arkit_depth.py` (removed 2026-11-23, never committed; the description here is all that is left) |
| 11-22 | same | **single-view refiner** (`lpba_refiner.py` (removed 2026-11-23, never committed; the description here is all that is left): ResNet18 on RGB + DA3 log depth + conf → 8-cell log-depth grid), L1 / per-frame scale-invariant loss, 4-6k steps | val L1 0.111 → 0.12-0.14 (worse than DA3 at every checkpoint), shape-only 0.069 → 0.070-0.074; train loss falls to 0.05-0.07 | `refiner_g8*.log` |
| 11-22 | same, at B′'s state | **multi-view refiner** (`lpba_mvrefiner.py` (removed 2026-11-23, never committed; the description here is all that is left): ResNet18-FPN on RGB + B′ log depth + conf + residual maps against the 10 spread neighbours → per-pixel log-depth correction), control without maps | val 0.116 (B′) → 0.126 (maps) / 0.140 (no maps, larger val); in-sample on 40 train windows 0.144 → 0.114 (both), shape 0.094 → 0.084 (a GT-fitted 16-cell correction reaches ~0.02): the maps add nothing, the network barely fits | `mvref_train*.log` |
| 11-22 | same, val = 96 windows of 48 videos | learning curve of the residual-map refiner: 50 / 100 / 200 training videos, 5k steps | val L1 0.144 / 0.143 / 0.137 vs B′ 0.130 and DA3 0.1225 (shape 0.096 / 0.095 / 0.089 vs 0.085 / 0.071): improves slowly with data, worse than no correction at every size | `curve*.log` |
| 11-22 | all 81 pose scenes, B′ `iters=25`, seed 0 | **2×2 cameras × depth**: LARGE's cameras via `init_from` with SMALL's depth; B′ on LARGE itself; BASE's depth with SMALL's cameras | AUC@3 (HiRoom / 7-Scenes / ScanNet++ / DTU / ETH3D; mean): LARGE cam + SMALL depth 72.5 / 26.2 / 60.8 / 88.6 / 43.5; 58.3 · LARGE / LARGE 87.4 / 27.8 / 74.3 / 95.1 / 64.3; 69.8 · SMALL cam + BASE depth 58.4 / 28.0 / 38.7 / 85.5 / 42.7; 50.7 | `lpba_tb_*`, `lpba_pa_base_s0` |
| 11-26 | all 81 pose scenes, B′ `iters=25`, seed 0, 3080 | missing cell of the cameras × depth table: LARGE-1.1's cameras (`init_from`) + BASE's depth (`lpba_depth_oracle.py model:workspace/evaluation/DA3-BASE`, scaled to SMALL's median), SMALL's run as the source | AUC@3 78.7 / 29.9 / 66.6 / 92.2 / 52.6; **64.0** (LARGE/SMALL 58.3, LARGE/LARGE 69.8); AUC@30 93.4; 0.81 s TTO. Figure `results/pba/figures_depth/depth_camera_swap.png` (`scripts/plot_depth_bottleneck.py`) | `lpba_tb_largecam_basedepth` |
| 11-23 | ARKit val 96 windows (LiDAR) | DA3-LARGE-1.1 as teacher: its depth error | abs. rel. SMALL 12.6 %, LARGE 8.0 % (one scale per window) | `teacher_val.log` |
| 11-23 | DA3-SMALL, teacher LARGE on 400 ARKit + 132 outdoor windows (`lpba_distill_depth.py` (removed 2026-11-23, never committed; the description here is all that is left)), 3080 shared | **depth distillation**, head only (3.9M) 3k steps / head + last 6 blocks (14.5M, lr / 10, camera-token loss) 4k steps, 6-8 frames | ARKit val abs. rel. 12.31 → 12.01 % (head) / 11.76 % (6 blocks); benchmark (61 scenes without the 100-frame ones, OOM) with SMALL's cameras: HiRoom 49.8 → 50.6, DTU 81.2 → 74.5, ETH3D 27.3 → 31.2, depth error HiRoom 6.05 → 5.86 %, ETH3D 5.67 → 5.89 %; with its own cameras HiRoom 23.9 | `lpba_du6_*`, `lpba_distill/dist_*.log` |
| 11-23 | DA3-SMALL, teacher = LARGE's saved predictions on the 59 held-out scenes (no GT), dev evaluation | in-domain depth distillation (6 blocks, 4k steps, 6 frames) | teacher loss 0.11 → 0.038; dev depth error HiRoom / ScanNet++ / ETH3D 5.37 → 5.23 / 6.05 → 6.20 / 5.24 → 5.26 %; **B′ dev 43.3** (SMALL 42.9 ± 1.3 over 3 seeds; LARGE depth 52.1); own cameras 26.9 | `lpba_db_*` |
| 11-23 | SMALL, HiRoom / 7-Scenes / ScanNet++ / ETH3D (68 scenes), seed 0, B′ `iters=25 depth_grid=0`; refs lerp:0 23.3, smooth:4 / 8 / 16 37.6 / 42.9 / 46.6, GT 49.4 (3-set means w/o 7-Scenes) | **feature-span oracle** (`scripts/lpba_basis.py` + `lpba_depth_oracle.py ofile:`): GT-fitted per-frame ridge LS correction of log depth in the span of DA3's own head features (h32 = the 32 channels log depth is linear in; f64 = fused map; pNN = per-scene PCA; s4+/s8+ = plus grid hat functions), fit clipped to its range on GT pixels (unclipped, the features extrapolate to 1e34 / inf off GT and crash PBA) | AUC@3 H / 7-Sc / S / E (3-set mean), coefficients/frame: h32 38.6 / 20.9 / 26.3 / 33.1 (32.7), 33; f64 40.1 / 19.2 / 24.8 / 34.7 (33.2), 65; f64p16 35.5 / 21.2 / 25.9 / 33.3 (31.6), 17; s4+f64p16 47.2 / 20.6 / 28.8 / 38.5 (38.2), 28; s4+h32 49.5 / 19.2 / 29.6 / 41.6 (40.3), 44; s8+f64 52.0 / 18.2 / 30.8 / 46.1 (43.0), 112. In-sample RMS log error left (HiRoom) h32 .047, f64 .047, smooth4 .053, smooth8 .043, s8+f64 .035, smooth16 .036. **Negative:** the features fit the GT better than smooth4 but buy less pose (f64 depth abs. rel. HiRoom 2.80 % vs smooth4 2.92 %, AUC 40.1 vs 43.7); s4+h32 < smooth8, s8+f64 = smooth8: the frozen head features add nothing pose-relevant to a grid; the correction PBA needs is low-frequency and not in DA3's feature span | `lpba_bas_*`, `lpba_bas_s8_f64_s0/fit_summary.json` |

## Open / don't repeat

- Don't build an "adjustable depth basis" from DA3's frozen head features (h32 / f64, BLASt3R-style Y = exp(αF + β)):
  the GT-fitted oracle in their span is worse than a 4-cell grid at equal or more coefficients, and adds nothing on top
  of an 8-cell grid (feature-span oracle, 11-23). A basis would have to be trained for it (and bounded off-GT).
- The matched-compute question: LARGE's depth costs LARGE's forward pass, which also gives better cameras; the 2×2
  (cameras × depth from SMALL / LARGE) decides whether the gap SMALL + TTO → LARGE + TTO is depth or cameras. A cheap
  depth source (monocular model, lower resolution, a subset of frames) is what would make "small cameras + good depth"
  a compute win; see [da3-resolution] on the resolution branch (LARGE 336 px + RoPE + TTO 66.6).
- The GT oracles only sample pixels with GT depth (ETH3D: 15 % of the pixels), which may explain why LARGE's depth + free
  grid beats GT depth + free grid (60.8 / 37.5 / 39.4) on all three sets; not separated yet.
- Learned refiners: at 200 videos neither a single-view nor a residual-map refiner learns a transferable correction;
  the learning curve (50 / 100 / 200 videos) improves slowly but stays below "no correction". A small network that
  learns depth from scratch competes with DA3's own training; the prior has to come from a pretrained model.
- Depth distillation ("distil only what TTO can't fix"): the idea stands (upper bound SMALL cameras + LARGE depth
  52.5 vs 43.6 at SMALL's forward cost) but 4k steps on 59-400 windows don't generalise; the teacher loss falls, the
  error on unseen scenes doesn't. This is the experiment to scale on the 5090: many more clips (ARKitScenes has 4.5k
  training videos, plus ScanNet / CO3D / outdoor), longer training, the full backbone with the cameras taken from the
  frozen model (two passes) or a stronger camera constraint; measure depth error on held-out scenes before running PBA.

- 2026-11-24: which part of the depth error matters (low-frequency; first-order sensitivity model) and the attempts
  to fine-tune DA3 for PBA are in [learned-depth-basis](learned-depth-basis.md); test-time augmentation of DA3's depth
  and cameras in [da3-tta](da3-tta.md).

Related: [learned-refinement-step](learned-refinement-step.md), [depth-parametrisation](depth-parametrisation.md),
[depth-prior](depth-prior.md), [matched-compute](matched-compute.md)
