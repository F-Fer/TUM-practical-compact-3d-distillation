# A learned refinement step on DA3's explicit state (L-PBA)

Status: **partly (G0, checks 1-2 done)**: V1 (learned per-frame prior strength towards DA3) **rejected by its oracle**;
the weak-direction error is **not predictable** from per-frame state statistics (check 2); more pairs or VGG features
improve the conditioning ≤ 2x (check 1); the main source of the flat directions is the **pose ↔ depth-grid trade-off**
(fixing the grid stiffens them ~29x); **GT depth closes most of the gap** (depth oracle); alternating multi-view depth / pose refinement doesn't improve the
depth (+1.6 AUC@3 from the extra pose iterations only) · Area: TTO method · Updated 2026-11-22

**Answer:** G0 on dev DA3-SMALL, B′ T = 25 (3080, tum1). (a) Even with the GT choosing it, a per-frame pose prior
towards DA3 (w_i = c / e_i², e_i = GT error of frame i) gives no gain beyond seed noise: best c = 0.003 43.8 vs 42.9 ± 1.3
(3 seeds); larger c lose up to 25 points; choosing c per scene by GT (46.8) is no better than choosing the best of 3
seeds per scene (45.7). (b) Removing the GT error left in the weakest 10 % of pose directions after B′ lifts AUC@3
41.9 → 61.5 (25 %: 74.5), the strongest half 41.9 → 41.1. So the headroom is real, but DA3's own poses are not the
information that unlocks it. Check 1: all covisible pairs instead of 10 spread neighbours stiffen the weak directions
×2.0 (strong ×1.4), VGG + all pairs ×2.0 (strong ×1.0): at most 2x better conditioned, the weak10 headroom stays 61 →
61. Holding the depth grid fixed stiffens them ×28.7 (strong ×1.2), most on HiRoom ×52, DTU ×36, ETH3D ×16 (video sets
7-Scenes/ScanNet++ ×3-6, where all pairs help more, ×10-12 vs strong ×3-8). Check 2: on 81 scenes, ridge or boosted
trees on 40 per-frame features explain 0.0 of the weak-direction error across scenes (in-sample 0.69), and applying them
lowers AUC@3. Depth oracle (68 scenes with GT depth, 2 seeds): replacing DA3's depth (6.3 % abs. rel.) by the GT
depth lifts B′ on HiRoom / ScanNet++ / ETH3D 38.2 → 45.9 (grid free) → **49.1 (depth fixed)**, about the weak10 oracle's
49.5 on the same scenes; 7-Scenes falls (its GT depth is misaligned). The pose error PBA can't remove is mostly caused
by the depth. Self-consistent depth refinement doesn't get it: alternating a 32-cell depth stage (poses fixed) and a pose
stage (depth fixed) twice leaves the depth error 5.4-6.0 % → 5.3-6.4 % (ScanNet++ worse) and lifts AUC@3 43.6 → 45.2 on
81 scenes, most of it (+0.9) from one pose polish with the depth fixed. Proposal and plan: `reports/Learned refinement step between DA3 and DeepBA.md` (§8 G0, §9 checks 1-2).

## Evidence

| date | setting (model, set, seeds, baseline) | change | result | source |
|---|---|---|---|---|
| 11-22 | SMALL dev, B′ `iters=25`, seeds 0/1/2, 3080 | baseline | 41.9 / 44.5 / 42.3 (mean 42.9; ETH3D 17.5-28.1 decides the spread); AUC@30 86.3-86.5 | runs `lpba_pho_it25_s*` |
| 11-22 | same, seed 0 | global `w_pose` 1 / 3 / 10 / 30 / 100 | 37.4 / 35.3 / 30.7 / 26.4 / 22.1 (only 7-Scenes gains, +3 to +4.7); per-scene GT choice incl. 0: 44.9 | `lpba_wpose*` |
| 11-22 | same, seed 0 | **G0a** oracle per-frame prior `pose_prior_oracle=c`, c = 0.001 ... 3 | 42.9, **43.8**, 39.4, 34.7, 28.5, 23.2, 19.2, 17.6 (c = 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3); per-scene GT choice of c (incl. 0) 46.8 vs best-of-3-seeds 45.7 | `lpba_orac*` |
| 11-22 | same, seed 0, at the final state of B′ | **G0b** remove the GT error in eigen-directions of the pose Schur complement (16,384 samples/frame) | weakest 10 % 41.9 → 61.5 (w/o gross frames 63.3), 25 % 74.5, 50 % 84.9, strongest 50 % 41.1; share of the left error in the weakest 10 %: 54-73 % per dataset; median condition number 6e4-6e5 | `scripts/lpba_weak_oracle.py`, `lpba_pho_it25_s0/weak_oracle_s0.json` |
| 11-22 | same, 22 dev scenes, B′ final state, 8,192 samples/frame | **check 1**: Schur complement for grey/VGG × spread10/all pairs; information along the weak (ref. grey:spread10) and strong directions, total (× visible warps) | median cond 3.2e5 / 2.0e5 (grey all) / 1.7e5 (vgg 10) / 1.1e5 (vgg all); weak ×1.96 / ×1.22 / ×2.00, strong ×1.37 / ×0.66 / ×0.96; weak10 oracle 61.7 / 61.1 / 61.3 / 60.8 | `lpba_pho_it25_s0/info_dev.json`, `scripts/lpba_info.py` |
| 11-22 | same | depth grid (and focal) held fixed instead of eliminated | cond 7.5e4; weak ×28.7 (×30.4 with focal fixed too), strong ×1.18, along the error ×4.4; per dataset weak ×52 HiRoom, ×3.2 7-Scenes, ×5.7 ScanNet++, ×36 DTU, ×16 ETH3D; weak10 oracle 60.0 (error stays in the weakest directions of the stiffer matrix) | `info_dev_fixgrid.json` |
| 11-22 | SMALL dev, B′ `iters=25` with all pairs (`neighbors=1000 nbr_mode=top`), seed 0 | direct run | OOM on the 3080 for the 64/100-frame scenes; HiRoom 47.1 (base 43.8-45.5), ETH3D 12.7 (base 17.5-28.1) | `lpba_allpairs_s0` |
| 11-22 | SMALL held-out, B′ `iters=25`, seed 0, 3080 | baseline for the probe | 43.9 (HiRoom 52.7, 7-Scenes 22.4, ScanNet++ 34.7, DTU 78.7, ETH3D 31.0); held-out B′ T = 50 on the 5090 was 44.0 | `lpba_pho_it25_ho_s0` |
| 11-22 | SMALL dev + held-out (81 scenes, 4,123 frames) | **check 2**: probe of the weak-direction error per frame (update from DA3, relative poses to index neighbours, reduced gradient, block eigenvalues, confidence, position, dataset); zero / shrink / ridge / GBM, projected onto the weak basis | explained in-sample 0.00 / 0.00 / 0.15 / 0.69; 5-fold CV over scenes 0.00 / 0.00 / −0.01 / 0.00, AUC@3 43.6 / 43.3 / 34.5 / 40.3 (oracle 58.2); leave-one-dataset-out all < 0 | `lpba_pho_it25_s0/probe.log`, `scripts/lpba_probe.py` |
| 11-22 | SMALL, HiRoom / 7-Scenes / ScanNet++ / ETH3D, all scenes (30/7/20/11 = 68), seeds 0/1, 3080 | **depth oracle** (`scripts/lpba_depth_oracle.py`): depth0 = GT depth (one scale per scene to DA3's median, unsampled where GT is missing), grid free / `depth_grid=0`; control = DA3 depth through the same wrapper | AUC@3 HiRoom 50.1 → 60.8 / 61.1, 7-Scenes 26.8 → 23.9 / 20.2, ScanNet++ 33.6 → 37.5 / 39.8, ETH3D 30.8 → 39.4 / 46.3; mean without 7-Scenes 38.2 → 45.9 / **49.1** (weak10 oracle on the same scenes, seed 0: 36.8 → 49.5); dev / held-out alike (4-set mean +4.1/+8.8 dev, +5.1/+5.3 held-out); AUC@30 +0.1 to +2.4 except 7-Scenes; DA3 depth abs. rel. vs GT 6.3 % (scene scale) | `lpba_dor_*` |
| 11-22 | SMALL, all 81 pose scenes, seed 0, from the saved B′ states (dev + held-out) | **alternation** (`scripts/lpba_stage.py`): pose stage (`depth_grid=0 levels=2,1,1 lm_levels=2`); depth stage (`poses=false focal=none depth_grid=32 levels=2,1 lm_levels= samples=4096`) | B′ 43.6 (0 s extra); pose stage only 44.5 (+0.40 s); depth 1 + pose 1 44.6; ×2 45.2 (+2.8 s); depth abs. rel. B′ HiRoom/ScanNet++/ETH3D 5.38/5.95/5.16 % → after 2 depth stages 5.45/6.44/5.33 % (DA3: 6.3 % over 4 sets) | `lpba_alt_*` |

## Open / don't repeat

- Don't train V1 (learned per-frame damping / prior towards DA3): its GT oracle has no gain on SMALL. Untested: LARGE,
  where a global `w_pose=10` helped (+3.5 dev, [pose prior](pose-prior-and-flat-directions.md)); an oracle there would
  bound a model-dependent prior, not a learned step.
- V2 as a learned step on per-frame state statistics: the probe (check 2) finds nothing predictable across scenes. Not
  tested: image-level inputs (DA3 tokens, a CNN on the images), scene-level models (a GNN over all frames).
- **Depth is the lever** (depth oracle): correct depth recovers about the whole weak-direction headroom on HiRoom /
  ScanNet++ / ETH3D, and with correct depth the grid hurts (fixed > free by 3.2). Next: non-oracle depth sources at
  matched compute (a larger DA3's depth with SMALL's cameras; a monocular depth model), and how much of DA3's 6 % depth
  error is fixable by multi-view consistency at a finer level than the 8-cell grid. 7-Scenes can't be judged with its
  misaligned GT depth.
- Don't repeat: multi-view depth refinement with poses fixed (alternation, 32-cell grid, photometric + geometric terms)
  doesn't reduce the depth error; the information is not in the photometric objective. A cheap gain that is real but
  small: one pose polish with the depth fixed after B′ (+0.9 on 81 scenes, one seed, 0.4 s).
- Learned features trained through the solver: check 1 says better features (VGG) and more pairs change the conditioning
  by ≤ 2x, while fixing the depth grid changes it ~29x. The flat directions are mostly a geometric pose ↔ depth trade-off,
  which features alone can't remove. Next, the oracle that decides whether *depth* is the missing information: PBA with
  the depth grid fitted to the GT depth (or GT depth as depth0), poses free; plus a `w_prior` sweep as the non-oracle
  version.
- The baseline's seed spread on dev SMALL (±1.3, mostly ETH3D) is larger than the ±0.4 in [evaluation](../evaluation.md):
  use ≥ 3 seeds per configuration.
- Related negative results to keep in mind: correspondences inside PBA lowered AUC
  ([correspondences](correspondences-in-pba.md)); DA3 tokens as features were neutral ([dino](dino-features.md));
  TTO-refined targets didn't help DéjàView ([tto-teacher](dejaview-tto-teacher.md)).

Related: [pose-prior-and-flat-directions](pose-prior-and-flat-directions.md), [depth-prior](depth-prior.md),
[objective-bias-gt-start](objective-bias-gt-start.md), [matched-compute](matched-compute.md)
