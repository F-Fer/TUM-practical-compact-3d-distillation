# Geometry-guided match refinement: a learned feature-metric step between feed-forward reconstruction and BA

Exploration of 2026-11-25 (Finn's branch `lpba`, written by Claude in a 10 h autonomous session on tum1, RTX 3080).
Nothing is committed. Numbers are AUC@3 in %, DA3-SMALL unless noted, one seed unless noted; the sets and their noise
are in [knowledge/evaluation.md](../knowledge/evaluation.md). Evidence tables with run names:
[knowledge/ideas/guided-match-refinement.md](../knowledge/ideas/guided-match-refinement.md) and
[local-windows.md](../knowledge/ideas/local-windows.md).

**Summary.**

- **Method (GMR).** The feed-forward reconstruction already gives a coarse correspondence for every covisible pair:
  back-project with the predicted depth and project with the predicted cameras. GMR feeds this geometric warp to RoMa
  v2's small convolutional refiners and skips RoMa's DINOv3 descriptor and global matcher. A GPU bundle adjustment with
  free 3D points then runs on the refined matches. Only the 2.55M-parameter 1/4-resolution refiner is fine-tuned, on
  the errors of feed-forward starts.
- **Held-out pose.** On the 59 held-out scenes, with clean cross-fold splits so that no refiner saw the scene it
  processes:
  - DA3-SMALL: B′ 44.1 → **71.5** AUC@3 (NCC-PBA 57.2, RoMa + BA 67.1);
  - BASE: 57.0 → **76.7**;
  - LARGE-1.1: 69.9 → **80.2**;
  - the refiner was trained only on DA3-SMALL states, so the BASE and LARGE gains need no retraining;
  - recon F1 (SMALL held-out) is 58.8, against 57.0/53.2/51.1 for RoMa + BA / NCC-PBA / B′.
- **No PBA needed.** Started from the raw feed-forward prediction on dev, it reaches 79.8 from DéjàView (alone 56.1) and
  79.2 from LARGE (alone 48.8).
- **What makes it work.** Not the BA: from GT cameras the BA gives the same result. With RoMa's weights the refiners
  capture only errors ≤ 2-3 px and stay confident about larger ones. 15 minutes of fine-tuning moves the capture range
  out by ~4x and makes the confidence honest. That is +12-17 AUC@3 over the same pipeline with RoMa's weights.
- **Training data.** The gain saturates after ~8 training scenes and ~1k steps. A refiner trained only on ARKitScenes,
  with no benchmark data at all, keeps most of it: held-out SMALL **68.1** (benchmark-trained 71.5, RoMa + BA 67.1, NCC-PBA 57.2).
- **Cost.** A cheap configuration costs 4.2 s per 100-frame scene on the 3080 for 68.3 on dev (full: 16.7 s for
  68.9). That is not yet a matched-compute win for SMALL against LARGE + B′; the gain is that every model moves up.
- **Negative result.** Running DA3 on local 16-frame windows and stitching them (§2) gave a worse start.
- **What to scale on the 5090 (§7).** A purpose-built refiner, rather than more data for this one: all three refiners,
  a recurrent update with a correlation pyramid for a wider reach, the refiner on DA3's own feature maps (makes the
  measurement almost free), predicted precision in the BA, and depth corrected from the BA points.

## 1. Where the branch stood

The branch had shown that photometric BA (PBA) on a compact model is limited by three things: DA3's depth (the pose
↔ depth-grid trade-off; GT depth is worth +11), an objective whose optimum is set by that depth (PBA started from RoMa-BA's better cameras walks back to its own
depth-limited optimum), and a per-pixel bias (NCC helps). Everything learned so far (depth refiners, depth
bases, head fine-tuning, distillation of the test-time ensemble) failed to transfer at tum1 scale. The reference to
beat without a global matcher was dense RoMa tracks + BA: ~65-67 from SMALL at 20-36 s, against 44 (B′, 1 s) and 57
(NCC-PBA, 15 s).

The learned-refinement report ended with "the missing information has to come from a new measurement". This session
looked for that measurement.

## 2. First attempt: local windows (negative)

DA3-SMALL is much better inside a 16-frame window than in its 100-frame pass (ScanNet++, pairs inside a window: AUC@3
19.0 vs 4.9, median rotation 1.55° vs 3.84°). But no way of stitching the windows (Sim(3) chain, alignment to the full
pass with or without averaging) was a better start for PBA: 27.5-30.3 and 9.3 against 34.0 from the full pass. The
window boundaries leave discontinuities that PBA doesn't remove. Details: `knowledge/ideas/local-windows.md`.

## 3. The idea

A dense matcher such as RoMa v2 has two parts: a large coarse stage (DINOv3 ViT-L descriptor and a transformer
matcher, which output a warp at 1/16 resolution) and three small convolutional refiners (at 1/4, 1/2 and full
resolution, on VGG19-BN features) that correct a given warp with a local correlation. The coarse stage is what makes
RoMa cost ~70 ms per pair.

With a feed-forward reconstruction, the coarse warp of every covisible pair already exists: back-project the source
pixels with the predicted depth and project them with the predicted cameras. So the pipeline is

1. DA3 (optionally + B′, 0.3-1 s) → depth and cameras;
2. for every frame and its 10 spread covisible neighbours: geometric warp → refiners → corrected warp + overlap
   confidence (no descriptor, no global matcher);
3. 1,024 random source pixels per frame become 3D points with one observation in each neighbour that the confidence
   keeps; reprojection BA with free points on the GPU (`scripts/sba_fast.py`'s solver: poses, shared focal, radial).

Compared with PBA, the linearised photometric residual (a basin of 1-2 px per blur level) is replaced by a learned
local correlation and regression, and the structure is free, so DA3's depth error no longer biases the pose. Compared
with RoMa + BA, the expensive coarse stage is gone, every frame is a query, and pairs come from the predicted
covisibility. Code: `scripts/gmr_ba.py`.

## 4. With RoMa's weights as they are (no training)

Dev (22 scenes), from DA3-SMALL + B′ (`iters=25`, 42.3), one round, on the 3080:

| refiner schedule (RoMa's weights) | HiRoom | 7-Sc | ScanNet++ | DTU-64 | ETH3D | mean |
|---|---|---|---|---|---|---|
| B′ (the start) | 44.9 | 37.0 | 24.8 | 87.0 | 17.8 | 42.3 |
| `4,2,1` (RoMa's own) | 53.5 | 36.4 | 32.0 | 99.9 | 48.1 | 54.0 |
| `4@2,4,2,1` (1/4 refiner first on half-resolution images) | 57.8 | 36.5 | 34.2 | 100 | 57.3 | 57.2 |
| `4@2,4,4,2,1` (+ the 1/4 refiner twice) | 54.7 | 36.4 | 34.2 | 100 | 58.2 | 56.7 |
| `4@4,4@2,4,2,1` (+ quarter-resolution images) | 51.1 | 36.7 | 41.1 | 100 | 80.4 | 61.8 |
| *NCC-PBA, T = 800 (13.5 s)* | *54.7* | *33.9* | *42.0* | *93.3* | *54.2* | *55.6* |
| *RoMa turbo, 8 query frames + BA (26 s)* | *56.7* | *34.4* | *47.2* | *100* | *39.9* | *55.6* |
| *RoMa, 30 covisible frames per query + BA (20 s)* | *56.8* | *32.6* | *66.5* | *100* | *66.9* | *64.6* |

- Already with RoMa's untouched refiners, one round after B′ reaches the level of NCC-PBA and of RoMa with 8 query
  frames. Wider search (coarse stages) helps ETH3D and ScanNet++ a lot but breaks two HiRoom scenes.
- From DA3 alone (no PBA) the plain schedule does nothing (16.2 → 18.4): the start is out of range. With
  `4@4,4@2,4,2,1` it reaches 48.7, above B′.
- More rounds with the same matches-near-the-current-geometry don't help (53.9 after 3 rounds).

Two diagnostics say why it stops there.

**The BA is not the limit.** Started from the ground-truth cameras with the matches that were predicted from B′'s
geometry, the BA returns to the same solution and the same cost as from B′'s cameras (HiRoom / ScanNet++ / ETH3D 53.1 /
32.1 / 50.9 vs 53.5 / 32.0 / 48.1). Where a scene stays wrong, the matches say so.

**The refiners capture only 2-3 px here, and are confident about what they miss.** Match error against GT
correspondences (HiRoom dev, GT depth and cameras), by the error of the geometric warp they started from:

| initial error | share of matches | median error after, RoMa's weights | < 2 px | confidence > 0.5 |
|---|---|---|---|---|
| < 1 px | 38 % | 0.44 px | 97 % | 100 % |
| 1-2 px | 19 % | 0.78 px | 88 % | 100 % |
| 2-4 px | 17 % | 1.6 px | 59 % | 99 % |
| 4-8 px | 13 % | 4.0 px | 24 % | 98 % |
| 8-16 px | 7 % | 9.1 px | 7 % | 96 % |
| 16-32 px | 3 % | 19.8 px | 1 % | 93 % |

Below 2 px the refiners are sub-pixel accurate; from 4 px on they return the warp they were given, with full
confidence. So the BA is told that the current geometry is right.

RoMa trained its refiners on the small errors of its own matcher. A feed-forward start has larger ones.

## 5. Fine-tuning the refiner for this use

`scripts/gmr_train.py` fine-tunes only RoMa's 1/4-resolution refiner (2.55M parameters; the 1/2 and 1/1 refiners have 0.19M and 0.02M; the finer refiners and
the VGG features stay frozen). A step takes one source frame and four neighbours from a scene with GT depth and cameras;
the initial warp is the geometric warp of DA3-SMALL's prediction or of its B′ refinement, the target the GT warp where
the GT says the point is visible. The schedule is `4@2,4,4,2,1` (the 1/4 refiner once on half-resolution images, twice
at full resolution, then the frozen finer ones), with a robust L1 loss in pixels and a BCE loss on the overlap logit
towards "visible and within 3 px". 3,000 steps take 15 minutes on the 3080.

**Checkpoint `a`: trained on the held-out scenes of HiRoom (20) and ScanNet++ (17), DA3-SMALL states only.** The dev
scenes are unseen; 7-Scenes, DTU and ETH3D are unseen datasets; BASE and LARGE states are unseen.

Dev (22 scenes), one round from B′, `stages=4@2,4,4,2,1`, 1,024 points per frame:

| start | refiner | HiRoom | 7-Sc | ScanNet++ | DTU-64 | ETH3D | mean |
|---|---|---|---|---|---|---|---|
| SMALL + B′ (42.3) | RoMa's weights | 54.7 | 36.4 | 34.2 | 100 | 58.2 | 56.7 |
| SMALL + B′ | **fine-tuned `a`** | 62.8 | 36.3 | 55.9 | 100 | 89.4 | **68.9** |
| SMALL alone (16.2), no PBA | fine-tuned `a` | 56.7 | 35.9 | 47.9 | 100 | 72.0 | 62.5 |
| SMALL alone, no PBA | RoMa's weights, `4@4,4@2,4,2,1` | 38.0 | 36.1 | 21.5 | 100 | 47.8 | 48.7 |
| BASE + B′ (59.6) | fine-tuned `a` | 85.7 | 36.1 | 73.7 | 100 | 86.2 | **76.3** |
| LARGE-1.1 + B′ (70.5) | fine-tuned `a` | 86.3 | 35.7 | 85.6 | 100 | 98.5 | **81.2** |
| *references: NCC-PBA SMALL / BASE / LARGE* | | | | | | | *55.6 / 65.0 / 72.0* |
| *RoMa (30 covisible frames per query) + BA, SMALL / BASE / LARGE* | | | | | | | *64.6 / 66.8 / 68.1* |

- The same schedule with RoMa's weights gives 56.7: **the 15 minutes of fine-tuning are worth +12.2** on unseen scenes,
  +31 on ETH3D, which is not in the training data.
- It transfers to the other models without retraining: BASE +16.7, LARGE +10.7 over B′. RoMa + BA lowered LARGE
  (62-68 on dev, [fast-sparse-ba](../knowledge/ideas/fast-sparse-ba.md)); this doesn't. BASE with it (76.3) is above
  LARGE + B′ (70.5) and LARGE + NCC-PBA (72.0); LARGE with it (81.2) is above the best GIANT number we have on dev
  (77.2, VGG + NCC, on the 5090).
- A second round (re-match after the BA) changes nothing (HiRoom / ScanNet++ / ETH3D 59.9 / 56.4 / 89.3 vs 62.8 / 55.9 /
  89.4).
- Recon F1 (dev, 4 sets, TSDF of the benchmark; depth = DA3's, scaled per frame to the BA points): **56.5** (HiRoom
  63.5, 7-Scenes 54.9, ScanNet++ 45.4, ETH3D 62.1) vs B′ 47.8, NCC-PBA 51.6, RoMa + BA 46.8, DA3-SMALL alone 31.5.

What the fine-tuning changed in the refiner (HiRoom dev, same table as §4):

| initial error | < 2 px, RoMa's weights | < 2 px, fine-tuned | kept by the confidence | < 2 px among the kept |
|---|---|---|---|---|
| < 1 px | 97 % | 91 % | 96 % | 93 % |
| 1-2 px | 88 % | 80 % | 91 % | 85 % |
| 2-4 px | 59 % | 63 % | 83 % | 71 % |
| 4-8 px | 24 % | 44 % | 67 % | 57 % |
| 8-16 px | 7 % | 30 % | 50 % | 51 % |
| 16-32 px | 1 % | 16 % | 31 % | 40 % |

The capture range moved out by a factor of ~4, and the confidence now drops matches it didn't capture (before, it kept
92-99 % in every bin). The price is a few points on matches that were already within 2 px; training the finer
refiners too, or longer, should recover that. On ScanNet++ dev the 8-16 px bin goes from 5 % to 52 % below 2 px.

**Without PBA.** From the raw feed-forward prediction (dev, checkpoint `a`): SMALL 16.2 → 62.5 (two rounds 66.9),
BASE 27.1 → 68.9, LARGE-1.1 48.8 → 79.2 (with B′ first: 81.2), DéjàView (released, K = 16, 56.1) → **79.8**. B′ matters
for SMALL (+6.4), hardly for LARGE (+2.0); the refiner, trained only on DA3-SMALL states, works on DéjàView's output.

**Held-out with clean splits.** Two more refiners, each trained (3k steps) on the dev scenes plus one half of the
held-out HiRoom / ScanNet++ scenes and applied to the other half; 7-Scenes, DTU and ETH3D are in neither. So every one
of the 59 held-out scenes is processed by a refiner that never saw it, and three of the five datasets were never
seen at all. From B′ of each model, same configuration as above:

| model | B′ | NCC-PBA (T = 800, 15 s) | RoMa + GPU BA (36 s) | **GMR** (20 s, 3080) |
|---|---|---|---|---|
| SMALL | 44.1 | 57.2 | 67.1 | **71.5** (86.9 / 23.4 / 73.6 / 100 / 73.8) |
| BASE | 57.0 | 68.0 | 67.9 (59.2 with prior) | **76.7** (92.9 / 23.9 / 85.7 / 99.9 / 81.0) |
| LARGE-1.1 | 69.9 | 74.3 | 68.5 (70.4 with prior) | **80.2** (95.9 / 23.6 / 96.7 / 100 / 85.1) |

(AUC@3 per dataset in brackets: HiRoom / 7-Scenes / ScanNet++ / DTU-64 / ETH3D.) Seeds on dev (SMALL, checkpoint `a`): 68.9 / 67.8 / 68.7, mean 68.5.

**How much training it needs.** Fold `a` (dev + half of held-out HiRoom / ScanNet++, 32 scenes), evaluated on the other
half (HiRoom 9 + ScanNet++ 9 scenes) and on 5 held-out ETH3D scenes, in the cheap configuration of §6 (240 x 320, 6
neighbours, no 1/1 refiner, 15 LM iterations):

| refiner | HiRoom | ScanNet++ | ETH3D (unseen dataset) | mean of 3 |
|---|---|---|---|---|
| RoMa's weights | 66.9 | 45.8 | 54.1 | 55.6 |
| 3k steps, 1/4 of the scenes (8) | 75.9 | 60.9 | 72.4 | 69.7 |
| 3k steps, 1/2 of the scenes (16) | 83.6 | 60.4 | 73.0 | 72.3 |
| 3k steps, all 32 scenes (`cva`) | 83.8 | 61.2 | 73.1 | 72.7 |
| 10k steps, all 32 scenes (one-cycle LR; 1k / 3k / 10k checkpoints) | 80.6 / 84.5 / 85.3 | 64.2 / 65.6 / 64.0 | 72.3 / 72.5 / 73.5 | 72.4 / 74.2 / 74.3 |

Most of the gain is there after 8 scenes and 1k steps; on the unseen dataset (ETH3D) more data or steps add ~1 point.
So at this size the fine-tuning mainly teaches the refiner the *kind* of error a feed-forward start makes (larger,
smooth, depth-induced) and to report it in the confidence, not scene content. Scaling the same recipe on more data
alone is unlikely to give much more; the capacity and the reach of the refiner (§7, items 1-2) are the next levers.

**Trained without any benchmark data.** A refiner trained only on ARKitScenes (400 16-frame windows of 200 training
videos, LiDAR depth, ARKit poses interpolated to the frame timestamps; DA3-SMALL and B′ states; 3k steps) gives
held-out (59 scenes, all unseen) **68.1** (HiRoom 78.5, 7-Scenes 23.6, ScanNet++ 72.0, DTU-64 99.9, ETH3D 66.5) from
SMALL + B′ (44.1), against 71.5 for the benchmark-trained folds, 67.1 for RoMa + BA and 57.2 for NCC-PBA; on dev 64.8
(RoMa's weights 56.7). The ARKit poses had to be fixed first: the windows stored the nearest pose of the 10 Hz
trajectory, ~1.3° (~10 px) off on two thirds of the frames, and a refiner trained on those targets didn't learn at
all (`scripts/gmr_arkit_interp.py` interpolates them). The gap to the benchmark-trained refiner is mostly ETH3D (66.5
vs 73.8) and HiRoom (synthetic, 78.5 vs 86.9), i.e. domain, which more varied training data (outdoor, synthetic)
should close.

**Depth.** GMR keeps DA3's depth and only rescales each frame to the BA points. The depth error vs GT (one scale per
scene) barely moves (HiRoom dev 4.34 → 4.15 % etc.), and on two ScanNet++ dev scenes it doubles (4.8 → 9.5 %,
8.5 → 10.1 %): the per-frame scales from the BA points are noisy there. A smooth 16-cell correction fitted to the
points (`depth_fit=grid:16`) lowers it by only another 0.05-0.8 points and leaves the pose unchanged (68.9). The F1
gain above comes from the cameras; the depth side is untouched and is the obvious next target.

Recon F1 (TSDF of the benchmark, `recon_unposed`, 4 sets; GMR's depth is DA3's, scaled per frame to the BA points):

| | HiRoom | 7-Scenes | ScanNet++ | ETH3D | mean |
|---|---|---|---|---|---|
| SMALL held-out: B′ | 48.0 | 46.7 | 45.8 | 63.9 | 51.1 |
| NCC-PBA | 59.0 | 45.7 | 48.7 | 59.5 | 53.2 |
| RoMa + BA | 56.7 | 42.9 | 54.7 | 73.6 | 57.0 |
| **GMR** | 64.5 | 43.5 | 54.0 | 73.2 | **58.8** |
| BASE dev: B′ / GMR | 71.2 / 82.3 | 61.0 / 59.3 | 52.6 / 59.2 | 63.0 / 73.8 | 61.9 / **68.7** |
| LARGE dev: B′ / GMR | 87.5 / 87.2 | 59.1 / 54.9 | 65.1 / 69.6 | 68.9 / 75.8 | 70.2 / **71.9** |

7-Scenes is the one set where GMR's F1 is lower (its GT is known to be unreliable, `knowledge/evaluation.md`); its
pose AUC doesn't move under any method.

## 6. Cost

All on the RTX 3080 (tum1), TTO time per scene as `opt_s` of the PBA runs (image decoding excluded, models warm), dev
means; the 5090 was not used in this session.

| configuration (checkpoint `a`, from B′) | AUC@3 dev | s per scene | of which matching / BA |
|---|---|---|---|
| refiners at 384 x 512, 10 neighbours, `4@2,4,4,2,1`, 25 LM it. | 68.9 | 16.7 | 12.7 / 0.6 |
| 288 x 384 | 69.0 | 10.1 | 7.3 / 0.6 |
| 288 x 384, 6 neighbours | 68.0 | 6.7 | 4.5 / 0.6 |
| without the full-resolution refiner | 68.7 | 14.4 | 10.8 / 0.6 |
| **240 x 320, 6 neighbours, no full-resolution refiner, 15 LM it.** | **68.3** | **4.2** | 2.8 / 0.35 |
| *B′ (the start), on the same GPU* | *42.3* | *1.0* | |
| *NCC-PBA T = 800* | *55.6* | *13.5* | |
| *RoMa v2 turbo + GPU BA (8 queries / 30 covisible frames)* | *55.6 / 64.6* | *26 / 20* | |

Per pair the 1/4 refiner costs 11.4 ms at 384 x 512 and the 1/2 and 1/1 refiners 5.5 and 4.9 ms (bf16, 3080); VGG
features 3.1 ms per frame. The cost is dominated by the 1/4 refiner's eight 512-channel conv blocks at 96 x 128 and is
linear in the number of pairs, so a refiner designed for this (fewer channels, run on DA3's own feature maps) is
where the next factor comes from. On the 5090, PBA ran 2.5-3x faster than on the 3080; if GMR scales the same, the
cheap configuration is ~1.5 s plus B′'s 0.3 s (an estimate, not measured).

## 7. What this suggests to scale on the 5090

The prototype borrows a refiner that was built for another job. What a purpose-built one could add, in order of
expected value:

1. **Train the refiner on the errors of feed-forward starts, at scale.** 15 minutes on 37 scenes already moved the
   capture range and made the confidence honest (§5). The 5090 version: all three refiners, tens of thousands of steps,
   posed RGB-D / MVS data that isn't the benchmark (ScanNet++ train split, ARKitScenes, MegaDepth, synthetic), initial
   warps from DA3-SMALL / BASE / LARGE with and without B′ so that the error range is covered. The target metric is the
   capture table of §4 (fraction below 2 px per initial-error bin, and the kept fraction), which costs minutes to
   evaluate, before any BA.
2. **Make it iterative and wider.** The 1/4 refiner sees ±3 cells (±12 px at 384 x 512). A recurrent update with a
   correlation pyramid (RAFT / DROID style: look-up radius at 1/16, 1/8, 1/4) would reach the 16-64 px errors that
   DA3-SMALL makes on ScanNet++ and ETH3D without the detour over half-resolution images, and could make the B′ stage
   unnecessary (from DA3 alone the fine-tuned refiner already gives 62.5).
3. **Use the feed-forward model's own features.** The VGG19-BN features cost a pass per frame and 4 GB at 100 frames.
   DA3's DPT head already computes feature maps at H/3.5 for every frame; a refiner on those would make the measurement
   nearly free. RoMa-Ω (arXiv 2609.09507) shows that feed-forward 3D models' features carry matching information once a
   learned projection reads them out; it keeps the global matcher, we would drop it.
4. **Predicted uncertainty in the BA.** RoMa v2's refiners output a 2 x 2 precision per match; the BA here ignores it
   (isotropic Cauchy 0.5 px). Whitened residuals are the standard next step.
5. **Depth from the BA points.** The triangulated points are an independent depth measurement of each frame; a smooth
   correction of DA3's depth fitted to them (`depth_fit=grid:16`) is the non-oracle version of the "smooth GT-fitted
   correction" that gave +10 F1 on HiRoom in [shared-depth-field](../knowledge/ideas/shared-depth-field.md).

Why this is a research direction and not only a faster RoMa + BA: the measurement is conditioned on the feed-forward
geometry (which pairs, where to look, what is occluded), the refiner is trained on that model's error distribution, and
its capture range and honesty can be measured and trained directly. Nearest work: BLASt3R (own multi-view matcher +
depth-regularised BA), Glob3R (matching head on frozen π³X tokens, global SfM), RoMa-Ω (feed-forward features inside
RoMa), Refin3R / GGPT (RoMa + BA, then a learned densifier), MegaSaM / DROID-style dense BA (video, learned flow
updates from a cold start), DeepBA (learned BA step on eLoFTR matches). None of them warm-starts a local correspondence
refiner from the feed-forward geometry of an unordered set and trains it for those errors; check again before
committing, the field moves weekly.

Risks: (a) the refiner here was trained on held-out scenes of two benchmark datasets; §5 lists what is and isn't
in-domain; (b) scenes where the start is grossly wrong for most pairs stay wrong (one ScanNet++ dev scene, AUC@30
35 %), only a global matcher repairs those; (c) at matched compute SMALL + GMR is not yet a win: on the 5090 the cheap configuration would put SMALL + B′ +
GMR at ~2.2 s for ~68 (estimate), while LARGE + B′ gives 71.0 at 1.8 s (dev); the win is that every model gains
(BASE + GMR 76.3 dev is above LARGE + B′ and GIANT + any PBA), so the compute plot moves up as a whole;
(d) all numbers are one seed on 22 dev scenes unless stated.

## 8. Files

- `scripts/gmr_ba.py` (the method; options in its `DEFAULTS`), `scripts/gmr_train.py` (fine-tuning),
  `scripts/gmr_basin.py` (capture-range table), `scripts/lpba_scenes.py` (per-scene table), `scripts/lpba_win.py`
  (local windows).
- Runs: `workspace/tto/<model>/gmr/<run>/` (metrics in `metric_results/<dataset>_pba.json`, read by
  `scripts/sba_sum.py`); checkpoints `workspace/tto/DA3-SMALL/gmr/ft/{a,cva,cvb,ark}.pt` (`a`: held-out HiRoom / ScanNet++; `cva` / `cvb`:
  the two folds; `ark`: ARKitScenes only; `lc*`: learning curves); ARKit windows
  `workspace/tto/DA3-SMALL/gmr/arkit/`.
- Reproduce the headline run:
  `.venv/bin/python scripts/gmr_ba.py workspace/tto/DA3-SMALL/sba_ref/pho25 <out> pts=1024 refiner_ckpt=workspace/tto/DA3-SMALL/gmr/ft/a.pt stages=4@2,4,4,2,1`
