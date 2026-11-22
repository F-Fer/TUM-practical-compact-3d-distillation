# A learned refinement step between feed-forward reconstruction and bundle adjustment

Idea and motivation document, 2026-11-22 (Finn, written with Claude). To be continued on tum2. Read
[knowledge/README.md](../knowledge/README.md) first; numbers below are from there unless cited otherwise.

## In one paragraph

Train a small network that does what one iteration of our photometric BA (PBA) does, but better: it takes DA3's
cameras and coarse depth as an **explicit state**, looks at the **residuals of that state against the images** (the
same photometric/feature-metric warps, gradients and Gauss-Newton blocks `scripts/run_tto_pba.py` already computes),
adds **DA3's own prior** (confidence, pooled tokens) where the residuals carry no information, and outputs an update
of the poses, the shared focal and the depth grid. A few learned steps replace the 4 × T Adam iterations on blurred
levels; a short classical LM finishes at the tight thresholds. The research question is our project question one level
deeper: **at matched compute, does a learned step beat a hand-tuned optimiser on a feed-forward initialisation, and
does the gain come from the directions the photometric objective cannot see?**

## 1. Where we stand and why this is the next question

- PBA works: on dev, AUC@3 SMALL 16.2 → 43.8, BASE 27.1 → 59.2, LARGE-1.1 48.6 → 71.0 for 0.2-0.3 s of TTO; at
  matched time a model + TTO beats the next larger model alone ([matched-compute](../knowledge/ideas/matched-compute.md)).
- What limits it, from our own diagnosis:
  - **Flat directions.** The photometric objective has condition number ~6e4 in the relative poses; DA3's pose error
    sits mostly in the weakest 10 % of eigen-directions, and after PBA 60-90 % of the remaining error is there
    ([pose-prior-and-flat-directions](../knowledge/ideas/pose-prior-and-flat-directions.md)). A pose prior towards DA3
    helps LARGE and hurts SMALL: *how much to trust the prior depends on the initialisation*, and we set it by hand per
    model ([depth-prior](../knowledge/ideas/depth-prior.md), open question 7).
  - **Objective bias.** LM started at the ground truth walks away; NCC removes most of it
    ([objective-bias-gt-start](../knowledge/ideas/objective-bias-gt-start.md)).
  - **Hand-built schedule.** Blur pyramid, Adam then LM, stop rule on held-out pixels, per-model priors, gated radial
    coefficient: each was found by a failure, and they interact (the curve over T changed three times,
    [iterations-over-t](../knowledge/ideas/iterations-over-t.md)).
  - **Gross failures.** 13.6 % (ScanNet++) and 17.8 % (ETH3D) of DA3-SMALL frames are > 15° off; PBA repairs few, and
    our relocalisation broke more than it fixed ([gross-failures](../knowledge/ideas/gross-failures-and-reference-view.md)).
- A learned step attacks the first three directly: it can learn when to trust DA3 (prior strength as a function of the
  state, not a constant), learn a better-scaled step than Adam in the flat valley, and absorb the schedule. It does not
  obviously fix gross failures (see §6).

## 2. The design space (the conceptual part)

Three families refine iteratively. They differ less in "tokens vs geometry" than in **what each step looks at**:

| | state carried between steps | what one step sees | start | examples |
|---|---|---|---|---|
| looped transformer | latent tokens; geometry decoded at the end | the previous tokens (+ the image encoding z0 in Solv3R) — **no residual of the current estimate** | image encoding | DéjàView, Solv3R |
| latent solver | latent tokens | a **learned latent residual** between views | image encoding | BA-T |
| learned BA / learned optimiser | **explicit geometry** | **geometric or photometric residuals of the current geometry** | cold start or a prediction | DeepBA (matches, cold start), DROID-SLAM, DRO, BA-Net, ReSplat / GradSplat (3DGS) |
| classical BA | explicit geometry | residuals, Jacobians, a hand-written update rule | a prediction | our PBA, track BA, BLASt3R |
| diffusion / flow matching | explicit or latent | the noisy sample + the images; residuals only through guidance | noise (or a prediction) | PoseDiffusion (Sampson guidance), PosePlaner |

Two clarifications that came out of the discussion:

- **DeepBA is small because the expensive parts are not learned.** A matcher (eLoFTR) finds the correspondences, and
  projective geometry is hand-written (closed-form residuals). The 2.04M-parameter GNN only learns the step. The price:
  no monocular prior (depth only where matches are, 32 × 32 grid), it fails where matching fails, and it is far behind
  VGGT at 3° (ETH3D RRA/RTA@3 33.8 / 18.4 vs 84.1 / 51.4). The reprojection is not overhead: it is the measurement.
  Our idea keeps the measurement but replaces the matcher by the image warp, because a DA3 initialisation tells us
  where to look.
- **Diffusion vs a learned solver** is not the real axis. Diffusion learns a distribution and starts from noise; a
  solver is trained to contract to one fixed point and is fed its own residual. They blend (DeepBA trains from
  perturbed starts; PoseDiffusion adds residual guidance; DéjàView's t ∈ [0, 1] loop is structurally a flow
  integrator). A distribution only pays where the answer is ambiguous, i.e. DA3's gross failures, and there the hard
  part is scoring hypotheses, which our relocalisation showed we cannot yet do.

## 3. Literature (checked 2026-11-22, ~20 papers read; re-check before committing, the field moves weekly)

### Closest work and how ours differs

| paper | what it refines, from what | step sees | views / cameras | why it is not our idea |
|---|---|---|---|---|
| **DeepBA** 2611.18256 (TUM, Nov 13) | poses, FOV, 32² log-depth, **cold start** | closed-form residuals of eLoFTR matches | many, uncalibrated | needs a matcher, no feed-forward prior, weak at 3° |
| **Refin3R** 2611.23051 (KAIST, Nov 17) | DA3/π³X **depth** with SfM points (RoMa v2 + sparse BA + DLT, GGPT pipeline) | SfM depth vs FF depth, backbone features | 4-16, cameras from the SfM BA | one-shot depth refiner; poses come from a classical BA; needs dense matching. DA3 ETH3D 16-view AUC@1 cm 7.0 → 53.6 |
| **PosePlaner** 2610.34845 | relative **rotation**, from any FF pose | plane-warping residuals of matched patch planes; flow matching, one Euler step | **two-view**; n-view by rotation averaging + GLOMAP positioner | pairwise, rotation only, needs a matcher and a plane network (3.9M training pairs); slightly hurts already-accurate inits |
| **Solv3R** 2611.02857 | DéjàView latent state, stationary loop | previous state + z0 injection, no geometric residual | many | no explicit residual, no comparison with BA; trains stationarity (perturbed states, partial resets) |
| **BA-T** 2606.03287 (TUM) | latent camera + geometry tokens | learned latent residual | two-view (4-view extension) | latent, two-view, trained from scratch (8 H100 × 2 days) |
| **Diff3R** 2604.01030 (TUM) | 3DGS from **DA3** (pose-free) | bilevel: TTO inside training (implicit gradients); learns **per-parameter proximal weights** | 4 views | Gaussians/NVS, not poses; but the learned-prior idea transfers directly |
| **DeltaSplat** 2610.09853, **ReSplat**, **GradSplat** 2610.38951, **Learn2Splat** 2605.15760 | 3DGS from a FF model | render residuals (+ Plücker rays, depth) or Adam-normalised loss gradients | poses fixed (DeltaSplat: pose-free backbone, refines Gaussians only) | NVS; cameras are not the unknowns |
| **DRO** 2103.13201, BA-Net, DeepV2D, DeepSFM | depth + pose from a head | feature-metric cost map (GRU, alternating depth/pose) | two-view video, calibrated | not multi-view, not uncalibrated, not on a foundation model |
| classical BA on FF output: BLASt3R 2609.05210, GeoWeaver 2608.17389, EPO 2607.00579, VGGT+BA, Self-Geometry 2608.10708 (LoRA TTA on DA3) | | | | hand-written optimisers, the baselines |

**Open niche (not found in what we read):** a learned update operator on **explicit** cameras + coarse depth,
**warm-started from a multi-view foundation model**, fed by **dense photometric / feature-metric residual statistics**
(no matcher, no SfM), for **unordered, uncalibrated** multi-view sets, compared **at matched compute** against a tuned
classical photometric BA on the DA3 benchmark. Refin3R and PosePlaner (both Oct/Nov 2026) are the nearest threats and
show the direction is hot.

### Lessons the papers give us for the design

1. **The explicit residual is the ingredient.** BA-T: the crudest latent residual alone recovers 76 % of the rotation
   gain; DRO without its cost map is much worse; DeltaSplat without geometric conditioning loses 3/4 of its gain.
   → Feed residuals, not just the images.
2. **Learned refiners degrade past their training horizon** (ReSplat diverges after ~10 steps, DeltaSplat peaks at the
   trained K = 3, DéjàView falls to 0 AUC@5 at 128 calls). Fixes that worked: supervise every iterate (DeepBA, BA-T,
   DRO), train from perturbed and partially-reset states (Solv3R), a checkpoint buffer + rollouts + a stability loss
   that penalises any increase in error (Learn2Splat), a fixed-point penalty ‖Δs‖ (DeepBA).
3. **Learned steps drift too.** DeepBA tilts bonsai by 9.4° starting from the ground truth, like our GT-start test.
   → Always evaluate from the GT state as a drift test.
4. **A learned prior strength per parameter** (Diff3R's per-Gaussian proximal weights) beats a global one
   (+0.27 dB vs +0.12 dB after TTO in their ablation) and learns to anchor unconstrained regions without labels. That
   is our "prior strength should follow init quality" problem.
5. **Classical BA after the learned step still helps** (DeepBA; PosePlaner regresses on already-accurate inits).
   → Keep an LM polish at the end; the learned part should bring the state into the basin quickly.
6. **Gradients as input work if normalised** (GradSplat divides by RMS, keeps a second moment; Learn2Splat uses
   Adam-style moments) and can be treated as constants during training (no second derivatives of the warp).

## 4. Proposed method: a learned PBA step (L-PBA)

### State (unchanged from `run_tto_pba.py`)

Pose update per frame (frame 0 fixed), one shared log focal (median init), gated radial coefficient, 8-cell log-depth
grid per frame, relative brightness. Init from saved DA3 predictions, exactly as now.

### Inputs per iteration (all already computed by `Kernel` in `scripts/run_tto_pba.py`)

- `Kernel.normal_eq` returns the Gauss-Newton blocks `Hpp` (poses + focal + radial), `Hgp`, `Hgg`, `Hx` and the
  gradients `gp`, `gg` of the IRLS-weighted objective. Per frame: its 6-vector of the gradient, its 6 × 6 diagonal
  block of `Hpp` (or its eigenvalues / the gradient in its eigenbasis), the grid gradient, the loss.
- Per pair (the 10 spread neighbours from `select_neighbors`): mean robust residual, visible fraction, NCC, off-diagonal
  coupling block.
- The prior: DA3 depth confidence summary per frame, a pooled DA3 camera/patch token per frame (frozen), and the
  state's offset from the DA3 initialisation.
- Normalise gradients by RMS per parameter group with a running second moment (lesson 6).

### Two variants, cheapest first

- **V1, learned LM weights (recommended first).** The network outputs a per-frame damping / prior precision (and a
  prior precision per grid cell) for the existing `Kernel.solve` (`(H + λ D) δ = −g`), i.e. it decides per frame and
  per iteration how far to trust DA3. The step stays a Gauss-Newton solve, so it cannot do anything a well-tuned prior
  couldn't, but it chooses the prior from the state, which no hand rule did. Differentiable through
  `torch.linalg.cholesky_solve` (dense, ≤ ~700 unknowns at 100 frames). Small, generalises, and directly tests lesson 4
  on cameras.
- **V2, direct update (DeepBA-style).** A message-passing network over frames (nodes) and covisible pairs (edges)
  predicts Δpose, Δfocal, Δgrid from the inputs above and is applied K times with shared weights. It can move along
  flat directions using the prior (e.g. DA3's systematic errors), which V1 cannot. Then LM polish.

Both run inside the current loop: replace the Adam levels by K learned steps, keep `lm_step` at the end.

### Training

- **Data without benchmark overlap:** ARKitScenes clips (`scripts/distill_steps/prep_arkit.py`, GT poses from
  `lowres_wide.traj`; 150 videos × 240 frames already used for distillation) and outdoor clips
  (`prep_outdoor.py`: Mip-NeRF 360, Tanks and Temples / Deep Blending; their COLMAP poses as GT). ETH3D/DTU-like
  outdoor geometry is the domain risk (the ARKit-only DéjàView student lost ETH3D, [dejaview-speed](../knowledge/ideas/dejaview-speed.md)).
- **Precompute** DA3-SMALL (and BASE) predictions on windows of 8-48 frames once; training then never runs DA3.
- **Loss:** scale-free pose loss as in DeepBA (geodesic relative rotation + translation-direction cosine over all
  pairs, Sim(3)-free), log-focal error, optional depth where GT exists; summed over iterates with later ones weighted
  higher; a penalty on any iterate that is worse than the previous (Learn2Splat) and on ‖Δs‖ at late iterates.
- **States:** start from DA3, from DA3 + noise, from GT + noise, and from states stored in a checkpoint buffer after
  rollouts (lessons 2, 3).
- **Gradients:** treat the kernel inputs as constants and detach the state between iterates (DeepBA, GradSplat): no
  autograd through the warp, memory O(1) in K.
- **Cost estimate (to verify):** one kernel evaluation is 1-2 ms at 1,024 px, the network is tiny (≤ 100 nodes,
  ~1,000 edges), so a training step of K = 8-16 iterates on one window is ~50 ms → 50k steps in ~1-2 h on a 5090.
  Fits the 6 h Slurm limit with checkpoints.

## 5. Experiments, in order, with a gate after each

| step | what | cost | gate / question |
|---|---|---|---|
| G0 | **Oracles, no training.** (a) Per-frame prior weight chosen with GT (best λ per frame by grid search) on dev SMALL: upper bound of V1. (b) Remove the GT error in the weakest 10 % of directions after B′ (`weak_modes.py`, commit `1c1a921`): upper bound for "fixing the flat directions" | 1 day, CPU/GPU | if (a) gives < +3 AUC@3 over B′, V1 is not worth training; go to V2 or stop |
| G1 | V1 on DA3-SMALL, dev, vs B′ T = 25 (0.71 s total) and the NCC config, at matched time | ~1 day incl. data prep | must beat B′ at equal time by more than the dev noise (±0.4) |
| G2 | V2 + LM polish, same comparison; ablate: no residual inputs (images/prior only), no prior inputs (residuals only), no per-iterate supervision, no perturbed starts, K beyond training | 2-4 days | which input carries the gain; does it stay stable at 2-4 × K |
| G3 | held-out (59 scenes); transfer **without retraining** to BASE, LARGE-1.1, GIANT-1.1 and DéjàView inits; GT-start drift test | 1-2 days | generalisation across init quality is the claim that matters for the paper |
| G4 | compute plot: add L-PBA to `results/pba/figures5/compute_all_models.png` (`plot_compute.py`); error decomposition weak/strong directions before/after | 1 day | the figure for the intermediate presentation / paper |

Report per step: AUC@3 / AUC@30, F1 (4 sets), total seconds per scene (DA3 + TTO, RTX 5090), and where the error went
(weak vs strong directions, gross frames). Log every run into [runs.csv](../knowledge/runs.csv) and an idea file
(see below).

## 6. Risks and what would kill it

- **No information in the flat directions.** If the residuals are flat there, a learned step can only use the prior,
  i.e. learn DA3's dataset biases. Part of that is already taken by the shared focal (+4.7 to +9.9 held-out). G0(b)
  measures how much is left.
- **Domain gap** ARKit/Mip-NeRF → ETH3D/DTU; mitigate with the outdoor clips and by keeping the update local
  (per-frame features, shared weights).
- **Gross failures** are out of reach of a local step; don't promise them. A sampler (flow matching over the
  per-frame rotation, scored by the residual) is a later, separate idea.
- **Chaos and noise.** PBA diverges from 1e-9 perturbations; compare on the dev noise level, several seeds for the
  final numbers ([evaluation](../knowledge/evaluation.md)).
- **Novelty.** Refin3R and PosePlaner appeared within the last four weeks; DeepBA's group (TUM, Cremers) could do the
  warm-start version next. The defensible claims are: warm start from a foundation model + photometric residuals
  without a matcher + matched-compute comparison against a strong tuned classical TTO + analysis by direction.

## 7. Where things are in the repo

- `scripts/run_tto_pba.py`: `PBA` (state), `Kernel.gradient`, `Kernel.normal_eq` (GN blocks), `Kernel.solve`
  (dense Cholesky with damping `lam * D`), `Kernel.lm_step`, `select_neighbors`, `refine_scene` (the loop to put the
  learned steps into). Older autograd path and `weak_modes.py`: commit `1c1a921`.
- Data: `scripts/distill_steps/prep_arkit.py`, `prep_outdoor.py` → `workspace/distill_data/arkit_frames/`.
- Scoring: `scripts/eval_tto.py`, `scripts/plot_compute.py`; dev/held-out sets in `knowledge/evaluation.md`; job
  system `$SCRATCH/jobs/pmba` ([infrastructure](../knowledge/infrastructure.md)).
- Knowledge entry: [learned-refinement-step](../knowledge/ideas/learned-refinement-step.md).

## 8. Results

### G0 (2026-11-22, dev DA3-SMALL, B′ `iters=25`, RTX 3080 on tum1)

Baseline over seeds 0/1/2: AUC@3 41.9 / 44.5 / 42.3 (mean 42.9), AUC@30 86.3-86.5. ETH3D alone ranges 17.5-28.1 and
makes the seed spread ±1.3, larger than the ±0.4 in `knowledge/evaluation.md`.

**G0a, oracle per-frame prior (V1's upper bound): fails the gate.** `pose_prior_oracle=c` adds a pose prior towards DA3
with weight w_i = c / e_i² per frame (e_i = GT error of DA3's pose of frame i, in PBA's parameters), applied in the Adam
and LM levels. AUC@3 (seed 0):

| c | 0 | 0.001 | 0.003 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|---|---|---|
| AUC@3 | 41.9 | 42.9 | 43.8 | 39.4 | 34.7 | 28.5 | 23.2 | 19.2 | 17.6 |

A global `w_pose` (1-100) gives 37.4 → 22.1. Choosing c per scene with the GT gives 46.8, but choosing the best of three
seeds per scene gives 45.7: the scene-level choice mostly picks favourable chaos. So even a perfect per-frame decision of
"how much to trust DA3" adds at most ~1 point on SMALL, below the +3 gate. V1 isn't trained.

**G0b, weak-direction oracle: large headroom.** At B′'s final state, removing the remaining GT error in the weakest
eigen-directions of the pose Schur complement (`scripts/lpba_weak_oracle.py`):

| removed | none | weakest 10 % | 10 % w/o gross frames | 25 % | 50 % | strongest 50 % | all |
|---|---|---|---|---|---|---|---|
| AUC@3 | 41.9 | 61.5 | 63.3 | 74.5 | 84.9 | 41.1 | 100 |

54-73 % of the remaining error lies in the weakest 10 % of the directions (median condition number 6e4-6e5).

**What this means for the plan.** The flat directions hold ~20 AUC@3 points, and the strong ones none, so a step that
only does better optimisation of the same objective can't gain. DA3's poses don't carry the missing information either
(G0a). V2 can only win if the network predicts the weak-direction error from something else: DA3's systematic biases
learned from data, wide-baseline residuals, or features. The next cheap check, before building training data, is a
probe of how predictable the weak-direction component of the error is from per-frame statistics (residuals, gradients,
confidence, tokens) on held-out scenes. If it isn't, the learned-step line stops here and the information has to come
from a new measurement (the wide-baseline term of open question 3 in `knowledge/README.md`).

## 9. Checks 1 and 2 (2026-11-22, same setting)

After G0 the candidate was to learn the *measurement* instead of the step: features trained through the solver (BA-Net,
Diff3R-style implicit gradients), on the assumption that the flat directions come from weak comparisons (short
baselines, grey intensities) that better features or more pairs would fix. Two checks before building it.

**Check 1: does more measurement stiffen the weak directions?** (`scripts/lpba_info.py`, 22 dev scenes, at B′'s final
state, Schur complement of the poses, information summed over visible warps, ratios to grey + 10 spread neighbours)

| objective | median cond | weak directions | strong directions | weak10 oracle AUC@3 |
|---|---|---|---|---|
| grey, 10 spread neighbours (B′) | 3.2e5 | 1 | 1 | 61.7 |
| grey, all covisible pairs | 2.0e5 | ×1.96 | ×1.37 | 61.1 |
| VGG, 10 spread | 1.7e5 | ×1.22 | ×0.66 | 61.3 |
| VGG, all pairs | 1.1e5 | ×2.00 | ×0.96 | 60.8 |
| grey, 10 spread, **depth grid fixed** | 7.5e4 | **×28.7** | ×1.18 | 60.0 |
| grey, 10 spread, grid and focal fixed | 7.5e4 | ×30.4 | ×1.18 | 58.3 |

Per dataset, fixing the grid stiffens the weak directions ×52 (HiRoom), ×36 (DTU), ×16 (ETH3D), but only ×3-6 on the
100-frame videos (7-Scenes, ScanNet++), where all pairs help more (×10-12, strong ×3-8). The direct all-pairs PBA run
ran out of memory on the 3080 for the 64/100-frame scenes; HiRoom 47.1 vs 43.8-45.5, ETH3D 12.7 vs 17.5-28.1.

So better features and more pairs buy ≤ 2x in conditioning; the flat directions mostly exist because a smooth per-frame
change of the depth correction can absorb a pose change. That is a geometric ambiguity of the model (pose ↔ coarse
depth), not a weakness of the comparison, and a learned feature can't remove it. In every variant the remaining error
stays in the weakest directions of the new matrix (weak10 oracle ~60 throughout).

**Check 2: is the weak-direction error predictable?** (`scripts/lpba_probe.py`, all 81 dev + held-out scenes, 4,123
frames; 40 per-frame features: update from DA3, poses relative to index neighbours, reduced gradient, block
eigenvalues, confidence, position, dataset)

| model | explained in-sample | explained 5-fold CV | AUC@3 5-fold CV |
|---|---|---|---|
| none (B′) | 0 | 0 | 43.6 |
| shrink (α · update) | 0.00 | 0.00 | 43.3 |
| ridge | 0.15 | −0.01 | 34.5 |
| boosted trees | 0.69 | 0.00 | 40.3 |
| oracle (= G0b) | 1 | 1 | 58.2 |

Leave-one-dataset-out is negative for every model. The trees memorise scenes but nothing transfers: there is no
systematic DA3/PBA bias in the weak directions that per-frame state statistics reveal. Untested: image-level inputs
(tokens, a CNN) and scene-level models (a GNN over frames).

**Conclusion.** Neither learned steps on the current state nor learned features look promising as the next
contribution: the missing information is about **depth** (or the pose ↔ depth trade-off), not about the comparison or
the update rule. The natural next oracle is PBA with correct depth (GT depth as depth0, or the grid fitted to GT), poses
free: if it closes most of the 20-point gap, the contribution is a better depth constraint (multi-view depth
consistency, a learned depth-uncertainty prior from DA3's confidence, or re-running the depth head on refined
cameras); if not, the gap is the photometric objective's bias itself ([objective-bias](../knowledge/ideas/objective-bias-gt-start.md)).

## 10. Depth oracle (2026-11-22)

Check 1 pointed at the depth: the weak directions are a pose ↔ depth trade-off. The oracle: run B′ (`iters=25`)
unchanged, but with the GT depth as depth0 (`scripts/lpba_depth_oracle.py`; one scale per scene to DA3's median depth, so
DA3's translations stay consistent; pixels without GT are not sampled). All scenes of the four datasets with GT depth
(DTU-64 has none), seeds 0 and 1, AUC@3:

| depth | HiRoom (30) | 7-Scenes (7) | ScanNet++ (20) | ETH3D (11) | mean w/o 7-Scenes |
|---|---|---|---|---|---|
| DA3 (control) | 50.1 | 26.8 | 33.6 | 30.8 | 38.2 |
| GT, grid free | 60.8 | 23.9 | 37.5 | 39.4 | 45.9 |
| GT, fixed (`depth_grid=0`) | 61.1 | 20.2 | 39.8 | 46.3 | **49.1** |
| reference: B′ + weak10 oracle (seed 0) | 64.2 | 54.1 | 46.0 | 38.2 | 49.5 (from 36.8) |

- **Correct depth recovers about the whole weak-direction headroom** (+10.9 vs +12.7) without touching the optimiser,
  the objective or the poses' initialisation. Dev and held-out agree (4-set mean: dev +4.1 / +8.8, held-out +5.1 /
  +5.3). On ETH3D the depth oracle even beats the weak10 oracle (46.3 vs 38.2): there, good depth also fixes errors in
  stronger directions.
- DA3-SMALL's depth is only 6.3 % off (abs. rel. after one scale per scene), but that is enough to cost ~11 points of
  AUC@3: at 3° the pose is sensitive to small, smooth depth distortions that the photometric term can trade against it.
- With correct depth, the free grid costs 3.2 points: the grid's freedom absorbs pose error (check 1).
- 7-Scenes falls (−2.9 / −6.6): its GT depth is registered to the depth camera and shifted 4-12 px against the RGB
  images (knowledge/evaluation.md), so it is not a valid oracle there.

**What it means.** The pose accuracy of PBA on a DA3-SMALL initialisation is limited by the depth, not by the optimiser,
the update rule or the comparison. That reframes the project question ("which initialisation errors can TTO correct?"):
TTO corrects the pose errors that are consistent with the predicted depth; the rest needs better depth. Directions:

1. **Matched compute, non-oracle:** a larger model's depth with SMALL's cameras, or a monocular depth model (one forward
   per frame, parallel) as depth0. Cheap: the predictions exist (`workspace/evaluation/DA3-LARGE-1.1`).
2. **Multi-view depth refinement inside the TTO** at a finer level than the 8-cell grid, constrained so that it can't
   absorb pose (e.g. a depth update only from the geometric/photometric terms of frames whose pose is held, or
   alternating).
3. **Learned:** a depth refiner that uses the multi-view residuals (Refin3R uses SfM points instead), trained so that
   PBA's poses become right: the bilevel idea of §9, but for the depth instead of the features.

## 11. Multi-view depth refinement by alternation (2026-11-22)

Direction 2 of §10, as a one-hour test: does refining the depth from the multi-view terms alone, with the poses held so
that it can't trade against them, make the depth better? `scripts/lpba_stage.py` starts each scene from a saved state
and runs one stage of `run_tto_pba.py`:

- depth stage: poses and focal fixed, a finer 32-cell log-depth grid, Adam on σ = 2 and 1 (no LM: the dense grid
  coupling of 32 cells doesn't fit), 4,096 samples per frame;
- pose stage: depth fixed (`depth_grid=0`), poses + focal + radial, Adam on σ = 2, 1, then LM.

All 81 pose scenes, seed 0, from the B′ (`iters=25`) states of the dev and held-out runs:

| stage | AUC@3 (5 sets) | HiRoom / ScanNet++ / ETH3D | depth abs. rel. HiRoom / ScanNet++ / ETH3D | extra TTO time |
|---|---|---|---|---|
| B′ | 43.6 | 49.8 / 33.2 / 27.3 | 5.38 / 5.95 / 5.16 % | — |
| pose stage only | 44.5 | 50.9 / 34.8 / 29.5 | unchanged | 0.40 s |
| depth 1 → pose 1 | 44.6 | 51.2 / 35.0 / 28.4 | 5.36 / 6.18 / 5.20 % | 1.4 s |
| depth 2 → pose 2 | 45.2 | 52.1 / 35.8 / 29.4 | 5.45 / 6.44 / 5.33 % | 2.8 s |

The depth doesn't get better: the depth stages leave HiRoom and ETH3D unchanged and make ScanNet++ worse. The AUC gain
(+1.6 after two rounds) comes from the extra pose iterations, and most of it (+0.9) from a single pose polish with the
depth fixed, which is cheap and could be added to B′ (one seed; needs seeds before adopting). Together with checks 1-2
and the depth oracle: the photometric/geometric objective contains the pose ↔ depth trade-off but not the information to
resolve it, so the better depth has to come from outside, either a stronger depth predictor (direction 1) or a learned
prior (direction 3).

## References (arXiv IDs)

DeepBA 2611.18256 · Refin3R 2611.23051 · PosePlaner 2610.34845 · Solv3R 2611.02857 · DéjàView 2605.30215 ·
BA-T 2606.03287 · Diff3R 2604.01030 · DeltaSplat 2610.09853 · GradSplat 2610.38951 · Learn2Splat 2605.15760 ·
DRO 2103.13201 · BA-Net 1806.04807 · DeepV2D 1812.04605 · DeepSFM 1912.09697 · GASfM 2308.15984 ·
PoseDiffusion 2306.15667 · BLASt3R 2609.05210 · GeoWeaver 2608.17389 · EPO 2607.00579 · Self-Geometry 2608.10708 ·
Test3R 2506.13750 · Pixel-Perfect SfM 2108.08291.
