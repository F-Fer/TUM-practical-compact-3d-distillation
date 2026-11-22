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

## References (arXiv IDs)

DeepBA 2611.18256 · Refin3R 2611.23051 · PosePlaner 2610.34845 · Solv3R 2611.02857 · DéjàView 2605.30215 ·
BA-T 2606.03287 · Diff3R 2604.01030 · DeltaSplat 2610.09853 · GradSplat 2610.38951 · Learn2Splat 2605.15760 ·
DRO 2103.13201 · BA-Net 1806.04807 · DeepV2D 1812.04605 · DeepSFM 1912.09697 · GASfM 2308.15984 ·
PoseDiffusion 2306.15667 · BLASt3R 2609.05210 · GeoWeaver 2608.17389 · EPO 2607.00579 · Self-Geometry 2608.10708 ·
Test3R 2506.13750 · Pixel-Perfect SfM 2108.08291.
