# In-house measurements and code facts (coordinator, 2026-11-15)

Sources: `knowledge/README.md`, `knowledge/ideas/dejaview-steps.md`, `knowledge/ideas/distillation.md`, Antonin's
logs in `~/TUM-dvlt/dvlt_tto_fast_18*.log` (averaged by the coordinator), DéjàView code
`third_party/dvlt/src/dvlt/model/dvlt/model.py` and the HF checkpoint `nvidia/dvlt` config.json.

## DéjàView's loop is an interval-conditioned trajectory on t in [0, 1] (read in the code)

- Checkpoint config: `k_sampling="linspace"`, `time_conditioning="interval"`, `recurrence_mode="gated"`,
  `min_steps=8`, `num_steps=16`, `k_sampler_beta_a=2`, `k_sampler_beta_b=1`, `inference_steps=12`, DINOv2 ViT-B/14
  encoder (`dinov2_vitb14_reg`), embed_dim 768, img_size 504.
- Inference with K steps runs `ts = linspace(0, 1, K)` and calls one shared block K times with the pair
  `(t_now, t_next) = (ts[i], ts[i+1])`, the last call being the terminal sentinel `(1, 1)` (delta_t = 0). The block
  gets sinusoidal embeddings of both times (multiplicative "gated" state update, step-conditioned depth scaling).
- Training samples K ~ Beta(2, 1) scaled to [8, 16] (biased towards 16) on the same uniform grid, with
  stochastic depth 0.3. So trained interval sizes are 1/15 to 1/7, and **K < 8 is outside the training
  distribution**: the collapse below K = 6 is extrapolation to unseen large intervals, not an inherent limit.
- Consequence: the loop is structurally like a few-step ODE/flow sampler whose step is conditioned on (t, t + d).
  This is exactly the conditioning of shortcut models (Frans et al., "One Step Diffusion via Shortcut Models",
  conditioning on step size d) and of flow maps / consistency trajectory models: K = 2-4 inference only needs the
  shared block to learn larger intervals, e.g. by self-distillation "one step of 2d = two steps of d" with the
  frozen K = 12/16 trajectory as teacher. No new architecture is needed; intermediate states along the trajectory
  are not decoded during training (only the final state is decoded).
- Encoder alone ~0.4 s of the 0.96 s K = 12 forward (100 frames, RTX 5090), so each loop step is ~0.045 s; K = 4
  would cost ~0.56 s.

## DéjàView K sweep (full DA3 benchmark, RTX 5090; mean pose AUC@3 over 5 datasets)

| K | 1 | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|---|
| AUC@3 | 0.0 | 0.2 | 2.1 | 9.0 | 31.1 | 47.6 | 58.7 |
| AUC@30 | 5.1 | 23.9 | 52.5 | 72.0 | 87.9 | 91.9 | 94.0 |
| F1 (4 sets) | — | — | 17.6 | 32.6 | 54.3 | 63.4 | 67.7 |
| forward s | 0.42 | 0.47 | 0.52 | 0.56 | 0.66 | 0.76 | 0.96 |

DA3 on the same machine (pose only): SMALL 14.4 (0.37 s), BASE 27.3 (0.76 s), LARGE-1.1 51.3 (1.54 s), GIANT-1.1 67.0
(2.85 s). DéjàView K = 8 ≈ DA3-LARGE-1.1 at half the time.

Note: AUC@30 stays high (72 at K = 4) while AUC@3 collapses: small-K outputs are roughly right but imprecise,
i.e. coarse-to-fine. Also: K < 8 is very sensitive to image resampling (cv2 resize moved AUC@3 by ~10 points).

## Photometric TTO on DéjàView outputs (dev subset, photometric BA T = 25, ~0.3-1 s; AUC@3 before -> after)

Averaged per dataset by the coordinator from the per-scene lines of Antonin's logs (official dev set: HiRoom 10,
7-Scenes 2, ScanNet++ 3, DTU-64 4, ETH3D 3; dataset-mean). Corrected after the report draft: each log holds the
photometric pass (`pho_it25`) and then the VGG feature-metric pass (`fmh_it25`); an earlier version of this table
averaged both together.

Photometric (`pho_it25`):

| K | HiRoom | 7-Scenes | ScanNet++ | DTU-64 | ETH3D | mean | mean TTO s |
|---|---|---|---|---|---|---|---|
| 12 | 65.2 -> 73.6 | 41.5 -> 38.6 | 65.3 -> 72.6 | 75.2 -> 94.2 | 33.6 -> 74.0 | 56.2 -> 70.6 | 0.25 |
| 8 | 51.5 -> 70.9 | 43.0 -> 39.0 | 49.4 -> 68.7 | 45.4 -> 92.0 | 29.9 -> 65.7 | 43.8 -> 67.3 | 0.25 |
| 6 | 32.6 -> 60.3 | 35.2 -> 38.8 | 25.4 -> 55.5 | 28.5 -> 87.2 | 15.7 -> 48.1 | 27.5 -> 58.0 | 0.23 |
| 4 | 9.5 -> 29.4 | 10.8 -> 36.0 | 3.7 -> 21.9 | 10.4 -> 65.4 | 4.2 -> 13.1 | 7.7 -> 33.2 | 0.32 |
| 3 | 2.2 -> 17.3 | 3.5 -> 25.6 | 0.8 -> 4.6 | 2.4 -> 53.1 | 1.2 -> 4.1 | 2.0 -> 20.9 | 0.36 |

VGG + Huber (`fmh_it25`), mean only: K = 12 56.2 -> 71.6 (1.41 s), 8 -> 67.3 (0.85 s), 6 -> 55.1 (0.63 s),
4 -> 29.4 (0.78 s), 3 -> 17.4 (0.84 s).

For reference, DA3 + the same kind of TTO on dev: SMALL 16.2 -> 43.8, BASE 27.1 -> 59.2, LARGE-1.1 48.6 -> 71.0,
GIANT-1.1 67.8 -> 74.9 AUC@3. So DéjàView K = 12 + TTO ≈ DA3-LARGE + TTO, and K = 8 + TTO (67.3 at ~1 s total) is
close to it. TTO's basin starts around K ≈ 6; at K ≤ 4 the init is outside it (except DTU). Track-based BA
(RoMa tracks) on DéjàView K = 12 also ran (Antonin, ~14-25 s per scene), not summarised here.

## Earlier distillation attempt (2026-10-28/29, branch origin/distillation-test, scripts/distill/)

- Students `dvlt:<K>` (e.g. K = 6) with the DINOv2 encoder and camera/register tokens frozen, teacher `dvlt:12`,
  optional feature matching of student step tokens to teacher step tokens (matched by schedule time, or student
  step i <- teacher step 2i+1); also DA3 students with DéjàView/DA3/GT teachers. lr 3e-5, 24-32 views, ~600
  batch-1 steps, 10 ScanNet++ scenes (benchmark scenes split even/odd).
- Outcome: all students data/optimisation limited; training on 8-16-view subsets did not transfer to the 100-frame
  benchmark. Numbers stayed on tum1 and are not available. It was not framed as interval/shortcut distillation,
  and it trained on benchmark scenes.

## What our TTO knows (relevant to "TTO-aware" ideas)

- Photometric BA optimises poses, a shared focal, an 8-cell log-depth grid per view and a radial coefficient
  (Adam on 4 blurred levels, then LM). Best pose: LM on NCC; best F1 on large models: VGG features + Huber.
- DA3's pose error lies in the weak (flat) directions of the photometric objective; gross failures depend on the
  reference view; correspondence terms inside PBA hurt; re-running DA3 on refined cameras ("re-conditioning")
  did not help; DA3/DINO tokens as features did not help.

## Probe submitted (tum2 job 2832, may not have run before the report)

Training-free schedule probe through `run_bench.py` with a patched step grid (script in
`/scratch/vc17/pr26rb00/jobs/dvlt_probe/`, outputs `workspace/evaluation/dvlt_probe/<name>/`): K = 6 uniform
(control), K = 6 with a big last interval (0, .1, .2, .3, .5, 1), K = 6 with a big first interval (0, .5, .7,
.8, .9, 1), K = 16 (top of the trained range), K = 24 (beyond), K = 8 uniform (control), K = 8 with six trained-size
intervals then one jump. Question: where along t does the loop tolerate large intervals, and does K > 12 help.
