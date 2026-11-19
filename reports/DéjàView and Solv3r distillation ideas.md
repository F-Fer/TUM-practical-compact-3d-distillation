# Shorten DéjàView's loop and let BA finish

Our best bet is to teach DéjàView's tiny interval gates to take big steps, and to train them so that the photometric BA can finish the job, not so that they copy the teacher. The released checkpoint was only ever trained with 8 to 16 loop steps on a `linspace` grid over t ∈ [0, 1] (step sizes 1/15 to 1/7). Our collapse below K = 8 is therefore an extrapolation in step size. A fine-tune of the 0.32M gate parameters, a LoRA or the 14.5M shared block can learn larger steps from self-generated targets, without ground truth. Ranked by novelty, cost and fit, the directions are:

1. **TTO-aware few-step distillation of DéjàView**: targets from K = 16 + BA, and a loss that ignores the errors BA removes anyway.
2. **Solv3R with our BA as an external verifier and competing iteration axis**: training-free first.
3. **Heads trained through the BA's own Gauss-Newton system**, so they correct only the directions BA cannot.
4. **Latent-space TTO** through the frozen decoders.
5. **Per-scene allocation** of loop steps versus BA iterations.

Three facts change the setup before any training:

- **The paper runs DéjàView at K = 16, not at the K = 12 config default we benchmarked.** In the paper's ablation, 12 → 16 is worth +3.7 AUC@3.
- **The prize from fewer steps is small in seconds.** A loop step costs about 0.05 s next to a fixed ~0.42 s encoder, so fewer steps save at most ~0.4 s per scene. The real value is moving K = 4-6 into BA's basin.
- **Solv3R (2 Nov 2026) is a fine-tuned DéjàView with released weights.** Its evaluation code pins our exact `dvlt` commit, so it is a cheap baseline. It is also a direct competitor that has already claimed adaptive halting.

By my estimate the whole programme fits in roughly 150-300 of our 100-500 GPU-hours. Week 1 should stay training-free: read the pending schedule probe, re-baseline at K = 16, add Solv3R and measure BA's basin. The first 6-hour gate-only training job should run in week 2, and only if the probes show that the collapse is about step size.

## DéjàView collapses below K = 8 because it never saw big steps

DéjàView's loop is not a stack of identical refinements. It is an interval-conditioned trajectory over a fixed horizon. At inference the code builds `ts = linspace(0, 1, K)`, applies the single shared frame-plus-global attention block to each pair (ts[i], ts[i+1]) and ends with a zero-width terminal step (1, 1) ([model.py](../third_party/dvlt/src/dvlt/model/dvlt/model.py)). So K = 8 takes steps of 1/7 and K = 16 steps of 1/15, while K = 6, 4 and 3 take steps of 1/5, 1/3 and 1/2. Training drew K per batch from Beta(2, 1) scaled into [8, 16] (mean ≈ 13.3), supervised only the final state and used no teacher ([DéjàView](https://arxiv.org/abs/2605.30215)). **Every K below 8 therefore queries the step conditioning at intervals it has never seen.**

Only two zero-initialised gate MLPs see (t_now, t_next), about 0.32M parameters in total. They output channel-wise scales for the attention residual, the MLP residual and the carried state ([depth_scaling.py](../third_party/dvlt/src/dvlt/model/dvlt/depth_scaling.py)). Shortcut models use the same kind of conditioning: they learn large jumps by self-distillation of the form "one step of 2d equals two steps of d" ([Shortcut models](https://arxiv.org/abs/2410.12557)). The closest existence proof from another domain is "Depth as time" ([Depth as time](https://arxiv.org/abs/2610.03626)):

- It compressed a MeanFlow image transformer into one interval-conditioned block, trained data-free from the teacher's hidden states.
- With half as many applications it reached FID 5.25, against 4.01 for the teacher.
- Removing the interval conditioning more than doubled FID.

The open empirical question is whether DéjàView's loop needs some minimum number of block applications for what the authors call its "emergent iterative correspondence search" ([DéjàView](https://arxiv.org/abs/2605.30215)).

**Our DéjàView baseline is also four steps short of the paper's operating point.** The paper states "We run K_inf = 16 steps at inference". The Hugging Face config and the `DVLT` class default to `inference_steps = 12`, and our adapter uses that default ([baselines.py](../scripts/baselines.py)). The repo's own Hydra path, used by its test script and demo, falls back to 16 ([model.py](../third_party/dvlt/src/dvlt/model/dvlt/model.py)). Three results from the paper and from Solv3R bear on this:

- In the paper's 100K-iteration ablation, the variable-K model scores AUC@3 29.6 at K_inf = 12 and 33.3 at K_inf = 16 ([DéjàView](https://arxiv.org/abs/2605.30215), Table 5).
- A full K-step schedule beats early-stopping a 16-step run: AUC@3 0.44 vs 0.31 at K = 8 (same paper, supp. E).
- Beyond 16 steps the model degrades. Solv3R measures DéjàView at AUC@5 76.5 after 16 calls and 0.0 after 128 ([Solv3R](https://arxiv.org/abs/2611.02857)).

From our sweep, four extra steps cost about 0.2 s ([dejaview-steps](../knowledge/ideas/dejaview-steps.md)). If the probe confirms the gain on the DA3 benchmark, K = 16 should become our DéjàView baseline and the distillation teacher.

The released model constrains how we can train it:

- **Readouts.** The loop has no input injection, so there is no channel to feed cameras or depth back into it. Inference poses come from a RANSAC fit to the decoded rays inside `torch.no_grad`, while the camera MLP head is differentiable. `compute_world_rays` turns any camera set, including BA-refined ones, into a ray-loss target ([model.py](../third_party/dvlt/src/dvlt/model/dvlt/model.py)).
- **Training code.** The release contains the trainer, all losses and `finetune_mode` switches, but only a ScanNet++ data loader. The other loaders depend on an unreleased package ([dvlt README](../third_party/dvlt/README.md)). Self-distillation needs only images, so a minimal image-folder loader is enough.
- **Lessons from October.** Students trained on 8-16-view subsets did not transfer to the 100-frame benchmark. At K < 8 DéjàView is so sensitive to resampling that a cv2-resized cache moved AUC@3 by about 10 points. That attempt also trained on 10 benchmark ScanNet++ scenes, which must not happen again ([distillation](../knowledge/ideas/distillation.md)).
- **Cost.** The DINOv2 ViT-B encoder holds 86.6M of the 117M parameters and costs ~0.42 s (the K = 1 forward time). Each loop step costs about 0.05 s on the RTX 5090 ([dejaview-steps](../knowledge/ideas/dejaview-steps.md)). Going from K = 12 to K = 4 saves ~0.4 s, and going from K = 6 to K = 4 saves only ~0.1 s. An analytic FLOP count in our notes suggests that a step's share grows with the number of views (≈32 TFLOPs per step vs ≈20 for the encoder at 100 views), so the median understates the savings on 100-frame scenes. That count is an estimate, not a measurement.

| Model | Forward s (full bench, 5090) | AUC@3 full bench | AUC@30 full bench | Dev AUC@3 alone → + PBA T = 25 |
|---|---|---|---|---|
| DéjàView K = 3 | 0.52 | 2.1 | 52.5 | 2.0 → 20.9 (TTO 0.36 s) |
| DéjàView K = 4 | 0.56 | 9.0 | 72.0 | 7.7 → 33.2 (0.32 s) |
| DéjàView K = 6 | 0.66 | 31.1 | 87.9 | 27.5 → 58.0 (0.23 s) |
| DéjàView K = 8 | 0.76 | 47.6 | 91.9 | 43.8 → 67.3 (0.25 s) |
| DéjàView K = 12 | 0.96 | 58.7 | 94.0 | 56.2 → 70.6 (0.25 s) |
| DA3-SMALL | 0.37 | 14.4 | — | 16.2 → 43.8 (0.71 s total) |
| DA3-BASE | 0.76 | 27.3 | — | 27.1 → 59.2 (1.11 s) |
| DA3-LARGE-1.1 | 1.54 | 51.3 | — | 48.6 → 71.0 (1.83 s) |
| DA3-GIANT-1.1 | 2.85 | 67.0 | — | 67.8 → 74.9 (3.32 s) |

Sources: the DéjàView sweep and DA3 forward times are from [dejaview-steps](../knowledge/ideas/dejaview-steps.md). DéjàView + PBA are Antonin's logs averaged per dataset ([local measurements](../research_notes/D%C3%A9j%C3%A0View%20and%20Solv3r%20distillation%20ideas/local_measurements.md)). DA3 + PBA is from [matched-compute](../knowledge/ideas/matched-compute.md).

The table shows a coarse-to-fine loop. At K = 4, AUC@30 is still 72 while AUC@3 is 9, so small-K outputs are roughly right but imprecise. **BA's basin starts around K = 6**:

- K = 6 + PBA (58.0, ~0.9 s total) is slightly above K = 12 alone (56.2, 0.96 s).
- K = 8 + PBA reaches 67.3 at roughly 1 s. That is above DA3-BASE + PBA (59.2 at 1.11 s) and close to DA3-LARGE + PBA (71.0 at 1.83 s).
- At K ≤ 4, BA mostly rescues only DTU (10.4 → 65.4 at K = 4); ETH3D stays at 13.1 ([local measurements](../research_notes/D%C3%A9j%C3%A0View%20and%20Solv3r%20distillation%20ideas/local_measurements.md)).
- The feature-metric variant (VGG + Huber, same logs) is no better on DéjàView at 3-6x the TTO time: K = 12 71.6 (1.41 s TTO), K = 8 67.3, K = 6 55.1, K = 4 29.4, K = 3 17.4.

Four caveats apply before these DéjàView + PBA numbers are quoted:

- They are read from the per-scene log lines of Antonin's `slurm_dvlt_tto_fast.sh` runs (official dev set, 10/2/3/4/3 scenes; each log holds the photometric pass first, then the VGG pass), not from the evaluator's metric files. Rescore them with `run_tto_dev.sh` / the official evaluator before quoting.
- 7-Scenes does not move under any TTO, because of a ground-truth problem.
- ETH3D has only 3 dev scenes but carries a large share of every TTO gain.
- Several dev gains have not survived the held-out set ([evaluation](../knowledge/evaluation.md)).

### Solv3R is a fine-tuned DéjàView we can run this week

Solv3R (Kwak, Tagliasacchi, Kagami, Ono, Yi; UBC, Wayve, SFU, Toronto, Sony; arXiv 2611.02857, 2 Nov 2026, no venue yet) turns DéjàView's block into a *stationary* solver ([Solv3R](https://arxiv.org/abs/2611.02857)):

- **Architecture.** Constant gates (initialised from the average of DéjàView's 16 step gates) replace the step-conditioned ones. The encoded images z_0 are re-injected at every call through a zero-initialised linear map, and the state is rescaled to a fixed RMS.
- **Frozen parts.** The encoder and heads are DéjàView's, frozen and therefore bit-identical, so our DéjàView adapter and its preprocessing carry over.
- **Training.** 16-40 calls run without gradients, then 8 differentiable calls are supervised at the last one. With probability 0.15 the state is first perturbed by noise or view resets. A distillation loss on DéjàView's 16-step outputs is added. The full run took about a week on 8×H100, roughly 1,350 H100-hours, and the 3k-step ablations already changed the dynamics measurably.
- **Release.** Inference code, evaluation code and weights came out on 5 Nov 2026: weights `jgkwak/solv3r` under the NVIDIA non-commercial research licence, code under Apache-2.0. Training code is neither released nor on the TODO list. The `eval` extra pins `dvlt` at commit `f59bb18`, exactly our submodule ([Solv3R repo](https://github.com/ubc-vision/solv3r), [weights](https://huggingface.co/jgkwak/solv3r)).

**What Solv3R wins, and what it leaves open.** At 16 calls Solv3R roughly matches DéjàView. Adaptive stopping (relative state update below 0.02) averages 17.5 calls and lifts ETH3D AUC@3 from 66.0 to 70.4, at 91.3 instead of 75.9 TFLOPs ([Solv3R](https://arxiv.org/abs/2611.02857)). Solv3R therefore spends *more* compute than DéjàView, includes no explicit optimiser and no comparison with BA, and reports no table below 16 calls.

The one failure mode the authors name is a wrong but stable fixed point: views with little overlap sit at 60° rotation error after 4 calls and are still at 57° after 64. Its experiments also break DéjàView in three ways:

- Perturbing the state at call 8 leaves rotation errors of 6.3-15.6°, against 0.3-1.3° for Solv3R.
- Warm starts diverge.
- With views at mixed refinement stages, AUC@3 falls to 58.7 or 38.8, against 83.6 for Solv3R ([Solv3R](https://arxiv.org/abs/2611.02857)).

So any plan that writes BA results back into DéjàView's state is dead on arrival without retraining. Solv3R's tables use the DéjàView benchmark protocol (for example every 11th 7-Scenes frame), not the DA3 protocol, so we have to run it through our own adapter before comparing anything.

## The open niche is few learned steps followed by explicit optimisation

Most of the neighbouring ground is taken:

- **Looping as a compute knob** belongs to DéjàView, and the **stationary solver with adaptive halting, warm starts and streaming** to Solv3R. Post-hoc looping of a trained ViT is RAPTOR, cited by [DéjàView](https://arxiv.org/abs/2605.30215).
- **Compact students distilled from scratch** stay far from their teachers out of distribution. OTT3R spent ~384 GPU-hours distilling π³ into 102M parameters. Its general student reaches 10.16/34.84 cm accuracy/completion zero-shot on 7-Scenes, against 1.52/2.03 for the teacher, and the authors write that it "does not replace the teacher on out-of-distribution multi-view geometry" ([OTT3R](https://arxiv.org/abs/2609.36374)).
- **Token merging** buys nothing at our sizes: FastVGGT takes 8.15 s against VGGT's 8.12 s at 100 frames ([GoToHunt](https://arxiv.org/abs/2605.23892)).
- **Weight-space test-time adaptation** is crowded and weak at matched compute. Self-Geometry moves DA3-SMALL's mean AUC@3 from 0.10 to 0.12 at 0.3-2.5 minutes per scene ([Self-Geometry](https://arxiv.org/abs/2608.10708)). Our output-space BA moves dev AUC@3 from 16.2 to 43.8 in about 0.3 s ([matched-compute](../knowledge/ideas/matched-compute.md)). The protocols differ, but the gap is two orders of magnitude.

What remains open is the *interface* between a short learned loop and a classical optimiser:

- **No few-step distillation of a looped 3D model exists.** Shortcut-style consistency inside looped transformers has only been done for language models trained from scratch. LoopFormer conditions each loop on (t, Δt) and aligns short and full trajectories ([LoopFormer](https://arxiv.org/abs/2602.11451)).
- **No distillation objective targets a downstream optimiser's basin.** The closest is Diff3R, which trains a feed-forward 3DGS model, including a pose-free DA3 variant, through its test-time optimisation with implicit differentiation, but for novel-view synthesis rather than cameras and depth ([Diff3R](https://arxiv.org/abs/2604.01030)).
- **No paper plots learned iterations against explicit BA iterations at matched time.** Solv3R also never asks whether a BA residual can catch its sticky fixed points.

Evidence from flow and SLAM supports coupling the two kinds of iteration:

- DROID-SLAM found that training without its BA layer and adding BA only at inference makes the system "unstable and prone to failure" ([DROID-SLAM](https://arxiv.org/abs/2108.10869)).
- BA-T found that an explicit latent residual is the dominant factor: its crudest residual recovers 76% of the rotation gain, and BA-T beats DROID-SLAM's classical BA from the same initialisation (rotation AUC@5 0.68 vs 0.53, 31 ms vs 161 ms) ([BA-T](https://arxiv.org/abs/2606.03287)).
- PosePlaner's learned refiner lifts weak initialisations (rotation AUC@5 11.5 → 29.5, where BA reaches 13.4) but gives nothing on top of accurate ones ([PosePlaner](https://arxiv.org/abs/2610.34845)). Our K-sweep shows the same pattern.

Concurrent work is the main threat. Self-Geometry, PURE, Solv3R, PosePlaner, BLASt3R and CSS-BA all appeared between September and November 2026, so the novelty claims below must be re-searched before we commit to a submission.

## Five directions, ranked by novelty, cost and fit

| Rank | Direction | What is trained, on what | GPU-h (my estimate) | Closest prior work and the gap | Cheapest first experiment |
|---|---|---|---|---|---|
| 1 | TTO-aware few-step DéjàView | Interval gates (0.32M) → LoRA → block (14.5M); encoder and heads frozen; unlabeled clips outside the benchmark, targets from K = 16 and K = 16 + BA | 60-150 | DéjàView trains only K ∈ [8, 16]; Solv3R distils K = 16 for stationarity, not fewer steps; shortcut training exists only for LMs and images; no basin-targeted distillation | Job 2832 grids, truncated K = 8 grid, K = 16 re-baseline, basin analysis |
| 2 | Solv3R calls vs BA iterations, BA as verifier | Nothing at first; later a zero-initialised camera → state injection | 5-60 | Solv3R has no optimiser and no BA comparison; DROID-SLAM and BA-T couple learned and BA steps but are trained jointly from scratch | Solv3R adapter, calls × T frontier, reset the worst views by BA residual |
| 3 | Heads trained through the BA (weak subspace, implicit gradient) | Per-frame prior weights, corrective depth channels or a feature head on frozen DA3/DéjàView features; GT train scenes or GIANT + BA | 30-60 | Diff3R (3DGS novel-view synthesis only), BLASt3R (depth basis trained without BA), Selfi (track-BA features); nothing targets an optimiser's weak directions | Oracle swap of poses in the weak subspace; per-frame prior from DA3 confidence |
| 4 | Latent-space TTO through the frozen decoders | Nothing | 5-15 | Measurement guidance in diffusion samplers and PoseDiffusion; Test3R tunes weights; never on a deterministic looped regressor | 10-20 Adam steps on Solv3R's state at call 16 vs PBA at matched time |
| 5 | Per-scene allocation of K vs T | Nothing, or a logistic regression | < 5 | Solv3R halts on its latent update norm; no joint allocation between learned and classical iterations | Oracle K* from existing per-scene runs |

### 1. Train the gates for big steps, and train them for the BA basin

**Hypothesis and novelty.** The collapse below K = 8 is mainly the gates extrapolating to unseen step sizes. If we train the interval conditioning on coarse intervals, K = 3-6 lands inside the photometric BA's basin, and K ≈ 4 + BA approaches K = 16 + BA. In time, that is about 0.86 s against an estimated 1.45 s (K = 16 forward ≈ 1.15 s plus ~0.3 s of BA). It would also beat every DA3 size + BA at equal time.

The stronger, more novel version changes the target. The student does not try to match the network teacher. It is trained towards *teacher + BA* outputs, with a pose loss that is zero below the BA's basin radius and steep above it. Capacity then goes into "roughly right everywhere", and BA does the sub-degree work.

Closest prior work:

- DéjàView itself only covers K ∈ [8, 16].
- Solv3R distils DéjàView-16 to make it stationary, at more calls, not fewer.
- LoopFormer and "Depth as time" show that interval and shortcut training works in language models and image generators.
- Nothing distils a looped 3D model to fewer steps, and nothing trains any predictor for a downstream optimiser's basin.

**What is trained, on what data and at what cost.** Use a ladder that escalates only when the previous rung fails, keeping the encoder (under `no_grad`) and heads frozen throughout:

1. The two gate MLPs (0.32M parameters).
2. The gates plus LayerScale.
3. A LoRA on the block's attention and MLP.
4. The full 14.5M block.

Two losses come first:

- **Endpoint distillation** on decoded rays (36×27 per view, cheap and linear) and depth, or on the final tokens.
- **Shortcut consistency**: one coarse student step against two finer teacher steps.

Keep a slice of K ∈ [8, 16] in every batch so the existing behaviour survives. Mix K values within one optimiser update by gradient accumulation: ELR found that sampling one synchronised K per step falls short of the baseline ([ELR](https://arxiv.org/abs/2610.12582)). Results on language models suggest that depth elasticity can be added cheaply as a short finishing stage on a trained model ([Adaptive-depth finishing stage](https://arxiv.org/abs/2610.14252)).

The second stage holds the novelty. Its targets become K = 16 + PBA:

- BA cameras are turned into ray targets with `compute_world_rays`.
- Depth is the network depth times the refined log-depth grid.
- Targets are filtered by photometric residual, because BA labels inherit BA's failures (7-Scenes K = 12 goes 41.5 → 38.6 under PBA).
- The pose loss is truncated at the radius that the basin analysis measures, or weighted by idea 3's implicit gradient.

The validation metric is AUC@3 *after* PBA, at matched total time, on held-out scenes.

**Data and compute.** Use unlabeled clips of 24-32+ views from outside the benchmark: ScanNet++ v2 train minus the 20 benchmark IDs, ARKitScenes, ScanNet v1 and DL3DV, plus BlendedMVS or MegaDepth for outdoor scenes. Exclude 7-Scenes, ETH3D, DTU and HiRoom entirely ([constants.py](../third_party/Depth-Anything-3/src/depth_anything_3/utils/constants.py)). Always use DéjàView's own preprocessing and turn off `drop_path`.

My estimates for compute:

- **Teacher cache.** Labels cost roughly 1.5-2 s of GPU time per scene, plus ~1.7 s of image decoding that our timings do not count ([matched-compute](../knowledge/ideas/matched-compute.md)). That is on the order of 1,000 scenes per 5090-hour.
- **Memory.** A fine-tune step at 32 views with a frozen encoder needs an estimated 8-12 GB. This is not yet validated.
- **Training.** A gate-only run of a few thousand steps fits one 6-hour job. LoRA or full-block runs at Solv3R's ablation scale (3k steps) take several jobs.

That puts the whole direction at 60-150 GPU-hours.

**Cheapest first experiment, risk and fit.** The pending probe (tum2 job 2832) is exactly the right first step. It compares, all training-free:

- uniform K = 6 with a big-last-interval grid (0, .1, .2, .3, .5, 1) and a big-first-interval grid (0, .5, .7, .8, .9, 1);
- K = 16 and K = 24;
- uniform K = 8 with an eight-point grid whose six steps stay in the trained range and whose last step is one large jump.

Two further training-free probes cost minutes on the dev set: running the K = 8 grid but stopping after m = 3/4/6 steps before the terminal step, and LoopCD-style extrapolation z_K + ω(z_K − z_{K−1}). LoopCD matched full depth at half the loop iterations in language models ([LoopCD](https://arxiv.org/abs/2610.02185)).

Two outcomes are possible:

- If grids that keep the trained interval sizes clearly beat the uniform grid at the same number of block applications, the collapse is a conditioning problem, and gates-only training is the right adapter.
- If they don't, the number of applications is what matters. Then target K = 6 rather than K = 3-4, and start at the LoRA rung.

The risks are real:

- **Language-model evidence is mixed.** LoopFormer beat early-exit baselines at a 12× budget but lost at the most aggressive 6× budget: perplexity 14.30 against 12.61 for naive early exit ([LoopFormer](https://arxiv.org/abs/2602.11451)).
- **There may be a floor.** Theory on looped depth extrapolation says the lower end of the useful window is set by the hardness of the instance ([Yang et al.](https://arxiv.org/abs/2610.14456)).
- **The payoff is capped** at about 0.4 s per scene.
- **Others may get there first.** NVIDIA or the Solv3R group could publish few-step variants.

The fit with our project question is direct. The project asks whether a small model + TTO can replace a large model at matched compute, and this idea makes the loop count a model-size axis below 8 without training a new model.

### 2. Pit Solv3R's learned calls against BA iterations, then let BA police it

**Hypothesis and novelty.** Learned solver calls and explicit BA iterations are complementary, and BA can detect and repair Solv3R's sticky fixed points. Each call is the same block as a DéjàView step, so it should cost ≈0.05 s. Our whole photometric BA (0.2-0.3 s) then equals roughly 4-6 calls, which gives a concrete matched-compute question: does k calls + BA beat k + 5 calls? Separately, the per-view photometric residual after BA should flag views stuck at a wrong solution. Resetting those views to z_0 is Solv3R's own "continuation" property, and re-solving them should recover them.

Solv3R advertises that it needs no separate optimisation backend and never compares against BA. DROID-SLAM and BA-T couple learned and classical steps only in systems trained jointly from scratch. A stationary multi-view foundation model steered by an external BA outer loop is not in the literature we found.

**What is trained, on what data and at what cost.** At first, nothing. Inference should take about 4 GiB at 100 frames, the same block as our DéjàView adapter at 3.9 GiB, so it fits the RTX 3080 and 3070. The sweeps cost a few GPU-hours.

If the training-free outer loop helps, a small training step follows: a zero-initialised Plücker or ray embedding of BA-refined cameras, added to Solv3R's camera tokens. It would be fine-tuned with Solv3R-style perturbation training for a few thousand steps on cached z_0, using the same unlabeled clips as idea 1. The training loop must be re-implemented from the paper's supplement, since the training code is unreleased. My estimate is 20-50 GPU-hours.

**Cheapest first experiment, risk and fit.** The steps, in order:

1. Add `third_party/solv3r` as a pinned submodule and write an adapter copied from the DéjàView one (same preprocessing, cameras fitted to the rays).
2. Check that 16 calls reproduce DéjàView K = 16 on the DA3 benchmark.
3. Sweep calls ∈ {4, 8, 12, 16, 24, 32, adaptive} on dev, with and without PBA, and plot the result with `plot_compute.py`.
4. On low-AUC dev scenes, reset the 10-25% of views with the worst photometric residuals, run 8 more calls and compare against "8 more calls without reset".

The main risks:

- **A tie.** If BA adds the same amount on top of Solv3R as on top of DéjàView, the contribution shrinks to a strong baseline.
- **Unknown short-call quality.** Below 16 calls, Solv3R's quality is unreported.
- **The wrong direction for our theme.** Adaptive Solv3R spends more compute, not less.
- **Resets may not help.** Reset views may fall back into the same wrong basin.

The fit is strong anyway. Solv3R is the only looped model where "more learned iterations" does not degrade past the trained range. That gives the matched-compute plot a clean "more calls" line next to "more BA iterations". A SIGGRAPH reviewer will expect Solv3R as a baseline regardless.

### 3. Teach heads only what the BA cannot fix

**Hypothesis and novelty.** BA fixes errors in stiff directions and leaves the flat ones. On DA3 the photometric objective has a condition number of ~6e4. 42-76% of DA3-SMALL's pose error lies in the weakest 10% of eigen-directions, and 60-90% of what remains after BA sits there ([flat directions](../knowledge/ideas/pose-prior-and-flat-directions.md)). Diff3R's implicit gradient for a proximally regularised inner problem is Λ(H + Λ)⁻¹. It is ≈0 along stiff directions, where the optimiser fixes the error regardless of the start, and ≈1 along flat ones ([Diff3R](https://arxiv.org/abs/2604.01030)). Training heads with the post-BA loss through this gradient therefore spends network capacity exactly where BA cannot act.

The same machinery supports three companion heads:

- **Learned per-frame pose-prior weights.** A global prior helped LARGE (dev 56.7 → 60.2) and hurt SMALL (31.4 → 30.8) ([flat directions](../knowledge/ideas/pose-prior-and-flat-directions.md)).
- **Corrective depth channels** in place of the 8-cell grid. BLASt3R reports that 16 channels instead of 1 improve pointmap accuracy after BA by 30-40% ([BLASt3R](https://arxiv.org/abs/2609.05210)).
- **A feature head trained for the optimiser** in place of PCA-VGG. PixLoc's ablation moves from 1.2/3.3 with photometric alignment to 59.8/79.0 with features, confidences and damping trained through LM ([PixLoc](https://arxiv.org/abs/2103.09213)).

Nothing we found trains a multi-view camera/depth predictor in its optimiser's weak subspace.

**What is trained, on what data and at what cost.** Only small heads, on frozen DA3 or DéjàView features. On DéjàView the camera head is the differentiable readout. The implicit gradient costs one extra solve with the damped normal matrix that `run_tto_pba.py` already builds analytically.

The targets must be accurate in the weak directions. GT train scenes are the clean choice: ScanNet++ train, ARKitScenes, or synthetic Hypersim and TartanAir. GIANT + long BA (74.9-77.2 dev AUC@3) is weaker, because its errors also live in weak directions, and self-distilling from our own BA cannot teach them at all. My estimate is 30-60 GPU-hours, at most a day per head on one 5090.

**Cheapest first experiment, risk and fit.** The first test is an oracle with no training. Replace DA3-SMALL's poses (or DéjàView K = 4-6's) by GIANT's or GT's *only* inside the BA Hessian's weak subspace, run BA, and measure AUC@3. The `weak_modes.py` tool at commit `1c1a921` does the projection. If the oracle does not approach teacher + BA, stop. The second test is a non-learned per-frame prior strength from DA3 confidence, which separates gross-failure frames at AUROC ~0.90 ([gross failures](../knowledge/ideas/gross-failures-and-reference-view.md)).

The risks:

- Weak-direction errors may come from global ambiguity that image features cannot predict.
- The implicit gradient assumes a converged LM solution, which our Adam-then-LM schedule with a stop rule may not reach.
- Fixing gross frames by oracle is worth only +1.5 AUC@3 on SMALL dev, which caps one sub-variant.

The fit is excellent. It improves "small model + TTO" at zero test-time cost, it works for every DA3 size and for DéjàView at any K, and it supplies idea 1's basin-aware loss.

### 4. Optimise the loop state instead of the outputs

**Hypothesis and novelty.** Take a few gradient steps on the latent state against our photometric or NCC objective, through the frozen per-frame decoders, and then continue looping. This lets the network's prior regularise the optimisation, a neural re-parameterisation of BA. On Solv3R the edited state is a legal input, because subsequent calls are trained to pull perturbed states back. The approach mirrors measurement guidance in diffusion samplers and PoseDiffusion's geometry-guided sampling, and it differs from Test3R's prompt tuning of weights ([Test3R](https://arxiv.org/abs/2506.13750)). It has not been tried on a deterministic looped multi-view regressor. BA-T's finding that explicit residuals matter more than iteration itself is indirect support ([BA-T](https://arxiv.org/abs/2606.03287)).

**What is trained, at what cost, first experiment, risk and fit.** Nothing is trained. Probing on dev costs 5-15 GPU-hours, and gradients at 100 views need the 5090.

The first experiment: at call 16 of Solv3R, run 10-20 Adam steps on the camera and patch tokens. Compute the loss from decoded depth and from camera-head poses, because the ray RANSAC is not differentiable. Then run 4 more calls and compare against PBA on the same decoded outputs at matched time. Solv3R's API is wrapped in `@torch.no_grad`, so call the network's block and decoders directly ([Solv3R repo](https://github.com/ubc-vision/solv3r)).

The risks are substantial:

- **The released DéjàView does not tolerate edited states.** Solv3R's perturbation test left it at 6.3-15.6° rotation error.
- **Gradients through the depth head may be poorly conditioned.**
- **Each step costs a decoder pass.** Our explicit BA costs 1.2-2.2 ms per Adam iteration ([feature-metric-vgg](../knowledge/ideas/feature-metric-vgg.md)), so latent TTO must win on quality, not speed.

It fits as a third TTO family on the matched-compute plot and gives the cleanest test of coupling a learned prior to an optimiser.

### 5. Spend loop steps only on scenes that need them

**Hypothesis and novelty.** The number of loop steps a scene needs varies widely. The K = 6/K = 12 AUC@3 ratio ranges from 0.42 on ScanNet++ to 0.73 on 7-Scenes ([dejaview-steps](../knowledge/ideas/dejaview-steps.md)). Choosing (K, T) per scene from a cheap signal could therefore save compute at equal quality. Candidate signals: the photometric residual after a few BA iterations, the disagreement between the camera head and the ray fit after a short probe, or Solv3R's relative update norm. Solv3R halts only on its own latent norm, and no work allocates jointly between learned and classical iterations with an external geometric signal.

**What is trained, at what cost, first experiment, risk and fit.** Nothing, or a logistic regression trained in minutes. The first experiment uses the per-scene outputs we already have. Compute the oracle K* per scene (the smallest K whose K + BA reaches K = 12 + BA), sum the compute it saves, then test which observable predicts K*.

The risks:

- **DéjàView fixes its step size up front**, so a wrong guess costs a full re-run (draft and verify).
- **The fixed 0.42 s encoder caps the savings.**
- **Dev overfitting.** Our dev set has only 3 ETH3D scenes, so a policy tuned on it may not transfer.

This idea is a cheap adaptive point for the paper's frontier rather than a paper on its own.

## Directions that would waste the GPU budget

**Plain distillation into a smaller DéjàView stays ruled out.** OTT3R's ~384 GPU-hours is the realistic reference, and it shows the outcome: good in-distribution depth and poor out-of-distribution pose ([OTT3R](https://arxiv.org/abs/2609.36374)). A ViT-S encoder regressing DéjàView's z_0 is closest to this ruled-out path, even though the encoder is 74% of the parameters and the fixed ~0.42 s. Only its training-free cousin, an input-resolution sweep on Solv3R, is worth an hour. **Making DéjàView stationary or adaptive** is exactly what Solv3R spent ~1,350 H100-hours on ([Solv3R](https://arxiv.org/abs/2611.02857)). **Writing BA results back into DéjàView's state** has no input channel and breaks under perturbation.

The remaining dead ends are cheaper but no more promising:

- **Weight-space test-time adaptation** is crowded and two orders of magnitude less compute-efficient than our BA.
- **Token merging and sparse attention** give little or nothing at ≤ 100 views and are not novel.
- **Dataset-scale self-distillation of DA3 on its own BA outputs** most likely leaves the student + BA at the teacher's level, because the BA's residual errors sit in exactly the directions it cannot teach. It is a clean negative result at best.
- **Few-call self-distillation of Solv3R and Anderson acceleration of its calls** should wait until we have measured how much worse 4 calls are than 20. They push towards anytime solving, not compactness.

## A two-week plan that trains only after the probes pass

| When (2026) | Suggested owner | Task | Output or decision |
|---|---|---|---|
| Mon 16 Nov | Antonin | Read the job 2832 results into `knowledge/ideas/dejaview-steps.md` | Gates G1 and G2 |
| 16-17 Nov | Antonin | DéjàView K ∈ {4, 6, 8, 12, 16} + PBA on the official dev set via `run_tto_dev.sh` (`metric_file=pba`); held-out at K = 8, 12, 16 | DéjàView points on the compute plot; official metrics instead of log-line averages |
| 17-19 Nov | Antonin | `third_party/solv3r` submodule and adapter; 16 calls vs DéjàView K = 16; calls {4 … 32, adaptive} × PBA on dev | Gate G4 |
| 17-19 Nov | Finn | Basin analysis: per-scene initial vs post-PBA rotation, translation and focal error for DéjàView K = 3-16 and DA3 S-G | Basin radius, truncation threshold (G3) |
| 19-20 Nov | Finn | Weak-subspace oracle swap; truncated-grid and LoopCD probes if job 2832 didn't cover them | Go or no-go for idea 3 |
| Fri 20 Nov | either | 5-minute profiling job: one fine-tune step at 32 views, frozen encoder; download ARKitScenes and request ScanNet++ v2 if we lack it; teacher-cache script | Measured memory; data on scratch |
| 23-24 Nov | Antonin | Cache K = 16 rays, depth and cameras plus K = 16 + PBA targets for 1-2k unlabeled scenes | Cache, a few GPU-hours |
| 24-26 Nov | Antonin | Gate-only interval distillation, K_s ∈ {3, 4, 6} mixed with [8, 16], 1-3 jobs of 6 h | Gate G5 |
| 24-26 Nov | Finn | Solv3R reset/verifier test; latent TTO probe on Solv3R; oracle K* | Go or no-go for ideas 2, 4, 5 |
| Fri 27 Nov | both | Decide the main line for the 10.12 intermediate presentation; update `knowledge/` and `runs.csv` | Plan for weeks 3-4 |

The gates turn the probes into decisions. Seed noise on the dev mean is about ±0.2-0.4 AUC@3, and held-out with one seed about ±0.5 ([evaluation](../knowledge/evaluation.md)), so a few points is a real signal. The thresholds below are my suggestions, not measured cut-offs.

- **G1 (step size).** If a grid that keeps trained-size intervals beats uniform K = 6 by several AUC@3 points at the same number of block applications, start with gates-only training. Otherwise start at the LoRA rung and target K = 6.
- **G2 (re-baseline).** If K = 16 beats K = 12 by ≥ 2 AUC@3 for ≈0.2 s, it becomes our DéjàView baseline and teacher, and every plot moves.
- **G3 (loss shape).** If K ≤ 4 errors are mostly far outside the basin radius, the truncated basin loss of idea 1 is the right objective. If they are mostly small, plain endpoint distillation suffices.
- **G4 (Solv3R).** If PBA adds nothing on top of Solv3R, idea 2 shrinks to a baseline row. If BA residuals flag its wrong fixed points, idea 2 becomes a co-equal line.
- **G5 (first training).** Success means gate-only training lifts K = 4 + PBA well above today's 33 dev AUC@3, towards the 67 of K = 8 + PBA, while K = 12 and 16 stay within noise. If it doesn't, escalate once to LoRA. If that also fails, make idea 3 the training contribution and keep the DéjàView K axis as evaluation.

Week 1 needs only inference, likely a few tens of GPU-hours. Week 2's cache and first training jobs add roughly 15-30 GPU-hours (my estimates). That leaves most of the 100-500 GPU-hour budget for scaling whichever line passes.

## Conclusion

The project question has a sharper form than it did at the proposal. DéjàView's loop count is already a model-size axis that needs no new models. With a 0.3 s BA on top, the current DéjàView numbers sit above the DA3-BASE + BA point at similar time and near DA3-LARGE + BA at about 60% of its time, if the rescoring on the official dev and held-out sets confirms them. The scientific question that remains is therefore not whether a compact model + TTO can match a big one. It is how little learned computation the optimiser needs, and whether a network can be trained for the optimiser rather than for the teacher. "Train for the basin, not the teacher" is the principle that unifies the top-ranked loop distillation and the head-level weak-subspace training, and it transfers to DA3. That makes it more defensible as a SIGGRAPH contribution than step distillation alone, whose payoff in seconds is capped by the encoder.

Solv3R reframes the competition. It went the opposite way: more calls, no optimiser, and errors that stability makes sticky. That leaves "learned iterations versus explicit iterations at matched time" as a crisp axis that nobody has plotted, with BA as the external verifier a latent solver lacks. The biggest single uncertainty is whether three or four applications of the shared block are enough for its correspondence search. Pending job 2832 is the cheapest way to find out, and it should be read before any GPU-hour goes into training.
