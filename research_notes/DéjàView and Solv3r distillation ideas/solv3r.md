# Solv3R (3D looped transformer as a multi-view solver): dossier and distillation/TTO hooks

Sources read: the full arXiv text (main paper and the whole supplementary, A to D; only the reference list was skimmed),
the GitHub repo (README, `solv3r/model.py`, `solv3r/network/network.py`, `solv3r/network/block.py`, `pyproject.toml`,
eval configs), the project page and the Hugging Face model card. Local cross-checks against our repo
(`third_party/dvlt`, `scripts/baselines.py`, `knowledge/ideas/dejaview-steps.md`) are marked as such.
Short links: Paper = https://arxiv.org/abs/2611.02857, Repo = https://github.com/ubc-vision/solv3r,
Page = https://ubc-vision.github.io/solv3r/, HF = https://huggingface.co/jgkwak/solv3r.

## 1. Identity: title, authors, institution, arXiv id, date, venue

### Takeaway
"Solv3R: 3D Looped Transformer as a Multi-view Solver" by Kwak, Tagliasacchi, Kagami, Ono and Yi (UBC, Wayve, SFU,
U Toronto, Sony), arXiv 2611.02857, submitted 2 Nov 2026. It is an arXiv preprint with no venue yet. It is the only
multi-view reconstruction paper with this name that I found.

### Cited Findings
- Title "Solv3R: 3D Looped Transformer as a Multi-view Solver"; authors Jeong-gi Kwak (UBC), Andrea Tagliasacchi (Wayve Labs, Simon Fraser University, University of Toronto), Sho Kagami (Sony Semiconductor Solutions), Yuki Ono (Sony Corporation), Kwang Moo Yi (UBC) — [Paper](https://arxiv.org/abs/2611.02857), [Page](https://ubc-vision.github.io/solv3r/)
- arXiv:2611.02857v1 [cs.CV], 2 Nov 2026 — [Paper](https://arxiv.org/abs/2611.02857); listing date confirmed by [papers.cool](https://papers.cool/arxiv/2611.02857)
- Venue: the project page cites only the arXiv preprint and names no conference; the BibTeX in the README is `@article{... journal = {arXiv preprint arXiv:2611.02857}}` — [Page](https://ubc-vision.github.io/solv3r/), [Repo README](https://github.com/ubc-vision/solv3r)
- Funding: Sony Faculty Innovation Award, NSERC, Digital Research Alliance of Canada — [Paper, Acknowledgments](https://arxiv.org/abs/2611.02857)
- An alphaXiv search for "Solv3r" returned only this paper as a match; the other hits, Kosmo-GFM (2611.16840) and Fast3R (2501.13928), are unrelated names — [alphaXiv search](https://www.alphaxiv.org/abs/2611.02857)
- It builds directly on DéjàView (Burzio et al., NVIDIA et al., cited as NeurIPS'26, arXiv 2605.30215) — [Paper §3.1, ref. 12](https://arxiv.org/abs/2611.02857), [alphaXiv DéjàView](https://www.alphaxiv.org/abs/2605.30215)

### Inferences
- The paper is 13 days old (as of 2026-11-15) and has no venue, so it is probably under review, likely at CVPR 2027. It is a direct competitor in "iterative or looped GFMs plus refinement". A SIGGRAPH 2027 submission from us would have to cite it and position against it.

### Gaps
- I found no papers that cite Solv3R yet. That is expected at 13 days old.
- The submission venue is unknown.

## 2. Method: architecture, what "solv" means, outputs, training, iterations as a test-time knob

### Takeaway
"Solv" means a *learned stationary iterative solver*, in the sense of the Jacobi method. Solv3R is DéjàView's single
shared transformer block (frame attention plus global attention), fine-tuned so that its update has no step index.
It gets constant gates instead of DéjàView's step-conditioned gates, the encoded images are re-injected at every call,
and the state is rescaled to a fixed RMS after every call. It is trained to be stable at a fixed point and to correct
perturbed or partially reset states. It has no explicit optimisation layer, no geometric residual and no
differentiable BA inside the loop. The "solver" is purely a learned latent fixed-point iteration. The number of calls
is a test-time knob, and an adaptive stopping rule on the relative state change (γ = 0.02) is supported.

### Cited Findings
- **Motivation:** DéjàView updates z_{k+1} = F_θ(z_k; τ_k) with conditioning τ_k on the iteration index. As a result it "loses reconstruction accuracy when iterated beyond its trained range", and it cannot jointly update views that have received different numbers of iterations. "Sharing weights alone does not give the block an update that is independent of the iteration index." — [Paper §3.1](https://arxiv.org/abs/2611.02857)
- **The solver analogy:** a stationary iteration x_{k+1} = Φ(x_k; A, b), as in Jacobi, with three desired properties: (P1) stability, the solution is a fixed point; (P2) correction, perturbed states converge back; (P3) continuation, new unknowns can be added and solved jointly with already-solved ones in fewer calls than a restart — [Paper §3.2](https://arxiv.org/abs/2611.02857)
- **Update:** z_{k+1} = T_θ(z_k; z_0) with z_0 = E(I), the encoded images, which play the role of the "constraints (A, b)". In detail, u = U_θ(z_k + W·LN(z_0)) and z_{k+1} = ρ·u/‖u‖_RMS. W is a zero-initialised linear projection. ρ ≈ 0.85 is the RMS of the final DéjàView state, measured once and fixed. A single scale factor covers the whole state across views — [Paper §3.3, Supp. A, Eq. 6](https://arxiv.org/abs/2611.02857)
- **Architecture (code):** each view is a token sequence [camera token, 4 register tokens, patch tokens], width 768. The encoder is DINOv2 ViT-B/14, frozen, taken from DVLT/DéjàView. The block is one `SolverBlock` = a frame-attention `GatedAttentionBlock` (with RoPE) followed by a global-attention `GatedAttentionBlock`, 12 heads, MLP ratio 4, constant gates `g_frame`, `g_global` of shape (3, C) for attention residual, MLP residual and output carry, plus the injection `inj` (Linear 768→768) and `target_rms`. The output heads are a ray head (6-D rays per pixel: direction and origin), a depth head (depth and confidence, conv) and a camera head (9-D pose encoding), all frozen from DVLT — [network.py](https://github.com/ubc-vision/solv3r/blob/main/solv3r/network/network.py), [block.py](https://github.com/ubc-vision/solv3r/blob/main/solv3r/network/block.py)
- **Outputs** after any call: depth (S,H,W,1), depth_conf, rays (S,H,W,6), world_points = ray origin + direction × depth, pose_enc (S,9). Cameras come either from the camera head or "fitted to the predicted rays (the benchmark protocol)" via `cameras_from_rays`. It does not predict tracks — [model.py](https://github.com/ubc-vision/solv3r/blob/main/solv3r/model.py)
- **Gate initialisation:** the constant gates are initialised by averaging DéjàView's gate values over its 16 refinement steps — [Paper Supp. A](https://arxiv.org/abs/2611.02857)
- **Training: truncated unrolling, no implicit gradients.** It samples 4–64 views (size classes 4–16, 24–32 and 48–64 with probabilities 0.4/0.3/0.3) and runs 16–40 solver calls *without gradients*, then 8 differentiable calls supervised at the last one. Before the differentiable calls, with p = 0.15, it either adds Gaussian noise (σ ~ U[0, 0.5] × state RMS) or resets up to half of the views to z_0. Supervision on varying calls trains it to be "accurate at any call between the 24th and the 48th". Earlier calls are not supervised — [Paper §3.4, §5, Supp. A](https://arxiv.org/abs/2611.02857)
- **Loss:** L = L_geo + λ_dist·L_dist with λ_dist = 0.5. L_geo has an L1 loss on ray directions and scale-aligned origins, geodesic losses on relative rotations and translation directions, scale-aligned log-L1 depth (weight 0.5) and a multi-scale depth-gradient loss (0.5) where depth is dense, plus two scale-consistency terms (1.5 each). **L_dist distils DéjàView's 16-step predictions** (rays, log-depth, cameras). Datasets with sparse or noisy GT (Mip-NeRF 360, Waymo) use L_dist only — [Paper §3.4, Supp. A](https://arxiv.org/abs/2611.02857)
- **What is trained:** only the shared block, the gates and the projection W. The image encoder and the prediction heads of DéjàView are frozen. AdamW with weight decay 0.01, LR 2e-5 for the block and W and 2e-4 for the gates, EMA 0.999. Two stages: ~70k iterations at batch 8 with a curriculum on window size and number of calls, then ~20k iterations at batch 32. In the last 2k iterations, views are rotated by 90° with p = 0.15 — [Paper Supp. A](https://arxiv.org/abs/2611.02857)
- **Iterations as a knob:** fixed mode uses 16 calls, "the budget of DéjàView". Adaptive mode stops when r_k = ‖z_{k+1} − z_k‖/‖z_k‖ < γ = 0.02, with at least 4 and at most 64 calls. On 148 eight-view windows, 144 stop within 64 calls, after 13–32 calls (mean 17.5, median 17), and 143 of them have rotation error < 5°. The stopping call correlates with the rotation error at that call (Spearman 0.77) — [Paper §4, Supp. C.1](https://arxiv.org/abs/2611.02857)
- **Threshold sensitivity (pose AUC@3 / mean calls):** γ = 0.01: DTU 85.9/36.1, ETH3D 69.8/54.9, 7-Scenes 14.2/35.8, ScanNet++ 81.5/39.8, nuScenes 41.8/31.1. γ = 0.02: 85.5/19.7, 70.4/20.5, 14.2/22.6, 81.2/17.3, 41.5/18.2. γ = 0.05: 81.5/13.0, 62.0/13.5, 13.9/12.9, 77.0/11.8, 40.9/11.9 — [Paper Supp. Table 6](https://arxiv.org/abs/2611.02857)
- **KV cache:** held views' global-attention keys and values depend only on their own state, so caching them is exact. A single cache serves all calls; DéjàView would need one per step and layered GFMs one per layer — [model.py `cache`/`step_cached`](https://github.com/ubc-vision/solv3r/blob/main/solv3r/model.py), [Paper §4](https://arxiv.org/abs/2611.02857)
- **API:** `encode(images)` → Context(x0, pos); `solve(ctx, calls, z=None)`; `add_views(z, ctx, images)`; `residual(z, ctx)` (the stopping criterion); `cache`/`step_cached`; `decode`; `decode_cameras`. All methods are `@torch.no_grad()` and run in bf16 autocast — [model.py](https://github.com/ubc-vision/solv3r/blob/main/solv3r/model.py)
- **Stochastic subset updates:** the project page also describes "stochastic block-coordinate-style updates", which refine a subset of views per call while the rest supply cached context: "128 photos by updating 8 views per call using 1.2 GB of GPU memory, versus 12.8 GB for VGGT". The README lists "stochastic subset updates" examples as "coming soon" — [Page](https://ubc-vision.github.io/solv3r/), [Repo README](https://github.com/ubc-vision/solv3r)
- **Attention analysis:** at call 16, the three global-attention heads best at finding GT correspondences find them for 51–74% of query patches. Solv3R keeps that accuracy up to call 64 (attention similarity > 0.96 to call 16), while DéjàView's heads drop below 3% by call 64 — [Paper Supp. C.3](https://arxiv.org/abs/2611.02857)

### Inferences
- The "solver" is learned latent fixed-point iteration in the style of a DEQ, trained with truncated BPTT (no-grad burn-in plus 8 differentiable calls), not an optimisation layer. Nothing in it minimises a geometric or photometric residual. Our photometric/feature-metric BA is therefore complementary rather than redundant: it adds explicit pixel-level evidence that the latent solver only sees implicitly through attention.
- Because the encoder and heads are bit-identical to DéjàView's (frozen DVLT modules), z_0 for Solv3R equals the DéjàView encoder output. Our DéjàView adapter, its preprocessing and any cached encoder tokens from the `distillation-test` branch carry over directly.
- Solv3R takes only images as input. There is no input path for poses, depths or intrinsics. Its solver therefore cannot be applied to DA3 outputs or to our BA results without training a new geometry-to-state encoder.

### Gaps
- The paper gives the shape of the accuracy-vs-calls curve below 16 calls (Fig. 5 left) only as a plot. The text gives AUC@5 = 77.2 at 16 calls and 79.1 at 128 calls only, so how good 2–8 calls are is unknown. This matters for the matched-compute question.
- There is no FLOPs or latency figure per call in the paper. See the estimate in §3.

## 3. Cost: parameters, FLOPs, runtime, memory, number of views, training data and compute

### Takeaway
Solv3R has 117M parameters, the same as DéjàView: a frozen DINOv2 ViT-B encoder, one shared block and frozen heads. It
costs 76.5 TFLOPs for 24 views at 16 calls and 91.3 TFLOPs in adaptive mode. Inputs are 504 px on the long side, and
it is trained on 4–64 views but runs on 96–128+ views. Training took about a week on 8×H100 (≈1,350 GPU-hours) on a
25-dataset mixture, starting from DéjàView. Inference is cheap enough for our GPUs; full retraining is not.

### Cited Findings
- 117M parameters; images with long side 504 px — [Paper Supp. A](https://arxiv.org/abs/2611.02857); loading: "long side 504, sides multiples of 14, no crop" — [Repo README](https://github.com/ubc-vision/solv3r)
- TFLOPs at 24 views: Solv3R fixed (16 calls) 76.5, adaptive 91.3; DéjàView 75.9; DA3-L 71.4; DA3-G 178.7; VGGT 190.0; VGGT-Ω 99.8; π³ 153.8; MapAnything 148.4; MASt3R-SfM 1150.1 — [Paper Table 1](https://arxiv.org/abs/2611.02857)
- Adaptive stopping improves "nearly all reconstruction metrics" over the fixed budget "with only 3–4 additional solver calls on average" — [Paper §5.1](https://arxiv.org/abs/2611.02857)
- Relocalisation query latency on one H100: Solv3R 51 ms with a 2.86 MB/view cache; DéjàView (all-step cache) 88 ms at 45.8 MB/view; DA3-L 62 ms at 30.4 MB/view; VGGT 186 ms at 97.6 MB/view; π³ 161 ms at 89.7 MB/view — [Paper Table 2](https://arxiv.org/abs/2611.02857)
- Streaming on H100, including the encoder and heads: L8 (default) 95 ms/frame and 5.0 GB; L4 65 ms and 5.0 GB; S4 51 ms and 2.1 GB. Memory stops growing after ~1030 frames for L8 and ~260 for S4. Offline VGGT runs out of the 80 GB at 320 frames, StreamVGGT after 365 — [Paper Table 3, Supp. Table 8, Supp. C.2](https://arxiv.org/abs/2611.02857)
- Pipelining gives the same latency as DéjàView at 16 calls with "16× more frames per second (10× end to end)" — [Paper §4](https://arxiv.org/abs/2611.02857)
- View counts: training windows of 4–64 views; the Mill19 aerial evaluation uses 96 images in a single window; the qualitative offline comparisons use 100 images; the project page shows 128 photos via 8-view subset updates in 1.2 GB — [Paper Supp. A, D.1, Figs. 21–23](https://arxiv.org/abs/2611.02857), [Page](https://ubc-vision.github.io/solv3r/)
- **Training compute:** "approximately one week on 8×H100 GPUs"; stage 1 ~70k iterations at batch 8, stage 2 ~20k at batch 32. The ablations are fine-tuned for only 3k steps from the same DéjàView checkpoint — [Paper §5, Supp. A, §5.5](https://arxiv.org/abs/2611.02857)
- **Training data (stage-2 sampling weights):** indoor ScanNet++ 10.8%, ScanNet 7.4, CA-1M 3.3, Taskonomy 3.3; object CO3D 7.4, NAVI 3.3, Skoltech3D 2.5, OmniObject3D 2.5, Objaverse renders 2.5; outdoor DL3DV 8.2, BlendedMVS 4.1, MegaDepth 3.3, Oxford Spires 1.6, Mip-NeRF 360* 0.8; driving Waymo* 4.1, Argoverse 2 3.3, ParallelDomain-4D 3.3, Virtual KITTI 2 1.6; synthetic Infinigen 9.1, TartanAir 5.8, Hypersim 5.8, MVS-Synth 1.6, Spring 1.6, MatrixCity 1.6, UnrealStereo4K 1.2 (* = distillation loss only) — [Paper Supp. Table 5](https://arxiv.org/abs/2611.02857)
- (Local) Our DéjàView adapter runs the same encoder and block at 100 frames in 3.9 GiB peak on the 3080 with `decode_chunk_size=8`. Solv3R's network also decodes in chunks of 8 views by default — `CLAUDE.md` (Baselines), [network.py](https://github.com/ubc-vision/solv3r/blob/main/solv3r/network/network.py)
- (Local) Our full-benchmark DéjàView K sweep on tum2 (RTX 5090) gives a median forward time of 0.76 s at K = 8 and 0.96 s at K = 12 — `knowledge/ideas/dejaview-steps.md`

### Inferences
- **Per-call cost estimate.** The 0.6 TFLOPs gap between Solv3R fixed (76.5) and DéjàView (75.9) equals 16 injection projections: 768² × 2 FLOPs × ~31k tokens (24 views × 1301 tokens, assuming square 504×504 inputs; the paper doesn't state the resolution used for its FLOPs) ≈ 37 GFLOPs per call. This is consistent with DéjàView's 75.9 TFLOPs being a 16-step figure. From the adaptive/fixed difference (14.8 TFLOPs for ~3.7 extra calls, using the mean call count over the five datasets in Table 6), one call costs ≈ 4 TFLOPs at 24 views, and encoder plus heads ≈ 12 TFLOPs. These are rough derivations, not paper numbers.
- **Wall-clock estimate for us.** From our K sweep, one DéjàView step costs ≈ 0.05 s on a 5090 at benchmark sizes (0.20 s per 4 steps), so a Solv3R call should cost about the same since the block is the same. Our whole photometric TTO (0.2–0.3 s) then equals roughly 4–6 solver calls. Adaptive Solv3R at ~20 calls is ≈ 0.4 s more than DéjàView at K = 12.
- **Memory.** It is almost certainly the same as our DéjàView adapter (≈ 4 GiB at 100 frames), and the subset/KV-cache mode reduces it further. It therefore fits the 3070 and 3080 for inference. Backprop through a few calls at 100 views would need the 5090.
- **Training cost.** ≈ 1,350 H100-hours. At H100 parity that is ~14 days of all four tum2 nodes; on 5090s it would be longer. Full retraining is infeasible for us. A 3k-step fine-tune of the block alone, the paper's ablation scale, is plausibly feasible on one 5090 within several 6 h jobs if z_0 is cached, but the ablation batch size isn't stated.
- Paper Table 1 lists DA3-L at 411M parameters, while our `CLAUDE.md` gives DA3-LARGE-1.1 0.35B and DéjàView's table 356M. This is an inconsistency between sources, possibly because different parts were counted.

### Gaps
- No per-call latency or memory on consumer GPUs (3080/3070/5090) is reported. The local estimates above are inferred from our DéjàView measurements.
- The batch size of the 3k-step ablation fine-tunes is not stated.

## 4. Results: benchmarks, metrics, comparisons, ablations

### Takeaway
On DéjàView's offline benchmark (DTU, ETH3D, 7-Scenes, ScanNet++, nuScenes; pose AUC@3/@30, point-map Rel-L2 and
inlier ratio), Solv3R at 16 calls roughly matches DéjàView. Adaptive stopping gives the best or near-best pose among
all methods, with the largest gain on ETH3D AUC@3 (66.0 → 70.4) and a clear win over DA3-L and DA3-G, VGGT, π³ and
MapAnything at 3–11× fewer parameters. The real novelty is in the solver properties: it stays accurate at 1024 calls,
it can add views mid-solve, it reuses a KV cache for relocalisation, and it handles streaming without extra training.
None of the comparisons includes a feed-forward model followed by BA refinement, other than MASt3R-SfM.

### Cited Findings
Offline benchmark (pose AUC@3 / AUC@30; DVLT protocol; TFLOPs at 24 views) — [Paper Table 1](https://arxiv.org/abs/2611.02857):

| Method | Params | TFLOPs | DTU | ETH3D | 7-Scenes | ScanNet++ | nuScenes |
|---|---|---|---|---|---|---|---|
| MASt3R-SfM | 690M | 1150.1 | 40.2/91.4 | 52.8/85.0 | 16.4/80.3 | 38.7/85.0 | 9.1/70.9 |
| VGGT | 1.26B | 190.0 | 96.5/99.8 | 35.4/82.8 | 11.1/79.1 | 15.1/74.7 | 39.2/83.8 |
| VGGT-Ω | 1.14B | 99.8 | 77.4/98.1 | 64.1/95.4 | 21.3/87.0 | 29.9/87.3 | 42.4/85.5 |
| π³ | 959M | 153.8 | 70.2/97.3 | 43.2/84.9 | 11.3/81.3 | 76.5/97.3 | 24.8/82.4 |
| MapAnything | 1.23B | 148.4 | 18.1/88.8 | 37.1/73.8 | 8.2/74.6 | 70.6/96.7 | 37.8/82.8 |
| DA3-L | 411M | 71.4 | 79.4/98.4 | 42.0/75.6 | 11.4/79.7 | 7.0/71.8 | 10.9/74.6 |
| DA3-G | 1.32B | 178.7 | 86.4/99.1 | 63.2/83.2 | 15.4/81.3 | 24.3/80.1 | 39.8/82.3 |
| DéjàView | 117M | 75.9 | 83.3/98.8 | 66.0/95.4 | 13.9/81.7 | 79.5/98.0 | 43.4/85.3 |
| Solv3R fixed (16) | 117M | 76.5 | 84.5/99.0 | 65.4/94.8 | 13.8/81.8 | 80.7/98.2 | 41.5/85.3 |
| Solv3R adaptive | 117M | 91.3 | 85.5/99.1 | 70.4/96.3 | 14.2/81.9 | 81.2/98.2 | 41.5/85.3 |

- Geometry (Rel-L2 / inlier ratio %), Solv3R adaptive vs DéjàView: DTU .009/97.1 vs .009/97.1; ETH3D .026/77.9 vs .026/78.3; 7-Scenes .034/75.8 vs .035/74.2; ScanNet++ .015/92.6 vs .015/93.3; nuScenes .059/60.4 vs .067/58.5. DA3-L for reference: .010/97.1, .210/49.9, .039/69.7, .051/47.8, .141/27.0 — [Paper Table 1](https://arxiv.org/abs/2611.02857)
- The README's "expected results" differ slightly from the paper for the adaptive mode (DTU inlier 97.0 vs 97.1; ETH3D Rel-L2 .027 vs .026) — [Repo README](https://github.com/ubc-vision/solv3r) vs [Paper Table 1](https://arxiv.org/abs/2611.02857)
- **Protocol** (eval configs): ETH3D and ScanNet++ use max_frames 100; nuScenes max_frames 50; 7-Scenes every 11th frame; DTU "middle_first" view ranking. The configs run inside DVLT's evaluation pipeline, which the `eval` extra pins to commit `f59bb18` — [eval configs](https://github.com/ubc-vision/solv3r/tree/main/eval/config/data/test_datasets), [pyproject.toml](https://github.com/ubc-vision/solv3r/blob/main/pyproject.toml). (Local) Our `third_party/dvlt` submodule is at exactly that commit, `f59bb18259ab446a7585c4f47731a77cfdefa426`.
- **Solver dynamics:** DéjàView's accuracy drops beyond its 16 trained steps, while Solv3R keeps it. Perturbation at call 8 followed by 8 more calls: Solv3R rotation errors 0.3–1.3° vs DéjàView 6.3–15.6° (relative or fixed noise at 1× and 2×, and reset) — [Paper Fig. 5](https://arxiv.org/abs/2611.02857)
- **Mixed-progress states** (each view taken from a random call 0–16): Solv3R pose AUC@3 83.6, DéjàView 58.7 (one clock) and 38.8 (per-view clock). The mean rotation difference across independent trials is 0.08° — [Paper §4, Fig. 7](https://arxiv.org/abs/2611.02857)
- **Incremental reconstruction:** 4 views are solved for 8 calls, then 12 views are added. Solv3R also improves the original views, while DéjàView fails. Growing from 8 to 64 views, 8 calls per insertion of continued solving reaches AUC@10 92.2, against 84.8 for re-solving with 8 calls and 91.8 for re-solving with 16. The result is insensitive to insertion order (within 2 points) — [Paper §5.2, Figs. 8–9, Supp. C.4, Table 10](https://arxiv.org/abs/2611.02857)
- **Map-based relocalisation** (7-Scenes median cm/° and recall at 5 cm/5°; Cambridge recall at 25 cm/2°): Solv3R 3.58/0.95, 72.8%; Cambridge 63.5/0.74, 24.9%. These are the best recalls; π³ has 66.9% and 14.7%, DA3-L 36.7% and 12.7% — [Paper Table 2](https://arxiv.org/abs/2611.02857)
- **Streaming** (ATE in m): Sintel .157, TUM-dyn .015, ScanNet .077, at 95 ms and 5.0 GB, against LingBot-Map .149/.032/.089 at 119 ms and 14.3 GB, and CUT3R .204/.170/.843. Depth and 3D metrics are competitive (e.g. KITTI AbsRel .078, δ .947). This needs no streaming-specific training — [Paper Table 3](https://arxiv.org/abs/2611.02857)
- **Mill19 aerial** (96 images, AUC@3/@30): Rubble Solv3R 48.9/94.1 vs DéjàView 39.4/93.0 and DA3-L 3.8/74.0; Building 22.5/89.6 vs 11.0/87.0 and 1.1/55.8 — [Paper Supp. Table 11](https://arxiv.org/abs/2611.02857)
- **Ablation** (AUC@5 at 16 / 128 / 1024 calls; 3k-step fine-tunes): Solv3R 72.5/70.8/65.4; w/o input injection 71.6/33.1/0.0; w/o state normalisation 74.3/70.5/0.0; w/o random-depth pre-calls 73.1/3.4/0.0; w/o perturbation 71.0/69.1/61.7 (but join and cache errors 4.24°/3.51° vs 0.62°/0.76°); step-conditioned gates (P = 16) 71.5/52.1/23.5; released DéjàView 76.5/0.0/0.0; final Solv3R 77.2/79.1/78.8 — [Paper Table 4](https://arxiv.org/abs/2611.02857)
- The authors conclude: "accurate fixed-budget predictions are insufficient for reusable geometric solving, which requires both stable continuation and corrective updates" — [Paper §5.5](https://arxiv.org/abs/2611.02857)

### Inferences
- In the offline setting, the gain over DéjàView at equal budget is within noise: some datasets go up and others down, with nuScenes −1.9 AUC@3. The gain from adaptive stopping is real but costs +20% FLOPs. Pose AUC@3 on 7-Scenes stays at 8–21 for every method (Solv3R 14.2, best VGGT-Ω 21.3), which suggests a precision ceiling of current feed-forward and latent refinement that an explicit pixel-level BA might break. This is untested and is exactly our niche.
- Their DéjàView numbers use the 16-step budget, but our DéjàView adapter defaults to `inference_steps=12` (`scripts/baselines.py`; the DVLT constructor default is 12 while `num_steps = 16` is the training budget, in `third_party/dvlt/src/dvlt/model/dvlt/model.py`). So our DéjàView baseline is slightly under-computed relative to the papers' numbers.
- Their protocol is the DVLT benchmark, not the DA3 benchmark we use (no HiRoom, no TSDF-F1). To compare, we would have to run Solv3R through our DA3-bench adapter.

### Gaps
- There is no comparison with "feed-forward + BA" hybrids such as VGGT + BA, DA3 + BA or our PBA. Whether Solv3R's extra calls beat a few BA iterations at matched time is open.
- There are no results at fewer than 16 calls in the tables.
- There are no DA3-benchmark (HiRoom, TSDF F1) numbers.

## 5. Code, weights, license

### Takeaway
The inference, evaluation and demo code and the pretrained weights were released on 5 Nov 2026. The code is under
Apache-2.0, the weights under the NVIDIA non-commercial (research and evaluation) license inherited from DVLT. The
training code is **not** released and is not even on the TODO list.

### Cited Findings
- News: "[2026-11-05] Inference and evaluation code and the pretrained weights are released"; "[2026-11-10] An interactive demo is now available on HuggingFace Space" — [Repo README](https://github.com/ubc-vision/solv3r)
- TODO list: inference code, pretrained models, offline (Table 1), streaming (Tables 3, 8) and relocalisation (Table 2) evaluation code and the demo are all ticked; "Release solver property examples" is open. Training is not listed — [Repo README](https://github.com/ubc-vision/solv3r)
- The network module states "no step schedule, training or configuration code" — [network.py header](https://github.com/ubc-vision/solv3r/blob/main/solv3r/network/network.py)
- License: "The code is released under the Apache License 2.0. The model weights are built on DVLT and are released under the NVIDIA License (non-commercial, research and evaluation use only)" — [Repo README](https://github.com/ubc-vision/solv3r), [HF model card](https://huggingface.co/jgkwak/solv3r)
- Weights: HF repo `jgkwak/solv3r`, file `solv3r.safetensors`, loaded by `Solv3R.from_pretrained()` — [model.py](https://github.com/ubc-vision/solv3r/blob/main/solv3r/model.py)
- Dependencies: Python ≥3.11,<3.13; torch ≥2.5; huggingface_hub, safetensors, numpy, onnxruntime, opencv-python, pillow, scipy. The extra `eval` pulls in `dvlt @ git+...@f59bb18...`, `demo` pulls in gradio ≥6 — [pyproject.toml](https://github.com/ubc-vision/solv3r/blob/main/pyproject.toml)
- Examples: `examples/offline.py` (`--adaptive 0.02`), `incremental.py`, `streaming.py` (`--config L8/L4/S4`), `any_order.py`; the demo exports `.ply`, COLMAP and cameras plus depth — [Repo README](https://github.com/ubc-vision/solv3r)

### Inferences
- It is compatible with our uv environment (Python 3.11, torch 2.5.1 or 2.7.0). `opencv-python` is already overridden to headless in our `pyproject.toml`. It could be added as a pinned submodule, `third_party/solv3r`, with an adapter in `scripts/baselines.py` modelled on the DéjàView one. The non-commercial weights license is fine for academic research and a paper.
- Fine-tuning would require writing our own training loop: truncated unroll, perturbations and the losses in Supp. A. The loss details are specific enough to re-implement, and DVLT's own training stack exists in `third_party/dvlt`.

### Gaps
- The safetensors file size was not shown on the HF card.
- Whether training code will be released later is unknown.

## 6. Limitations: stated by the authors and obvious weaknesses

### Takeaway
The authors name one failure mode: the solver may converge to a wrong but *stable* fixed point when views form weakly
overlapping groups, and it stays there. Stability makes errors sticky. Other weaknesses: no explicit geometric
residual, no comparison with BA refinement, fixed 504 px resolution, unchanged parameter count and per-call cost
versus DéjàView, more FLOPs in adaptive mode, and expensive training with no released training code.

### Cited Findings
- Failure case: "The solver may occasionally stop at a suboptimal solution." In one window, views 5–8 share little content with views 1–4, and Solv3R places them at a wrong position and orientation within 16 calls and "keeps them there up to call 64" (rotation error 60° at K = 4, 57° at K = 16 and K = 64) — [Paper Supp. C.5, Fig. 18](https://arxiv.org/abs/2611.02857)
- Of 148 analysis windows, 4 do not reach γ = 0.02 within 64 calls, though they are still accurate (< 3°) at call 64. DéjàView reaches the threshold in none — [Paper Supp. C.1](https://arxiv.org/abs/2611.02857)
- A looser γ (0.05) saves about a third of the calls but loses 4–8 AUC points on DTU, ETH3D and ScanNet++. A tighter γ costs 1.6–2.7× the calls "without consistent gains" — [Paper Supp. C.1, Table 6](https://arxiv.org/abs/2611.02857)
- Few calls per frame in streaming need warm starts: with 1 call per frame, the warm start reduces ATE ~6× on TUM-dyn and 3.5× on 7-Scenes compared with a cold start — [Paper Supp. C.1, Fig. 15](https://arxiv.org/abs/2611.02857)
- The streaming results depend on refreshing frozen views: without the refresh, ATE grows more than 10-fold on TUM-dyn and ScanNet — [Paper Supp. C.2, Table 9](https://arxiv.org/abs/2611.02857)
- The step-conditioned gate variant is "sensitive to the planned refinement horizon P" — [Paper §5.5](https://arxiv.org/abs/2611.02857)
- The KV-cache step computes "their contribution to the state scale ... with attention among the held views only, which is close to the full call when the reference view (view 0) is held". It is therefore an approximation, not exact — [model.py `cache` docstring](https://github.com/ubc-vision/solv3r/blob/main/solv3r/model.py)

### Inferences
- **Sticky errors** are the flip side of P1, a stable fixed point. A latent solver has no external verifier: nothing tells it the fixed point is geometrically wrong. A photometric or feature-metric BA residual, or a track reprojection residual, is exactly such a verifier. This is the strongest hook for us (Idea 2 below).
- There is no sub-pixel geometric objective. The ~14 AUC@3 on 7-Scenes and the modest ETH3D AUC@3 suggest that precise poses still benefit from explicit refinement. This is consistent with VGGT's IMC result (BA lifts AUC@3 from 39.2 to 66.4), cited in our `CLAUDE.md`.
- The compactness is DéjàView's (117M), not new. The cost per call stays quadratic in total tokens for global attention unless the subset/KV mode is used.
- Training is unrolled with no-grad burn-in plus 8 differentiable calls. No implicit (DEQ) gradient is used, even though the model is a fixed-point iteration, so memory-cheaper fine-tuning by implicit differentiation is unexplored.

### Gaps
- The authors have no explicit "Limitations" section. Only the failure case in Supp. C.5 is stated.
- There is no analysis of robustness to resolution changes or of behaviour at more than 64 views offline without subset updates.

## 7. Hooks for us: distillation and TTO ideas on our compute

### Takeaway
Solv3R is less useful as a size-distillation teacher, because it is already 117M parameters and plain distillation
was ruled out. It is more useful as a **learned TTO baseline and partner**. Its stationary, correctable latent state
means that compute can be traded continuously between learned solver calls and explicit BA iterations, and that BA
can feed back into the solver by resetting, holding or editing views. All first experiments below are training-free
or need only small fine-tunes, and run on the 3080/3070 for inference and on the 5090s for anything with gradients.

### Cited Findings
- Solv3R exposes `solve(ctx, calls, z)`, `residual(z, ctx)`, `add_views`, `cache`/`step_cached` and `decode` after any call, so it can be interleaved with external code at call granularity — [model.py](https://github.com/ubc-vision/solv3r/blob/main/solv3r/model.py)
- It is trained to correct noisy states and reset views (P2/P3) — [Paper §3.4](https://arxiv.org/abs/2611.02857)
- The stopping call correlates with error (Spearman 0.77) — [Paper Supp. C.1](https://arxiv.org/abs/2611.02857)
- Related recent work on feed-forward plus optimisation (abstract-level only): BA-T, "An Iterative Transformer for Two-View Bundle Adjustment" (TUM, MCML, ETH, 2606.03287) — [alphaXiv](https://www.alphaxiv.org/abs/2606.03287); BLASt3R, BA of any image set with multi-view matching and monocular priors (NAVER LABS Europe, 2609.05210) — [alphaXiv](https://www.alphaxiv.org/abs/2609.05210); PosePlaner, "Denoising Any Feed-Forward Pose Predictor from Pairwise Planar Geometry" (2610.34845) — [alphaXiv](https://www.alphaxiv.org/abs/2610.34845); Glob3R, global SfM with 3D foundation models (2607.09225) — [alphaXiv](https://www.alphaxiv.org/abs/2607.09225); PURE, refining feed-forward reconstruction with unlabeled post-training (2610.14698) — [alphaXiv](https://www.alphaxiv.org/abs/2610.14698)
- (Local) DéjàView quality collapses below K = 6 on the DA3 benchmark (AUC@3 31.1 at K = 6, 58.7 at K = 12). Our photometric TTO costs 0.2–0.3 s and lifts DA3 AUC@3 substantially, e.g. SMALL 16.2 → 43.8 on dev — `knowledge/ideas/dejaview-steps.md`, `CLAUDE.md` (TTO status)

### Inferences (idea sketches; everything below is our proposal, not from the paper)

**Idea A: learned calls vs explicit BA at matched compute (training-free; the most direct fit to our question).**
- Sweep (number of solver calls k, PBA iterations T) on Solv3R. Report the Pareto front of AUC@3, AUC@30 and F1 against wall-clock time, and compare it with DA3-{S,B,L,G} + PBA and DéjàView-K + PBA.
- Questions: does PBA still add on top of a converged learned solver, especially for sub-pixel AUC@3 on 7-Scenes and ETH3D? Is a k = 8 solver plus T = 25 BA better than k = 24 alone? One call costs ≈ 0.05 s on a 5090 (inferred), so the whole PBA equals roughly 4–6 calls.
- Novel: neither Solv3R nor DéjàView compares learned iterations with classical refinement at matched compute. Solv3R is the first model where "more learned iterations" is a stable axis that doesn't degrade past the trained range, which makes the comparison clean.
- Cheapest first experiment: add `third_party/solv3r` and a `Solv3RAdapter` (copy the DéjàView adapter: same preprocessing, `cameras_from_rays` for cameras). Run the dev subset at k ∈ {4, 8, 12, 16, 24, 32, adaptive}, run `run_tto_pba.py` on each, and plot with `plot_compute.py`. This takes about a day of work and a few 5090 hours, or it can run on the 3080 at ~4 GiB.

**Idea B: BA as an external verifier and preconditioner for a stable learned solver (training-free).**
- Run k calls, decode, and run PBA or track BA. Then use the per-view BA residuals, or the disagreement between Solv3R and BA poses, to reset suspicious views to z_0 (P3) or to re-add them via `add_views`. At the same time, hold BA-consistent views fixed as KV-cached context and continue solving only the rest. Iterate.
- This targets Solv3R's stated failure mode, a wrong but stable fixed point (Supp. C.5), and DA3/DéjàView gross failures (our `knowledge/ideas/gross-failures-and-reference-view.md`).
- Novel: the classical optimiser acts as an outer loop that steers a learned stationary solver, closer to DROID-SLAM's GRU + DBA but training-free and with a GFM.
- Cheapest first experiment: on dev scenes with low per-scene AUC, compute per-view photometric residuals after PBA. Reset the worst 10–25% of views, run 8 more calls, re-decode and re-score. Compare against "8 more calls without reset" as a control, the analogue of their no-join control. Inference only.

**Idea C: TTO in the solver's latent state rather than in output or weight space (small engineering; 5090).**
- Treat the state z_k, or a per-scene low-rank offset on the injection W·LN(z_0), as the TTO variable. Backprop our photometric or feature-metric loss through the frozen decode heads, optionally also through 1–2 solver calls. Then hand z back to the solver, which projects it onto its learned fixed-point manifold (P2 says perturbed states get corrected).
- This differs from Test3R (prompt tuning of network weights), TTT3R (recurrent state as test-time training) and our PBA (explicit poses and depth grid). Stationarity is what makes an edited state a legal input; in VGGT or DA3, editing layer-l tokens has no consistent meaning.
- Cheapest first experiment: at call 16, run ~10–20 Adam steps on z, using only patch tokens and the camera token, with the photometric loss computed from decoded depth and cameras. Then run 4 more calls and compare against PBA on the same decoded outputs. The `@torch.no_grad()` wrappers must be bypassed by calling `model.net.block` and `model.net.decode` directly.
- Risk: the gradients through the DPT-style heads may be poorly conditioned. Fall back to optimising only the camera tokens.

**Idea D: compact initial computation + solver calls + BA, the latent analogue of "small model + TTO vs large model" (training-free first, small fine-tune second).**
- Training-free: reduce the encoder's cost by lowering the input resolution (e.g. long side 336 or 392 instead of 504; RoPE permits it) or by subsampling the views the encoder sees. Let more solver calls and then PBA, at full resolution, compensate. Measure where the solver's correction capacity (P2) recovers the loss.
- Small fine-tune: replace the ViT-B/14 encoder with ViT-S/14 trained to regress z_0, with block and heads frozen. The loss is feature regression to the ViT-B z_0 plus a fixed-point consistency term (decoded outputs after k calls match the teacher's). This needs only unlabeled multi-view images. z_0 targets can be cached offline as in OTT3R, and the frozen 117M teacher runs once.
- Novel: tests whether a stable learned solver tolerates a weaker initialisation better than a step-conditioned or layered model. That is our central question, moved into latent space.
- Cheapest first experiment: a resolution sweep at fixed calls plus adaptive stopping on dev; record AUC and the stopping call. Caveat from our notes: DéjàView at low K is very sensitive to resampling (`knowledge/ideas/distillation.md`), so the effect may be large either way.

**Idea E: few-call self-distillation of the solver's fixed point, as a "warm-start predictor" (moderate training; 5090s).**
- Train a LoRA on the block, or a tiny separate module, so that 2–4 calls reach the state of adaptive Solv3R at ~20 calls. Self-distillation with no GT needed (consistency or progressive distillation, as in diffusion). Use the student only for the first calls and hand over to the stationary Solv3R block and then BA, so the solver properties are kept.
- Feasibility: the block is about 14M parameters (inferred: two width-768 transformer layers ≈ 2 × 7.1M plus the 0.6M injection). With frozen encoder and heads and cached z_0, training is cheap per step. The paper's ablations used 3k-step fine-tunes.
- Novelty is moderate: it is distillation of iterations, not of model size. It is interesting only combined with Idea A's matched-compute curves.
- Cheapest first experiment: measure the gap between call 4 and call 20 on dev, since Fig. 5 shows it only graphically. If the gap is small, skip this idea.

**Idea F: convergence residual as a compute-allocation and uncertainty signal (training-free, cheap, combines with A and B).**
- Use r_k and the per-view residual (`residual(z, ctx, rows=...)`) at a fixed k to predict per scene and per view whether more calls, BA or a reset will help, and allocate the matched-compute budget adaptively.
- Cheapest first experiment: log r_16, the adaptive stopping call and per-view residuals on dev. Correlate them with the AUC gain from PBA and with per-view pose error.

**Feasibility summary.** A, B, F and D (training-free part) need only inference: about 4 GiB at 100 views (inferred), fine on the 3080/3070, or on tum2 in 6 h jobs. C needs gradients through the heads, so use the 5090. D (encoder distillation) and E need multi-day 5090 training with cached z_0: risky but bounded. A full Solv3R-style retraining (≈1,350 H100-hours) is out of reach.

**Positioning risk.** Solv3R already claims "no separate optimization backend" as a virtue. A paper from us must show that explicit optimisation still pays at matched compute (A), or that it fixes the learned solver's failure modes (B, C). Showing that Solv3R works is not enough.

### Gaps
- None of these ideas has been tested. Whether Solv3R runs correctly through our DA3-bench adapter (frame selection, 504 px preprocessing, ray-fitted cameras) has to be verified first, as was done for DéjàView.
- The per-call wall-clock time on our GPUs, and the accuracy at fewer than 8 calls, are unknown until measured.
