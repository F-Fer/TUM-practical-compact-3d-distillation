# How fast and how good can track-based (sparse) BA on DA3 be, compared with PBA?

Status: **answered (2026-11-24)** · Area: TTO method / track BA / speed · Updated 2026-11-26 · Report: `reports/Fast sparse BA vs photometric BA.md`

**Answer (RTX 3080):** BA itself is cheap on the GPU (own LM: 0.1-0.2 s on keypoint tracks, 0.5-4 s on 0.15-1.7M RoMa
observations; pycolmap 15-125 s, same accuracy); the tracks cost the time and decide the accuracy. **Dense RoMa tracks +
GPU BA reach held-out AUC@3 67.1 / 67.9 / 70.4 (SMALL / BASE / LARGE + prior) at ~36 s** (19 s with 30 covisible frames
per query: 65.0 / 66.1 / 67.3), nearly independent of the model, vs PBA B′ 44.1 / 57.0 / 69.9 at ~1 s and NCC 57.2 / 68.0 / 74.3 at 15 s. SuperPoint + LightGlue + GPU BA costs 1-4 s but reaches only 45.8 / 47.0 /
48.7 held-out (61.4 LARGE with a pose prior). Recon F1 (SMALL held-out): RoMa BA 57.0 > NCC 53.2 > B′ 51.1; on dev PBA
has the higher F1 for every model. Skipping tracks therefore buys ~20-35x over accurate tracks (at -10 AUC for SMALL
NCC, 0 for LARGE B′) but only 2-4x over cheap keypoint tracks. Starting PBA from sparse cameras doesn't help (PBA returns
to its own depth-limited optimum).

## Method (`scripts/sba_fast.py`, summary `scripts/sba_sum.py`)

Everything on the GPU, timed like PBA's `opt_s` (image decoding excluded): pairs from DA3's covisibility
(`run_tto_pba.select_neighbors`, 10 spread neighbours per frame), SuperPoint / ALIKED / DISK + LightGlue (batched pairs)
or RoMa v2 query tracks (`feat=roma`, `scripts/tracks_roma.py`), tracks by greedy union-find (a union that would put
two keypoints of one frame into a track is skipped), points from DA3's depth (mean over the track), own LM BA
(analytic Jacobians, dense Schur complement over point chunks, Cauchy loss, one shared focal + radial coefficient,
frame 0 fixed), DA3's depth scaled per frame to the BA points.

Pitfall found on the way: connected components of the match graph with conflicting components dropped (the obvious
vectorised track builder) keeps ~1.5 % of the observations (DTU scan, 66k matches → 975 observations; greedy:
23.7k observations in 0.03 s).

Second pitfall: clamping the camera depth z of a point to ≥ 1e-6 before projecting (to avoid division by zero) wrecks
BA on large scenes: a point behind a camera gets Jacobians ~1e6, its camera's diagonal dominates the LM damping and that
camera freezes (ScanNet++ dev scene on identical RoMa tracks: AUC@3 0.07 vs 0.47 for `ba_lm.py`; fixed by keeping the
sign, as `ba_lm.py` does). Speed: on GeForce cards fp64 runs at 1/64 rate, and cuBLAS batched GEMM on 2x8 blocks costs
~27 ms per call at 600k observations; per-observation products as broadcast multiply-adds in fp32, block sums
(`segment_reduce`) and the Schur product in fp64, the latter as three fp32 GEMMs (hi/lo split, `mm_split`): 50 LM
iterations at 615k observations 3.5-5.5 s (pycolmap: 52 s for the same scene, ba_lm.py ~35 s), same final cost as
ba_lm.py.

## Evidence

All on the RTX 3080 (tum1), DA3 predictions of `workspace/evaluation/<model>`, AUC@3 per dataset (H / 7-Sc / S / DTU / E)
and the 5-set mean, mean TTO s per scene (opt_s; image decoding excluded, networks warmed up). PBA references re-run on
the same GPU (`workspace/tto/<model>/sba_ref`): B′ `iters=25`, NCC `iters=800 lm_photo_loss=ncc lm_iters=40 lm_stop=0`.

| date | setting | change | result | source |
|---|---|---|---|---|
| 11-24 | SMALL dev, seed 0 | PBA references on the 3080 | B′ 42.3 at 0.99 s (44.9 / 37.0 / 24.8 / 87.0 / 17.8); NCC 55.6 at 13.5 s (54.7 / 33.9 / 42.0 / 93.3 / 54.2); VGG + Huber OOMs on the 100-frame scenes (3080) | `sba_ref/*` |
| 11-24 | same | SuperPoint 2k + LightGlue, 10 spread neighbours, greedy tracks, GPU LM 25 it, Cauchy 0.5 px, shared focal + radial | **48.6 at 2.8 s** (34.4 / 36.8 / 41.3 / 93.9 / 36.5); median stages: features 0.18, LightGlue 1.73, tracks 0.2, BA 0.16 s | `sba/s3_sp2k` |
| 11-24 | same | variants: 50 it / 4k kp / ALIKED / LightGlue fp16 / 5 neighbours / all pairs / every match a 2-view point / no radial / Cauchy 0.25, 1 px | 48.5 / 46.3 / 48.0 / 47.2 at 1.8 s / 47.5 at 1.7 s / 36.8 at 16 s / 44.4 / 45.1 / 46.4, 47.7: SuperPoint 2k, 10 neighbours, greedy tracks, radial and Cauchy 0.5 are the best; fp16 LightGlue or 5 neighbours trade 1 AUC for 1 s | `sba/s3_*`, `s4_*` |
| 11-24 | same | RoMa v2 turbo, 8 query frames (Antonin's tracks), GPU LM 50 it, Cauchy 0.25 | **55.6 at 26 s** (56.7 / 34.4 / 47.2 / 100 / 39.9): RoMa 4.4-61 s per scene (~70 ms per pair), BA 0.5-4 s; Antonin's pycolmap radial BA on the same kind of tracks 54.1 at 106 s (tracks 17.6 + BA 22 s median) | `sba/s4_r8_clean`, `rad_frame_rej` |
| 11-24 | same | RoMa 4 query frames, only the 10 most covisible frames per query | 42.3 at 7.1 s (ScanNet++ 5.5, DTU 43): restricting the pairs breaks the tracks | `sba/s3_r4p10` |
| 11-24 | same | **hybrids**: PBA started from the sparse cameras (`init_from`) | sparse SP → B′ 46.2 (< 48.6 sparse alone); sparse SP → NCC 55.4 (= NCC alone 55.6); RoMa → B′ 52.0 (< 56.0); depth-only polish keeps the sparse poses (48.6): PBA walks to its own (depth-limited) optimum whatever the start | `sba/h_*` |
| 11-24 | BASE dev | sparse SP / RoMa / PBA B′ / NCC | SP 49.5 at 2.8 s, + pose prior 30: 46.9; RoMa 61.4 (prior 100: 49.7); B′ 59.6 at 0.84 s; NCC 65.0 at 13.5 s | `DA3-BASE/sba/s4_*`, `sba_ref` |
| 11-24 | LARGE-1.1 dev | same | SP 51.5 at 2.9 s, **+ prior 30: 64.2**; RoMa 62.2, prior 100: 63.4; B′ **70.5 at 0.71 s**; NCC 72.0 at 13.5 s: for good inits sparse BA without a prior moves good poses away (as Antonin found) and is below PBA at any budget | `DA3-LARGE-1.1/sba/s4_*`, `sba_ref` |

| 11-24 | SMALL dev | RoMa 8 queries, only the 30 / 20 most covisible frames per query; 4 queries all frames | p30 **64.6 at 20 s** (56.8 / 32.6 / 66.5 / 100 / 66.9), p20 53.1 at 14 s (DTU 71.7), q4 55.2 at 19 s; BASE / LARGE p30 (SMALL's tracks) 66.8 / 68.1 | `sba/s6_*` |
| 11-24 | SMALL / BASE / LARGE dev | fastest keypoint setting: SuperPoint 1k, LightGlue fp16, 5 neighbours | 46.2 / 51.3 / 50.6 at 1.2 s; 2k fp16 + prior 30 on LARGE 64.3 at 1.9 s (SMALL 36.3, BASE 46.9: the prior only helps LARGE) | `sba/s5_*` |
| 11-24 | **held-out (59)**, SMALL | B′ / NCC / SuperPoint 2k / RoMa q8 / RoMa p30 | 44.1 at 1.1 s (54.1 / 22.4 / 34.6 / 77.9 / 31.7); NCC **57.2** at 15 s (76.6 / 24.6 / 49.8 / 83.9 / 50.9; = the 5090 value); SP 45.8 at 3.9 s (51.0 / 20.6 / 38.0 / 89.5 / 30.0); **RoMa 67.1 at 36 s** (73.2 / 23.5 / 79.1 / 99.9 / 60.0); p30 65.0 at 19 s | `DA3-SMALL/sba_ho/*` |
| 11-24 | held-out, BASE | B′ / SP / RoMa (SMALL's tracks) / + prior 100 / p30 | 57.0 / 47.0 / **67.9** / 59.2 / 66.1 | `DA3-BASE/sba_ho/*` |
| 11-24 | held-out, LARGE-1.1 | B′ / SP / SP + prior 30 / RoMa / + prior 100 / p30 | **69.9 at 0.8 s** / 48.7 / 61.4 / 68.5 / **70.4 at 36 s** / 67.3 | `DA3-LARGE-1.1/sba_ho/*` |
| 11-24 | held-out, BASE / LARGE-1.1 | NCC (`iters=800 lm_photo_loss=ncc lm_iters=40 lm_stop=0`) on the 3080 | BASE **68.0** at 15 s (85.8 / 24.6 / 63.2 / 93.5 / 72.7) = RoMa BA 67.9 at 36 s; LARGE **74.3** at 15 s (90.0 / 24.9 / 76.8 / 97.3 / 82.5) > RoMa BA + prior 70.4 (5090 values 68.3 / 74.8) | `DA3-*/sba_ho/ncc800` |
| 11-24 | recon F1 (4 sets), dev / SMALL held-out | `eval_tto.py recon_unposed` | dev SMALL B′ 47.8, NCC 51.6, RoMa 46.8, SP 43.2; BASE B′ 61.9, RoMa 55.7, SP 50.3; LARGE B′ 70.2, RoMa 61.0, SP 55.6; **held-out SMALL RoMa 57.0, NCC 53.2, B′ 51.1** (SP: fusion of a HiRoom scene OOM-killed the scorer) | `*.recon.log` |
| 11-26 | SMALL held-out (59 scenes), existing runs `sba_ho/{pho25,ncc800,r8}` + GMR `t_cv_ho` | per frame pair: share brought below 3°, binned by DA3-SMALL's initial pair error (max of rot / trans angle, evaluator functions; each scene weighted equally), `scripts/plot_error_basin.py` | initial 0-3° (25 % of pairs): B′ 93 / NCC 96 / RoMa 97 / GMR 98 %; 5-10° (28 %): 55 / 75 / 83 / 85; 10-20° (15 %): 30 / 57 / 70 / 72; 20-45° (6 %): 9 / 34 / 52 / 50; > 45° (3 %): 2 / 13 / 30 / 23. Photometric BA's basin ends at ~5-10°, sparse BA on dense matches still fixes half of the 20-45° pairs. Figure `results/pba/figures_depth/error_basin.png` | `plot_error_basin.py` |

## Open / don't repeat

- Don't compare PBA's speed with RoMa + pycolmap: the fair sparse baseline is this GPU pipeline (2-4x slower than B′
  with keypoints; 20-35x with RoMa).
- Dense-track BA is ~68 held-out for any model: the bar a no-track method has to reach from SMALL. The gap to NCC-PBA is
  ScanNet++ (50 vs 79), ETH3D (51 vs 60), DTU (84 vs 100); PBA wins HiRoom (77 vs 73).
- Untested: a matching head on DA3's tokens (Glob3R / BLASt3R style) as a cheap dense front-end; RoMa tracks with
  DA3 depth as a BA prior (BLASt3R's adjustable depth); RoMa at lower resolution; the sparse pipeline on the 5090.

## Literature (2025-2026), what the field does after a feed-forward model

| work | refinement | front-end / cost | note |
|---|---|---|---|
| VGGT+BA (CVPR 25) | COLMAP BA on VGGSfM tracks | track head + BA, minutes at 100+ views | the reference "+BA" |
| BLASt3R (NAVER, 2609.05210) | GPU LM BA on tracks + depth-regularised **adjustable depth maps** (linear combination of predicted depth channels) | own fast multi-view matcher (MUSt3R-style memory) | ETH3D mAA@30 98.5; with COLMAP BA on its tracks as good; depth regularisation matters for low-parallax / video |
| Glob3R (2607.09225) | rotation + translation averaging, then BA | dense matching head on frozen Pi3X tokens (faster and better than RoMa v2 or VGGT tracks for this) | ETH3D RRA@1 91.6 / RTA@1 73.3; 2 FPS |
| GGPT (2603.11174), Refin3R (2611.23051) | sparse BA on RoMa v2 matches, then a learned densifier | RoMa v2 + BA + triangulation; GGPT 4-13 s, Refin3R 0.2-0.9 s for the network | refine DA3's depth with SfM points |
| InstantSfM (2510.13310), BA in the eager mode | GPU sparse LM in PyTorch, depth priors in GP and BA | – | 1.5-40x faster than COLMAP on large scenes |
| DeepBA (TUM, 2611.18256) | learned GNN update of poses / FOV / 32x32 log-depth from matches, no tracks | eLoFTR matching dominates (∼7 s for 10 views on an H100) | track-free, between learned SfM and VGGT |
| **EPO** (2607.00579) | **trackless**: align Canny edges through the predicted depth (distance transforms), Adam on an MLP pose refiner, then per-pixel affine depth | no matching | beats VGGT+BA on ScanNet++ / TerraSky3D / MipNeRF360 (AUC@5 77.1 vs 70.0) at ~15 s on a 4090 vs 70-260 s; the closest prior work to "PBA on a feed-forward model" |
| VGGT-X (2509.25191) | epipolar (Sampson) loss on XFeat matches, poses only, no tracks, 300 Adam steps | XFeat | AUC@30 0.951 → 0.988 on MipNeRF360 in ~30 s vs 157 min for VGGSfM-BA |

Related: [track-ba](track-ba.md), [ba-iterations](ba-iterations.md), [depth-free-pose](depth-free-pose.md),
[matched-compute](matched-compute.md), [speed](speed.md)
