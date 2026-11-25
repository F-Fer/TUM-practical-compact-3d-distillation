# Geometry-guided match refinement (GMR): the feed-forward geometry as the coarse matcher, a small learned refiner, then BA

Status: **open, positive: the direction to scale (clean-fold held-out SMALL / BASE / LARGE 71.5 / 76.7 / 80.2 vs B′ 44.1 / 57.0 / 69.9; one seed, RTX 3080; write-up `reports/Geometry-guided match refinement.md`)** · Area: TTO method /
learned refinement step · Updated 2026-11-25

**Answer.** Yes. Feeding the feed-forward geometric warp to RoMa v2's refiners and running BA works only once the 1/4
refiner is fine-tuned on feed-forward errors (15 min). With RoMa's weights it captures only errors of ≤ 2-3 px. Fine-tuned,
held-out with clean folds: SMALL 71.5, BASE 76.7, LARGE 80.2 AUC@3, against NCC-PBA 57.2 / 68.0 / 74.3 and RoMa + BA
67.1 from SMALL. The gain saturates after ~8 training scenes. Trained on ARKitScenes only, it reaches 68.1. Cost is
4-17 s per scene on the 3080. The next step is a purpose-built refiner (wider, iterative, on DA3's own features), not
more data.

**Idea.** A dense matcher such as RoMa v2 is a large coarse stage (DINOv3 ViT-L descriptor + transformer matcher, warp at
1/16) followed by three small convolutional refiners (1/4, 1/2, 1/1 on VGG19-BN features; local correlation of ±3 cells at
1/4) that correct a given warp. With a feed-forward reconstruction the coarse warp of every covisible pair is already
there: back-project the source pixels with the predicted depth and project them with the predicted cameras. So: skip
the descriptor and the matcher, let the refiners correct the geometric warp, and run a reprojection BA with free 3D
points on the corrected matches. This is the "learned step that sees a new measurement" the learned-refinement report
asked for ([learned-refinement-step](learned-refinement-step.md)): PBA's linearised photometric residual is replaced
by a learned local correlation + regression, and the BA has free points, so DA3's depth no longer limits the pose
([depth-bottleneck](depth-bottleneck.md)).

**Code.** `scripts/gmr_ba.py <src> <dst>` (src = any saved state: DA3's predictions or a PBA run). Per source frame:
`geo_warp` into its 10 spread neighbours (`run_tto_pba.select_neighbors`) → RoMa v2's `refiners` on its
`refiner_features` (the model's `f` and `matcher` are deleted) → 1,024 random confident source pixels become BA points
with one observation in the source frame and one per neighbour whose overlap confidence is > 0.5 → `sba_fast.BA`
(GPU LM, shared focal + radial, Cauchy 0.5 px). `stages` = refiner schedule: `4,2,1` is RoMa's; `4@2` / `4@4` run the
1/4 refiner on images downscaled by 2 / 4 first (wider search); a repeated stage is iterated. `rounds`, `loss_rounds`,
`gnc`: re-match after the BA, robust scale per round. `eval_matches=1`: match error against GT depth + cameras, binned
by the initial error (`scripts/gmr_basin.py`). `ba_init=gt`: BA from the GT cameras with the same matches.
`scripts/gmr_train.py`: fine-tunes the refiners for this use. `scripts/lpba_scenes.py`: per-scene table of runs.

## Evidence

SMALL, dev (22 scenes), seed 0, 3080. References on the same GPU ([fast-sparse-ba](fast-sparse-ba.md)): B′ `iters=25` 42.3
at 1.0 s; NCC-800 55.6 at 13.5 s; RoMa v2 turbo 8 query frames + GPU BA 55.6 at 26 s; RoMa 30 covisible frames per query
64.6 at 20 s. AUC@3 per dataset H / 7-Sc / S / DTU / E.

| date | setting | change | result | source |
|---|---|---|---|---|
| 11-25 | from B′ (`sba_ref/pho25`), 2,048 pts, stages 4,2,1, 1 round | first run | **53.8** (53.4 / 36.3 / 31.9 / 99.9 / 47.5) at 12.2 s (match 7.8 s median, 20 ms per pair; BA 1.3 s) | `gmr/d_bp` |
| 11-25 | same from DA3 alone | no PBA first | 18.4 (init 16.2): DA3-SMALL's warp is outside the refiners' range | `gmr/d_da3` |
| 11-25 | from B′, 3 rounds | re-match after each BA | 53.9: scenes that are stuck stay stuck | `gmr/d_bp_r3` |
| 11-25 | from B′, 1,024 pts | half the points | 54.0, BA 0.66 s median | `gmr/e_p1k` |
| 11-25 | same | + coarse stages: `4@2,4,2,1` / `4@4,4@2,4,2,1` | 57.2 (57.8 / 36.5 / 34.2 / 100 / 57.3) / **61.8** (51.1 / 36.7 / 41.1 / 100 / 80.4): ETH3D delivery_area 15 → 96, ScanNet++ 21d970d8de 28 → 54, but two HiRoom scenes collapse with `4@4` (22 → 0.5, 49 → 14) | `gmr/e_c2`, `e_c4` |
| 11-25 | from NCC-800 (55.6) | stages 4,2,1 | 60.7 (57.8 / 35.1 / 50.7 / 99.9 / 60.1) | `gmr/e_ncc` |
| 11-25 | from DA3 alone | `4@4,4@2,4,2,1`, 1 round | **48.7** from 16.2 (38.0 / 36.1 / 21.5 / 100 / 47.8): above B′ without any PBA | `gmr/e_da3_c4` |
| 11-25 | from B′, stages 4,2,1 | **BA started at the GT cameras**, same matches (`ba_init=gt`; H / S / E) | same result and same final cost as from B′'s cameras (53.1 / 32.1 / 50.9 vs 53.5 / 32.0 / 48.1): the BA has one optimum and it is where the matches put it; stuck scenes are stuck because of the matches | `gmr/f_gt` |
| 11-25 | same with `4@4,4@2,4,2,1` | BA from GT | 53.5 / 53.0 / 85.7 vs 51.1 / 41.1 / 80.4 at equal final cost: with the wider (noisier) matches the Cauchy 0.5 px BA has several optima | `gmr/f_c4_gt` |
| 11-25 | from B′, HiRoom dev (GT depth + cameras), stages 4,2,1 | **match error by initial error** | init < 1 px (38 % of matches): median after 0.44 px, 97 % < 2 px; 1-2 px: 0.78 px, 88 %; 2-4 px: 1.6 px, 59 %; 4-8 px: 4.0 px, 24 %; 8-16 px: 9.1 px, 7 %. Overlap confidence > 0.5 for 92-99 % in every bin. **RoMa's refiners correct ≤ 2-3 px here and don't flag what they miss** | `gmr/m_eval`, `scripts/gmr_basin.py` |
| 11-25 | same, `4@4,4@2,4,2,1` | | HiRoom 4-8 px: 3.7 px, 35 %; 8-16 px: 5.9 px, 25 %; but < 1 px: 0.70 px, 79 % (the coarse stages damage good matches). ScanNet++ (one scene): 4-16 px → 1.2 px median, 64-66 % < 2 px | `gmr/m_eval_c4` |

### Fine-tuning the 1/4 refiner for this use (`scripts/gmr_train.py`)

Checkpoint `a` (`workspace/tto/DA3-SMALL/gmr/ft/a.pt`): RoMa v2's 1/4 refiner only (the 1/2 and 1/1 refiners and the
VGG features frozen, BatchNorm in eval mode), 3,000 steps x 1 source frame x 4 neighbours, 15 min on the 3080, AdamW
1e-4. Data: the **held-out** scenes of HiRoom (20) and ScanNet++ (17) with GT depth + cameras; the initial warp of a
step is the geometric warp of DA3-SMALL's prediction or of its B′ refinement (random), the target the GT warp where
the GT says visible; schedule `4@2,4,4,2,1` with a robust L1 loss (px) on the trained stages' outputs and a BCE loss on
the overlap logit towards "visible and within 3 px". The dev scenes and 7-Scenes / DTU / ETH3D are never seen.

| date | setting | change | result | source |
|---|---|---|---|---|
| 11-25 | SMALL dev, from B′, 1,024 pts, `stages=4@2,4,4,2,1`, 1 round | **fine-tuned refiner `a`** | **68.9** (62.8 / 36.3 / 55.9 / 100 / **89.4**) vs RoMa's weights 61.8 at best (`4@4,4@2,4,2,1`), B′ 42.3, NCC-800 55.6, RoMa p30 + BA 64.6; ETH3D is an unseen dataset; 16.7 s unoptimised (match 12.7 s) | `gmr/t_a` |
| 11-25 | same, from DA3 alone (16.2) | no PBA | **62.5** (56.7 / 35.9 / 47.9 / 100 / 72.0) vs 48.7 with RoMa's weights | `gmr/t_a_da3` |
| 11-25 | HiRoom / ScanNet++ dev, from B′ | match error by initial error, refiner `a` | HiRoom 4-8 px: 44 % < 2 px (RoMa's weights 24 %), of the kept ones 57 %, 67 % kept; 8-16 px: 50 % kept, of those 51 % < 2 px, median 2.0 px (9.1 px); ScanNet++ 8-16 px: 67 % kept, 65 % < 2 px; 16-32 px: 52 % kept, 63 % < 2 px. Kept fraction now falls with the initial error (96 → 8 % from < 1 px to 32-64 px): **the confidence became honest**. < 1 px: 0.47 px median (0.44), 91 % < 2 px (97 %) | `gmr/t_a_eval` |

### Held-out with clean splits, cost, other training data

| date | setting | change | result | source |
|---|---|---|---|---|
| 11-25 | **SMALL held-out (59 scenes)**, from B′ (44.1), `4@2,4,4,2,1`, 1,024 pts | two refiners, each fine-tuned 3k steps on dev + one half of the held-out HiRoom / ScanNet++ scenes (`subset=cv_a` / `cv_b`) and applied to the other half; 7-Scenes / DTU / ETH3D (never in training) with `cv_a` | **71.5** (86.9 / 23.4 / 73.6 / 100 / 73.8) at 20.1 s vs NCC-PBA 57.2 (15 s), RoMa 8 queries + BA 67.1 (36 s), RoMa p30 + BA 65.0 (19 s), SuperPoint + LightGlue + BA 45.8 | `gmr/t_cv_ho`, `ft/cv{a,b}.pt` |
| 11-25 | SMALL dev from B′, checkpoint `a` | cheaper settings: refiner resolution 288 x 384 / + 6 neighbours / no full-resolution refiner / 288 x 384 + 6 nbrs + no 1/1 + 15 LM it. / **240 x 320 + 6 nbrs + no 1/1 + 15 it.** | 69.0 at 10.1 s / 68.0 at 6.7 s / 68.7 at 14.4 s / 68.3 at 5.5 s / **68.3 at 4.2 s** (match 2.8 s, BA 0.35 s) vs 68.9 at 16.7 s | `gmr/s_*` |
| 11-25 | refiner stage timing, 3080, bf16, per pair | `prof.py` (scratchpad) | 384 x 512: 1/4 refiner 11.4 ms, 1/2 5.5, 1/1 4.9, VGG features 3.1 ms per frame; 240 x 320 about 1/3 of that | `gmr/prof.log` |
| 11-25 | ARKitScenes windows of `lpba_prep_arkit_pose.py` | check of the GT poses | lowres_wide.traj is 10 Hz; vga frames sit -33 / +33 / 0 ms from the stored nearest pose in turn, median 3.9° per trajectory step: ~1.3° (~10 px) pose error on two thirds of the frames. A first ARKit training run (100 steps) didn't reduce the match error at all (in 15.9 → out 15.8 px). Fixed by `scripts/gmr_arkit_interp.py` (slerp + centre interpolation; change median 0.5-0.8°, max 2.5°). **The earlier ARKit fine-tuning results that used GT poses (`lpba_ft_head.py` ark_sens / ark_photo) used the nearest poses** and may have suffered from it (the depth-loss runs and `lpba_selfdistill.py` don't use them) | `gmr/arkit_interp.log` |

| 11-25 | held-out (59), clean folds as above | BASE / LARGE-1.1 from their B′ (57.0 / 69.9), refiners trained on SMALL states only | **BASE 76.7** (92.9 / 23.9 / 85.7 / 99.9 / 81.0) vs NCC 68.0, RoMa + BA 67.9; **LARGE 80.2** (95.9 / 23.6 / 96.7 / 100 / 85.1) vs NCC 74.3, RoMa + BA + prior 70.4 | `DA3-*/gmr/t_cv_ho` |
| 11-25 | SMALL held-out, recon F1 (4 sets) | | **58.8** (64.5 / 43.5 / 54.0 / 73.2) vs RoMa + BA 57.0, NCC 53.2, B′ 51.1 | `gmr/t_cv_ho.recon.log` |
| 11-25 | dev recon F1, BASE / LARGE (checkpoint `a`) | | BASE 68.7 vs B′ 61.9; LARGE 71.9 vs B′ 70.2 (7-Scenes the only set that drops: 59.1 → 54.9) | `DA3-*/gmr/t_a.recon.log` |
| 11-25 | SMALL dev, checkpoint `a`, seeds 0 / 1 / 2 | | 68.9 / 67.8 / 68.7 | `gmr/t_a`, `t_a_s1`, `t_a_s2` |
| 11-25 | same | depth: per-frame scale to the BA points (default) / + smooth 16-cell grid fitted to them (`depth_fit=grid:16`) | depth abs. rel. vs GT barely moves (HiRoom 4.34 → 4.15 / 4.07 %), two ScanNet++ scenes get much worse (4.8 → 9.5 %, 8.5 → 10.1 %: noisy per-frame scales); pose identical (68.9) | `gmr/t_a_g16` |

| 11-25 | dev, checkpoint `a`, `4@2,4,4,2,1`, **from the raw feed-forward prediction (no PBA)** | SMALL / SMALL 2 rounds / BASE / LARGE-1.1 / DéjàView (released, K = 16) | SMALL 16.2 → 62.5 / 66.9 (33 s); BASE 27.1 → 68.9; LARGE 48.8 → 79.2 (with B′ first 81.2); **DéjàView 56.1 → 79.8** (79.3 / 36.6 / 86.9 / 100 / 96.4): the refiner, trained on DA3-SMALL states, works on another model's output | `gmr/t_a_da3*`, `t_a_dvlt`, `DA3-*/gmr/t_a_raw` |

| 11-25 | **ARKitScenes only** (400 windows of 200 training videos, interpolated poses, LiDAR depth; states DA3-SMALL + B′ on 150 of them), 3k steps; no benchmark scene or dataset | SMALL dev from B′ | **64.8** (50.8 / 32.0 / 51.2 / 100 / 89.9) vs RoMa's weights in the same schedule 56.7, benchmark-trained `a` 68.9: most of the fine-tuning gain survives training on a different dataset; HiRoom (synthetic) gains least (54.7 → 50.8, below RoMa's weights), ETH3D most | `gmr/t_ark_dev`, `ft/ark.pt` |
| 11-25 | same ARKit-only refiner, **held-out 59 scenes** (`subset=all` on `sba_ho/pho25`), SMALL from B′ (44.1) | | **68.1** (78.5 / 23.6 / 72.0 / 99.9 / 66.5) vs clean-fold benchmark-trained 71.5, RoMa + BA 67.1, NCC-PBA 57.2: no benchmark data at all and still above RoMa's full pipeline; gap mostly ETH3D / HiRoom (domain) | `gmr/t_ark_ho`, `t_ark_ho2.log` |
| 11-25 | learning curve, fold `a` (32 training scenes) → its 18 HiRoom / ScanNet++ test scenes + 5 held-out ETH3D, cheap config (240x320, 6 nbr, `4@2,4,4,2`, 15 it.) | RoMa weights / 3k steps on 8 / 16 / 32 scenes / 10k steps (one-cycle; checkpoints 1k, 3k, 10k) | HiRoom/ScanNet++/ETH3D: RoMa 66.9/45.8/54.1; 8 sc. 75.9/60.9/72.4; 16 sc. 83.6/60.4/73.0; 32 sc. 83.8/61.2/73.1; 10k: 80.6/64.2/72.3 → 84.5/65.6/72.5 → 85.3/64.0/73.5. **Saturates after ~8 scenes / 1k steps** (unseen ETH3D +1 from more data or steps): the fine-tuning teaches the error type and the confidence, not content; capacity / reach of the refiner are the next levers | `gmr/lc_*`, `ft/lc10k*.pt`, `ft/lcd{2,4}.pt` |

## Reading

- The correspondences, not the BA, are the limit: the pretrained refiners were trained on the small errors of RoMa's own
  matcher, so in this use they capture only errors up to 2-3 px and are confident about the rest. Everything that
  widened their range (coarse stages) helped, at the price of noise on already-good matches.
- That is a training-distribution problem with a measurable target (the table above): a refiner trained on the errors
  of a feed-forward initialisation, iterated, with an honest confidence.

## Open / don't repeat

- Pitfall: `gmr_ba.py subset=held-out` drops the first DEV[ds] scenes of the source; with a held-out-only source
  (`sba_ho/*`) use `subset=all`. `t_ark_ho` was first run with the wrong subset (37 of 59 scenes), `t_a_ho3` too (13).

- More rounds with the same stages and Cauchy 0.5 px don't help (matches near the current geometry dominate).

Related: [fast-sparse-ba](fast-sparse-ba.md), [learned-refinement-step](learned-refinement-step.md),
[depth-bottleneck](depth-bottleneck.md), [correspondences-in-pba](correspondences-in-pba.md), [track-ba](track-ba.md)
