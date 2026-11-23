# Scaling resolution distillation: DA3-GIANT teacher, DéjàView student at 252-420 px

Status: **answered: negative** for the GIANT teacher (G0 / S0 / S1 on tum2, 2026-11-21); **positive** for camera
diversity (S0b, 2026-11-23) · Area: distillation + feed-forward compute · Owner: Finn (+ Claude) · Updated 2026-11-23

**Answer: no.** The question was: does a stronger teacher (DA3-GIANT-1.1 at 504 px, 67.0 AUC@3 alone vs 61.3 for
the released DéjàView K = 16) raise the resolution-distilled DéjàView student beyond what more data alone gives
(self-teacher control)? G0 passed (GIANT + TTO 73.7 vs 72.1 on the benchmark; on ARKit clips GIANT beats DéjàView at
AUC@3 / 10 / 30 by +13.6 / +11.4 / +3.7 once portrait videos go in upright), but the GIANT-taught student S1 is
worse than the self-taught S0 at all six (res, K): **−4.0 to −4.5 AUC@3 alone, −0.2 to −2.3 after TTO** (all 81
scenes), on every dataset but DTU. S1 did learn GIANT on held-out ARKit clips (val ray loss vs GIANT −37 %), so the
targets were followed; why that hurts the benchmark is not tested (candidates below).

**Camera diversity (S0b, 2026-11-23): yes.** S0's students were held back by a single training camera (DTU focal
error 6-8 %). Adding DL3DV and random crops / aspects / zoom to the self-distillation (S0b) lifts every setting by
+3 to +5 after TTO: **336 px K = 12 71.2 after TTO with fhk poses, 70.8 with camera-head poses (no RANSAC)** vs S0
67.0 / 66.4 and R2 66.0. DTU after TTO 85.5 → 94.6. Like for like (camera-head poses, 5090 forward), the released model
gets 71.7 / 71.8 at 504 px K = 16 / 12 (1.06 / 0.87 s): the 336 K12 student is 1.0 behind at 0.26 s. The
multi-setting run S0b_mix is within 1 point at 336 K12 (69.9 head) and gives the best student: **420 px K = 12 with
head poses 71.7 at 0.49 s, equal to the released 504 px K = 12 (71.8, 0.87 s)** and +8.2 over the released model at
420 px (63.5); 252 px K = 16 63.1 (S0 59.0). Its 504 px K = 12 lost 1.9 (69.9).

## Why (from the R2 / R3 results in [dejaview-speed](dejaview-speed.md))

- At 420 px K = 12 the R2 student is already close to its teacher (58.5 vs 61.3 alone, 70.3 vs 72.1 after TTO with
  clean intrinsics); a student cannot pass its teacher.
- R3 (1.75 h more, lower lr, 25 % of clips from 11 outdoor scenes) changed nothing after TTO.
- GIANT leads the current teacher on SN++ (+7.8), DTU (+11.1) and ETH3D (+11.7). After TTO its lead is smaller on dev
  (74.9 vs 73.1 for DéjàView with clean K); the G0 ceiling run measures it on all 81 scenes.
- C1 ([dejaview-tto-teacher](dejaview-tto-teacher.md)): targets only slightly better than the teacher (+5.5 AUC@3 vs
  ARKit GT, mostly focal) gave nothing, hence the G0 gate on ARKit clips.

## Setup

- **Data** (no GT, nothing from the benchmark): 765 ARKitScenes Training videos (`prep_arkit.py --n 1200 --seed 0`,
  a superset of R2's 200; the clip list was frozen at the 765 videos complete on 2026-11-21 08:14 to catch a free
  node, the download continues for later runs), the same 8 Validation videos as R2, and the 11 outdoor scenes (Mip-NeRF 360 `images_4`,
  T&T + Deep Blending; `prep_outdoor.py` from the local zips). Frames in `/scratch/vc15/pr26rb00/distill_data/arkit_frames`
  (`workspace/distill_data` links there). DL3DV-10K would add outdoor variety but is gated on Hugging Face (needs the
  user's license acceptance and a token on tum2), so not used yet.
- **Fixed clips** (`giant_cache.py --make_clips --per_video 4`, `clips.json`): 3,390 clips = 4 per ARKit video + 30
  per outdoor scene, 134k frames,
  16-64 views (contiguous window of 2-4x the view count); one 24-view clip per val video (= R2's val clips).
- **Teacher cache** (`giant_cache.py`, 2 shards, `slurm_giant_cache.sh`): on the same DéjàView-preprocessed 504-px
  tensor, DA3-GIANT-1.1 (ImageNet-normalised, no DA3 resize, ref view "first") and released DéjàView K = 16. GIANT is
  moved into DéjàView's gauge: similarity mapping GIANT camera 0 onto DéjàView camera 0, scale = median depth ratio
  (DéjàView's losses normalise scale, so its output scale is a convention the student keeps). Stored: aligned c2w,
  GIANT intrinsics, aligned depth (fp16). Portrait videos (sky Left/Right) go into GIANT turned upright and its
  predictions are turned back (`--upright` / `--redo_rot`, see the G0 rows).
- **Training** (`train_res.py --clips ... --teacher dvlt|giant`, `slurm_res_train.sh`): init released, res
  {252, 336, 420} x K {8, 12, 16}, lr 8e-6, 300 warmup, cosine, 5.0 h on one RTX 5090, losses as R2 (rays at the
  student resolution, log depth and 0.1 log conf at 504 px, pose_enc). S0: online released DéjàView 504 px K = 16 as
  teacher. S1: same online teacher (for the conf target only, so both runs cost the same per step), rays from the
  aligned GIANT cameras (`compute_world_rays`, pinhole, as DéjàView's own GT supervision), depth and pose_enc from
  GIANT. S1 ran in two parts with the same 5.0 h schedule: until `--deadline` inside job 3661, then
  `slurm_res_resume.sh` (`--resume`: same step-indexed data order and lr schedule) followed by the eval.
- **Scripts removed 2026-11-23** (never committed, the GIANT-teacher line is closed): `giant_cache.py` (fixed clips,
  GIANT teacher cache, G0 diagnostics), `slurm_giant_{cache,ceiling,smoke}.sh`, `slurm_res_train.sh`,
  `slurm_res_resume.sh`, `submit_res_scale.sh` and `train_res.py --clips / --teacher giant`. Copies (with the full
  `train_res_with_giant.py`) in `/cluster/vc17/pr26rb00/archive/2026-11-23_giant_teacher_scripts/` on tum2 (HDD, not
  backed up, deleted after 90 days untouched). S0b needs only `prep_dl3dv.py`, `slurm_res2_train.sh`,
  `slurm_res_eval.sh` and `compare_rs.py`.
- **Eval** (`slurm_res_eval.sh`, `compare_rs.py`): all 81 pose scenes, RoPE stretched, ray-fit extrinsics +
  camera-head intrinsics (fhk), photometric TTO T = 25; (res, K) = 336/16, 420/12, 252/16, 336/12, 252/12, 420/8.
  Forward times are the 3080 medians of the architecture at that (res, K) from dejaview-speed.md.

- **S0b (2026-11-22, jobs 3775-3779):** S0 reworked for camera diversity, evaluated with camera-head poses
  (the deployment mode: no RANSAC ray fit, `POSE=head slurm_res_eval.sh`, stored as run `<name>-head`) and fhk at
  336 K12. Data: a fresh random clip per step (as R2, not S0's fixed clips), 12-48 views, from ARKit (all 1,200
  videos, 50 %), DL3DV-10K (1,000 scenes of the 1K / 2K batches of `DL3DV/DL3DV-ALL-960P`, 120 frames each, 960x540,
  `prep_dl3dv.py`, 40 %) and the 11 outdoor scenes (10 %); one centred crop per clip (`train_res.py --aspects
  native,4:3,1:1,3:4 --crop_min 0.5 --p_zoom 0.5`: square / portrait inputs, half the clips zoomed to 50-100 %, i.e.
  ARKit 62° → 33-62° field of view). `slurm_res2_train.sh`, lr 8e-6, 5 h, released init. Two runs on the same data
  and time: **S0b_mix** res {252, 336, 420, 504} x K {12, 16} (K = 8 dropped, 504 back as anchor) and **S0b_336K12**
  (single setting), to answer whether training several (res, K) helps beyond flexibility. Plus S0 in camera-head
  mode as the baseline (jobs 3775-3779).

## Evidence

| date | setting (model, set, seeds, baseline) | change | result | source |
|---|---|---|---|---|
| 2026-11-21 | DA3-GIANT-1.1 504 px, saved predictions of the 2026-11-02 full run, all 81 pose scenes, 5090 | G0 ceiling: photometric TTO T = 25 (defaults) | AUC@3 67.0 → **73.7** (dev 67.8 → 75.1, held-out 66.2 → 72.9); per dataset after TTO HiRoom 90.3 / 7Sc 28.6 / SN++ **80.6 (alone 84.7: TTO hurts)** / DTU 94.3 / ETH3D 74.9. vs DéjàView K = 16 + TTO 70.9 (fit) / 72.1 (clean K): the teacher's lead after TTO is +1.6-2.8, alone +5.7 | job 3657, `workspace/tto/DA3-GIANT-1.1/pho_it25_all` |
| 2026-11-21 | smoke cache, 5 clips | round trip GIANT cameras → `compute_world_rays` → DéjàView RANSAC ray fit | rotation ≤ 0.04° (patch-resolution fit), centre ~1e-7, focal ~1e-6: the conversion is exact; depth GIANT vs DéjàView after scale: abs-rel 0.03-0.12; ~2-5 s per clip (24-59 views) | job 3657 |
| 2026-11-21 | G0 gate, full cache: 3,390 train clips (2,813 with ARKit traj) + 8 val, GIANT on the frames as stored | GIANT vs DéjàView K = 16 (ray fit) vs ARKit VIO poses | **gate failed as defined** (AUC@10/30): AUC@3 / 10 / 30 GIANT 21.6 / 48.0 / 65.2 vs DéjàView 16.7 / 53.1 / 79.1. Cause: **portrait videos** (metadata `sky_direction` Left, 46 % of the clips, frames stored rotated by 90°): GIANT fails (AUC@30 > 0.3 below DéjàView) on 61 % of them, on Up / Down videos on 3 %. Up: GIANT 34.3 / 69.0 / 86.6 vs DéjàView 21.2 / 59.9 / 83.3; Down: 26.0 / 62.9 / 83.0 vs 18.5 / 55.1 / 80.1; Left: 10.8 / 26.4 / 41.7 vs 12.7 / 47.4 / 75.7 (DéjàView also loses on Left, less). Depth abs-rel GIANT vs DéjàView does not flag the failures (0.08 vs 0.04 median). Round trip over all clips: rotation ≤ 0.056°, centre ≤ 2e-6, focal ≤ 6e-5 | jobs 3658/3659, `giant_cache/diag_{0,1}.jsonl` |
| 2026-11-21 | the 1,414 portrait clips (+ 2 val) | `giant_cache.py --redo_rot`: GIANT gets the clip turned upright (rot90), predictions turned back (K, c2w, depth) | **G0 gate passed**: all 2,813 GT clips AUC@3 / 10 / 30 GIANT **30.3 / 64.5 / 82.8** vs DéjàView 16.7 / 53.1 / 79.1; GIANT better on AUC@10 in 86 % of clips; portrait clips (Left) now 29.8 / 62.4 / 80.1 vs 12.7 / 47.4 / 75.7; GIANT still fails (AUC@30 > 0.3 below DéjàView) on 7.4 % of all clips (~12 % of the portrait ones, 3 % of the others); val (7 GT clips) 27.2 / 63.8 / 83.7 vs 16.5 / 54.1 / 80.0. Conversion exact (`rot_ray_err` ≤ 3e-7). 69 min for 1,416 clips. The first S1 attempt (job 3661, rotation-bug targets) was stopped after ~150 steps; S1 restarted inside the same allocation at 13:23 (1.80 s/step, 10,003 steps scheduled) | `giant_cache/diag_rot_0.jsonl`, `$SCRATCH/jobs/s1_inalloc.sh` |

| 2026-11-21 | S0 (self-teacher control): released init, 3,390 fixed clips, res {252, 336, 420} x K {8, 12, 16}, 10,000 steps / 5.3 h on a 5090; all 81 pose scenes, fhk, photometric TTO T = 25 (5090) | vs R2 (200 videos, res {336, 420, 504} x K {12, 16}, 2,550 steps / 3 h on the 3080) | all 81, alone → TTO: **420 K12 52.5 → 68.3** (R2 58.5 → 70.3; HiRoom 60.5 vs 76.0, the main loss; 7Sc / SN++ / DTU / ETH3D 26.8 / 70.8 / 76.2 / 28.3 vs 30.2 / 74.5 / 78.5 / 33.4); 420 K8 40.6 → 65.5; **336 K16 55.6 → 66.1** (R2 53.9 → 66.7); **336 K12 54.5 → 67.0** (R2 52.0 → 66.0); **252 K16 47.6 → 59.0**, 252 K12 46.5 → 58.5 (released at 252 K16: 9.7 → 21.3). 5090 forward s: 0.58 / 0.48 / 0.40 / 0.35 / 0.24 / 0.22. So ~4x the data and steps with the wider (res, K) mix help at 336 alone (+1.7 / +2.5) and make 252 usable, but cost 6 points at 420 (alone; 2 after TTO): the likely cause is the mix without 504 and with 252 / K = 8 pulling the shared weights further from the high resolutions (as R3 noted), not tested | job 3662, `workspace/distill_runs/compare_rs.md`, `results/pba/runs/dvlt_rs/` |
| 2026-11-21 | S1: as S0 but rays / depth / pose_enc targets from the cached aligned GIANT (conf from DéjàView), 10,003 steps / 5.25 h (8,697 in job 3661 until `--deadline`, rest in 3677 after `--resume`) | GIANT teacher vs S0 | all 81, alone → TTO (S0 in brackets): **420 K12 48.3 → 67.1** (52.5 → 68.3), 420 K8 36.6 → 63.2 (40.6 → 65.5), **336 K16 51.3 → 65.6** (55.6 → 66.1), **336 K12 50.3 → 65.1** (54.5 → 67.0), **252 K16 43.1 → 58.2** (47.6 → 59.0), 252 K12 42.2 → 58.3 (46.5 → 58.5). 420 K12 per dataset HiRoom / 7Sc / SN++ / DTU / ETH3D 50.1 / 22.2 / 66.1 / **77.5** / 25.6 vs 60.5 / 26.8 / 70.8 / 76.2 / 28.3; held-out after TTO 66.7 vs 68.2. ARKit val (8 clips), 336 K16, start → end: ray loss vs GIANT 0.0485 → 0.0304, pose 0.0348 → 0.0211; vs DéjàView ray 0.0390 → 0.0222 (S0 → 0.0175). (S0's val losses vs GIANT used the pre-fix GIANT targets it loaded at its start, so they are not comparable.) **Negative**: the uniform ~4.3 loss alone at every (res, K) points to something systematic, not to resolution | job 3677, `workspace/distill_runs/compare_rs.md`, `results/pba/runs/dvlt_rs/` |
| 2026-11-22 | S0 / S1 vs R2 per dataset after TTO (no new runs; TTO metric JSONs) | why S0 gains ~nothing over R2 after TTO | **At 336 px the average hides a trade:** after TTO S0 vs R2 HiRoom 79.5 vs 73.5 (K16) / 80.5 vs 72.1 (K12), ETH3D +1.1 / +3.2, but **DTU 84.0 vs 93.6 / 85.5 vs 93.1**. DTU focal error (fhk intrinsics, mean) before → after TTO: S0 336 K16 5.8 → 5.6 %, 336 K12 6.2 → 5.9 %, 420 K12 7.6 → 6.5 % vs released 504 K16 (fit, 5090) 3.0 → 4.9 %, released 336 (fit) 3.8 → 4.8 %; the TTO does not correct S0's focal (S1 starts as bad, 5.2-7.1 %, but its TTO brings it to 2.5-3.0 % and DTU to ~90). With R2's DTU, S0 would be 68.0 / 68.5 at 336 K16 / K12 (R2 66.7 / 66.0). At 420 K12 S0's HiRoom focal error is 2.7 % (released 420 1.0 %, S0 336 0.8 %), HiRoom alone 60.5 vs 76.0. Val ray loss vs the teacher on the same 8 clips: S0 336 K16 0.0175 vs R2 0.020, 420 K12 0.0186 vs 0.020, 504 K12 0.0183 vs 0.015: S0 imitates better in-domain at 336 / 420, which does not show on the benchmark. Likely (untested): all training videos share one iPad camera, so 4x the steps teach a fixed field of view (R2's focal on DTU not available on tum2; its DTU 93.6 after TTO suggests it is fine). Horizontal FoV, training clips (GIANT intrinsics): ARKit 61.9° (p5-p95 61.6-62.2°, 90 % of S0's clips), the 11 outdoor scenes 51.7-80.6°; benchmark GT: DTU **30.9°**, 7-Scenes 57.4°, HiRoom 60.0°, ScanNet++ 67.1°, ETH3D 83.0°. DTU's camera is far outside anything trained on; HiRoom, where S0 gains, matches ARKit. R2 trained on ARKit only (62°) and kept DTU, so the narrow camera set hurts only with S0's longer training | `results/pba/runs/dvlt_rs/*/pho_it25`, `workspace/tto/dvlt_fs/{base_K16,rope_res*}` (5090), `knowledge/runs.csv`, `giant_cache/train/*.npz`, `workspace/evaluation/DA3-GIANT-1.1/model_results/*/gt_meta.npz` |
| 2026-11-23 | S0b_mix (res {252, 336, 420, 504} x K {12, 16}, 8,748 steps / 4.67 h) and S0b_336K12 (336 px K = 12 only, 12,488 steps / 4.66 h), released init, lr 8e-6, same data and time on a 5090; S0 re-evaluated in head mode as baseline; all 81 pose scenes, photometric TTO T = 25 (5090); one training seed each | camera diversity: random clips (12-48 views) from ARKit 50 % / DL3DV 1,000 scenes 40 % / outdoor 10 %, one centred crop per clip (native / 4:3 / 1:1 / 3:4, half zoomed to 50-100 %); camera-head poses (`POSE=head`) as the main mode | AUC@3 all 81, alone → TTO. **336 K12:** S0b_336K12 fhk 57.0 → **71.2**, head 53.2 → **70.8**; S0b_mix fhk 56.3 → 70.3, head 52.7 → 69.9; S0 fhk 54.5 → 67.0, head 50.8 → 66.4; R2 (fhk) 52.0 → 66.0. **S0b_mix head:** 252 K16 44.6 → 63.1 (S0 head 43.3 → 58.2), 336 K16 54.0 → 70.3, 420 K12 55.1 → **71.7** (S0 head 49.9 → 68.7, R2 fhk 58.5 → 70.3), 504 K12 53.5 → 69.9 (released fhk 58.7 → 71.7). **DTU fixed:** after TTO 336 K12 S0 85.5 → S0b_336K12 94.6 (head 94.3), mix 93.6 (released 504 K16 94.1); DTU focal error before TTO 6.2 % → 2.4 % (single) / 3.2 % (mix; mix 420 K12 4.4 %, 504 K12 5.4 %). HiRoom after TTO 88.2 (S0 80.5, released K16 89.1), ETH3D 65.0 (S0 62.5, released 62.0); 7-Scenes flat (~28.5). **Head vs fhk:** −3.8 alone, −0.4 after TTO (HiRoom 66.0 vs 73.9 alone, 87.6 vs 88.2 after): the TTO repairs the head's poses, and head mode saves the ray fit (5090 forward 0.26 vs 0.35 s at 336 K12). **Single vs mix at 336 K12:** +0.5 alone / +0.9 after TTO (head) for 43 % more steps on that setting; the single-setting run forgets the others (val ray loss 504 K12 0.0448 vs 0.0177 mix, 420 K12 0.0278 vs 0.0194). Val ray loss on the ARKit val clips mix vs S0: 336 K16 0.0180 vs 0.0175, 420 K12 0.0194 vs 0.0186, 504 K12 0.0177 vs 0.0183, i.e. in-domain imitation unchanged, the gain is the cameras. Not separated: DL3DV vs crop / zoom vs random clips | `results/pba/runs/dvlt_rs/{S0-head,S0b_*}_*`, `S0b_{mix,336K12}_train`, jobs 3775-3779, `knowledge/runs.csv` |
| 2026-11-23 | released DéjàView, camera-head poses (`POSE=head`, run `released-head`), all 81 pose scenes, photometric TTO T = 25, 5090 (job 3889) | head-mode baseline for the S0b rows | alone → TTO: 504 K16 57.2 → **71.7** (1.06 s forward), 504 K12 55.2 → **71.8** (0.87 s), 420 K12 + RoPE 52.0 → 63.5 (0.50 s). After TTO head poses beat the ray fit (504 K16 70.9) and are close to the clean-K ray fit (72.1). DTU after TTO 95.2 / 93.8 / 80.0, ETH3D 62.8 / 67.2 / 57.0. vs S0b (head): 336 K12 70.8 at 0.26 s = −1.0 vs released 504 K12 at 30 % of its forward time; mix 420 K12 71.7 at 0.49 s = −0.1 at 56 %; mix 504 K12 69.9 = −1.9 at the same time | `results/pba/runs/dvlt_rs/released-head_*`, `knowledge/runs.csv` |

## Open / don't repeat

- Why S1 loses (untested candidates): (a) the student follows GIANT on ARKit but that does not transfer (ARKit-only
  data, as C1 suggested) — test: AUC vs ARKit GT of S0 / S1 / GIANT on unseen ARKit clips (more clips from the 8 val
  videos or the Validation fold, ~20 min of GPU); (b) GIANT's remaining failures (7.4 % of clips; depth abs-rel does
  not flag them, a pose-disagreement filter might); (c) mixed targets (GIANT rays / depth / pose, DéjàView conf)
  move the weights further from the released ones (val ray loss vs DéjàView 0.0222 vs 0.0175 for S0), which costs
  generalisation as the wider (res, K) mix did for S0 at 420 px.
- Don't repeat S1 with the same data hoping for a different answer; a GIANT teacher only makes sense again with (a)
  answered or with data where GIANT's benchmark lead lives (outdoor / object, DL3DV).
- S0's DTU focal error (2026-11-22 row): fixed by S0b (camera diversity, 2026-11-23 row); which part did it (DL3DV,
  crop / zoom, random clips) is not separated. The mix's 504 K12 still has 5.4 % on DTU (released 3.0 %).
- Single setting vs mix: one seed each and a ≤ 1-point difference at 336 K12; a second seed would say whether it is
  real. A mix trained at {336, 420} x K = 12 only (fewer settings to share the steps) is the obvious middle ground.
- Head mode alone is 3.8 points behind fhk (HiRoom / ETH3D poses); weighting `w_pose` up in training might close it.

- **ARKitScenes frames are stored as captured**: check `sky_direction` (raw/metadata.csv) before feeding them to a
  model that expects upright images. DA3-GIANT breaks on the 90°-rotated portrait videos; DéjàView degrades less.
  The student and the online DéjàView teacher still see the stored orientation (as in R1-R3).

- GIANT + TTO targets (now possible on DL3DV, where GIANT's lead lives); amount-of-data curve; GIANT ->
  DA3-LARGE at low resolution as a second architecture (S2, only if S1 beats S0).

Related: [dejaview-speed](dejaview-speed.md), [dejaview-tto-teacher](dejaview-tto-teacher.md),
[da3-resolution](da3-resolution.md), [matched-compute](matched-compute.md)
