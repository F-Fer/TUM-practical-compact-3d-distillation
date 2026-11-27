# Where DéjàView spends its time, and cheaper inputs (resolution, encoder)

Status: **in progress** · Area: feed-forward compute · Owner: Finn (+ Claude) · Updated 2026-11-18

**Answer so far:**
- The 12-16 loop steps and the decoders take the time; the ViT-B encoder is only 4-9 % of a K = 16 forward
  (5090 and 3080). Replacing it with a ViT-S can therefore save ≤ 5 %. The decoder cost is the depth head's conv
  stack (~93 % of the depth decoder).
- The step cost grows about quadratically with the token count, so input resolution is the big lever: at 100 views
  K = 16 costs 3.4 s at 504 px and 1.1 s at 336 px (5090; 11.4 / 3.4 s on a 3080).
- The released model collapses at lower resolution because the loop's RoPE uses integer patch coordinates.
  **Stretching the RoPE positions to the native grid (training-free, `DVLT_ROPE=1`) fixes 420 px:** AUC@3 59.2
  (504 px: 61.3), after TTO 67.3 (70.9). It only partly fixes 336 px (34.0, after TTO 49.0) and not 252 px.
- At matched time on one GPU, 420 px + RoPE ties fewer loop steps after TTO (420 px K = 16 67.3 vs 504 px K = 8
  67.1 at ~2.2 s); 420 px K = 12 (66.7 at 1.76 s) is the cheapest point near there.
- **Camera-head poses instead of the RANSAC ray fit** (training-free, `DVLT_POSE=head`): −4 AUC@3 alone, but
  better after TTO at 504 px. **K = 12 + camera head + TTO reaches 71.6 at 2.80 s forward (3080), above the released
  K = 16 + ray fit + TTO (70.9 at 3.67 s).** At 420 px the head hurts (66.7 → 63.6).
- The depth head's output stage at 216x288 saves ~1 % and costs up to 2 AUC@3 after TTO: dropped.
- **Resolution distillation works (R2, 3 h on 200 ARKit videos, 3080, 2026-11-18):** a student trained to reproduce
  the released 504-px K = 16 output at 336/420/504 px (RoPE stretched) reaches at 336 px K = 16 **53.9 alone / 66.7
  after TTO at 1.27 s** (released 34.0 / 49.0), at 336 px K = 12 52.0 / 66.0 at 1.08 s, at 420 px K = 12 58.5 / 70.3
  at 1.76 s (released 56.0 / 66.7). Indoor-only data costs ETH3D and ~1.3 after TTO at 504 px.
- **The camera head's gain after TTO comes from its intrinsics, not its poses (2026-11-18):** ray-fit extrinsics +
  camera-head intrinsics (`DVLT_POSE=fit_headK`) reach **72.1** after TTO at K = 16 (held-out 71.5), head poses + fit
  intrinsics 70.4. The fit's focal is as accurate (same median focal / GT), but its principal point is ~1 px off
  centre and fx != fy by ~0.5 %; the TTO keeps the principal point fixed. Centring the fit's principal point and
  setting fx = fy gives the same 72.1 without the head.

Goal (from the strategy discussion, 2026-11-16): after K-reduction by self-distillation failed
([dejaview-few-step-distillation](dejaview-few-step-distillation.md)), cut DéjàView's fixed cost instead, so that a
cheaper DéjàView + TTO beats a slower one, or DéjàView fits weaker GPUs.

## Setup

Code in `scripts/distill_steps/`:
- `profile_dvlt.py` (CUDA-event timings, bf16, decode chunks of 8, fp32 pose fit, on real ARKitScenes frames);
- `eval_student.py` with `DVLT_RES`: it runs the net at a smaller longest side (antialiased resize of the 504-px
  input), then upsamples depth / conf to the 504 frame and rescales the intrinsics, so the evaluator and
  `run_tto_pba.py` see 504-px predictions;
- `slurm_profile.sh`; at commit f3ab013: `slurm_res_sweep.sh`, `slurm_encoder.sh` + `train_encoder.py` (ViT-S student,
  `DVLT_ENC` in `eval_student.py`);
- `rope_scale.py` (float RoPE positions stretched to the native grid, `DVLT_ROPE=1` in `eval_student.py`),
  `rope_dec_check.py` (RoPE equivalence check + decoder timing breakdown), `slurm_rope.sh` (commit f3ab013;
  `slurm_res_eval.sh` now covers both sweeps: `released` with any res:K, RoPE stretched).

`profile_dvlt.py` and `rope_dec_check.py` take `--images <folder>` to run on a local GPU without the ARKit zips.

Raw logs and metric JSONs are in `results/distill_steps/` (local, not in git).

## Evidence

### Profile (job 2886, RTX 5090, 2026-11-16)

Seconds per forward. Fixed = encoder + decoders + pose fit; step = one loop application.

| views | input | encoder ViT-B | ViT-S | 1 step | decoders | pose fit | fixed | K = 16 | peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| 24 | 504x378 | 0.033 | 0.012 | 0.0157 | 0.101 | 0.044 | 0.177 | 0.43 | 2.7 |
| 24 | 420x308 | 0.021 | 0.008 | 0.0087 | 0.067 | 0.042 | 0.130 | 0.27 | 2.1 |
| 24 | 336x252 | 0.013 | 0.005 | 0.0048 | 0.038 | 0.043 | 0.095 | 0.17 | 1.7 |
| 100 | 504x378 | 0.143 | 0.057 | 0.166 | 0.422 | 0.178 | 0.742 | 3.40 | 6.5 |
| 100 | 420x308 | 0.096 | 0.038 | 0.086 | 0.279 | 0.177 | 0.552 | 1.92 | 4.9 |
| 100 | 336x252 | 0.062 | 0.023 | 0.043 | 0.161 | 0.180 | 0.403 | 1.09 | 3.5 |

- **Share of a K = 16 forward at 504 px:**
  - At 24 views: loop 59 %, decoders 24 %, pose fit 10 %, encoder 8 %.
  - At 100 views: loop 78 %, decoders 12 %, pose fit 5 %, encoder 4 %.
- **Step cost** grows about quadratically with the token count: it is global attention over S × 972 tokens.
- **Decoders** cost 3x the encoder. They run per frame in chunks of 8, so a larger `decode_chunk_size` may be a free
  speed-up on the 32 GB card (untested).
- **Pose fit** (RANSAC on the ray map) costs the same at every resolution.
- **Why the decoders cost 3x the encoder with 1/6 of its parameters** (checkpoint: encoder 86.6M, loop block
  14.5M, depth decoder 3.9M transformer + 7.6M conv stack, ray decoder 3.9M transformer + 0.45M linear head):
  - The depth head is a MoGe-style conv stack: three x2 stages at 54x72, 108x144 and 216x288 with 256 / 128 / 64
    channels (transposed conv, 3x3 conv, two residual blocks of two 3x3 convs at 2x width), then bilinear to
    378x504 and a 3x3 output conv there in fp32 (autocast off).
  - Conv weights are reused at every pixel. The last stage has 62k positions, 64x the 972 tokens per frame the
    encoder sees, so each stage costs ~40 GFLOPs: ~140 GFLOPs per frame vs ~200 for the ViT-B encoder.
  - Narrow convs on large maps (plus replicate padding and UV concatenation copies per stage) are memory-bound,
    unlike the encoder's wide matmuls. The ray head's linear + pixel shuffle also runs in fp32.
  - FLOPs are estimated by hand; `rope_dec_check.py` (job 3179) measures the split.

### Released model at lower input resolution (job 2887, all 81 pose scenes, TTO = `run_tto_pba.py` defaults T = 25)

| input (longest side) | K | fwd s | AUC@3 | AUC@30 | HiRoom / 7Sc / SN++ / DTU / ETH3D | all: alone → TTO |
|---|---|---|---|---|---|---|
| 504 (released) | 16 | 1.16 | 61.2 | 94.3 | 79.1 / 30.4 / 77.0 / 82.8 / 36.6 | 61.2 → 70.6 |
| 504 (released) | 8 | 0.76 | 47.6 | 91.9 | 64.1 / 27.6 / 55.7 / 59.0 / 31.7 | 47.6 → 66.9 |
| 420 | 16 | 0.69 | 26.8 | 84.2 | 25.5 / 15.1 / 12.4 / 75.2 / 5.8 | 26.8 → 37.0 |
| 336 | 16 | 0.41 | 12.5 | 66.1 | 2.9 / 4.5 / 1.1 / 52.8 / 1.0 | 12.5 → 9.0 |
| 336 | 8 | 0.30 | 10.8 | 68.7 | 4.4 / 4.1 / 1.5 / 42.9 / 1.1 | 10.8 → 8.5 |
| 252 | 16 | 0.25 | 3.7 | 45.7 | 0.4 / 0.6 / 0.1 / 17.2 / 0.2 | 3.7 → 2.0 |

- Pose AUC does not depend on the intrinsics rescaling, so the drop is the model's own.
- DTU (object-centric, 64 views) degrades least; the scene datasets collapse.
- **Suspected cause:** the loop block uses 2D RoPE on integer patch coordinates (`_get_rope_positions`,
  `F.embedding` lookup in `model_components/layers/rope.py`). At a lower resolution the same image content gets
  smaller position distances than in training. The DINOv2 encoder interpolates its own position embedding.
- **TTO at 336 px lowers AUC:** these initialisations are outside the photometric basin.

### RoPE equivalence check and decoder breakdown (tum1, RTX 3080, 2026-11-16)

- **Bug in the first `rope_scale.py` (fixed, uncommitted at the time):** upstream builds the cos/sin table with
  `torch.einsum("i,j->ij", positions, inv_freq)`. Under bf16 autocast that is a bf16 matmul, so the released model
  uses **bf16-rounded angles**. The first version computed the angles in fp32. At scale 1 that changed depth by up to
  1.17 (mean 1.7e-3) and conf by 0.26 on average (`rope_dec_check.log`). With the same einsum, rays / depth / conf are
  **bit-identical** to the unpatched model (`rope_dec_check2.log`). Under autocast the stretched float positions are
  rounded to bf16 as well (step 0.125 between 16 and 32), the same precision as the native angles.
  **tum2 job 3179 ran the buggy version**, so its RoPE pose numbers are inexact (its decoder timings are fine).
- **Decoder split** (504 px, bf16, per chunk of 8 frames): depth decoder 70.7 ms, of which transformer 5.0 ms, so
  the **conv stack is ~93 %**; ray decoder 5.6 ms (transformer 4.9 ms). Params: depth 11.40M, ray 4.30M.
- **`decode_chunk_size`:** full `_decode` at 24 views: 0.231 s (chunk 8) vs 0.218 s (16). At 100 views: 0.973 s
  (8, 4.4 GiB peak), 0.898 (16, 5.5 GiB), 0.856 (32, 7.4 GiB), 64 OOM. `cudnn.benchmark`: no gain, and chunk 32 OOMs
  with it. Larger chunks therefore save at most ~12 % of the decoder.

### Weak-GPU profile (tum1, RTX 3080 10 GB, 7-Scenes chess seq-01 frames, `profile_3080.{log,jsonl}`)

Seconds. K = 16 is fixed + 16 steps.

| views | input | enc ViT-B | ViT-S | 1 step | decoders | pose fit | fixed | K = 16 | peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| 24 | 504x378 | 0.117 | 0.045 | 0.0556 | 0.232 | 0.081 | 0.430 | 1.32 | 2.7 |
| 24 | 420x308 | 0.078 | 0.030 | 0.0323 | 0.159 | 0.080 | 0.316 | 0.83 | 2.1 |
| 24 | 336x252 | 0.050 | 0.018 | 0.0182 | 0.108 | 0.080 | 0.237 | 0.53 | 1.7 |
| 24 | 252x196 | 0.032 | 0.012 | 0.0101 | 0.066 | 0.079 | 0.177 | 0.34 | 1.3 |
| 100 | 504x378 | 0.481 | 0.177 | 0.598 | 0.982 | 0.340 | 1.80 | 11.37 | 6.5 |
| 100 | 420x308 | 0.324 | 0.118 | 0.304 | 0.673 | 0.333 | 1.33 | 6.19 | 4.9 |
| 100 | 336x252 | 0.208 | 0.075 | 0.149 | 0.459 | 0.331 | 1.00 | 3.39 | 3.5 |
| 100 | 252x196 | 0.121 | 0.045 | 0.064 | 0.282 | 0.326 | 0.73 | 1.75 | 2.5 |

- The 3080 is ~3.3x slower than the 5090 (100 views, 504 px, K = 16: 11.4 s vs 3.4 s), and the loop's share is
  larger: at 504 px loop 67 % / decoders 18 % / encoder 9 % / pose fit 6 % at 24 views, 84 / 9 / 4 / 3 % at 100.
- At 100 views 420 px is 1.8x and 336 px 3.4x faster than 504 px; peak memory 6.5 → 4.9 → 3.5 GiB.

### Training-free RoPE scaling at lower resolution (tum1, RTX 3080, fixed `rope_scale.py`, all 81 pose scenes)

`eval_student.py` with `DVLT_RES=<px> DVLT_ROPE=1`, official evaluator, TTO = `run_tto_pba.py` iters = 25; logs and
`compare.md` in `workspace/dvlt_speed/`. fwd s = median per scene on the 3080 (not comparable with the 5090 table
above).

| run | fwd s | AUC@3 | AUC@30 | HiRoom / 7Sc / SN++ / DTU / ETH3D | dev: alone → TTO | held-out | all: alone → TTO | TTO s |
|---|---|---|---|---|---|---|---|---|
| 420 px + RoPE, K = 16 | 2.13 | **59.2** | 93.9 | 74.2 / 28.8 / 67.1 / 78.0 / 48.1 | 55.2 → 68.0 | 60.6 → 66.9 | 59.2 → **67.3** | 0.69 |
| 336 px + RoPE, K = 16 | 1.27 | 34.0 | 86.0 | 31.1 / 25.8 / 28.4 / 64.2 / 20.6 | 31.0 → 50.5 | 35.0 → 48.5 | 34.0 → **49.0** | 0.78 |
| 420 px + RoPE, K = 12 | 1.76 | 56.0 | 93.5 | 71.2 / 28.5 / 62.0 / 73.8 / 44.6 | 51.5 → 67.3 | 57.6 → 66.3 | 56.0 → **66.7** | 0.70 |
| 420 px + RoPE, K = 8 | 1.40 | 41.8 | 90.8 | 55.0 / 26.4 / 37.0 / 53.6 / 37.2 | 36.8 → 62.5 | 43.8 → 63.1 | 41.8 → **62.8** | 0.75 |
| 252 px + RoPE, K = 16 | 0.73 | 9.7 | 63.8 | 0.7 / 14.0 / 1.2 / 30.5 / 2.1 | 11.2 → 22.8 | 9.1 → 20.5 | 9.7 → **21.3** | 1.13 |
| 504 px released, K = 16 (`K16_3080`) | 3.67 | 61.3 | 94.3 | 79.3 / 30.5 / 76.9 / 82.8 / 36.8 | 58.6 → 70.9 | 62.3 → 70.7 | 61.3 → **70.9** | 0.69 |
| 504 px released, K = 12 (local run `dvlt`) | 2.98 | 58.7 | 94.0 | 76.0 / 29.8 / 72.0 / 79.5 / 36.0 | | | | |
| 504 px released, K = 8 (`K8_3080`) | 2.29 | 47.7 | 91.9 | 64.3 / 27.7 / 55.8 / 59.0 / 31.7 | 43.9 → 67.2 | 49.3 → 67.0 | 47.7 → **67.1** | 0.76 |

- **The RoPE positions were the cause of the low-resolution collapse.** At 420 px stretching them to the native grid
  recovers almost everything: 26.8 → 59.2 alone (504 px K = 16: 61.2), 37.0 → 67.3 after TTO (504 px: 70.6). ETH3D
  is even better than at 504 px (48.1 vs 36.6).
- At 336 px the recovery is partial: 12.5 → 34.0 alone (about K = 6 at 504 px, 31.1), and TTO now helps
  (9.0 → 49.0), but stays 10 points below K = 6 + TTO (59.4).
- At 252 px it fails (3.7 → 9.7 alone, 21.3 after TTO).
- **Machine check:** released K = 16 at 504 px gives 61.3 / 70.9 on the 3080 (5090: 61.2 / 70.6).
- **Matched time (same GPU), resolution vs fewer steps:**
  - 420 px K = 16 (2.13 s) vs 504 px K = 8 (2.29 s): alone 59.2 vs 47.7, but after TTO **67.3 vs 67.1**. TTO
    closes the gap, so for DéjàView + TTO a lower resolution and fewer steps are equally good trades.
  - The best cheap point is **420 px K = 12: 1.76 s, 56.0 → 66.7**, i.e. 52 % less forward time than K = 16 at
    504 px for −4.2 after TTO, and 23 % less than 504 px K = 8 at the same accuracy after TTO (−0.4).
  - Nothing cheap reaches the released K = 16 + TTO (70.9): every cut so far costs ≥ 3.6 points after TTO.

### Training-free decoder cuts: camera-head poses, depth output at 216x288 (tum1, RTX 3080, 2026-11-17)

`scripts/distill_steps/fast_decode.py` replaces `_decode` (`eval_student.py`: `DVLT_POSE=head`, `DVLT_DEPTH_LR=1`);
`fast_decode_check.py` checks it. With both off it is bit-identical to the released decode.
- **`DVLT_POSE=head`:** cameras come from the camera head's `pose_enc` (c2w translation, quaternion, FoV, centred
  principal point; trained with `CameraLoss` weight 1) instead of the RANSAC fit on the ray map. The ray map's
  per-pixel linear output is skipped; its transformer still runs for the camera token.
- **`DVLT_DEPTH_LR=1`:** the depth head's output stage runs on the last upsampling map (216x288 at 378x504) and
  only the two logits are upsampled, instead of upsampling the 64-channel features first.

Timings on 7-Scenes chess, 504 px, K = 16 (`fast_decode_check.py`), decoder + pose:

| | 24 views | 100 views |
|---|---|---|
| released decode + RANSAC fit | 0.231 + 0.083 s | 0.976 + 0.346 s |
| depth at 216x288 | 0.212 s (−8 % of the decoder) | 0.898 s |
| camera head (no fit, no ray output) | 0.227 + 0.001 s | 0.965 + 0.001 s |
| both | 0.209 + 0.001 s | 0.887 + 0.001 s |

On chess the head's poses agree with the fit within 0.26° (mean, max 0.75°), focal 412 vs 414, same scale. Depth at
216x288 differs by 0.47 % abs-rel on average (p99 7.6 %).

All 81 pose scenes, TTO = `run_tto_pba.py` iters = 25, fwd s = median per scene on the 3080:

| run | fwd s | AUC@3 | HiRoom / 7Sc / SN++ / DTU / ETH3D | dev: alone → TTO | held-out | all: alone → TTO |
|---|---|---|---|---|---|---|
| 504 K = 16, ray fit (`K16_3080`) | 3.67 | 61.3 | 79.3 / 30.5 / 76.9 / 82.8 / 36.8 | 58.6 → 70.9 | 62.3 → 70.7 | 61.3 → 70.9 |
| 504 K = 16, depth 216x288 | 3.63 | 61.3 | (same) | 58.6 → 69.5 | 62.3 → 68.6 | 61.3 → 69.0 |
| 504 K = 16, camera head | 3.49 | 57.3 | 70.1 / 30.6 / 72.1 / 80.6 / 32.9 | 55.2 → 72.6 | 58.1 → 71.2 | 57.3 → **71.6** |
| 504 K = 16, head + depth 216x288 | 3.45 | 57.3 | (same) | 55.2 → 71.9 | 58.1 → 69.8 | 57.3 → 70.5 |
| **504 K = 12, camera head** | **2.80** | 55.2 | 67.6 / 29.6 / 68.6 / 78.1 / 32.1 | 52.7 → 72.4 | 56.1 → 71.4 | 55.2 → **71.6** |
| 420 + RoPE K = 12, ray fit | 1.76 | 56.0 | 71.2 / 28.5 / 62.0 / 73.8 / 44.6 | 51.5 → 67.3 | 57.6 → 66.3 | 56.0 → 66.7 |
| 420 + RoPE K = 12, camera head | 1.59 | 51.9 | 62.0 / 28.5 / 62.7 / 71.2 / 35.1 | 47.4 → 64.0 | 53.5 → 63.4 | 51.9 → 63.6 |
| 420 + RoPE K = 12, head + depth 216x288 | 1.56 | 51.9 | (same) | 47.4 → 63.2 | 53.5 → 63.3 | 51.9 → 63.3 |

- **Camera head: worse alone, better after TTO at 504 px.** Alone −4.0 (mostly HiRoom −9, SN++ −5, ETH3D −4), but
  after TTO +0.7 at K = 16 (held-out +0.5, dev +1.7). Per dataset after TTO (fit → head): HiRoom 88.4 → 90.0, SN++
  79.6 → 81.9, DTU 94.4 → 95.3, ETH3D 63.1 → 62.8, 7-Scenes 28.9 → 28.3. Why the head starts lower but ends higher is
  not known (its intrinsics or a more globally consistent pose set are candidates; untested).
- **K = 12 + camera head + TTO = 71.6 at 2.80 s fwd**, the same as K = 16 + head and above K = 16 + fit (70.9 at
  3.67 s): −0.87 s (−20 % of fwd + TTO) for +0.7 AUC@3. (K = 12 + fit + TTO on the 5090: 70.1.)
- **At 420 px + RoPE the head hurts** (66.7 → 63.6 after TTO): the camera token degrades more at the lower
  resolution than the ray map does.
- **Depth at 216x288 is not worth it:** it saves ~1 % of a forward and costs 0.3-1.9 AUC@3 after TTO (SN++ most,
  79.6 → 76.6). Photometric TTO needs the full-resolution depth detail.

### Head vs fit: extrinsics and intrinsics swapped (tum1, 2026-11-18)

Predictions of `K16_3080` (ray fit) and `K16_head` mixed per scene (`workspace/dvlt_speed/make_hybrid.py`, same depth),
then the evaluator (`eval_only`) and `run_tto_pba.py` iters = 25, all 81 pose scenes:

| cameras | alone AUC@3 | dev → TTO | held-out → TTO | all → TTO | HiRoom / 7Sc / SN++ / DTU / ETH3D after TTO |
|---|---|---|---|---|---|
| fit extrinsics + fit intrinsics (released) | 61.3 | 58.6 → 70.9 | 62.3 → 70.7 | 61.3 → 70.9 | 88.4 / 28.9 / 79.6 / 94.4 / 63.1 |
| head extrinsics + head intrinsics | 57.3 | 55.2 → 72.6 | 58.1 → 71.2 | 57.3 → 71.6 | 90.0 / 28.3 / 81.9 / 95.3 / 62.8 |
| head extrinsics + fit intrinsics | 57.3 | 55.2 → 70.4 | 58.1 → 70.4 | 57.3 → 70.4 | 87.5 / 29.0 / 79.7 / 94.0 / 61.9 |
| **fit extrinsics + head intrinsics** | 61.3 | 58.6 → **73.5** | 62.3 → **71.5** | 61.3 → **72.1** | 89.7 / 28.7 / 81.9 / 95.3 / 65.1 |
| **fit extrinsics + fit focal, principal point centred, fx = fy** (`hyb_fit_cleanK`) | 61.3 | 58.6 → 73.1 | 62.3 → **71.6** | 61.3 → **72.1** | |

- Intrinsics don't enter the pose AUC alone, so "alone" only depends on the extrinsics.
- Focal / GT focal (median per scene) is the same for both (e.g. ScanNet++ 0.992 vs 0.989, DTU 0.966 vs 0.965).
  The fit's principal point is 0.15-0.27 % of the width off centre (~1 px) and fx differs from fy by 0.35-0.9 %;
  the head's principal point is the image centre and fx = fy by construction. The TTO optimises one shared focal
  scale but not the principal point (`focal-and-intrinsics.md`: principal point rejected), so the fit's noise stays.
  **Confirmed:** the fit's own focal with the principal point centred and fx = fy gives the same 72.1 (held-out
  71.6). The cause is the fit's principal point / pixel aspect, not the head: a training-free +1.2 AUC@3 after TTO
  for any ray-fit DéjàView prediction (`workspace/dvlt_speed/make_cleanK.py`).
- `eval_student.py DVLT_POSE=fit_headK` now produces this combination directly.

### Resolution distillation (tum1, RTX 3080, 2026-11-18)

`scripts/distill_steps/train_res.py`: the student runs at a lower longest side with RoPE positions stretched to the
native grid (`rope_scale.py`) and K_s loop steps; the teacher is the frozen released model at 504 px, K = 16, on the
same frames. Loss vs the teacher: L1 rays at the student resolution (teacher area-downsampled), L1 log depth and
0.1 L1 log conf at 504 px (student bilinearly upsampled, as at test time), L1 on the camera head's `pose_enc`. Whole
model trained (encoder 0.1x lr), AdamW, cosine to 10 %, clip 1, decoders checkpointed per chunk, 12-32 views per
step (6.3 GiB peak at 32 views). Data: 200 ARKitScenes Training videos, 240 frames each (`prep_arkit.py`, HTTP range
reads of the raw `vga_wide` zips, JPEG q95), contiguous windows of 2-4x the view count; 8 Validation videos for the
validation loss. No GT, nothing from the benchmark.

- **R1** (lr 3e-5, res {336, 420}, K {12, 16}): stopped at step 500. Validation got worse at 420 px (ray 0.024 →
  0.032, depth 0.045 → 0.073) and 504 px (2.4x). The released model starts close to the teacher there, so lr 3e-5
  mostly adds noise. Don't repeat.
- **R2** (lr 8e-6, 300 warmup, res {336, 420, 504}, K {12, 16}, 3 h, 2550 steps): validation (ray / depth / pose vs
  the teacher, 8 val clips): 336 K16 0.039 / 0.067 / 0.028 → 0.020 / 0.037 / 0.016; 420 K12 0.024 / 0.045 / 0.018 →
  0.020 / 0.032 / 0.016; 504 K12 0.016 / 0.032 / 0.011 → 0.015 / 0.026 / 0.010.

Benchmark (all 81 pose scenes, TTO = `run_tto_pba.py` iters = 25, fwd s = median per scene on the 3080; `fhk` =
`DVLT_POSE=fit_headK`, ray-fit extrinsics + camera-head intrinsics):

| run | fwd s | AUC@3 | HiRoom / 7Sc / SN++ / DTU / ETH3D | dev: alone → TTO | held-out | all: alone → TTO |
|---|---|---|---|---|---|---|
| released 336 px + RoPE, K = 16, fit | 1.27 | 34.0 | 31.1 / 25.8 / 28.4 / 64.2 / 20.6 | 31.0 → 50.5 | 35.0 → 48.5 | 34.0 → 49.0 |
| released 336 px + RoPE, K = 16, fhk | 1.27 | 34.0 | (same) | 31.0 → 44.6 | 35.0 → 44.3 | 34.0 → 44.5 |
| **R2 336 px, K = 12, fhk** | 1.08 | 52.0 | 62.9 / 28.6 / 61.4 / 71.5 / 35.5 | 47.6 → 66.6 | 53.5 → 65.8 | 52.0 → **66.0** |
| **R2 336 px, K = 16, fhk** | 1.27 | **53.9** | 64.3 / 28.2 / 65.6 / 74.4 / 37.1 | 50.3 → 68.5 | 55.2 → **65.9** | 53.9 → **66.7** |
| released 420 px + RoPE, K = 12, fit | 1.76 | 56.0 | 71.2 / 28.5 / 62.0 / 73.8 / 44.6 | 51.5 → 67.3 | 57.6 → 66.3 | 56.0 → 66.7 |
| released 420 px + RoPE, K = 12, fhk | 1.76 | 56.0 | (same) | 51.5 → 67.6 | 57.6 → 64.2 | 56.0 → 65.2 |
| **R2 420 px, K = 12, fhk** | 1.76 | **58.5** | 76.0 / 30.2 / 74.5 / 78.5 / 33.4 | 56.1 → 72.2 | 59.5 → **69.4** | 58.5 → **70.3** |
| released 504 px, K = 12, fhk | 2.98 | 58.7 | 76.0 / 29.8 / 72.0 / 79.5 / 36.0 | 56.1 → 73.0 | 59.7 → 71.0 | 58.7 → 71.7 |
| R2 504 px, K = 12, fhk | 2.98 | 56.7 | 72.9 / 30.8 / 70.5 / 79.6 / 29.7 | 56.8 → 71.8 | 56.9 → 69.7 | 56.7 → 70.4 |
| R3 336 px, K = 16, fhk (R2 + 25 % outdoor) | 1.28 | 54.2 | 66.8 / 27.8 / 65.0 / 73.9 / 37.5 | 50.7 → 67.1 | 55.6 → 64.7 | 54.2 → 65.3 |
| R3 420 px, K = 12, fhk | 1.76 | 58.4 | 76.6 / 29.6 / 72.5 / 78.2 / 35.2 | 55.5 → 71.1 | 59.6 → 69.5 | 58.4 → 70.0 |
| R3 504 px, K = 12, fhk | 2.98 | 56.4 | 72.7 / 29.9 / 69.1 / 80.5 / 29.7 | 56.4 → 72.3 | 56.6 → 70.1 | 56.4 → 70.7 |
| released 504 px, K = 16, fit | 3.67 | 61.3 | 79.3 / 30.5 / 76.9 / 82.8 / 36.8 | 58.6 → 70.9 | 62.3 → 70.7 | 61.3 → 70.9 |

- **336 px:** +19.9 alone and +17.7 after TTO over the best released option, at the same 1.27 s. It now equals the
  released 420 px K = 12 + TTO (66.7 at 1.76 s) at 28 % less forward time.
- **420 px K = 12:** alone it equals the released 504 px K = 12 (58.5 vs 58.7) at 59 % of the time; after TTO +3.6
  (70.3, held-out 69.4 vs 66.3). At low resolution the released camera head hurts even as intrinsics only (fhk 65.2
  vs fit 66.7); after R2 it works again.
- ETH3D alone drops at 420 px (44.6 → 33.4): the released model was unusually good there at 420 px; the 504-px
  teacher has 36.8. One seed, one run.
- **At 504 px the R2 student is slightly worse** than the released model (K = 12: 56.7 vs 58.7 alone, 70.4 vs 71.7
  after TTO), again mostly ETH3D (36.0 → 29.7). Training only on indoor ARKit clips biases the model; outdoor /
  object data (Mip-NeRF 360, T&T, DL3DV) is the obvious next fix. Use R2 only below 504 px.
- **R3 (2026-11-18): outdoor data doesn't fix it.** R2 + 1.75 h more with 25 % of clips from 11 outdoor/object
  scenes (Mip-NeRF 360 images_4, T&T + Deep Blending; `prep_outdoor.py`), lr 5e-6. ETH3D alone recovers a little at
  420 px (33.4 → 35.2) but not at 504 px (29.7), and after TTO R3 is within noise of R2 (336: 65.3 vs 66.7; 420: 70.0
  vs 70.3; 504: 70.7 vs 70.4). 11 scenes are too few to matter; the 504-px drop is more likely the general cost of
  fine-tuning on few videos (C0, 504 px only, kept 71.9, see [dejaview-tto-teacher](dejaview-tto-teacher.md)), i.e.
  training the low resolutions pulls the shared weights away from 504 px. Keep R2 as the low-res checkpoint.

## Open / next

- **Done (2026-11-16, tum1):** RoPE scaling at 420 / 336 / 252 px, decoder breakdown and the 3080 profile (above).
  tum2 job 3179 ran the buggy `rope_scale.py`, so use the tum1 numbers for RoPE.
- **Put the 420 px + RoPE points into the DéjàView compute plot** next to the K sweep (forward + TTO time on one GPU).
- **Camera head (2026-11-17):** why it ends higher after TTO (intrinsics vs poses: try head extrinsics + fit
  intrinsics and the reverse); camera head at K = 8 + TTO; held-out check on the 5090 and put K = 12 + head into the
  DéjàView compute plot.
- **Decoder ideas left (untested):** depth output stage in bf16 / channels-last at full resolution (216x288 output
  was tested and costs TTO accuracy).
- **Resolution distillation: works (R2 above).** Next: longer / more data, 252-336 px, K = 8 at 336-420 px,
  held-out check on the 5090. Original note: fine-tune the model at
  336 px, starting from the RoPE-scaled model, to reproduce its own 504-px K = 16 output (the `train.py` loop with a
  downsampled student input).
  - This is a distribution shift rather than removed computation, so it may fine-tune more easily than K = 4 did.
  - It would cut the 100-view forward by ~3x.
- **Encoder distillation** (tum2 job 2888, ViT-S/14 + projection, regressing z_0, `train_encoder.py`, commit f3ab013): result never
  collected (no access to tum2 from tum1 since). By the profile it cannot give more than ~5 % speed, so it is parked.

Related: [dejaview-steps](dejaview-steps.md), [dejaview-few-step-distillation](dejaview-few-step-distillation.md),
[matched-compute](matched-compute.md)

Code: removed in the merge cleanup of 2026-11-26 (superseded or parked); the last version is at commit `f3ab013`, e.g. `git show f3ab013:scripts/distill_steps/<name>`: `slurm_res_sweep.sh`, `slurm_rope.sh` (superseded by `slurm_res_eval.sh`), `train_encoder.py` + `slurm_encoder.sh` (parked encoder distillation).
