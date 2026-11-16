# Where DéjàView spends its time, and cheaper inputs (resolution, encoder)

Status: **in progress** · Area: feed-forward compute · Owner: Finn (+ Claude) · Updated 2026-11-16

**Answer so far:**
- The 12-16 loop steps and the decoders take the time; the ViT-B encoder is only 4-8 % of a K = 16 forward.
  Replacing it with a ViT-S can therefore save ≤ 5 %.
- The step cost grows about quadratically with the token count, so input resolution is the big lever: at 100 views
  K = 16 costs 3.4 s at 504 px and 1.1 s at 336 px.
- But the released model is locked to 504 px. At 420 px AUC@3 falls 61.2 → 26.8 (after TTO 70.6 → 37.0), and at
  336 px pose collapses (12.5), with TTO making it worse.
- Lower resolution needs fine-tuning, or a training-free fix of the loop's RoPE positions (untested).

Goal (from the strategy discussion, 2026-11-16): after K-reduction by self-distillation failed
([dejaview-few-step-distillation](dejaview-few-step-distillation.md)), cut DéjàView's fixed cost instead, so that a
cheaper DéjàView + TTO beats a slower one, or DéjàView fits weaker GPUs.

## Setup

Code in `scripts/distill_steps/`:
- `profile_dvlt.py` (CUDA-event timings, bf16, decode chunks of 8, fp32 pose fit, on real ARKitScenes frames);
- `eval_student.py` with `DVLT_RES`: it runs the net at a smaller longest side (antialiased resize of the 504-px
  input), then upsamples depth / conf to the 504 frame and rescales the intrinsics, so the evaluator and
  `run_tto_pba.py` see 504-px predictions;
- `slurm_profile.sh`, `slurm_res_sweep.sh`, `slurm_encoder.sh` + `train_encoder.py` (ViT-S student);
- `rope_scale.py` (float RoPE positions stretched to the native grid, `DVLT_ROPE=1` in `eval_student.py`),
  `rope_dec_check.py` (RoPE equivalence check + decoder timing breakdown), `slurm_rope.sh`.

`profile_dvlt.py` and `rope_dec_check.py` take `--images <folder>` to run on a local GPU without the ARKit zips.

Raw logs and metric JSONs are in `results/distill_steps/{profile,res_sweep}/` on tum2 (local, not in git).

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

## Open / next

- **Training-free (job 3179, queued 2026-11-16):** RoPE positions stretched to the native grid at 420 / 336 / 252
  px (pose + TTO), plus the decoder timing breakdown (ray vs depth, transformer vs conv, `decode_chunk_size`
  8-64, `cudnn.benchmark`).
- **Decoder ideas (untested):** depth output stage in bf16 / channels-last, depth at 216x288 upsampled (TTO uses a
  coarse depth grid), poses from the camera head instead of the RANSAC fit on the ray map (also removes the pose fit).
- **Resolution distillation:** fine-tune the model at 336-420 px to reproduce its own 504-px K = 16 output (the
  `train.py` loop with a downsampled student input).
  - This is a distribution shift rather than removed computation, so it may fine-tune more easily than K = 4 did.
  - It would cut the 100-view forward by ~3x.
- **Encoder distillation** (job 2888, ViT-S/14 + projection, regressing z_0): running. It is now mainly a test of
  whether the loop tolerates a different encoder; by the profile, it cannot give more than ~5 % speed.

Related: [dejaview-steps](dejaview-steps.md), [dejaview-few-step-distillation](dejaview-few-step-distillation.md),
[matched-compute](matched-compute.md)
