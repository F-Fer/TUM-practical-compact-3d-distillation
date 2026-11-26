# Few-step DéjàView by self-distillation (fine-tune the loop to work at K = 4)

Status: **open (first pilot 2026-11-16, see answer)** · Area: distillation + TTO · Owner: Finn (+ Claude) ·
Updated 2026-11-16

**Answer:** 2 h of whole-model self-distillation from K = 16 lifts DéjàView at K = 4 from 9.0 to ~15 pose
AUC@3 (AUC@30 72 → 78) and after photometric TTO from 36.3 to 43-44 (all 81 scenes; held-out 37.8 → 44-45). Token
distillation (B) and 4.4x more scenes (C) change nothing beyond ~1 point. Simply running the released model at
K = 6 (+0.1 s) gives 31.1 alone and 59.4 after TTO, so at this budget few-step distillation is not worth it;
training only at K = 4 also degrades K = 8 (47.6 → 35.7).

Idea 1 of `reports/DéjàView and Solv3r distillation ideas.md`: DéjàView was trained only with K ∈ [8, 16] loop steps
on a `linspace(0, 1, K)` grid with interval conditioning ([dejaview-steps](dejaview-steps.md)), so K < 8 is
extrapolation in step size. Fine-tune the released model so that K = 4 reproduces its own K = 16 output, then judge
it alone and after photometric TTO.

## Setup (pilot, 2026-11-16)

- Code: `scripts/distill_steps/`: `train.py`, `eval_student.py` (benchmark adapter with a checkpoint
  and K), at commit f3ab013: `slurm_train_eval.sh` (train, then pose at K = 4 + photometric TTO T = 25 on all 81 pose scenes, pose at
  K = 6, 8), `slurm_eval_ckpt.sh`, `slurm_base_tto.sh`, `compare.py` (the table below). Runs in
  `$SCRATCH/distill_runs/<name>/` (log.jsonl, config.json, checkpoints), predictions in
  `workspace/evaluation/dvlt_fs/`, TTO in `workspace/tto/dvlt_fs/`.
- Student: the whole model (encoder 86.6M at 0.1x lr, loop block + heads 30.5M), K_s = 4, drop_path off, bf16,
  gradient checkpointing, AdamW lr 3e-5, 300 warmup steps, cosine to 10 % fitted to the time budget, clip 1.0,
  batch = 1 scene with 16-48 random sorted views. Teacher: frozen copy at K_t = 16, online.
- Loss: L1 rays + L1 log depth + 0.1 L1 log conf against the teacher. Variant B adds MSE between layer-normed tokens
  after student steps 0, 1, 2, 3 and teacher steps 4, 9, 14, 15 (same t), weighted to 0.5x the output loss at step 1.
- Data (no GT used; nothing from the DA3 benchmark): ARKitScenes `vga_wide` (Training fold, 160 random videos),
  Mip-NeRF 360 (7 scenes, `images_4`), Tanks and Temples truck/train + Deep Blending playroom/drjohnson, read from
  the zips in `$SCRATCH/distill_data/`. A and B used only the 31 ARKit videos downloaded at the time (27 train +
  4 val) + 11 others; C used 156 + 11. Same 4 ARKit val clips (32 views) for all runs.
- One RTX 5090 per run, ~1.55-1.65 s/step, 20.8 GB peak at 48 views.

## Evidence

Official evaluator, pose AUC@3 (mean of 5 datasets) for the model alone, and photometric TTO (`run_tto_pba.py`
defaults, T = 25) on its predictions: dataset means over dev (22 scenes), held-out (59) and all 81 pose scenes,
alone → after TTO. RTX 5090, one seed. Table from `scripts/distill_steps/compare.py` (commit f3ab013).

| date | model | fwd s | AUC@3 | AUC@30 | HiRoom / 7Sc / SN++ / DTU / ETH3D | dev | held-out | all |
|---|---|---|---|---|---|---|---|---|
| 11-15/16 | released K = 4 | 0.56 | 9.0 | 72.0 | 14.5 / 6.8 / 6.2 / 11.6 / 5.9 | 7.7 → 33.2 | 9.5 → 37.8 | 9.0 → 36.3 |
| | released K = 6 | 0.66 | 31.1 | 87.9 | 44.1 / 21.7 / 30.4 / 40.0 / 19.6 | 27.5 → 58.0 | 32.7 → 59.9 | 31.1 → 59.4 |
| | released K = 8 | 0.76 | 47.6 | 91.9 | 64.1 / 27.6 / 55.7 / 59.0 / 31.7 | 43.8 → 67.3 | 49.2 → 66.7 | 47.6 → 66.9 |
| | released K = 12 | 0.96 | 58.7 | 94.0 | 76.6 / 29.7 / 72.0 / 79.5 / 35.9 | 56.2 → 71.2 | 59.7 → 69.7 | 58.7 → 70.1 |
| | released K = 16 (teacher) | 1.16 | 61.2 | 94.3 | 79.1 / 30.4 / 77.0 / 82.8 / 36.6 | 58.4 → 70.7 | 62.2 → 70.4 | 61.2 → 70.6 |
| 11-16 | A (outputs, 38 scenes), 1233 steps | 0.56 | 11.4 | 76.4 | 14.2 / 8.7 / 6.2 / 21.0 / 7.0 | 12.6 → 39.5 | 11.0 → 40.7 | 11.4 → 40.2 |
| 11-16 | **A (outputs, 38 scenes), 4932 steps, 2.3 h** | 0.57 | **14.6** | 77.9 | 22.1 / 9.4 / 11.2 / 20.9 / 9.6 | 14.2 → 41.6 | 14.8 → 44.4 | 14.6 → **43.4** |
| 11-16 | A at K = 6 (trained for K = 4) | 0.67 | 30.8 | 88.1 | 35.9 / 19.9 / 26.5 / 51.7 / 19.8 | | | |
| 11-16 | A at K = 8 | 0.76 | 35.7 | 89.7 | 42.8 / 20.7 / 31.7 / 57.6 / 25.6 | | | |
| 11-16 | **B (outputs + tokens, 38 scenes), 4671 steps** | 0.57 | 13.0 | 76.2 | 18.9 / 9.8 / 10.3 / 17.3 / 8.7 | 12.9 → 41.5 | 13.0 → 43.0 | 13.0 → **42.3** |
| 11-16 | B at K = 6 | 0.67 | 28.0 | 87.3 | 34.3 / 20.0 / 25.2 / 43.7 / 16.8 | | | |
| 11-16 | **C (outputs, 167 scenes), 3917 steps, 1.8 h** | 0.57 | **14.9** | 78.7 | 19.4 / 13.0 / 11.7 / 21.0 / 9.5 | 15.1 → 41.8 | 14.9 → 45.3 | 15.0 → **44.0** |

Validation loss against the teacher (4 held-out ARKit clips of 32 views, noisy): ray L1 0.149 at the start →
0.091 (A), 0.093 (B), 0.110 (C) at the end, while A's training ray loss fell to ~0.013: with 38 scenes the student
overfits, but 4.4x more scenes (C) did not lower the validation loss or raise the benchmark at this budget.
Token loss in B fell from 0.47 to 0.25 without lowering the ray / depth validation loss below A's.
(The "alone" value in the TTO columns is the TTO script's own AUC of the same predictions and can differ from
the evaluator's by a few tenths.)

## Open / don't repeat

- The gain grows slowly (A: 11.4 at 25 %, 14.6 at 100 % of the steps) and the validation loss plateaus at ~0.09-0.11
  ray L1 from ~1,500 steps on, for 38 and for 167 scenes alike: neither more data nor much longer training looks
  likely to close a 16-point gap to K = 6 cheaply. K = 4 may simply be too few block applications for the loop's
  correspondence search (the schedule probe in [dejaview-steps](dejaview-steps.md) also showed that only the number
  of applications matters, not the interval sizes).
- Untested: K_s = 6 (the TTO basin starts there, so K = 6 → K = 8/12 quality would matter more), mixing K ∈ [8, 16]
  into the batches to keep large K intact, higher lr / longer schedule, a basin-aware loss (truncate errors BA fixes),
  loss on decoded cameras rather than ray maps.
- Don't repeat: B's token loss (layer-normed MSE at matched t, 0.5x weight) as a cure; 38 vs 167 scenes at 2 h.
- Later: A at 75 % (step 3699) 14.0 alone, 42.3 after TTO; B at K = 8 32.5; C at K = 6 31.3, K = 8 37.9.
  Next direction: [dejaview-speed](dejaview-speed.md).

Related: [dejaview-steps](dejaview-steps.md), [distillation](distillation.md), [matched-compute](matched-compute.md)

Code: removed in the merge cleanup of 2026-11-26 (the pilot's answer: not worth it at this budget); the last version is at commit `f3ab013`, e.g. `git show f3ab013:scripts/distill_steps/<name>`: `slurm_train_eval.sh`, `slurm_eval_ckpt.sh`, `slurm_base_tto.sh`, `compare.py` (`train.py` stays: `train_res.py` and `profile_dvlt.py` use its model loader).
