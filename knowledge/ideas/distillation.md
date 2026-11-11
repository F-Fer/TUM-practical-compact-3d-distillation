# Distillation into smaller students

Status: **dropped** (plan change after the proposal presentation, 2026-10-22: not enough compute to distill a large
teacher properly) · Area: original project plan · Updated 2026-11-10

Strategy-finding runs on the night of 2026-10-28/29 on tum1 (RTX 3080) on ScanNet++ (20 scenes split even/odd into
train/val). Code on branch `origin/distillation-test` (`scripts/distill/`, README there); the write-up
`workspace/distill/REPORT.md` is machine-local on tum1 and not in this repo.

## What was learned (from `scripts/distill/README.md` on that branch)

- Students: whole DA3 fine-tuned (`cam_enc` frozen) or DéjàView with fewer loop steps (encoder frozen, tokens shared
  with a DéjàView teacher); teachers DA3, DéjàView or GT; optional feature distillation.
- **Train with as many views as possible:** students trained on 8-16-view subsets improved on 32-frame validation but
  not on the 100-frame benchmark.
- **Data/optimisation limited:** with 10 train scenes and ~600-1000 batch-1 steps every student was limited by data and
  optimisation.
- Fine-tuning DA3 needs small learning rates (5e-6 backbone, 2e-5 heads at batch 1).
- DéjàView at K < 8 is so sensitive to resampling that the cv2-resized cache moves its AUC@3 by ~10 points: DéjàView
  teachers and students need its own preprocessing.
- In-training validation on exactly the evaluator's frames (seed 42) with each model's preprocessing reproduces the
  official pose AUC.

## Replaced by

Pretrained models of four sizes (DA3 SMALL to GIANT) plus TTO ([matched-compute](matched-compute.md)), and DéjàView's
K as a compute knob ([dejaview-steps](dejaview-steps.md)).
