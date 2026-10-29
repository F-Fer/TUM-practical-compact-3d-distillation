# Distillation experiments (ScanNet++)

Strategy-finding runs from the night of 2026-10-28/29 on tum1 (RTX 3080, 10 GB). Results and the
write-up are in `workspace/distill/REPORT.md` (machine-local, gitignored).

## Split and data

- The 20 DA3-BENCH ScanNet++ scenes are split by position in `SCANNETPP_SCENES`: even = train,
  odd = held-out val (`common.py`). Only train-scene images are used for training; GT is used only
  by the GT-supervised baseline (`--teacher gt`) and for validation.
- `prep_data.py`: iPhone frames at 504x378 (cv2 INTER_AREA, uint8, the preprocessing DA3's API
  applies) + GT depth/poses -> `workspace/distill/cache/`.
- `prep_dvlt_cache.py`: the same frames exactly as DéjàView's `preprocess_images` makes them
  (float16) -> `workspace/distill/cache_dvlt/`. DéjàView at K < 8 is so sensitive to resampling
  that the cv2 cache moves its AUC@3 by ~10 points; DéjàView teachers and students always get these.
- `prep_extra.py`: unlabeled 7-Scenes / ETH3D images for `--scenes train+extra` (not yet tested).

## Scripts

```sh
uv run scripts/distill/train.py --name NAME --student da3:depth-anything/DA3-SMALL --teacher dvlt:12 \
    --steps 600 --min_views 20 --max_views 24 --depth_loss log --ema 0.995
uv run scripts/distill/train.py --name NAME --student dvlt:6 --teacher dvlt:12 --steps 600 \
    --lr 3e-5 --lr_head 3e-5 --min_views 24 --max_views 32 [--w_feat 1 --feat_match double]
uv run scripts/distill/eval_student.py NAME        # official evaluator, val scenes -> workspace/distill/eval/NAME
uv run scripts/distill/eval_student.py dvlt:6      # untrained DéjàView with K=6
uv run scripts/distill/split_metrics.py [dirs]     # official metrics restricted to the val scenes
uv run scripts/distill/report.py                   # results.md + figures/val_curves_*.png
uv run scripts/distill/time_models.py              # time / memory / TFLOPs per model (idle GPU!)
tmux new -d -s distill scripts/distill/queue.sh    # runs workspace/distill/queue.txt line by line
```

- Students: `da3:<checkpoint>` (whole DA3 fine-tuned, `cam_enc` frozen) or `dvlt:<K>` (DéjàView
  run with K loop steps; DINOv2 encoder and camera/register tokens frozen, so the encoder tokens
  are computed once and shared with a DéjàView teacher). `dvlt:<K>u` unties the loop into K blocks.
- Teachers: `da3:<checkpoint>`, `dvlt:<K>`, `gt`. DA3 students get targets in DA3's format
  (depth, pose encoding, ray map) built from any teacher's depth + cameras, scale-normalized
  (`common.make_targets`, `common.da3_student_loss`).
- `--w_feat`: DA3 student <- DA3 teacher: per-layer linear projections of the 4 DPT input layers
  (cosine + smooth-L1 on layer-normed tokens). DéjàView student <- DéjàView teacher: student step i
  tokens vs teacher step tokens, matched by schedule time (`time`) or step i <- teacher step 2i+1
  (`double`).
- In-training validation (`--val_frames 100`, default) runs on exactly the frames the DA3 evaluator
  samples (seed 42) with each model's own preprocessing, so its pose AUC equals the official one.
- `baselines.py`, `run_bench.py`, `da3_lowmem.py` here are copies of the `scripts/` versions from
  branch `distillation-test`, so this directory works on any checkout.

## Notes for the 4x5090 run

- Train with as many views as possible: students trained on 8-16-view subsets improved on 32-frame
  validation but not on the 100-frame benchmark.
- Use more scenes (ScanNet++ train split, other indoor data): with 10 scenes and ~600-1000 batch-1
  steps every student is optimization/data limited.
- Fine-tuning DA3 needs small LRs (5e-6 backbone, 2e-5 heads at batch 1).
