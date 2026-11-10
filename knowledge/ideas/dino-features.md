# DA3 / DINO backbone tokens: coarse level and token matches

Status: **rejected for DA3 initialisations** (coarse level neutral, token matches wrong) · **kept as an option** for
poor initialisations (it widens the basin) · Area: PBA features · Updated 2026-11-10

DA3's backbone tokens are free (the forward pass computes them). `scripts/cache_da3_feats.py` re-runs DA3 exactly as
the evaluator with `export_feat_layers` and caches L2-normalised tokens PCA-reduced to 64 channels
(`workspace/tto/feats/`, 1.3 GB). Used as a level with `levels=local:0,vgg:4,vgg:2,vgg:1`.

## Evidence (REPORT2 §3, §5)

| setting | change | result |
|---|---|---|
| SMALL dev, 4x200 budget | `local` (last frame-local layer) / `mid` / `final` tokens as the first level | 34.8 / 34.8 / 32.5 vs 34.2 (VGG at all levels); `final` ScanNet++ 17.9 → 10.3: deeper, cross-view layers hurt |
| LARGE dev | `local` tokens first | 74.0 vs 72.5 (noise level) |
| SMALL, rotations perturbed 15° | `mid` level then VGG | A@30 HiRoom 67.9 → 76.1, ScanNet++ 53.4 → 58.5; neutral at 5°; no effect on A@3 |
| token mutual nearest neighbours (cos > 0.5, best 128 per pair), checked against refined LARGE geometry | match quality | only 6-32 % within one token (14 px), 46-84 % off by > 3 tokens; `local` layer has a positional bias (39 % HiRoom / 48 % DTU of matches at the same image position) |
| single scene | token matches as a Huber reprojection term on coarse levels | worse: A@30 0.687 → 0.52-0.61 (0.77 without) |

**Why no gain:** DA3's initial poses are already inside the basin of the blurred VGG level (A@30 85-96 %). A coarse
stage only helps where the fine stages fail to converge.

## Don't repeat

Token matches as correspondences (would need sub-token refinement + RANSAC; a dedicated matcher is better, see
[correspondences-in-pba](correspondences-in-pba.md)). A DINO level on DA3 inits.

Related: [optimizer](optimizer-adam-lm-hybrid.md) (basin test), [feature-metric-vgg](feature-metric-vgg.md).
