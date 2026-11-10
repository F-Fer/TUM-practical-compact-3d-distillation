# Robust kernel, colour and the geometric term

Status: **adopted:** Huber (δ 0.05 default), grey images (default since 2026-11-02), geometric term `w_geo=1` ·
**rejected:** L2, redescending kernels, minimum over neighbours, RGB · Area: PBA loss · Updated 2026-11-10

**Answer:** large residuals at the initialisation are misalignment, not outliers, so redescending kernels throw away
signal. Grey is as good as RGB and faster. Photometric and geometric terms are complementary.

## Evidence

| date | setting | change | result | source |
|---|---|---|---|---|
| 10-30 | SMALL dev, v1 | photometric only / geometric only / both | photo-only ≈ both on HiRoom (22.5 vs 24.0); geo-only best 7-Scenes recon (F1 63.1) but worse pose (19.0); both best overall | REPORT1 |
| 11-02 | SMALL dev, Adam 4x50 + LM 5, 19 runs `loss_*` | Huber δ 0.01 / 0.02 / 0.05 / 0.1 / 0.2 | 29.8-29.9 / … / 29.3 / worse / worse: smaller δ better | memory note, `results/pba/figures2/loss_sweep_small.png` |
| 11-02 | same | L2, Tukey, Geman-McClure | all worse than Huber, the more redescending the worse; after optimisation only 1-2 % of residuals are beyond 0.2 | memory note |
| 11-02 | same | grey vs RGB | equal quality, 6 % faster → `gray=True` default | memory note |
| 11-02 | same | δ 0.02 + grey | SMALL +0.6 over 3 seeds (29.9 vs 29.3, ranges don't overlap), −7 % time; BASE/LARGE (1 seed): pose same, F1 −1.5 → only for SMALL; never adopted | memory note |
| 11-02 | BASE / LARGE dev | grey alone | LARGE equal (55.8/69.9 vs 55.7/70.2), 5 % faster; BASE −0.8/−1.5, all on HiRoom (probably noise) | memory note |
| 11-02 | SMALL | minimum over neighbours instead of the sum | much worse | memory note |
| 11-02 | SMALL | `w_geo` 0 / 1 / 3 | 1 best: 0 costs F1, 3 costs both | memory note |
| 11-05 | SMALL dev | LM without the geometric term | 25.4, F1 39.9 (worse) | REPORT3 §4 |
| 11-05 | SMALL, GT start | Huber δ 0.02 | holds HiRoom slightly better (71.3/54.8 vs 66.5/51.9) | [objective-bias](objective-bias-gt-start.md) |
| 11-07 | SMALL, GT start / free depth | geometric row off | better (HiRoom 86.0 vs 78.9 at it. 20; patch stage 30.0 vs 27.3): the geometric row pulls depth back to DA3's errors | REPORT4 §4-5 |

## Open

The geometric term's weight was tuned on the old configuration (pre-NCC LM, pre-free-depth); lowering it where depth
is free (LM on NCC) is untested in the full pipeline.

Related: [ncc](ncc.md), [brightness-model](brightness-model.md), [feature-metric-vgg](feature-metric-vgg.md).
