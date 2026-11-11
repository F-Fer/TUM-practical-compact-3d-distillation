# Blur pyramid and image resolution

Status: **adopted:** 4 Adam levels at σ = 8, 4, 2, 1 px, then LM at σ = 1 (`levels=8,4,2,1,1 lm_levels=4`) at the
prediction resolution · **rejected:** 2x resolution · Area: PBA · Updated 2026-11-10

**Answer:** the coarse levels matter for the basin, not for accuracy; at a fixed budget the split across levels hardly
matters. Higher resolution doesn't help pose: the finest blur level is not where accuracy is lost.

## Evidence

| date | setting | change | result | source |
|---|---|---|---|---|
| 10-31 | SMALL, rotations perturbed 5° | Adam on the VGG pyramid | recovers almost all (HiRoom 30.7 vs 34.2 unperturbed) | REPORT2 §5 |
| 11-02 | SMALL dev, RGB fast, ablation | number of levels / split of iterations | no difference at 200 iterations; 1 level x 800 beats 4x200 at equal time (28.0/47.2 vs 27.2/44.7); 1 level x 1600 still drifts, so coarse levels don't cause the drift | memory note, `results/pba/figures2/ablation_small.png` |
| 11-09 | SMALL/LARGE subset, old path, 3070 | 2x resolution (`res_scale=2`, images resized from the originals) / + matched blur | mixed, mostly noise: SMALL HiRoom −2.3/−3.3, ETH3D +1.6/−0.2, DTU +1.2/+0.7; LARGE HiRoom +2.5/+0.6, ETH3D −4.7/−2.6. Small HiRoom recon gain with the sharp 2x polish (F1 +2.4 SMALL, +1.4 LARGE, 4 scenes) | REPORT6 |

## Open

The HiRoom recon gain at 2x resolution (4 scenes) if reconstruction matters. 1 level vs 4 wasn't rechecked after LM,
the stop rule and the brightness fix.

Related: [optimizer](optimizer-adam-lm-hybrid.md), [dino-features](dino-features.md).
