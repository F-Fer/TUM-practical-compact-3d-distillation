# Which and how many pixels to sample

Status: **adopted:** gradient-weighted sampling from confident pixels, 1,024 px per frame in Adam, 4,096 in LM ·
**useful in long LM runs:** stratified sampling (`strat_cells=16`) · **rejected:** more pixels as a fix for drift,
confidence filters, depth-edge masks · Area: PBA · Updated 2026-11-10

**Answer:** the pixel count is not what limits accuracy (64x the pixels: +0.7 AUC@3 at 26x the cost; 16x samples in
REPORT6: nothing). The distribution matters more: gradient weighting concentrates samples on a few textured places,
and stratifying it over a grid removes the long-run drift on DTU.

## Evidence

| date | setting | change | result | source |
|---|---|---|---|---|
| 11-05 | SMALL subset, LM 40 after Adam 4x50 | 1,024 / 4,096 / 16,384 / 65,536 px | 24.1 / 23.8 / 24.4 / 24.8 at 18 / 38 / 116 / 464 CPU s; HiRoom drift gone at 65k, the others don't respond; long Adam DTU 33.5 at every count | REPORT3 §3 |
| 11-05 | SMALL subset, LM 40 at 4,096 | gradient-weighted / uniform / stratified 4, 8, **16**, 32 cells | 23.8 / 26.0 / 25.3, 26.8, **27.5** (F1 45.1), 27.2; stratified removes the DTU drift on 2 subset scenes (54-55 from it. 5 to 40) | REPORT3 §3 |
| 11-05 | SMALL dev, 4 DTU scenes | LM 20 stratified vs LM 5 | DTU 45.7 vs 50.6: still worse there | REPORT3 §3 |
| 11-05 | SMALL dev, short budget | stratified | no pose gain (28.6 vs 29.1 at Fast), F1 47.9 vs 47.0 | REPORT3 §3 |
| 11-05 | SMALL | confidence quantile 0.5 (only the most confident pixels) | 21.8 vs 23.8 (worse), also worse from GT | REPORT3 §3-4 |
| 11-05 | SMALL | Adam at 256 px | 25.1 (too few) | REPORT3 §4 |
| 11-05 | SMALL | smaller pool, single pool for all levels, fixed LM samples, uniform/sqrt weighting at the short budget | neutral | REPORT3 §4 |
| 11-08 | SMALL dev, 3 seeds | depth-edge mask on samples (`depth_edge=0.05`) / on targets (`tgt_edge`) | 34.6 (ETH3D only) / 33.6 vs 33.7; gradient-weighted sampling puts more samples on bad depth (ScanNet++ 13.0 % vs 8.4 % uniform), the mask fixes that, but GT-start drift doesn't change | REPORT5 §4 |
| 11-09 | SMALL/LARGE, subset, old autograd path, 3070 | 16x samples per Adam step (4x LM, `sample_chunk`) | no change (SMALL HiRoom 39.1 vs 40.6, ETH3D 12.8 vs 12.1; LARGE within 0.5) at 11-15x the time | REPORT6 |

## Notes

Near convergence, LM steps at ≤ 4,096 px are mostly sampling noise; early steps are signal even at 1,024 (why cheap LM
works). `strat_cells` was tuned before the brightness fix; recheck (REPORT4 §11).

Related: [objective-bias](objective-bias-gt-start.md), [lm-stop-rule](lm-stop-rule.md), [neighbours-and-pairs](neighbours-and-pairs.md).
