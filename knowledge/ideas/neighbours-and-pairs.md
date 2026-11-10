# Which frame pairs: neighbours, spread, all pairs

Status: **adopted:** 10 targets per frame spread over the covisibility ranking (`nbr_mode=spread neighbors=10`,
covisibility ≥ 0.2) · **rejected:** fewer neighbours, all pairs (time), wider baselines via `min_covis` · **small gain:**
recomputing neighbours per level · Area: PBA · Updated 2026-11-10

**Answer:** wide baselines only help with a viewpoint-robust cost (VGG/NCC); with 10 spread neighbours small scenes
already use all pairs, and more pairs help only on scenes with many frames, inconsistently.

## Evidence

| date | setting | change | result | source |
|---|---|---|---|---|
| 10-30 | SMALL dev | spread k = 10 with RGB | 41.1 on DTU: nothing; with VGG + NCC DTU 41 → 48.5 (bas-relief ambiguity: top-6 PBA barely reduced DTU rotation error, 1.3-2.2° → 0.8-1.9°, sparse BA reaches 0.4°) | REPORT1 |
| 11-05 | SMALL subset | 5 or 3 neighbours | F1 39.6 / 39.4 (worse) | REPORT3 §4 |
| 11-08 | SMALL dev, 3 seeds | `nbr_refresh=true` (recompute per level) | 34.4 vs 33.7 (+0.7) | REPORT5 §4 |
| 11-08 | SMALL dev | `min_covis=0.1` | 33.5 (level) | REPORT5 §4 |
| 11-08 | SMALL dev | 14 neighbours | +1.8 on the 3 sets that fit in 8 GB (39.5 vs 37.7); 100-frame sets OOM on the 3070 | REPORT5 §4 |
| 11-09 | SMALL/LARGE subset, old path, 3070 | all pairs (`neighbors=1000 nbr_mode=top`) | identical with ≤ 11 frames; SMALL HiRoom 40.6 → 45.6 is one 23-frame scene (45 → 66); ETH3D +2.5 / −0.6, DTU +0.1; 4-6x time | REPORT6 |

## Open

14 neighbours or `nbr_refresh` on the 5090 (memory isn't the limit there); pairs chosen for the weak pose directions
([pose prior](pose-prior-and-flat-directions.md)).

Related: [pixel-sampling](pixel-sampling.md), [feature-metric-vgg](feature-metric-vgg.md).
