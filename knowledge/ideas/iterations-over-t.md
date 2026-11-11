# How many Adam iterations (T per level)

Status: **answered for the current configuration** (photometric: T = 25 is enough; NCC-LM and feature-metric gain up to
T = 800) · Area: PBA · Updated 2026-11-10

**Answer:** the curve over T changed three times, each time for a found cause. Today (relative brightness + shared
focal) the photometric line is nearly flat from T = 25 to 800 on all four models; the NCC-LM configuration and SMALL
still gain from long Adam (ScanNet++, ETH3D), 7-Scenes falls slightly with T in every configuration.

## How the curve changed

| date | configuration | curve | explanation then | later explanation |
|---|---|---|---|---|
| 10-31 | v3 Adam only (VGG+NCC) | SMALL 4x25 → 4x800: HiRoom 19.6 → 37.5, ScanNet++ 9.2 → 31.1 (keeps rising for every model); ETH3D peaks early then falls (LARGE 52.4 at 4x50 → 29.3 at 4x800); 7-Scenes falls monotonically | the iteration count acts as a regulariser along flat pose directions | ETH3D: lens distortion ([radial](radial-lens-model.md)); 7-Scenes: GT ([evaluation](../evaluation.md)) |
| 11-06 | REPORT3 Recommended (grey Huber + LM + radial), all models | AUC@3 peaks at T = 25-50 and loses 9.5-12.3 by T = 800 (DTU 54.6 → 26.9 SMALL, GIANT 88.8 → 65.8) | "Adam follows the same biased objective" | **brightness collapse** (REPORT4) |
| 11-07 | + `bright_rel=true`, SMALL, 3 seeds | 31.7 / 34.0 / 35.6 / 36.7 / 39.0 at T = 25 / 50 / 200 / 400 / 800, flat beyond (38.0 at 1600, 39.8 at 3200); held-out T = 50 39.2, T = 800 41.3 | — | — |
| 11-08 | + shared focal (B′), SMALL held-out | T = 50 44.0, T = 200 44.1, T = 800 48.6 | — | — |
| 11-10 | B′ on all models, dev (figures5) | SMALL 43.8 → 46.8, BASE 59.2 → 60.8, LARGE 71.0 → 71.3, GIANT 74.9 → 73.3 from T = 25 to 800 | flat: the remaining error is in weak directions that need information, not iterations | — |

- Run cost is linear in T: analytic photometric ~0.15 s + 4.6 ms x T per scene (5090, 4 levels); old autograd v3
  0.7 s + 0.08 s x T.
- F1 doesn't follow the poses over T: with relative brightness HiRoom F1 falls with T (56.9 → 47.7) while ScanNet++ rises;
  the depth grid keeps changing during long Adam (REPORT4 §3).
- The radial gate's threshold (0.01) was tuned at T = 50; at other T it decides differently (REPORT3 §11, REPORT4 §9).

## Don't repeat

Reading "more Adam makes it worse" from runs before 2026-11-07: every photometric run with brightness parameters had
the collapse ([brightness-model](brightness-model.md)).

Related: [optimizer](optimizer-adam-lm-hybrid.md), [pose prior](pose-prior-and-flat-directions.md), [matched-compute](matched-compute.md).
