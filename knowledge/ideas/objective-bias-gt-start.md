# Is the objective's minimum at the true poses? (the GT-start test)

Status: **tool adopted** (cheap test for any objective change) · finding: **with per-pixel grey Huber the optimum is
biased; NCC removes most of it** · Area: PBA · Updated 2026-11-10

**Answer:** started at the ground-truth poses and focal (AUC@3 = 100), LM walks away while the photometric error falls.
The drift doesn't depend on the pixel count (bias, not noise). It shrinks with more depth freedom, a lens model on
ETH3D and, most of all, NCC instead of per-pixel grey differences.

**How to run:** `init_gt=pose_focal levels=1 lm_levels=0 lm_iters=5` and again with `lm_iters=20` (LM from the GT poses
and focal, no Adam), on the `drift` scenes (HiRoom 5, DTU-64 2, ETH3D 3). The old code tracked AUC@3 per iteration
in one run (`track_every`, `scripts/pm_table.py --track`); that was removed in the 2026-11-10 refactor; the code is at commit `1c1a921`.

## Evidence (AUC@3 at LM iteration 5 / 20; REPORT3 §3, REPORT4 §4)

| objective | HiRoom (5) | DTU-64 (2) | ETH3D (3) |
|---|---|---|---|
| grey Huber, 8-cell grid, 4,096 px | 66.5 / 51.9 | 84.8 / 79.5 | 31.5 / 10.1 |
| 16,384 px | 66.4 / 52.6 | 84.8 / 79.7 | 31.7 / 11.3 |
| depth frozen | 39.3 / 24.6 | 68.3 / 31.2 | 30.8 / 11.0 |
| 4-cell / 16-cell grid | 60.6 / 40.3 · 70.4 / 56.0 | 82.8 / 75.1 · 88.5 / 85.4 | 33.7 / 10.1 · 27.9 / 11.5 |
| + radial coefficient | 66.1 / 61.5 | 84.2 / 79.4 | **43.1 / 40.7** |
| no geometric term | 63.5 / 48.9 | 78.3 / 75.4 | 35.8 / 11.7 |
| confidence quantile 0.5 | 57.9 / 52.4 | 80.1 / 77.2 | 22.2 / 9.8 |
| Huber δ = 0.02 | 71.3 / 54.8 | 86.7 / 79.8 | 39.2 / 11.2 |
| no focal | 67.5 / 52.9 | 83.2 / 67.3 | 33.8 / 13.4 |
| uniform / stratified-8 sampling | 54.9 / 40.9 · 65.7 / 53.6 | 81.5 / 73.6 · 85.1 / 78.3 | 34.1 / 15.7 · 35.1 / 13.0 |
| Huber, fixed samples, free per-sample depth | 63.7 / 50.7 | 89.0 / 88.0 | 40.0 / 17.1 |
| **NCC, depth frozen** | 88.3 / 64.6 | 99.4 / 91.4 | 85.0 / 50.5 |
| NCC, 8-cell grid | 92.2 / 73.8 | 99.3 / 94.8 | 81.9 / 49.3 |
| NCC, 16-cell grid | 93.0 / 78.7 | 99.4 / 95.6 | 81.5 / 47.8 |
| NCC, free depth | 91.8 / 78.9 | 99.6 / 97.4 | 83.3 / 56.4 |
| NCC, free depth, no geometric row | 89.5 / 86.0 | 99.5 / 96.9 | 86.5 / 61.9 |
| NCC, free depth, radial | 91.9 / 77.2 | 99.6 / 97.2 | 87.1 / 64.5 |
| depth-edge masks (REPORT5, 2 seeds) | 66 / 66 | 82 / 79 | 28 / 10 |

## Reading

- REPORT3 concluded "depth rigidity + ETH3D fisheye"; REPORT4 showed the Huber runs here kept brightness at zero, and
  per-pixel grey differences carry a bias NCC doesn't. Most of the gap is the residual type.
- Near convergence, steps at ≤ 4,096 px are largely sampling noise (step cosine between independent samples 0.15-0.44
  after 20 iterations) but at 65,536 px they agree and still move: something real keeps moving the poses (REPORT3 §3).
- The GT-start test is not the same as the gain from DA3's start: free depth passes it but loses from DA3's start
  ([depth-parametrisation](depth-parametrisation.md)).

Related: [ncc](ncc.md), [radial-lens-model](radial-lens-model.md), [pixel-sampling](pixel-sampling.md).
