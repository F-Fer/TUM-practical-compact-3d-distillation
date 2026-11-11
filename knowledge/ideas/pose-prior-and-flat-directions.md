# Weak pose directions: pose prior, damping, pose basis

Status: **diagnosis confirmed** (DA3's error lies in the directions the objective constrains least) · `w_pose=10`
helped LARGE in session 2, not used now · **rejected:** pose basis (dev overfit), pose Laplacian prior, Levenberg damping
as a fix · **open:** information for the weak directions (wide-baseline term), uncertainty-aware priors · Area: PBA · Updated 2026-11-10

**Answer:** the photometric objective is nearly flat along some relative-pose directions (condition number ~6e4).
Converging drifts along them, and DA3's own error is concentrated in them, so PBA removes the well-constrained part and
leaves the weak part. Regularising towards DA3 recovers ETH3D/7-Scenes for good inits at the cost of scenes that need
large corrections; it doesn't pay for SMALL. Faster movement along a smooth basis didn't generalise.

## Evidence

| date | setting | change | result | source |
|---|---|---|---|---|
| 10-31 | LARGE dev ETH3D | Adam 4x50 vs 4x800 | photometric error identical (0.0888 vs 0.0886), visible fraction too, but A@3 52.4 → 29.3; 4x larger LM batches don't change it | REPORT2 §2 |
| 10-31 | LARGE dev, Adam 4x50 + LM 5 (mean A@3) | none / `w_pose=10` / `w_pose=100` / `lambda_min=1e-2` / Levenberg `mu` 1e-3, 1e-2, 1e-1 | 56.7 / **60.2** / 57.3 / 58.2 / 56.9, 57.8, 58.4 (every regulariser recovers ETH3D 28 → 46-51, costs ScanNet++) | REPORT2 §2 |
| 10-31 | SMALL dev, same | `w_pose=10`, `lambda_min`, `mu` | 31.4 → 30.8 / 29.2 / 30.0: doesn't pay, DA3's poses are too poor to be a prior | REPORT2 §2 |
| 10-31 | LARGE full | Adam 4x50 + LM 5, `w_prior=100`, `w_pose=10` | ETH3D 46.0, ScanNet++ 76.3, DTU CD 1.62 in 6 s; −2.7 HiRoom A@3 vs no pose prior | REPORT2 §6 |
| 11-05 | SMALL, LM steps | noise analysis | after 5 LM iterations 95 % of the step's squared norm is in the weakest 10 % of directions (ETH3D 99 %) | REPORT3 §3 |
| 11-08 | SMALL dev, `scripts/weak_modes.py` (at commit `1c1a921`) | share of DA3's pose error in weakest 10 % / next 40 % / strongest 50 % of eigen-directions | HiRoom 42/42/16, 7-Scenes 60/34/6, ScanNet++ 62/30/8, DTU 76/21/3, ETH3D 70/18/12 (random: 10/40/50); after B50 60-90 % of the remaining error is in the weakest 10 % | REPORT5 §1 |
| 11-08 | SMALL held-out, `scripts/pose_change_decomp.py` (at `1c1a921`) | what long runs remove | ScanNet++ smooth rotation 2.2° → 1.7°, gross frames 8.2 → 6.9 %; HiRoom long runs add gross frames (0.8 → 2.3 %) while AUC rises | REPORT5 §1 |
| 11-08 | SMALL dev | pose basis K = 6 (Laplacian eigenvectors of the covisibility graph, Adam only) | ScanNet++ +8.6 (T = 50), +9.6 (T = 200, above B800), paid on 7-Scenes/DTU; B′ + K = 3/6: 46.9/47.0 vs 44.7 | REPORT5 §3 |
| 11-08 | SMALL held-out | same | **+0.2 to +0.5 only**, ScanNet++ level or lower, F1 −1.2: the dev gain was 3 ScanNet++ scenes; K = 12 breaks small scenes (HiRoom −12.5) | REPORT5 §3 |
| 11-08 | SMALL dev | world axes only (`pose_world=true`) | 34.6 vs 33.7 | REPORT5 §3 |
| 11-08 | SMALL dev | pose Laplacian prior `w_lap` 1 / 10 / 100 | 30.0 / 21.6 / 17.5 vs 33.7: harmful at every strength (neighbouring cameras need different corrections) | REPORT5 §4 |
| ≤ 11-05 | LARGE, track BA (Antonin) | prior on DA3's poses in own LM BA, weight 30/100/300/1000 | 100 best (61.0 dev) | [ba-iterations](ba-iterations.md) |

## Open

- Information instead of regularisation: a wide-baseline term chosen for the weak modes (tracks or NCC pairs) on top of
  B′ (REPORT5 §11). ScanNet++ held-out gains most from long Adam (40-41) and NCC (49-50).
- Uncertainty-aware pose prior (DA3 outputs no pose uncertainty; proxies: disagreement between camera head and
  ray-based pose, Hessian weak directions).
- `w_pose` was never retested with the current configurations, nor on GIANT.

Related: [optimizer](optimizer-adam-lm-hybrid.md), [objective-bias](objective-bias-gt-start.md), [depth-prior](depth-prior.md).
