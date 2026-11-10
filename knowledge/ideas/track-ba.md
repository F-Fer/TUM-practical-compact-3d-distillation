# Track-based BA (Antonin): RoMa / VGGSfM tracks + COLMAP-style BA

Status: **works, but now behind the photometric BA on every model** (dev pose, at 3-30x the time; the track runs
predate the shared focal and relative brightness) · Area: TTO method · Owner: Antonin · Updated 2026-11-10 (from
CLAUDE.md "TTO status" 2026-11-05, README, and the run metrics; Antonin should check this file)

`scripts/run_tto_ba.py` (+ `scripts/tracks_roma.py`): RoMa v2 (`tracker=roma`) or VGGSfM (`tracker=vggsfm`) tracks,
then pycolmap BA. Named configurations in `scripts/tto_configs.sh`; sweeps `run_tto_roma_loop.sh`; timing
`slurm_time_tracks.sh`, `slurm_time_vggsfm.sh`; tables `tracks_table.py`.

## Settings it needs (each found by a failure)

- `max_reproj=1000000`: the default 8 px gate against DA3's own projection removed the good tracks whenever DA3's focal
  was off (e.g. 554 vs 436 px true).
- `loss=cauchy loss_scale=0.25`: pycolmap's default loss is trivial, so the few wrong matches dominated.
- `shared_camera=true`; `extra_queries` for frames no query frame sees.
- `camera=radial` models the lens (ETH3D); `depth_fit=grid fit_grid=4 fit_ridge=10 reject=10` refits DA3's depth to the
  BA points.
- BA fixes the scene scale freely (0.2x to 125000x DA3's size on HiRoom): `run_tto_ba.py` rescales the saved depth with
  the poses (`scripts/rescale_depth.py` repairs older runs); otherwise fusion blows up the RAM. Pose AUC doesn't notice.

## Results (dev, one seed; AUC@3 includes DTU-64 = 100 for every BA run)

| run | SMALL | BASE | LARGE-1.1 | time per scene (5090) |
|---|---|---|---|---|
| DA3 alone | 16.2 | 27.1 | 48.6 | 0.4-1.6 s |
| RoMa tracks + radial BA (`time_rad`) | 53.7 | 54.6 | 56.9 | ~37 s (BA CPU-bound) |
| + depth-only polish (`time_depth_only`) | 53.7 | 54.6 | 56.9 | +1.3 s |
| + full polish (`time_full_polish`) | 46.9 | 51.8 | 54.0 | +1.0 s |
| RoMa + own LM BA, k = 2 / 10 / 25 / 100 iterations (`ba_fast_it<k>`) | 37.9 / 50.3 / 51.8 / 53.4 | 43.4 / 50.1 / 50.0 / 50.7 | **58.5** / 56.3 / 55.3 / 57.7 | 12-22 s |
| VGGSfM + BA (plain loss, 8 px gate; `time_vggsfm`) | 18.9 | 27.7 | 43.0 | 91-105 s |
| for comparison: photometric + LM on NCC, T = 800 (`ncc_it800`, shared focal) | 55.6 | 64.8 | 71.9 | ~4.7 s TTO |
| for comparison: photometric B′ T = 25 (`pho_it25`) | 43.8 | 59.2 | 71.0 | 0.2-0.3 s TTO |

- README/CLAUDE.md table (2026-11-05, F1 3 sets): SMALL tracks + radial BA + depth-only polish 54.1 / 46.6, + full
  polish 46.6 / 53.5; BASE 54.3 / 48.5 and 49.6 / 59.3; LARGE 57.4 / 62.2 and 55.2 / 70.2. Against the photometric
  polish of that time, tracks won on SMALL/BASE pose; for LARGE the polish alone was as good at 1/20 of the cost.
- REPORT3 §11 (11-06), RoMa tracks + BA (13-16 s) against REPORT3's Recommended photometric BA: BASE 50.1 (13.8 s) vs
  50.6 (1.19 s); LARGE 58.5 (13.6 s) vs 64.3 (1.92 s); SMALL 51.8 (15.0 s) vs 32.8 (0.77 s): then SMALL gained far
  more from tracks. F1 (3 sets): SMALL 48.2 / 48.0, BASE 49.4 / 59.0, LARGE 60.1 / 69.3. The shared focal and
  relative brightness (REPORT4-5) closed SMALL's gap.
- Session 1 (10-30): the earlier `vggt-ba` sparse BA beat PBA only on DTU (68.2 vs 62.3) and hurt LARGE's good poses
  (HiRoom 53.8 → 23.7, ETH3D 30.4 → 8.5).
- Track lines on the compute plots have no ETH3D recon: their F1 is a 3-set mean, not comparable with the 4-set series.

## Negative results (don't repeat)

Correspondence terms inside PBA, a per-scene gate between the methods, the initial reprojection residual as a predictor,
a focal-anchored polish for DTU-64 ([correspondences-in-pba](correspondences-in-pba.md)).

## Open (from CLAUDE.md)

SMALL/BASE with a weak pose prior (< 30); a faster solver so the anchored BA is a good compute point; a second seed;
recon of the anchored cameras without the polish; aligned F1 for the photometric runs; a tracker newer than RoMa v2.
New: rerun the track BA with the shared focal / current photometric polish, since the comparison above mixes ages.

Related: [ba-iterations](ba-iterations.md), [matched-compute](matched-compute.md).
