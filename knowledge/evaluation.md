# Evaluation: sets, noise, conventions and pitfalls

Read this before comparing any two numbers. Updated 2026-11-10.

## Scene sets

| set | scenes (HiRoom / 7-Scenes / ScanNet++ / DTU-64 / ETH3D) | use |
|---|---|---|
| **dev** | 10 / 2 / 3 / 4 / 3 = 22, the first scenes of each dataset | tuning, all compute plots |
| **held-out** | 20 / 5 / 17 / 9 / 8 = 59, the rest of the pose datasets | testing what dev chose (REPORT3 §6 onwards) |
| **full** | 30 / 7 / 20 / 13 / 11 = 81 (+ DTU-49 22 scenes for recon/Chamfer) | the DA3 benchmark itself |
| mini ("subset" in REPORT3) | 5 / 1 / 2 / 2 / 3 = 13 | cheap sweeps; correlation with the dev mean 0.98 over 109 runs |
| drift, tiny | 5/0/0/2/3, 2/0/0/0/1 | GT-start and debugging runs |

`scripts/index_runs.py` labels every run with its set (column `set` in [runs.csv](runs.csv)).

## Metrics and averages

- **Pose:** AUC@3 / AUC@30 in %, the mean over the five pose datasets (each dataset = mean over its scenes). Pose
  numbers in the PBA runs come from `<dataset>_pba.json` (per scene), the same values as the evaluator's `_pose.json`.
- **Recon F1:** TSDF fusion of the benchmark, `recon_unposed`. Two averages are in use: **4 sets** (HiRoom, 7-Scenes,
  ScanNet++, ETH3D; the compute plots since 2026-11-02, REPORT5 onwards) and **3 sets** (without ETH3D; REPORT1-4,
  CLAUDE.md/README tables, the track-BA runs `time_*`). Don't compare across the two. ETH3D recon on the 3 dev scenes
  takes ~80 s and ≤ 15 GB RAM; on the full set its fusion needs up to ~58 GB.
- **DTU:** DTU-64 for pose, DTU-49 Chamfer distance (mm) for recon. Every track-BA run scores DTU-64 AUC@3 = 100
  (saturated), which inflates the five-dataset mean of track runs by several points.
- Recon scores aren't exactly reproducible (Open3D's unseeded point sampling, ~1e-4 F1). On tum2 BASE's initial
  ETH3D F1 is 3-7 points above tum1's (different predictions, same pose).

## Noise: how big a difference has to be

| what | spread | source |
|---|---|---|
| dev AUC@3 mean, over seeds | ±0.2 to ±0.4 (sessions 3-5); ±0.6 (session 1, slower config) | REPORT1, REPORT3 §0, REPORT5 |
| dev F1 mean, over seeds | ±0.5 to ±1 (HiRoom F1 alone ±2) | REPORT3 §0, ablations 2026-11-02 |
| held-out AUC@3 mean, one seed | roughly ±0.5 | REPORT5 §9 |
| a single scene | several points | the optimisation is chaotic (below) |
| ETH3D dev mean at long T | 30.4 to 39.9 over seeds (the radial gate decides differently) | REPORT4 §3 |

- **Chaos:** PBA diverges after tens of iterations from a 1e-9 perturbation (REPORT3 §2). On the GPU runs aren't
  bit-identical (`index_add_` atomics); on the CPU they are. So compare seed means, not scenes.
- **Dev overfits.** Several dev gains didn't survive held-out: the pose basis (+8.6 ScanNet++ on dev, +0.2 held-out),
  the stop rule (+1.8 dev, +0.6 held-out), gate 0.01 vs 0.02 (+1.3 dev, +0.1 held-out). ETH3D has 3 dev scenes and
  decides much of the dev mean (one fifth of it). Report held-out numbers for anything that will be claimed.

## Timing conventions

- **TTO time** = `opt_s` per scene: feature extraction + Adam + LM. Not included: image decoding (~1.6-1.8 s per scene,
  "a pipeline has the images"), covisibility (4-23 ms), the DA3 forward pass.
- **Total time** on the compute plots = DA3 forward (`workspace/evaluation/<model>/timings.jsonl`) + TTO. DA3 forward on
  the 5090: SMALL 0.40, BASE 0.85, LARGE-1.1 1.60, GIANT-1.1 3.10 s (dev means); full-benchmark medians 0.37 / 0.76 /
  1.54 / 2.85 s. (Checked 2026-11-10: the latest `runs.jsonl` entry of all four models is an RTX 5090 on tum2; plots
  before 11-06 used RTX 3080 forward times.)
- The **first scene of each dataset** includes process warm-up (up to 16 s on the old LM path) and is replaced by the
  dataset's median (`plot_compute.py`, `index_runs.py`).
- **GPU:** RTX 5090 on tum2 for sessions 1-4 and 7 and the compute plots; RTX 3070 for session 5 (2.5x slower on small
  scenes, 5-6x on 100 frames) and session 6. Runs that shared a GPU have contaminated times (named in each report);
  `<run>_t` reruns hold clean times.
- The TTO cost doesn't depend on the model (same frames and pixels), so the TTO advantage shrinks as the forward pass grows.

## Pitfalls found in the evaluation itself

- **Frame order changes the pose AUC.** `compute_pose` gets world-to-camera matrices, so `inv(T_i) @ T_j` isn't the
  relative pose and its translation depends on which camera is first. Sorting DTU scan1's frames moves AUC@3 27.1 → 33.9
  with identical predictions. TTO runs must keep the frame order (failure-mode report).
- **The reference view is a hidden variable** of DA3: on one ScanNet++ scene AUC@3 ranges 2.3-26.1 with the reference
  view alone. Keep `ref_view_strategy=first` when comparing ([gross failures](ideas/gross-failures-and-reference-view.md)).
- **7-Scenes is an unreliable reference.** GT depth is shifted 4-12 px horizontally against the RGB images (4 scenes
  checked; controls < 1 px), GT rotations are slightly non-orthonormal, and the GT is KinectFusion with the depth
  camera's intrinsics (fx 585 vs ~525 RGB; DA3 predicts ~530) (Brachmann et al., ICCV 2021). No TTO method moves
  7-Scenes AUC@3 off ~26-28 (full) / ~35 (dev), and its "focal error" of 8-10 % never decreases. Treat 7-Scenes changes
  as noise unless large.
- **BA leaves the scale free.** Refined cameras can be 0.2x to 125000x DA3's size while the saved depth is DA3's; the
  evaluator takes the scale from the poses, so fusion blew up (RAM). `run_tto_ba.py` rescales the depth; `init_from`
  in `run_tto_pba.py` rescales translations to DA3's scale (without it: a degenerate solution with AUC@30 = 10).
- **`results.npz` stays pinhole.** Neither radial model is written out, so the evaluator's fusion never sees the lens:
  radial lens models don't help ETH3D recon directly.
- **Old PBA configurations no longer reproduce bit-exactly** (sampling probabilities from the blurred grey image since
  session 3; `gray=True` default since 2026-11-02, earlier runs are RGB). Within noise.
- `open3d` must stay 0.19.0 (0.20.0 makes empty TSDF meshes, F1 ≈ 1 % without an error); `pycolmap` 3.10.0.
