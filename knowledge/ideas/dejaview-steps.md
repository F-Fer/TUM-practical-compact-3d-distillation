# DéjàView with fewer loop steps K (and TTO on top)

Status: **in progress (Antonin)** · K sweep scored, not written up; TTO on DéjàView predictions queued · Area:
feed-forward compute knob · Owner: Antonin · Updated 2026-11-10 (numbers read from the metric files; Antonin should check)

DéjàView (117M parameters) applies one shared transformer block K times; K is an inference-time compute knob
(default 12). Branch `dvlt-fewer-steps`: `run_bench.py --dvlt-steps`, `scripts/slurm_dvlt_k.sh` (benchmark at K),
`slurm_dvlt_tto.sh` (the TTO methods of the DA3 tables on DéjàView predictions: init, photometric polish, RoMa tracks
+ radial BA, depth-only and full polish), `slurm_dvlt_tto_lm.sh` (track BA with `ba_lm` at 15 and 100 iterations).
Outputs: `workspace/evaluation/dvlt_K<K>` (K = 12 is `dvlt`), TTO in `workspace/tto/dvlt_K<K>/` (none yet on 11-10).

## K sweep (full benchmark, tum2; AUC@3 mean of 5 datasets, F1 mean of 4 sets, median forward time)

| K | AUC@3 | HiRoom / 7-Scenes / ScanNet++ / DTU-64 / ETH3D | AUC@30 | F1 (4 sets) | DTU-49 CD (mm) | forward s |
|---|---|---|---|---|---|---|
| 1 | 0.0 | 0 / 0 / 0 / 0 / 0 | 5.1 | — | — | 0.42 |
| 2 | 0.2 | | 23.9 | (partial) | — | 0.47 |
| 3 | 2.1 | 3.4 / 2.1 / 1.2 / 2.5 / 1.3 | 52.5 | 17.6 | 6.35 | 0.52 |
| 4 | 9.0 | 14.5 / 6.8 / 6.2 / 11.6 / 5.9 | 72.0 | 32.6 | 5.87 | 0.56 |
| 6 | 31.1 | 44.1 / 21.7 / 30.4 / 40.0 / 19.6 | 87.9 | 54.3 | 4.25 | 0.66 |
| 8 | 47.6 | 64.1 / 27.6 / 55.7 / 59.0 / 31.7 | 91.9 | 63.4 | 3.00 | 0.76 |
| **12** | **58.7** | 76.6 / 29.7 / 72.0 / 79.5 / 35.9 | 94.0 | 67.7 | 2.46 | 0.96 |

For comparison, DA3 on the same machine (full benchmark, pose only there): SMALL 14.4 (0.37 s), BASE 27.3 (0.76 s),
LARGE-1.1 51.3 (1.54 s), GIANT-1.1 67.0 (2.85 s) AUC@3.

- DéjàView K = 12 is above DA3-LARGE-1.1 on pose at 62 % of its forward time; K = 8 ≈ LARGE-1.1 at half the time.
  Quality collapses below K = 6.
- From the distillation runs: DéjàView at K < 8 is very sensitive to resampling (a cv2-resized cache moved AUC@3 by
  ~10 points), so always use its own `preprocess_images` ([distillation](distillation.md)).

## Open

TTO on DéjàView at K = 3-12 (jobs submitted 11-09/10): does a cheap K + TTO beat a larger K? This is the same
matched-compute question with a different compute knob ([matched-compute](matched-compute.md)).
