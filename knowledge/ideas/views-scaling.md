# DA3 quality vs number of input views (no TTO)

Status: **answered** for SMALL/BASE/LARGE-1.1 · Area: feed-forward compute knob · Updated 2026-11-10

Report: `results/views/REPORT.md` (2026-11-01, tum2), tables `results/views/views.md`; scripts `scripts/run_views.py`,
`scripts/plot_views.py`; outputs `workspace/views/` (~29 GB).

**Answer:** quality saturates at 32-64 views; larger models gain much more from extra views, and the gain is better
cameras/depth for the same views (context), not coverage. At matched forward time the larger model with fewer views
wins, so views trade worse than model size.

## Evidence (7-Scenes + ScanNet++ pooled, 27 scenes)

| model | AUC@3 at 8 / 16 / 32 / 64 / 100 views | F1 at 8 / 64 / 100 | forward s at 16 / 100 |
|---|---|---|---|
| SMALL | 8.1 / 9.6 / 11.2 / 11.6 / 10.4 | 18.2 / 35.5 / 34.3 | 0.10 / 0.85 |
| BASE | 16.9 / 19.4 / 21.1 / 22.1 / 20.5 | 25.2 / 48.7 / 48.4 | 0.23 / 1.73 |
| LARGE-1.1 | 33.6 / 43.6 / 48.2 / 50.3 / 49.0 | 31.5 / 62.7 / 63.9 | 0.44 / 3.74 |

- Same 8 target views, more context (8 → 64): LARGE AUC@3 33.6 → 51.9, AUC@30 77.4 → 93.0, precision 48.1 → 68.2;
  SMALL only 8.1 → 12.7. The fixed-target gain accounts for about all of the standard-metric gain.
- LARGE at 16 views (0.44 s) beats BASE (1.73 s) and SMALL (0.85 s) at 100 views on pose, F1 and precision.
- ScanNet++ drives the trend; 7-Scenes LARGE pose is flat beyond 8 views; DTU-64 saturates at 8.
- N = 2 is bimodal (pairs exact or failed) and not comparable; N = 4 is the worst on ScanNet++.
- Forward time roughly linear in N above 16; peak memory nearly flat with `da3_lowmem`.

## Implications

DA3's context is saturated at the benchmark's 100 frames, and small models can't use extra views: explicit multi-view
TTO is a way to give them that information (PBA raised SMALL's ScanNet++ AUC@3 9.0 → 38.5 vs LARGE 56.6 → 76.3).
Not run: a contiguous window (fixed density), posed mode, GIANT.

Scoring notes: DTU recon ignores frame subsets; recon needs ≥ 3 views; ≤ 2 fusion workers and 52 GB RAM.

Related: [matched-compute](matched-compute.md), [dejaview-steps](dejaview-steps.md).
