# Depth prior strength

Status: **partly adopted:** default `w_prior=1` everywhere in the current configurations; `w_prior=100` helped LARGE in
sessions 1-2 · **open:** confidence-weighted per-pixel prior, retest on the current configuration · Area: PBA · Updated 2026-11-10

**Answer:** the strength should follow the quality of the initialisation: a weak init (SMALL) needs the depth to move,
a good init (LARGE) is mostly hurt by free depth corrections. A prior fitted to DA3's error statistics is neutral.

## Evidence

| date | setting | change | result | source |
|---|---|---|---|---|
| 10-30 | SMALL dev, v3 | `w_prior` 1 / 10 / 100 | HiRoom A@3 34.2 / 33.5 / 30.3, F1 47.4 / 48.5 / 46.9; DTU CD 4.30 / 4.24 / 4.51 | REPORT1 |
| 10-30 | BASE dev | 1 / 10 / 100 | HiRoom A@3 50.8 / 50.8 / 49.7; 7-Scenes F1 58.7 / 63.8 / 63.0; DTU CD bad for every value (focal, not depth) | REPORT1 |
| 10-30 | LARGE dev | 1 / 10 / 100 / 1000 | HiRoom F1 77.4 / 84.5 / 83.4 / 82.0, 7-Scenes F1 54.8 / 60.1 / 65.7 / 58.6, DTU CD 2.59 / 2.40 / 1.94 / 1.59 | REPORT1 |
| 10-30 | full benchmark | w = 10 SMALL/BASE, 100 LARGE | ~neutral for SMALL/BASE; LARGE v3 w = 100 better than DA3-LARGE on every metric except a DTU-CD near-tie (HiRoom F1 69.5 → 82.8) | REPORT1 |
| 10-31 | LARGE full, hybrid | `w_prior=100` (+ `w_pose=10`) | best LARGE results of session 2 (DTU CD 1.62 mm, first PBA result below DA3's 1.81) | REPORT2 §6 |
| 11-08 | SMALL dev, 3 seeds | prior fitted by ML to DA3's depth error (`scripts/fit_depth_prior.py`): offset prior half the shape prior (`w_scale=0.5`), smoothness between cells (`w_smooth=0.14`) | 34.0 and 33.8 vs 33.7: neutral | REPORT5 §4 |

## Open

- No run since session 2 uses `w_prior ≠ 1`: the current configurations (shared focal, relative brightness, LM on
  NCC) were never re-tuned per model, and **GIANT has never been run with a depth or pose prior** (memory audit 11-05).
- Confidence-weighted per-pixel prior (DA3's depth confidence ranks depth outliers with AUROC 0.66-0.73,
  [da3-failure-modes](da3-failure-modes.md)). Proposed in REPORT1, never built.

Related: [pose-prior-and-flat-directions](pose-prior-and-flat-directions.md), [depth-parametrisation](depth-parametrisation.md).
