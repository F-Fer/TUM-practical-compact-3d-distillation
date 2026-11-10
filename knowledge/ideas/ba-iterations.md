# How many BA iterations the track BA needs (Antonin)

Status: **answered** (SMALL ~10-15 iterations; LARGE gains in 2-3 and then drifts; a pose prior of weight 100 keeps
it) · Area: track BA · Owner: Antonin · Updated 2026-11-10 (from CLAUDE.md "TTO status" 2026-11-05; Antonin should check)

pycolmap 3.10 binds no solver options, and every BA of the dev scenes ends with "Iterations 101, No convergence"
(COLMAP's convergence tolerances appear to be 0 by default; not checked in the source), so the iteration cap is the only
stop. To study the iterations, the BA problems are dumped (`run_tto_ba_iterations.py`) and re-solved with an own LM
(`scripts/ba_lm.py`; curves `ba_iteration_curves.py`; pipeline runs with a cap `run_tto_ba_truncated.py`).

## Evidence (dev)

- `ba_lm` has the same objective and starts at the same cost (rms 0.29818 px) but ends 2.3 AUC@3 below pycolmap on SMALL.
- Speed: 0.03-0.22 s per iteration on the RTX 5090 (0.22 s for 7-Scenes chess, 740k observations); it was 3.4 s before
  the Schur complement was built as one dense product (`step(schur="dense")`, matches the pairwise loop to 1e-11).
- SMALL needs ~10 iterations for 95 % of its gain and 15 for 99 %.
- LARGE-1.1 gains ~12 AUC@3 within 2-3 iterations, then often drifts back (HiRoom 828763: 59 → 70 → 26 at iterations
  0, 2, 100). A prior on DA3's poses keeps it close: weight 100 beats 30, 300 and 1000; 25 iterations with weight 100
  give LARGE-1.1 61.0 (dev), the best track-BA LARGE pose.
- Run metrics: `ba_fast_it<k>` (k = 2, 5, 10, 25, 100) per model in [runs.csv](../runs.csv): SMALL 37.9 → 53.4,
  BASE 43.4 → 50.7, LARGE 58.5 at k = 2 → 57.7 at 100.

## Open

SMALL/BASE with a weak prior (< 30); a faster (sparse Schur, analytic Jacobian) solver so the anchored BA becomes a
good compute point.

Related: [track-ba](track-ba.md), [pose-prior-and-flat-directions](pose-prior-and-flat-directions.md) (the same drift in PBA).
