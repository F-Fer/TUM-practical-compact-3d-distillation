# Depth-free pose refinement: dense patch matches + epipolar (Sampson) BA

Status: **open (pose AUC small positive, recon F1 not improved, no scaling)** · Area: TTO method / pose · Updated 2026-11-23

**Answer (dev, SMALL, seed 0):** a pose-only refinement on the Sampson distance of dense NCC patch matches (no depth
unknowns; the depth only predicts where to search) adds a little on top of B′ but converges after 2 rounds and does not
scale with more rounds. From B′: AUC@3 42.8 → 44.8 (DTU +9.4, HiRoom +3.4, 7-Scenes −1.5, ETH3D −1.7); from DA3 alone
16.2 → 30.8 (7-Scenes 41.0, better than B′'s 36.9); epipolar → B′ → epipolar 46.1 (B′ seeds 42.1-45.3). **Recon F1 doesn't follow:** B′ 49.9, epipolar from B′ 48.6 (HiRoom 56.3 → 49.3: B′'s depth no longer fits the moved cameras), + plane sweep on those cameras 46.1, epipolar → B′ → epipolar 48.6. Why it can't do
more: on B′'s cameras the matches are already explained to 0.15-0.6 px median Sampson, and the GT cameras explain the
same matches only marginally better (median lower on 7/10 HiRoom scenes, robust cost lower on ~6/10), although B′'s
relative rotations are off by up to ~4° on those scenes. B′'s remaining pose error lies mostly in directions that 2D
correspondences at this resolution barely see (rotation ↔ translation / bas-relief with free structure); removing the
depth from the objective doesn't remove them. Matching at 2× resolution doesn't sharpen the GT preference.

Related finding: the plane sweep (`scripts/lpba_mvs.py`) on Antonin's RoMa track-BA cameras (HiRoom, 10 dev scenes)
lifts recon F1 43.4 → 58.5 (re-scored with the current evaluator; B′ 56.3, B′ seeds 54.6-59.2) and the depth error
5.37 → 5.09 %: track BA's saved depth is inconsistent with its cameras, and the sweep repairs that.

## Method (`scripts/lpba_epi.py`)

Per round (`rounds` = sigma:radius:step): `samples` gradient-weighted pixels per frame; a 7×7 patch warped into each of
10 spread neighbours with the state's depth and the current cameras; 2D search ±radius px maximising NCC, parabolic
sub-pixel; kept if NCC > 0.8, not at the window border, unique by 0.02. Then Adam (300 steps) on the Cauchy (0.5 px)
Sampson distance over poses (frame 0 fixed) and the shared focal, with a penalty keeping the mean camera distance
(the epipolar cost has no scale). Prints AUC per round.

## Evidence

| date | setting (model, set, seeds, baseline) | change | result | source |
|---|---|---|---|---|
| 11-23 | SMALL HiRoom dev 10, B′ state, matches at 504 px (and 2×) | Sampson cost of the same matches under B′ vs GT cameras | median B′ / GT e.g. 0.32 / 0.26, 0.56 / 0.43, 0.46 / 0.39, 0.25 / 0.24, 0.58 / 0.61 px; robust (Cauchy 0.3) cost lower for GT on 6/10 (1×) and 6/10 (2×); a few wrong-but-B′-consistent matches inflate GT's Huber cost | `epi/gtcheck.py` |
| 11-23 | same | B′ pairwise errors | HiRoom relative rotation median 0.1-3.8°, translation direction 0.2-4.6°; ScanNet++ 09c1414f1b rotation 11° | `epi/errsplit.py` |
| 11-23 | SMALL dev 22, seed 0, B′ T=25 42.8 | epipolar refinement from B′ (rounds 2:8:1,1:4:1×3, Cauchy 0.5 px) | AUC@3 per round 42.8 → 44.3 / 44.7 / 44.8 / 44.8 (HiRoom 48.2, 7-Sc 35.4, ScanNet++ 25.8, DTU 96.0, ETH3D 18.7) | `epi/epi_bp` |
| 11-23 | same | from DA3 alone (rounds 4:24:2,2:8:1,1:4:1×4) | 16.2 → 30.8, flat after round 2 (HiRoom 21.3, 7-Sc 41.0, ScanNet++ 8.4, DTU 73.9, ETH3D 9.5) | `epi/epi_da3` |
| 11-23 | same | B′ (T=25) started from epi_da3, then epipolar 2 rounds | 43.9, then **46.1** (HiRoom 48.7, 7-Sc 36.5, ScanNet++ 25.3, DTU 95.5, ETH3D 24.5) | `epi/epi_da3_bp`, `epi_da3_bp_epi` |
| 11-23 | SMALL HiRoom dev 10 / all 30, RoMa turbo q8 track BA (Antonin) | plane sweep on its cameras (3 rounds, smooth σ 32) | F1 43.4 → 58.5 (dev), 39.7 → 60.0 (all 30); depth err 5.37 → 5.09 / 6.05 → 5.37 % | `mvs/mvs_roma`, `mvs/roma_rescore` |
| 11-23 | SMALL dev 22, seed 0, B′ F1 49.9 (56.3 / 51.7 / 35.8 / 55.6) | recon F1 (HiRoom / 7-Sc / ScanNet++ / ETH3D) of the epipolar cameras | epi_bp 48.6 (49.3 / 52.3 / 36.3 / 56.7); + sweep `c1_mvs` 46.1 (52.9 / 46.9 / 31.7 / 53.1, depth err H/S/E 4.80 / 6.81 / 4.48 %); epi → B′ → epi 48.6 (54.7 / 53.2 / 36.5 / 50.2) | `epi/*.recon.log` |
| 11-23 | same | 3 cycles plane sweep ↔ epipolar (matches predicted with the swept depth) from epi_bp | AUC@3 flat 44.8 / 44.8 / 44.9; depth err H/S/E rises 4.80 / 6.81 / 4.48 → 4.95 / 7.16 / 5.13 → 5.07 / 7.43 / 5.84 %; F1 sweep outputs 46.1 / 41.6 / 41.7 (B′ 49.9): the sweep drifts, as on B′'s cameras | `epi/c*` |

## Open / don't repeat

- The ceiling is observability, not the depth model: what would help is information that fixes the bas-relief-type
  directions (wider baselines / more frames in the optimisation, a learned pose prior, better initial cameras), or a
  depth prior good enough to fix them (GT depth fixed: +11 AUC, [depth-bottleneck](depth-bottleneck.md)).
- Not tried: feature-metric (VGG / DINO) matches instead of grey NCC; forward-backward match check; RoMa matches in the
  same Sampson BA; LM instead of Adam.

Related: [shared-depth-field](shared-depth-field.md), [depth-bottleneck](depth-bottleneck.md),
[track-ba](track-ba.md), [pose-prior-and-flat-directions](pose-prior-and-flat-directions.md)

Code: removed in the merge cleanup of 2026-11-26 (closed experiment); the last version is at commit `107ea16`, e.g. `git show 107ea16:scripts/<name>.py`: `lpba_epi.py`, `lpba_mvs.py` (plane sweep).
