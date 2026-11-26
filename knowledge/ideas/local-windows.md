# Local windows instead of one global pass: is DA3-SMALL better in small groups, and can PBA use it?

Status: **rejected (stitched window predictions are a worse PBA start than the full pass)**; the diagnosis stands:
DA3-SMALL is much more accurate inside a 16-frame window than in its 100-frame pass · Area: TTO init · Updated 2026-11-25

**Answer (SMALL, ScanNet++ all 20 scenes, 3080):** DA3-SMALL run on 16 consecutive frames (stride 8, 12 windows,
reference = middle frame) gives far better relative poses for the pairs inside a window than the 100-frame pass gives
for the same pairs (AUC@3 19.0 vs 4.9, AUC@30 71.7 vs 51.5, median rotation 1.55° vs 3.84°, translation direction 4.6°
vs 12.2°). On 7-Scenes there is no difference (10.6 vs 11.1). But no assembly of the windows was a better start for
B′ (`iters=25`) than the full pass: 34.0 (full pass) vs 30.3 (windows aligned to the full pass, poses averaged),
27.5-28.9 (each frame from its most central window, window / full-pass depth), 9.3 (Sim(3) chain over the shared
frames, drifts: AUC@30 53 at the start), 34.3 (full-pass cameras, depth averaged with the windows'). The windows cost
2x the full pass (SMALL's forward is about linear in the frames: 0.15 s per 16 frames, 0.85 s per 100 on the 3080).

## Evidence

| date | setting (model, set, seeds, baseline) | change | result | source |
|---|---|---|---|---|
| 11-25 | SMALL, ScanNet++ 20 scenes (dev + held-out), 100 frames each, 3080 | DA3-SMALL on 12 windows of 16 consecutive frames (`ref_view_strategy=middle`) vs the benchmark's pass; pairs inside a window | AUC@3 / AUC@30 / median rot / median trans. dir.: full pass 4.9 / 51.5 / 3.84° / 12.2°; windows **19.0 / 71.7 / 1.55° / 4.6°**; 7-Scenes dev (2): 11.1 / 75.7 vs 10.6 / 77.8 | `scripts/lpba_win.py`, `workspace/tto/DA3-SMALL/win/w16_all2.log` |
| 11-25 | same | all-pairs AUC@3 / AUC@30 of the assembled start | full pass 8.5 / 64.9; anchor (each window aligned to the full pass by the medoid of its per-camera Sim(3), poses averaged over 2 windows) 11.4 / 67.3; anchor1 (each frame from its most central window) 12.3 / 70.4; chain (window k+1 → k over the 8 shared frames) 10.7 / 53.1; depth abs. rel. unchanged (6-7 %) | same |
| 11-25 | same, B′ `iters=25`, seed 0 (full pass: **34.0**) | B′ from the assembled starts | anchor 30.3; anchor1 27.5; anchor1 with the full pass's depth 28.9; chain 9.3; full-pass cameras + depth averaged over pass and windows 34.3 | `win/*_pho25` |

## Open / don't repeat

- Don't stitch windows by rigid alignment or chaining and hand them to PBA: a better start AUC (12.3 / 70.4 vs
  8.5 / 64.9) became a worse result. The window boundaries leave discontinuities that B′ doesn't remove, and after B′ the
  pairs inside a window are already better than the windows' own accuracy.
- Not tried: a robust pose graph over windows at several temporal strides; window predictions as relative-pose
  factors inside PBA; re-running only the low-confidence frames with their neighbours (DA3's confidence flags gross
  frames with AUROC ~0.9, [gross failures](gross-failures-and-reference-view.md)).
- The local accuracy says DA3-SMALL's ScanNet++ problem is global consistency over 100 frames, not the images.

Related: [da3-tta](da3-tta.md), [gross-failures-and-reference-view](gross-failures-and-reference-view.md),
[views-scaling](views-scaling.md)

Code: removed in the merge cleanup of 2026-11-26 (closed experiment); the last version is at commit `107ea16`, e.g. `git show 107ea16:scripts/<name>.py`: `lpba_win.py`.
