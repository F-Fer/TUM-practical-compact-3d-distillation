# DA3's gross pose failures and the reference view

Status: **diagnosed** (the reference view decides which frames fail) · **rejected:** discrete relocalisation of flagged
frames · **open/untested:** several reference views per scene, choose or merge by confidence · Area: DA3 / TTO
init · Updated 2026-11-10

**Answer:** gross pose errors (> 15° or > 25 % of scene depth) are 13.6 % of ScanNet++ frames and 17.8 % of ETH3D frames
on DA3-SMALL and hold 94-96 % of the squared rotation error there. They come in runs, mostly as rotations about the
vertical axis (the camera faces the wrong way), and the reference view decides which frames fail. PBA repairs few of
them (7-9 % of frames stay wrong), and fixing them before PBA is worth at most +1.5 AUC@3 / +4.1 AUC@30 (oracle).

## Evidence

| date | setting | test | result | source |
|---|---|---|---|---|
| 11-07 | SMALL, ScanNet++ (4 scenes) | same reference view, other frames shuffled | same failing frames (overlap 0.97-1.00) | failure-mode report |
| 11-07 | same | only the reference view changed (12 choices) | `5f99900f09`: 43 failing frames → 2 to 20; AUC@3 2.3-26.1, AUC@30 27.8-83.4 | failure-mode report |
| 11-07 | ScanNet++ 20 scenes | full shuffle (2 seeds) / `saddle_balanced` reference / `use_ray_pose=True` | gauge-invariant AUC 8.5/65.3 → 13.0-14.0/72.6-73.6 / 14.7/70.8 / same frames fail (0.81) | failure-mode report |
| 11-07 | failing frames | run with only their 9 true neighbours | median relative rotation 9.2° → 0.84° (images aren't hard); not image quality (AUROC 0.37-0.56), not wrong GT | failure-mode report |
| 11-07 | failing frames | properties | P(next fails \| fails) 0.46 vs 13.6 % base; 4 similar views vs 8; scene size Spearman 0.72; most only ~11° off, 5 % > 90°; DA3 depth confidence low on them (AUROC ~0.90); their depth is fine | failure-mode report |
| 11-08 | SMALL dev, 3 seeds | oracle: gross frames fixed from GT before PBA (`init_fix_gross`) | 35.2 / 89.1 vs 33.7 / 85.0: the upper bound | REPORT5 §4 |
| 11-08 | SMALL, all scenes | relocalisation of low-confidence frames (`reloc`, 12 rotations about the vertical x other cameras' centres, scored by depth + grey consistency) | ScanNet++: moved 48 gross frames, fixed 1, broke 17 good; ETH3D fixed 0 of 14, broke 6; HiRoom broke 6. Dev 33.4. The score favours placing a frame where another camera already is | REPORT5 §4-5 |
| 11-08 | SMALL held-out | after PBA | gross frames 11-12 % → 7-9 % on ScanNet++/ETH3D, B800 a bit better (8.2 → 6.9 %) | REPORT5 §1 |

## Open

Several reference views per scene (a few forward passes), choosing or merging by DA3's confidence (failure-mode
report's suggestion, never run in the pipeline). `scripts/reference_view_sweep.py` exists from the analysis.
Covers DA3-SMALL and ScanNet++ only; larger models fail less ([views](views-scaling.md)).

Related: [da3-failure-modes](da3-failure-modes.md), [evaluation](../evaluation.md) (reference view as a hidden variable).
