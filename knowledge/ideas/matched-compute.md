# Matched compute: small model + TTO vs larger model

Status: **open, answer changing** · Area: project question · Updated 2026-11-10

**Current answer (dev, one seed, RTX 5090, figures5):** with the photometric BA of sessions 3-5, a model plus cheap
TTO now beats the next larger model alone at less time, for SMALL→BASE and BASE→LARGE, and on pose also LARGE→GIANT.
SMALL + photometric T = 25 (0.71 s) 43.8 AUC@3 / 46.8 F1 vs BASE alone (0.85 s) 27.1 / 44.7; BASE + photometric
(1.11 s) 59.2 / 62.8 vs LARGE alone (1.60 s) 48.6 / 59.9; LARGE + photometric (1.83 s) 71.0 / 70.4 vs GIANT alone
(3.10 s) 67.8 / 73.5 (pose wins, F1 doesn't). Held-out (59 scenes) confirms the first two steps (below). DéjàView with
K = 12 loop steps (0.96 s) is a strong competitor that hasn't had TTO yet ([dejaview-steps](dejaview-steps.md)).

## How the answer moved

| date | TTO used | answer | source |
|---|---|---|---|
| 10-30 | PBA v3 (VGG+NCC, Adam 4x200, 18 s) | larger model wins at matched time: LARGE alone (1.6 s) above BASE+PBA (2-4 s) and SMALL+PBA (17 s); best point LARGE + fast50 (~2.7 s) 48.5 → 56.6 / F1 65.7 → 68.1 | REPORT1 |
| 10-31 | hybrid Adam + LM (1.6-1.8 s) | still larger wins, margin shrank 3-10x: LARGE + RGB hybrid 55.8/70.2 > BASE + hybrid 43.5/61.7; BASE stays below LARGE alone | REPORT2 §7 |
| 11-01 | (views, no TTO) | more input views trade worse than model size: LARGE at 16 views (0.44 s) beats BASE and SMALL at 100 views | [views-scaling](views-scaling.md) |
| 11-02 | GIANT-1.1 added (13 configurations, pre-radial) | no PBA configuration helped GIANT meaningfully (best Adam 4x50 RGB 67.9/73.8 vs 67.8/73.5 alone); every LM configuration collapsed ETH3D (52.8 → 21-28), later explained by the lens ([radial](radial-lens-model.md)); GIANT alone beat every LARGE + PBA point | memory note, `results/pba/figures2/` |
| 11-06 | Recommended (analytic, stop rule, gated radial, 0.36 s) | SMALL + TTO (0.77 s) beats BASE alone (0.85 s) on all columns; BASE + TTO (1.19 s) ≈ LARGE alone (1.60 s); LARGE + TTO below GIANT alone. Most of the gain was 3 ETH3D scenes | REPORT3 §11 |
| 11-08 | B′ = + relative brightness + shared focal | held-out: SMALL + B′ 44.0 >> BASE alone 27.0; **BASE + B′ 57.4 > LARGE alone 52.0** at 3/4 of its time; with NCC each size reaches the next size up | REPORT5 §7 |
| 11-09/10 | photometric B′, VGG+NCC, VGG+Huber, photometric + LM on NCC; all 4 models, T = 25-800 | current answer above; LARGE + TTO beats GIANT alone on pose, not F1; the photometric line is flat in T | compute plot `results/pba/figures5/compute_all_models.png` |

## The current compute plot (figures5, dev, one seed, snapshot `f1`)

AUC@3 / F1 (4 sets) at total time per scene (DA3 forward + TTO). Families: `pho` = photometric B′ (grey Huber,
1,024 px, relative brightness); `fmh` = VGG + Huber, 1,024 px; `fm` = VGG + NCC, 4,096 px; `ncc` = `pho` followed by
40 LM iterations on NCC (not drawn on the plot, runs exist). All with shared focal, 8-cell grid, gated radial.

| model | DA3 alone | pho T=25 | pho T=800 | fmh T=50 | fm T=25 | fm T=800 | ncc T=200 | ncc T=800 |
|---|---|---|---|---|---|---|---|---|
| SMALL | 16.2/31.5, 0.40 s | 43.8/46.8, 0.71 | 46.8/50.8, 4.21 | 40.8/50.4, 1.16 | 45.4/49.1, 1.47 | 50.6/51.3, 16.1 | 50.7/50.4, 2.25 | **55.6/53.1**, 5.11 |
| BASE | 27.1/44.7, 0.85 | 59.2/62.8, 1.11 | 60.8/62.8, 4.62 | 58.7/60.9, 1.48 | 59.7/61.4, 1.76 | 63.2/62.1, 16.5 | 61.4/62.3, 2.69 | **64.8/63.0**, 5.60 |
| LARGE-1.1 | 48.6/59.9, 1.60 | 71.0/70.4, 1.83 | 71.3/70.9, 5.38 | 69.8/70.1, 2.19 | 72.3/68.2, 2.48 | **74.1**/69.0, 17.2 | 72.9/66.9, 3.44 | 71.9/66.9, 6.27 |
| GIANT-1.1 | 67.8/73.5, 3.10 | 74.9/76.1, 3.32 | 73.3/74.1, 6.90 | 75.4/**77.0**, 3.70 | **77.2**/74.7, 3.96 | 75.3/71.2, 18.8 | 74.0/70.1, 4.95 | 74.8/68.0, 7.75 |

All T values: [runs.csv](../runs.csv) (`pho_it*`, `fmh_it*`, `fm_it*`, `ncc_it*`; `ncc` has T = 50, 200, 800 only).

- The photometric line is nearly flat in T: T = 25 is enough (relative brightness + shared focal removed the decline
  of earlier curves, [iterations-over-t](iterations-over-t.md)).
- LM on NCC gives the best pose for SMALL and BASE (55.6, 64.8 at ~5 s) but lowers F1 on LARGE and GIANT (66-70 vs
  70-77).
- RoMa tracks + BA (Antonin, 12-37 s) are now below the photometric BA on every model on dev pose; they predate the
  shared focal ([track-ba](track-ba.md)).

## Caveats and open points

- Dev only for the plot; held-out exists for SMALL/BASE/LARGE at T = 50 and the NCC configuration (REPORT5), none for
  GIANT and none for the VGG configurations with the shared focal.
- ETH3D (3 dev scenes) carries a large part of every TTO gain; 7-Scenes never moves (GT problem, [evaluation](../evaluation.md)).
- Image decoding (~1.7 s) isn't counted; it would dominate the cheapest points.
- Not yet on the plot: DéjàView with fewer steps + TTO (Antonin's jobs), number of views as a third knob.
