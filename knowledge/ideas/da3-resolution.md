# Released DA3 models at a lower input resolution (training-free, RoPE stretched)

Status: **measured** · Area: feed-forward compute · Owner: Finn (+ Claude) · Updated 2026-11-20

**Answer so far:**
- DA3's backbone uses the same VGGT-style integer RoPE as DéjàView, but only in frame attention (global attention
  gets `pos_nodiff`, no relative position). Stretching the frame-attention positions to the native 504-px grid
  (`scripts/distill_steps/da3_res.py`, `DA3_ROPE=1`; bit-identical to upstream at scale 1) is what makes low
  resolution usable **after TTO**.
- **Naive downscaling barely hurts the pose alone but breaks the TTO:** e.g. BASE 420 px 26.8 alone (504 px: 27.3),
  but 26.7 after TTO (504 px: 57.5); LARGE 420 px 43.2 → 36.2 (TTO lowers it). The TTO converges to a
  self-consistent wrong solution (geo residual drops as much as with RoPE; DTU 50 → 22), and the focal is fine.
  Likely the wrong-scale positions distort the per-frame depth shape and the TTO bends the poses to fit it
  (hypothesis, untested).
- **With RoPE, 420 px is free:** after TTO SMALL 43.8 (504: 43.5), BASE **59.0 (57.5)**, LARGE 69.2 (69.7), at
  ~60 % of the forward time. 336 px costs 3-4 points, 252 px 15-17.
- **Across models at matched time:** BASE 336 px + RoPE + TTO (55.8 at 0.50 s fwd) beats SMALL 504 + TTO (43.5 at
  0.60 s); LARGE 336 px + RoPE (66.6 at 1.74 s) beats BASE 504 (57.5 at 1.36 s) by 9 points; LARGE 252 px (55.2 at
  0.96 s) is below BASE 420 (59.0 at 0.82 s). Dropping resolution of the larger model beats a smaller model at
  native resolution down to ~336 px.
- Unlike DéjàView (420 px naive: 61 → 27 alone), DA3 degrades smoothly without RoPE alone; DéjàView's loop applies
  RoPE in every step and all its attention.

## Setup

- `da3_res.py` wraps `DepthAnything3.forward`: the 504-px batch is resized (antialiased) to longest side RES
  (multiple of 14), depth / conf are upsampled back to 504 px, the intrinsics scaled multiplicatively (DA3's
  principal point is exactly W/2; the half-pixel formula of `eval_student.py` would put it 0.25 px off centre).
  `da3_res_check.py --check` (RoPE equivalence) and the 100-view profile; `da3_res_compare.py` makes the tables.
- Queue: `workspace/da3_res/run.sh` (tum1, RTX 3080, 2026-11-20). Pose on all 81 scenes, TTO = `run_tto_pba.py`
  iters = 25 (defaults, `save=false`). fwd s = median per scene incl. resize / upsampling. SMALL / BASE with
  `DA3_LOWMEM=0` (new switch in `run_bench.py`: the offload costs ~40 % of SMALL's forward), LARGE with `da3_lowmem`
  (504 px OOMs without it), so LARGE times are ~12-30 % pessimistic vs SMALL / BASE.
- Metrics and logs: `workspace/evaluation/da3_res/`, `workspace/tto/da3_res/`, `workspace/da3_res/` (predictions
  deleted for disk space; the disk was full during the run).

## Evidence

### Forward pass at 100 views (7-Scenes chess, RTX 3080, bf16, median of 3)

| model | input | tokens / frame | fwd s | peak GiB | fwd s (lowmem) | peak GiB (lowmem) |
|---|---|---|---|---|---|---|
| DA3-SMALL | 504x378 | 972 | 1.69 | 3.6 | 2.34 | 1.6 |
| DA3-SMALL | 420x308 | 660 | 0.95 | 2.7 | 1.41 | 1.3 |
| DA3-SMALL | 336x252 | 432 | 0.53 | 1.9 | 0.84 | 1.0 |
| DA3-SMALL | 252x196 | 252 | 0.27 | 1.3 | 0.47 | 0.8 |
| DA3-BASE | 504x378 | 972 | 3.74 | 6.9 | 5.01 | 3.3 |
| DA3-BASE | 420x308 | 660 | 2.14 | 5.1 | 3.01 | 2.6 |
| DA3-BASE | 336x252 | 432 | 1.21 | 3.6 | 1.79 | 2.0 |
| DA3-BASE | 252x196 | 252 | 0.62 | 2.5 | 0.97 | 1.5 |
| DA3-LARGE-1.1 | 504x378 | 972 | OOM | OOM | 11.27 | 5.5 |
| DA3-LARGE-1.1 | 420x308 | 660 | 5.74 | 8.4 | 6.54 | 4.6 |
| DA3-LARGE-1.1 | 336x252 | 432 | 3.24 | 6.3 | 3.76 | 3.8 |
| DA3-LARGE-1.1 | 252x196 | 252 | 1.65 | 4.6 | 1.96 | 3.2 |

### Pose AUC@3, all 81 pose scenes (fwd s = median per scene on the 3080; TTO = run_tto_pba.py T = 25)

| model | input | fwd s | AUC@3 alone | HiRoom / 7Sc / SN++ / DTU / ETH3D | + TTO (all) | dev: alone → TTO | held-out: alone → TTO |
|---|---|---|---|---|---|---|---|
| DA3-SMALL | 504 px (released) | 0.60 | 14.4 | 9.5 / 14.5 / 9.0 / 30.7 / 8.6 | **43.5** | 16.2 → 41.2 | 13.6 → 44.1 |
| DA3-SMALL | 420 px naive | 0.36 | 13.3 | 14.6 / 13.2 / 6.2 / 20.9 / 11.8 | **32.8** | 13.9 → 29.9 | 13.1 → 33.9 |
| DA3-SMALL | 420 px + RoPE | 0.36 | 14.3 | 14.5 / 12.7 / 8.2 / 24.4 / 12.0 | **43.8** | 15.5 → 44.2 | 13.7 → 43.3 |
| DA3-SMALL | 336 px naive | 0.22 | 10.9 | 14.3 / 10.5 / 5.4 / 18.7 / 5.5 | **39.0** | 10.8 → 37.5 | 10.9 → 39.5 |
| DA3-SMALL | 336 px + RoPE | 0.22 | 11.3 | 12.3 / 12.0 / 5.6 / 18.9 / 7.6 | **40.0** | 13.7 → 39.7 | 10.1 → 39.7 |
| DA3-SMALL | 252 px naive | 0.12 | 4.9 | 5.5 / 5.0 / 1.5 / 11.3 / 1.1 | **19.2** | 5.5 → 17.9 | 4.7 → 19.2 |
| DA3-SMALL | 252 px + RoPE | 0.12 | 4.1 | 4.7 / 6.9 / 1.5 / 5.3 / 1.9 | **26.9** | 5.4 → 28.9 | 3.5 → 25.8 |
| DA3-BASE | 504 px (released) | 1.36 | 27.3 | 18.9 / 21.4 / 20.2 / 60.9 / 14.9 | **57.5** | 27.1 → 59.5 | 27.0 → 56.5 |
| DA3-BASE | 420 px naive | 0.82 | 26.8 | 31.0 / 19.3 / 11.6 / 50.2 / 21.9 | **26.7** | 22.3 → 24.9 | 28.4 → 27.3 |
| DA3-BASE | 420 px + RoPE | 0.82 | 29.5 | 32.4 / 20.0 / 18.8 / 53.9 / 22.4 | **59.0** | 26.4 → 58.6 | 30.6 → 59.0 |
| DA3-BASE | 336 px naive | 0.50 | 22.7 | 30.2 / 15.8 / 8.8 / 47.9 / 10.7 | **34.6** | 19.2 → 32.6 | 24.1 → 35.4 |
| DA3-BASE | 336 px + RoPE | 0.50 | 24.8 | 29.2 / 18.0 / 14.6 / 46.4 / 15.9 | **55.8** | 22.5 → 57.7 | 25.8 → 54.8 |
| DA3-BASE | 252 px naive | 0.27 | 11.2 | 15.1 / 10.1 / 5.9 / 23.1 / 1.7 | **21.3** | 10.0 → 20.3 | 11.6 → 21.1 |
| DA3-BASE | 252 px + RoPE | 0.27 | 12.5 | 13.3 / 13.1 / 5.6 / 23.9 / 6.4 | **40.0** | 12.1 → 38.7 | 12.6 → 40.2 |
| DA3-LARGE-1.1 | 504 px (released) | 4.53 | 51.4 | 59.9 / 27.6 / 56.6 / 77.9 / 34.8 | **69.7** | 48.8 → 70.2 | 52.0 → 69.4 |
| DA3-LARGE-1.1 | 420 px naive | 2.83 | 43.2 | 61.0 / 28.6 / 21.6 / 71.7 / 33.1 | **36.2** | 42.4 → 35.2 | 43.1 → 36.5 |
| DA3-LARGE-1.1 | 420 px + RoPE | 2.82 | 50.7 | 61.7 / 25.7 / 54.4 / 74.7 / 36.9 | **69.2** | 48.4 → 70.9 | 51.2 → 68.4 |
| DA3-LARGE-1.1 | 336 px naive | 1.74 | 43.3 | 55.3 / 26.9 / 39.6 / 74.1 / 20.4 | **34.8** | 41.7 → 35.3 | 43.2 → 34.9 |
| DA3-LARGE-1.1 | 336 px + RoPE | 1.74 | 43.1 | 52.9 / 24.9 / 47.7 / 57.4 / 32.9 | **66.6** | 39.2 → 66.0 | 44.4 → 66.7 |
| DA3-LARGE-1.1 | 252 px naive | 0.89 | 26.7 | 39.2 / 19.7 / 22.9 / 48.9 / 3.1 | **30.9** | 25.7 → 31.7 | 27.0 → 30.5 |
| DA3-LARGE-1.1 | 252 px + RoPE | 0.96 | 26.9 | 34.2 / 18.5 / 25.9 / 38.4 / 17.5 | **55.2** | 24.3 → 57.2 | 27.6 → 54.2 |

The 504-px rows match earlier runs (held-out SMALL 13.6, BASE 27.0, LARGE 52.0 alone).

## Open / next

- Resolution distillation for DA3 (as for DéjàView R2), mainly for 336 / 252 px of BASE and LARGE.
- Test the depth-distortion hypothesis for the naive case (TTO from naive depth with RoPE poses, or the reverse).
- Re-time LARGE without the offload where it fits (420 px and below) for a fair matched-time plot; add these points
  to the compute plot ([matched-compute](matched-compute.md)).

Related: [dejaview-speed](dejaview-speed.md), [matched-compute](matched-compute.md)
