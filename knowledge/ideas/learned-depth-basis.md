# Fine-tune DA3 for PBA: a learned depth basis / low-frequency correction on frozen DA3-SMALL head features

Status: **rejected at tum1 scale (learned basis, mean head, head fine-tuning with depth / sensitivity / photometric
losses); adopted as an analysis tool (first-order pose sensitivity)** · Area: TTO method / depth · Updated 2026-11-24

**Answer (SMALL, B′ T = 25):** (1) **What PBA needs from the depth is its low-frequency shape**: GT-fitted corrections
rank in pose exactly as their residual after a Gaussian blur of σ = H/16-H/8 does, not as their RMS (a learned
8-map basis fits the GT better than a 4-cell grid in RMS but gives less pose: 30.3 vs 32.6 dev, depth fixed). A
first-order model of PBA's optimum (dp = -Hpp^-1 Hgp^T for an error field, `scripts/lpba_sens.py`) reproduces the
oracle ranking on HiRoom / ETH3D, the near-unbiasedness of the objective at the GT, and with the free grid eliminated
by its Schur complement predicts B′ on HiRoom (44.0 vs 43.5-45.9): the 8-cell grid absorbs the error up to 8 cells,
B′'s remaining bias comes from the 8-16-cell band; finer grids help in reality too (12 cells H 44.7 → 46.3, E 31.2 →
35.1; a smoothness prior on them hurts). (2) **Learning a better depth for PBA doesn't transfer at our data scale**:
a learned basis on frozen DA3 head features (held-out → dev) gives no pose gain fixed or free; a mean-correction head
is worse than no correction; fine-tuning DA3's main depth branch (0.56M, cameras unchanged) with the GT depth loss,
the first-order pose-sensitivity loss or a photometric loss with GT poses overfits in-domain (45 scenes) and on
ARKitScenes (1,000 windows, LiDAR + poses) moves the benchmark depth the wrong way (held-out B′ HiRoom 53.3 → 46.4,
ETH3D 32.5 → 23.6); the photometric loss diverges to degenerate depth. What did help on the model side without
training: test-time augmentation of depth and cameras ([da3-tta](da3-tta.md), held-out +1.7).

## Method

- `scripts/lpba_basisnet.py prep | train | export`: features of the benchmark clips (frozen DA3-SMALL, hooks on the main
  DPT branch, stored at refinenet2's resolution H/3.5; ScanNet++ also 4 extra random 40-frame clips per scene from its
  iPhone frames with rendered depth); loss = ridge LS projection residual of log(GT/DA3) on [1, B] (unit-RMS maps),
  optionally + `lowpass` x the blurred residual (σ = H/16, H/8) and a mean-correction head (`mean=1`). Folds: `ho`
  (train held-out, validate dev), `cv0`/`cv1` (train dev + one half of held-out, validate the other half).
- `scripts/run_tto_pba.py depth_basis=<export>/pred [basis_grid=G]`: PBA with the per-frame log-depth correction
  Σ_k c_k B_k(u, v) (dense weights; the 8-cell grid is the special case of hat functions). Checked against the grid
  path with the 4-cell hat functions as basis: cost, Adam gradient and all LM blocks equal to ~1e-5 relative (Huber
  and NCC).
- `scripts/lpba_devsum.py <runs>`: AUC@3 on these 16 dev scenes.

## Evidence

| date | setting (model, set, seeds, baseline) | change | result | source |
|---|---|---|---|---|
| 11-23 | SMALL, dev 16 (H 10 / S 3 / E 3), seed 0, B′ T = 25 (grid free: 28.7 / 32.8 / 29.2, seeds 0-2) | references on these scenes, depth fixed (`depth_grid=0`) | DA3 19.2, smooth4 32.6, smooth8 37.5, smooth16 43.3, GT 46.3; GT + free grid 40.2 | `lpba_pa_*`, `lpba_dor_*` |
| 11-23 | same; basis net K = 8, 8k steps, trained on held-out | in-sample RMS log err left (dev, H/3.5 res) H / S / E | K8 .0316 / .0674 / .0181; smooth4 .0360 / .0680 / .0235; smooth8 .0281 / .0614 / .0155; K16 .0263 / .0609 / .0130; K4 .0368 / .0715 / .0238 | `bnet/k*.log` |
| 11-23 | same | K8 span, GT-fitted, depth fixed / coefficients free / + 4-cell grid free | 30.3 (40.4 / 21.7 / 28.7) / 25.6 (37.2 / 24.3 / 15.3) / 31.2 (44.9 / 24.7 / 24.1) | `bnet/r_k8_*` |
| 11-23 | same | low-pass residual (σ = H/16) of the GT fits, H / S / E | none .044 / .041 / .030; smooth4 .0148 / .0233 / .0079; K8 .0175 / .0267 / .0083; smooth8 .0049 / .0108 / .0028; smooth16 .0018 / .0036 / .0007: same order as the pose oracles | scratch `lowfreq.py` |
| 11-23 | same | K8 trained with + 4 x low-pass loss | low16 H / S / E .0158 / .0256 / .0055 (K8 .0175 / .0267 / .0083) | `bnet/k8lp.log` |
| 11-23 | same | K8 with 20 x low-pass loss (`k8lp20`): GT-fitted fixed / free / free + 4-cell grid | 30.3 (39.6 / 24.1 / 27.2) / 27.6 / 29.3: no better than K8 | `bnet/r_k8lp20_*` |
| 11-23 | same | mean-correction head on frozen features (`m0lp`, `k8mlp`, 20 x low-pass): DA3 x exp(predicted correction), per-frame scale only | dev RMS H / S / E .063 / .084 / .057 (m0lp), .057 / .079 / .057 (k8mlp) vs DA3 .058 / .078 / .049: **no transferable correction even in-domain** | `bnet/m0lp.log`, `k8mlp.log` |
| 11-23 | same, GT poses / focal / depth as the state | **first-order sensitivity** (`scripts/lpba_sens.py`): dp = -Hpp^-1 Hgp^T for each error field, scored as AUC@3 ("linearised") | GT itself 96.7 / 97.7 / 98.5 (H / S / E): the objective is nearly unbiased at the GT. HiRoom DA3 15.0, K8 29.1, K8lp20 32.1, smooth4 36.3, smooth8 53.8, smooth16 60.7 (real fixed-depth oracles 25.9 / 40.4 / 39.6 / 40.9 / 48.0 / 54.1); ETH3D 28.1 / 54.1 / 48.4 / 52.6 / 58.9 / 72.3 (real 15.5 / 28.7 / 27.2 / 31.3 / 37.8 / 50.1): same ranking; ScanNet++ all < 10 (100-frame videos: first order too sensitive, the real oracles are flat 21-26 too) | `bnet/sens_dev.json` (camera scale matched to DA3's spread; later runs use the GT depth's scale) |
| 11-23 | SMALL, 45 held-out scenes (143 frame subsets of ≤ 16 frames) → dev (36 subsets) | **fine-tune DA3's main depth branch** (refinenet1-4, output_conv1/2, 0.56M; cameras unchanged, checked) with the sensitivity loss (LM-damped, λ = 0.1 diag, log1p-robust) + 0.1 x scale-invariant depth loss, 3k steps, lr 1e-4 (`scripts/lpba_ft_head.py`, `sens1`) | train subsets: median loss H / S / E 0.36 / 0.98 / 0.62 → 0.05 / 0.52 / 0.01; dev 0.40 / 0.62 / 0.10 → 0.45 / 1.12 / 0.21: **overfits, no transfer** at 45 scenes. PBA on dev with its depth (same export path as the unchanged model): depth fixed 19.4 → 13.9 (H / S / E 26.6 / 16.8 / 14.8 → 21.7 / 10.8 / 9.3), B′ HiRoom 43.5 → 32.3, ETH3D 28.2 → 36.0 (one seed; ETH3D B′ seeds 28.2-35.3); depth abs. rel. H 5.37 → 6.41 % | `ft/sens1.log`, `ft/r_sens1_*`, `ft/r_none_*` |

| 11-23 | same, in-domain control | fine-tune with the depth loss only (`depth1`, w_sens 0, w_depth 1) | dev sensitivity loss median H 0.40 → 0.31 (generalises better than the sensitivity loss); PBA (3 seeds, H / E): depth fixed 26.8 → 23.4 / 15.1 → 12.2; B′ 44.7 → 43.8 / 31.2 → 35.7 (ETH3D 3 scenes, seed spread ±3) | `ft/depth1.log`, `ft/r_depth1_*` |
| 11-24 | SMALL, **ARKitScenes** 500 training videos x 2 windows of 24 frames (vga 640 x 480, LiDAR depth conf ≥ 2, GT poses from lowres_wide.traj = world-to-camera, OpenCV; `scripts/lpba_prep_arkit_pose.py`) → benchmark dev + held-out (unseen) | fine-tune the main depth branch, sensitivity loss (`ark_sens`) / depth loss (`ark_depth`), 4k steps | ark_sens: training loss 0.19 → 0.16, benchmark validation mixed (median H dev 0.40 → 0.38, held-out ETH3D 0.61 → 0.50, ScanNet++ worse, depth loss worse). **PBA B′: dev H / E 44.7 / 31.2 → 42.2 / 29.3 (3 seeds); held-out H (20) / E (8) 53.3 / 32.5 → 46.4 / 23.6 (2 seeds)**, depth abs. rel. held-out E 5.4 → 6.7 %. ark_depth: benchmark depth loss flat (dev H 0.0096 → 0.0096): ARKit → benchmark doesn't transfer | `ft/ark_*.log`, `ft/r_arks_*`, `ft/r_ho_*` |
| 11-24 | same ARKit set | **photometric self-supervision** with the GT poses (min-over-4-neighbours L1 at H/2, clip scale from LiDAR) + 0.1 x LiDAR depth loss, lr 5e-5 | sanity: at the GT poses LiDAR depth 0.0755 < DA3 0.0766 < bent DA3 0.0783 (weak signal); training diverges (depth loss 0.009 → 0.017-0.038 by step 500) both with dropped out-of-view pixels (degenerate: warp out of view) and with a constant penalty for them; second run then OOM-killed by a parallel job | `ft/ark_photo*.log` |
| 11-24 | SMALL, HiRoom / ETH3D dev, GT state, basis at H/2 | first-order sensitivity **with a free grid** eliminated by its Schur complement (`lpba_sens.py free=G`), DA3's depth error | HiRoom linearised AUC@3 with free 4 / 8 / 12 cells: 30.2 / 44.0 / 54.4 (real B′ with 8 cells: 43.5-45.9 over 3 seeds); with 8 cells, smooth8-corrected depth 46.5 ≈ DA3 (the grid absorbs it), smooth16 59.0: B′'s pose bias comes from the error between 8 and 16 cells. ETH3D 85.9 vs real ~31: not predictive there (convergence, radial) | `bnet/sens_dev_free8.json`, `sens_free*.json` |

## Open / don't repeat

- Judge depth sources by the low-frequency residual (or the linearised AUC of `lpba_sens.py`), not RMS / abs. rel.
- Don't retry at tum1 scale: learned bases / correction heads on frozen features, head fine-tuning on ≤ 1k clips with
  any of the three losses. The data scale is the problem (as for the refiners and the distillation in
  [depth-bottleneck](depth-bottleneck.md)); the sensitivity loss is also heavy-tailed (a few clips dominate) even
  with LM damping (λ = 0.1 diag) and a log1p cap.
- If scaled (5090, thousands of clips): the sensitivity loss with the free-grid Schur complement (what B′ actually
  sees, the 8-16-cell band), a large teacher's depth/cameras as the pseudo-GT state (no GT needed), and a
  depth-anchored photometric loss with SSIM / auto-masking (the plain L1 version degenerates).
- The first-order model is too optimistic for finer free grids (12 cells 54.4 linearised vs 46.3 real) and doesn't
  predict ETH3D (convergence / radial gate) or ScanNet++ (100-frame videos: everything looks catastrophic).
- Code: `run_tto_pba.py depth_basis=` (dense per-frame basis, checked against the grid path), `w_smooth`,
  `basis_grid`; `lpba_basisnet.py`, `lpba_ft_head.py`, `lpba_sens.py`, `lpba_prep_arkit_pose.py`, `lpba_devsum.py`,
  `lpba_hesum.py`. Data (local): `workspace/tto/DA3-SMALL/lpba/{bnet,ft,arkit}`.

Related: [depth-bottleneck](depth-bottleneck.md), [depth-parametrisation](depth-parametrisation.md),
[shared-depth-field](shared-depth-field.md), [learned-refinement-step](learned-refinement-step.md)
