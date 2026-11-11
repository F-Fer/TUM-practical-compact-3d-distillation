# Radial lens model

Status: **adopted:** one division-model coefficient with a gate (`radial=true radial_max=0 radial_gate=0.01`, analytic
path) · **open:** whether two polynomial coefficients (`camera=radial`, REPORT6) are better in the same configuration ·
Area: PBA, camera model · Updated 2026-11-10

**Answer:** only ETH3D has a real lens (fisheye-like distortion, ~10 px at the corners at 504 px). Modelling it is the
largest ETH3D gain of all changes and explained why LM "collapsed" ETH3D in session 2. A free coefficient hurts
undistorted scenes (it trades off with the focal), hence barrel-only plus a gate that drops it when Adam hasn't moved it.

## Evidence

| date | setting | change | result | source |
|---|---|---|---|---|
| 11-05 | SMALL, LM from GT poses (ETH3D 3 scenes) | + radial coefficient | AUC@3 at LM it. 5/20: 31.5/10.1 → **43.1/40.7** | REPORT3 §3 |
| 11-05 | SMALL subset, Adam 4x50 | coefficient left free | k goes positive on undistorted scenes: HiRoom 32.5 → 24.6, ScanNet++ 17.1 → 5.1, a HiRoom focal error 0.015 → 0.087 | REPORT3 §3 |
| 11-05 | SMALL dev, 3 seeds, CPU | + radial k ≤ 0 / + gate 0.02 | 29.1 → 29.8 / 29.7 (ETH3D 9.0 → 15.8 / 11.9) | REPORT3 §3 |
| 11-05 | ETH3D 11 scenes | gate 0.02 | kept on 7 scenes, k −0.034 to −0.053 (kicker 46.7 → 77.8, facade 12.3 → 26.2, courtyard 8.3 → 23.0); misses electro (k −0.016 at the gate) | REPORT3 §3 |
| 11-05 | SMALL dev, 3 seeds | gate 0.01 vs 0.02 | 32.6 vs 31.6 (catches electro, delivery_area); drops all 30 HiRoom scenes but barely (max \|k\| 0.0088) | REPORT3 §3 |
| 11-06 | SMALL held-out | stop rule → + radial, gate 0.02 / 0.01 | 34.6 → 37.0 / 37.1 (ETH3D 17.8 → 27.3); 0.01 vs 0.02 level on held-out | REPORT3 §6 |
| 11-06 | all 4 models, dev | Recommended vs old polish | ETH3D 9.2 → 21.8, 23.6 → 56.6, 28.7 → 71.6, 23.2 → 76.0; courtyard k −0.043/−0.043/−0.043/−0.041 across models (a lens property) | REPORT3 §11 |
| 11-08 | SMALL dev | gate decided by held-out cost (`radial_gate_heldout`) | 33.8 vs 33.7: same decisions as 0.01 except 3 HiRoom scenes | REPORT5 §4 |
| 11-09 | SMALL / LARGE, all 11 ETH3D, old autograd path, VGG+NCC, no shared focal | `camera=radial` (k1, k2, radius check, no gate) vs pinhole | 20.9 → 48.6 / 46.0 → 68.1 (AUC@30 76.6 → 82.6 / 89.9 → 94.0); k1 −0.064 to −0.121, k2 0.061-0.098; HiRoom/DTU neutral without a gate; +5-12 % time (3070) | REPORT6 |
| 11-09 | same | `camera=radial` without the radius check | HiRoom worse (folded out-of-view points pull k2 negative) | REPORT6 |
| 11-10 | 8 held-out ETH3D scenes (= REPORT6's 11 minus the 3 dev ones) | ours (1 gated coefficient, shared focal) vs REPORT6 (2 coefficients) | SMALL: B′ T = 50 31.0, T = 800 + 40 LM on NCC 51.1, REPORT6 53.4; LARGE: B′ 61.8, NCC 82.8, REPORT6 67.8. Configurations differ, so not a clean test | REPORT5 runs `y_*_h`, `results/pba/runs6/eth3d_all` |
| 11-10 | current runs, dev ETH3D | fitted k | −0.035 to −0.045 on every scene, all models; gate switched it off in 1 of 72 scene fits | runs `pho_it*`, `fmh_it*`, `fm_it*` |

## Caveats

- **DTU-64 keeps a large k that looks wrong**: the gate keeps it on all DTU scenes, k reaches −0.22 (5x ETH3D's) and
  varies by model (−0.004 to −0.08 LARGE, −0.08 to −0.22 BASE): it may absorb another error (DTU calibration unchecked).
  GIANT loses on DTU with it (95.2 alone → 86.9/89.7). The principal point isn't what it absorbs (REPORT5 §4).
- The gate's HiRoom margin is thin (max \|k\| 0.0097 on SMALL vs 0.01), and the threshold fits T = 50 only.
- `results.npz` stays pinhole: ETH3D recon F1 doesn't benefit directly.
- `camera=radial` was only on the old autograd/jvp path (no `bright_rel`, no stop rule, 3-6x slower), removed in the 2026-11-10 refactor; the code is at commit `1c1a921`;
  porting it to the analytic kernels is a few hours of work plus a float64 check.

## Next

Compare 1 vs 2 coefficients in the same configuration on all 11 ETH3D scenes (SMALL, LARGE; VGG + Huber config, which
runs on both paths: analytic `radial=true`, old path pinhole, old path `camera=radial`, the old path from commit
`1c1a921`; ~20-30 GPU min). Only if 2
coefficients clearly win, port them to the analytic path. A per-camera (not per-scene) gate; export the undistortion
for recon.

Related: [focal-and-intrinsics](focal-and-intrinsics.md), [objective-bias](objective-bias-gt-start.md), [ncc](ncc.md).
