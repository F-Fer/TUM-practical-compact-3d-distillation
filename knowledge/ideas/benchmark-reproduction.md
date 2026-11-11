# DA3 benchmark reproduction and baselines

Status: **done** (within ±1.5 pose / ±3 F1 of the paper, deviations explained or listed) · Area: benchmark · Updated 2026-11-10

Numbers per model next to the paper's: `results/da3_bench/<model>/comparison.md` (tracked); metric JSONs stay local in
the same folder. README "Our results" has the discussion.

## Findings

- SMALL, BASE, LARGE-1.1 full benchmark (RTX 3080, 2026-10): most metrics within ±1.5 (pose) / ±3 F1 (recon); larger
  is better everywhere except 7-Scenes recon with GT pose (BASE 49.8 > LARGE-1.1 48.2).
- **ScanNet++ pose is lower for all three models** (SMALL AUC@30 −3.6, BASE AUC@3 −4.9 / AUC@30 −2.6, LARGE-1.1
  AUC@3 −3.6): data or evaluation, not a checkpoint. Not investigated.
- **LARGE-1.1 DTU pose AUC@3 +7.7**: the paper used the deprecated `DA3-LARGE` (DTU-64 70.16 vs paper 70.20;
  LARGE-1.1 77.89). LARGE-1.1 also ETH3D pose +2.6, 7-Scenes recon unposed −5.9 F1 (not investigated).
- Even the DA3 repo's own expected GIANT results differ from the paper by up to ~3 F1 (recalibrated ScanNet++ etc.).
- tum2 (RTX 5090, torch 2.7) matches tum1 within 0.25 on HiRoom; xformers SwiGLU matches plain PyTorch.
- `da3_lowmem` (CLAUDE.md) is bit-identical and halves peak memory; `open3d` must stay 0.19.0.
- Baselines (`scripts/baselines.py`): VGGT-1B, π³, DéjàView (K = 12). With DA3's preprocessing
  (`--da3-preprocess`) VGGT/π³ match the paper's Table 2 within 0.6 (VGGT DTU AUC@30 99.8 in the paper is the 49-view
  value); with native preprocessing off by up to 3. bf16 weights change depth ~0.1 %.
- DéjàView K = 12 on the full benchmark (tum2): AUC@3 58.7, F1 (4 sets) 67.7, 0.96 s per forward
  ([dejaview-steps](dejaview-steps.md)).

## Open

ScanNet++ pose gap and 7-Scenes recon gap of the paper; DA3-LARGE (deprecated) on the other datasets.

Related: [evaluation](../evaluation.md), [views-scaling](views-scaling.md).
