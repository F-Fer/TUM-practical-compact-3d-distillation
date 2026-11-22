# A learned refinement step on DA3's explicit state (L-PBA)

Status: **open (proposal, nothing run)** · Area: TTO method · Updated 2026-11-22

**Answer:** none yet. Proposal: a small network replaces PBA's Adam levels. It reads the residuals, gradients and
Gauss-Newton blocks `run_tto_pba.py` already computes, plus DA3's prior. It outputs either per-frame LM weights (V1)
or a direct update of poses, focal and depth grid (V2), followed by LM polish. Motivation, literature (DeepBA,
Refin3R, PosePlaner, Solv3R, BA-T, Diff3R, 3DGS learned optimisers) and the experiment plan with gates G0-G4 are in
`reports/Learned refinement step between DA3 and DeepBA.md`.

## Evidence

| date | setting (model, set, seeds, baseline) | change | result | source |
|---|---|---|---|---|

## Open / don't repeat

- First run G0 (oracle per-frame prior weight; oracle removal of the weak-direction error), before any training.
- Related negative results to keep in mind: correspondences inside PBA lowered AUC
  ([correspondences](correspondences-in-pba.md)); DA3 tokens as features were neutral ([dino](dino-features.md));
  TTO-refined targets didn't help DéjàView ([tto-teacher](dejaview-tto-teacher.md)).

Related: [pose-prior-and-flat-directions](pose-prior-and-flat-directions.md), [depth-prior](depth-prior.md),
[objective-bias-gt-start](objective-bias-gt-start.md), [matched-compute](matched-compute.md)
