# Patch BA: align patches once, then solve the geometry without the images

Status: **rejected** (settles, but at a worse fixed point; without Adam it fails) · Area: PBA alternatives · Updated 2026-11-10

The proposal (tested in session 4 with four experiments): per alignment, each of 200 patches per frame is warped into
each neighbour and aligned by Lucas-Kanade alone (3 iterations, zero-mean, gain); then LM on the aligned
correspondences with a free depth per patch (exact Schur), no image lookups. Code (removed in the 2026-11-10 refactor; the code is at commit `1c1a921`): `scripts/pba_patch.py`
(`patch=<sigma:alignments,...>`), checked against central differences and a dense solve (1e-13).

## Evidence (SMALL dev; REPORT4 §5-6)

| change | AUC@3 | note |
|---|---|---|
| Adam 4x50 + LM on Huber, stop rule (relative brightness, 3 seeds) | 34.0 | reference |
| Adam 4x50 + patch stage, 8 alignments x 10 LM | 27.7 | LM settles (cost changes < 1e-5 after 7 iterations), curve doesn't collapse, but the fixed point is worse |
| Adam 4x800 + patch stage | 38.8 | +0.8 over Adam 4x800 alone (38.0), −0.2 vs LM on Huber (39.0) |
| sweep of 10 settings (old brightness, base 27.3) | best 30.0 (no geometric row) | 800 patches 27.3, 5x5 patches 28.5, `patch_info=shape` 29.3, geo weight 0.1 29.5, depth prior 2 / 200 27.3 / 27.0, NCC threshold 0.8 28.4, 1 LK iteration 26.2, blur 2 then 1 27.9 |
| no Adam: patch stage at 4 blur levels | 17.4-23.1 | HiRoom below DA3 alone (10.6 → 5.4); LK on blurred patches doesn't reach SMALL's initial errors, and the radial gate never sees a moved k |
| no Adam: LM on NCC at 4 blur levels, grid / free depth | 29.5 / 23.1 | |

**Reading (untested):** a per-warp alignment is a hard decision from 9 pixels; its errors become constraints, whereas LM
on the images re-decides with all patches together. The patch stage costs about another Adam 4x50 (~0.3 s).

## If revisited

Softer alignment (several hypotheses per warp, or weights from the NCC over LK's basin). The direct counterpart, LM on
NCC with image lookups, works much better ([ncc](ncc.md)).

Related: [depth-parametrisation](depth-parametrisation.md) (free depth), [optimizer](optimizer-adam-lm-hybrid.md).
