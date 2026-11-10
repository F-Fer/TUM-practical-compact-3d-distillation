# Brightness model for per-pixel photometric losses

Status: **adopted: `bright_rel=true`** · Area: PBA · Updated 2026-11-10

**Answer:** the original per-frame affine brightness `r = I_j e^{a_j} + b_j − I_i e^{a_i} − b_i` (only frame 0 fixed)
has a degenerate minimum: the loss goes to 0 when every other frame's contrast goes to 0, and Adam finds it. This was
the main reason quality fell with T in every photometric run before 2026-11-07. The relative form
`r = I_j e^{a_j − a_i} + b_j − b_i − I_i` (DSO-style) fixes it at no cost. NCC and VGG configurations have no brightness
parameters (the code turns them off for those), which is part of why they looked better on DTU before.

## Evidence (SMALL, dev T = 800, 3 seeds unless noted; REPORT4 §1-3)

| change | AUC@3 | note |
|---|---|---|
| brightness as before (`rec_it800`) | 23.3 | mean log gain −2.6 to −4.4 at T = 800 (DTU contrast shrinks 78x), −0.3 to −0.8 already at T = 50 |
| learning rates of poses, focal, depth / 4 | 23.2 | Adam noise on the poses isn't the driver |
| `bright=false` (no brightness) | 34.4 | |
| all learning rates / 4 incl. brightness | 26.5 | |
| only `lr_bright` / 4 | 27.5 | |
| `lr_bright=1e-3` | 33.6 | |
| no brightness, other lr / 4 | 33.3 | |
| **`bright_rel=true`** | **39.0** | mean log gain −0.01 to +0.11 |
| relative, all lr / 4 | 36.0 | |

- Held-out (59): relative T = 50 39.2 / 85.0 (0.39 s) vs REPORT3 Recommended 37.1 / 84.4: **+2.1 at the same cost**;
  T = 800 41.3.
- F1 (dev, 4 sets): relative T = 50 48.1 (was 46.1), T = 800 46.1 (was 41.5).
- Earlier (11-02 loss sweep, before the fix, T = 50 + LM 5): without the brightness model pose −0.8.
- Implemented in the analytic gradient (`pba_analytic.Kernel.photo_res`, `gradient`; checked against central
  differences, ratio 1.00000) and autograd `PBA.residuals`; the LM levels keep brightness fixed, and `lm_residuals` of the
  jvp path still has the old form, so `bright_rel` needs `impl=analytic lm_impl=analytic`.

## Consequences

- REPORT3's "more Adam only makes it worse" and "Adam follows the same biased objective" were mostly this collapse
  ([iterations-over-t](iterations-over-t.md)). Sessions 1-3 photometric runs, `fast_it<T>`, the old polish and REPORT3 §11
  all had it to some degree.
- Open (REPORT4 §11): recheck REPORT3 conclusions tuned under the collapse: the stop rule's value at T = 50,
  `strat_cells`, the gate threshold.

Related: [loss](loss-kernel-and-colour.md), [ncc](ncc.md), [objective-bias](objective-bias-gt-start.md).
