# Stopping LM on held-out pixels

Status: **adopted** for the Huber LM level (`lm_stop=2e-3`, needs `lm_samples=4096`) · **open:** a stop rule for the
NCC LM (runs a fixed 40 iterations) · Area: PBA optimiser · Updated 2026-11-10

`lm_stop=2e-3` stops the LM level when the cost on 2,048 fixed held-out pixels per frame falls by less than 0.2 % in an
iteration (one extra cost evaluation per iteration). At 1,024 samples the held-out cost is too noisy.

**Answer:** the true gain of LM ends early and differs per scene; the same-sample cost never stops falling (~1/M per
iteration from noise). The rule stops after 2.5 iterations on 7-Scenes and 10.7 on ScanNet++ (SMALL), earlier on better
inits (ScanNet++ 11, 9, 3, 3 iterations from SMALL to GIANT); from T = 400 it usually stops after ~1 iteration.

## Evidence

| date | setting | change | result | source |
|---|---|---|---|---|
| 11-05 | SMALL dev, 3 seeds | Adam 4x25 / 4x50 + stop | 30.1 (0.25 s) / 30.6 (0.33 s) | REPORT3 §3 |
| 11-05 | SMALL dev | threshold 1e-3 / 2e-3 / 5e-3 | 30.7 / 30.6 / 30.9 (flat) | REPORT3 §3 |
| 11-06 | SMALL held-out | Adam 4x50 + stop vs original hybrid | 34.6 vs 34.0; dev gain was +1.8, held-out +0.6, mostly HiRoom (+4.1); dev's 2 7-Scenes scenes +2.3, the other five −0.6 | REPORT3 §6 |
| 11-05 | SMALL dev | LM 20 without stop, stratified + gated radial | 30.0 / F1 49.5 (best dev F1 then) vs LM 5 29.4: long LM only pays with stratified sampling | REPORT3 §3 |
| 11-07 | SMALL dev | NCC LM length (no stop) | keeps rising to 160 iterations on ETH3D (47.5 → 63.5), not on 7-Scenes | REPORT4 §7 |

## Open

Retest the stop rule's value under relative brightness and the shared focal (REPORT4 §11); a stop rule for the NCC LM.

Related: [optimizer](optimizer-adam-lm-hybrid.md), [pixel-sampling](pixel-sampling.md), [ncc](ncc.md).
