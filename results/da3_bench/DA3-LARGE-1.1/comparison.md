## DA3-LARGE-1.1

Ours (difference to paper). F1 and AUC in %, DTU Chamfer distance in mm.

| Metric | hiroom | eth3d | dtu | 7scenes | scannetpp |
|---|---|---|---|---|---|
| Pose AUC@3 | 59.89 (+1.19) | 34.82 (+2.62) | 77.89 (+7.69) | 27.64 (-1.56) | 56.57 (-3.63) |
| Pose AUC@30 | 94.40 (+0.20) | 87.69 (+0.79) | 97.63 (+0.93) | 86.03 (-0.57) | 94.06 (-0.64) |
| Recon w/o GT pose | 69.10 (-0.40) | 68.68 (+2.88) | 1.84 (-0.24) | 50.44 (-5.86) | 68.46 (+0.56) |
| Recon w/ GT pose | 87.48 (+0.38) | 76.46 (+1.26) | 1.54 (+0.31) | 48.21 (-0.99) | 76.18 (+0.48) |

206 inference calls, forward total 18.5 min, max peak memory 6.00 GiB.
