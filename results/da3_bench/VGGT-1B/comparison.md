## VGGT-1B

Ours (difference to paper). F1 and AUC in %, DTU Chamfer distance in mm.

| Metric | hiroom | eth3d | dtu | 7scenes | scannetpp |
|---|---|---|---|---|---|
| Pose AUC@3 | 47.79 (-1.31) | 29.00 (+2.70) | 79.65 (+0.45) | 23.20 (-0.70) | 59.30 (-3.30) |
| Pose AUC@30 | 87.87 (-0.13) | 81.48 (+0.68) | 85.26 (-14.54) | 84.75 (-0.25) | 94.33 (-0.77) |
| Recon w/o GT pose | 58.36 (+1.66) | 61.93 (+4.73) | 1.94 (-0.11) | 47.94 (+0.04) | 67.40 (+1.00) |
| Recon w/ GT pose | 70.10 (-0.10) | 70.97 (+4.27) | 1.21 (-0.23) | 50.74 (-0.66) | 69.34 (-1.36) |

206 inference calls, forward total 40.1 min, max peak memory 7.13 GiB.
