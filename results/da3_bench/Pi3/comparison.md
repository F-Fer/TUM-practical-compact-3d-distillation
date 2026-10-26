## Pi3

Ours (difference to paper). F1 and AUC in %, DTU Chamfer distance in mm.

| Metric | hiroom | eth3d | dtu | 7scenes | scannetpp |
|---|---|---|---|---|---|
| Pose AUC@3 | 65.10 (-1.90) | 32.22 (-2.98) | 62.98 (+0.48) | 25.89 (+0.39) | 55.00 (+4.30) |
| Pose AUC@30 | 94.50 (-0.30) | 85.54 (-1.76) | 94.90 (-0.00) | 86.10 (-0.20) | 93.76 (+1.66) |
| Recon w/o GT pose | 73.09 (-2.71) | 69.47 (-3.23) | 2.93 (-0.35) | 41.90 (-2.30) | 63.09 (-0.01) |
| Recon w/ GT pose | 83.63 (-1.37) | 78.18 (-2.42) | 1.70 (-0.02) | 58.26 (+0.76) | 76.76 (+3.46) |

206 inference calls, forward total 39.1 min, max peak memory 7.47 GiB.
