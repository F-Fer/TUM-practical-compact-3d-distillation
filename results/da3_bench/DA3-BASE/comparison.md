## DA3-BASE

Ours (difference to paper). F1 and AUC in %, DTU Chamfer distance in mm.

| Metric | hiroom | eth3d | dtu | 7scenes | scannetpp |
|---|---|---|---|---|---|
| Pose AUC@3 | 18.85 (-0.15) | 14.93 (-0.17) | 60.85 (+0.75) | 21.42 (+1.32) | 20.24 (-4.86) |
| Pose AUC@30 | 83.25 (+0.05) | 74.55 (-0.05) | 95.92 (+0.02) | 82.44 (-0.46) | 80.82 (-2.58) |
| Recon w/o GT pose | 23.36 (-2.54) | 48.14 (-1.36) | 2.91 (+0.04) | 49.68 (-0.22) | 47.71 (+0.51) |
| Recon w/ GT pose | 72.12 (+0.72) | 66.32 (-0.38) | 1.96 (-0.40) | 49.81 (-0.79) | 67.32 (-0.48) |

285 inference calls, forward total 6.3 min, max peak memory 7.02 GiB.
