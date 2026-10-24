## DA3-SMALL

Ours (difference to paper). F1 and AUC in %, DTU Chamfer distance in mm.

| Metric | hiroom | eth3d | dtu | 7scenes | scannetpp |
|---|---|---|---|---|---|
| Pose AUC@3 | 9.53 (+0.04) | 8.57 (-0.02) | 30.65 (+0.05) | 14.53 (+0.53) | 8.95 (-1.95) |
| Pose AUC@30 | 75.28 (+0.08) | 62.15 (+0.05) | 91.15 (-0.05) | 78.22 (-0.48) | 68.33 (-3.57) |
| Recon w/o GT pose | 18.47 (+0.17) | 38.98 (-2.62) | 5.76 (-0.07) | 38.99 (-2.01) | 32.89 (+0.59) |
| Recon w/ GT pose | 51.77 (-0.43) | 63.42 (+0.02) | 2.49 (+0.00) | 46.67 (-0.13) | 53.24 (-0.56) |

386 inference calls, forward total 3.2 min, max peak memory 3.87 GiB.
