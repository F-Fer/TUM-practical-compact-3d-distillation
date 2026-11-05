## dvlt

Ours (not in the paper). F1 and AUC in %, DTU Chamfer distance in mm.

| Metric | hiroom | eth3d | dtu | 7scenes | scannetpp |
|---|---|---|---|---|---|
| Pose AUC@3 | 76.04 | 36.05 | 79.51 | 29.77 | 71.99 |
| Pose AUC@30 | 96.91 | 90.06 | 97.91 | 88.08 | 96.85 |
| Recon w/o GT pose | 76.47 | 69.23 | 2.47 | 53.20 | 70.76 |
| Recon w/ GT pose | 80.95 | 75.20 | 1.86 | 59.32 | 76.46 |

206 inference calls, forward total 13.3 min, max peak memory 3.92 GiB.
