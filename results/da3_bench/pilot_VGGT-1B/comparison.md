## pilot_VGGT-1B

Ours (difference to paper). F1 and AUC in %, DTU Chamfer distance in mm.

| Metric | hiroom | eth3d | dtu | 7scenes | scannetpp |
|---|---|---|---|---|---|
| Pose AUC@3 | 47.79 (-1.31) | – | – | – | – |
| Pose AUC@30 | 87.87 (-0.13) | – | – | – | – |
| Recon w/o GT pose | 58.35 (+1.65) | – | – | – | – |
| Recon w/ GT pose | 70.11 (-0.09) | – | – | – | – |

60 inference calls, forward total 2.3 min, max peak memory 5.61 GiB.
