#!/usr/bin/env python3
# regenerates gpu.q: benches, then the speed ladder timed on the GPU (no scoring here; pose comes with the run, recon stays from the CPU runs)
import sys
snap=sys.argv[1]
FAST="nbr_mode=spread neighbors=10 lr_rot=3e-3 lr_trans=3e-3 lr_focal=3e-3"; A="impl=analytic lm_impl=analytic"
POL=f"{FAST} samples=1024 iters=50 levels=8,4,2,1,1 lm_levels=4 lm_iters=5 lm_samples=4096"
F=f"{FAST} levels=8,4,2,1,1 lm_levels=4 lm_iters=5 iters=25 samples=512 lm_samples=1024 {A}"
pre=f"cd ~/TUM-practical-compact-3d-distillation && .venv/bin/python $J/snap/{snap}/bench_pba.py"
L=[f"{pre} hiroom 20241230/828744/cam_sampled_04 time > $J/runlogs/gpu_bench_hir.log 2>&1", f"{pre} eth3d courtyard time > $J/runlogs/gpu_bench_eth.log 2>&1", f"{pre} scannetpp 09c1414f1b time > $J/runlogs/gpu_bench_sn.log 2>&1",f"{pre} eth3d courtyard time features=vgg photo_loss=ncc samples=4096 > $J/runlogs/gpu_bench_ncc.log 2>&1"]
for name,subset,ov in [("pm_G_I0","dev",POL),("pm_G_I1","dev",f"{POL} impl=analytic"),("pm_G_I2","dev",f"{POL} {A}"),("pm_G_T25","dev",f"{POL} {A} iters=25"),("pm_G_L1024","dev",f"{POL} {A} iters=25 lm_samples=1024"),("pm_G_F","dev",F),("pm_G_F_rad","dev",f"{F} radial=true radial_max=0 radial_gate=0.02"),("pm_G_F_st16","dev",f"{F} strat_cells=16"),("pm_G_F_d32","dev",f"{F} lm_solver=dense32"),("pm_G_F_r5","dev",f"{F} lm_solver=dense32 lm_refactor=5"),("pm_G_I0b","dev",POL)]:
    L.append(f"WANDB_MODE=disabled bash $J/run.sh {snap} {name} {subset} 0 {ov} > $J/runlogs/{name}.log 2>&1")
print("\n".join(L))
