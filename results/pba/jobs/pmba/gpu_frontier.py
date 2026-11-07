#!/usr/bin/env python3
import sys
snap=sys.argv[1]
FAST="nbr_mode=spread neighbors=10 lr_rot=3e-3 lr_trans=3e-3 lr_focal=3e-3"; A="impl=analytic lm_impl=analytic lm_solver=dense32"
H=f"{FAST} levels=8,4,2,1,1 lm_levels=4 samples=1024 {A}"
NEW="strat_cells=16 radial=true radial_max=0 radial_gate=0.02"
L=[]
def add(name,ov,subset="dev"):
    L.append(f"WANDB_MODE=disabled bash $J/run.sh {snap} {name} {subset} 0 {ov} > $J/runlogs/{name}.log 2>&1")
# plain sampling: where is the knee of the hybrid?
for t in (25,50):
    for n in (3,5,8):
        for m in (1024,4096):
            add(f"pm_GF_T{t}_N{n}_M{m}",f"{H} iters={t} lm_iters={n} lm_samples={m}")
# stratified + gated radial, longer LM
for t in (25,50):
    for n in (5,10,20):
        for m in (1024,4096):
            add(f"pm_GY_T{t}_N{n}_M{m}",f"{H} {NEW} iters={t} lm_iters={n} lm_samples={m}")
# feature-metric: the old implementation for its time, then the analytic one by channels and samples
V=f"{FAST} features=vgg photo_loss=ncc levels=8,4,2,1,1 lm_levels=4 lm_iters=5"
add("pm_GV_old",f"{V} samples=4096 iters=50")
for t in (25,50):
    for dim in (16,8,4):
        for s in (4096,1024):
            add(f"pm_GV_T{t}_d{dim}_s{s}",f"{V} {A} iters={t} feat_dim={dim} samples={s}")
print("\n".join(L))
