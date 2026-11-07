# Experiment 1 of the patch-BA proposal: rec_it800 with one change per run, 3 seeds. Seed 0 of every arm first.
R="nbr_mode=spread neighbors=10 lr_rot=3e-3 lr_trans=3e-3 lr_focal=3e-3 levels=8,4,2,1,1 lm_levels=4 samples=1024 impl=analytic lm_impl=analytic lm_solver=dense32 lm_iters=15 lm_samples=4096 lm_stop=2e-3 radial=true radial_max=0 radial_gate=0.01 save=false"
arms = {
    "ncc": "iters=800 photo_loss=ncc samples=4096",
    "vgg": "iters=800 features=vgg photo_loss=huber vgg_fast=true feat_dim=8",
    "nobright": "iters=800 bright=false",
    "lr4": "iters=800 lr_rot=7.5e-4 lr_trans=7.5e-4 lr_focal=7.5e-4 lr_depth=1.25e-3",
    "base": "iters=800",   # seed 0 is rec_it800
    "t200": "iters=200",   # seed 0 is rec_it200; same deterministic travel lr*T as lr4
}
for s in (0, 1, 2):
    for a, o in arms.items():
        if s == 0 and a in ("base", "t200"):
            continue
        print(f"WANDB_MODE=disabled bash $J/run.sh d15 x1_{a}_s{s} dev 0 {R} {o} seed={s}")
