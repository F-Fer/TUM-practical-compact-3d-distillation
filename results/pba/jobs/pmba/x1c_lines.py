# Follow-up: how far does the brightness walk (bright_*_abs, logged from snapshot e2); does a small lr_bright keep the short-T gain; NCC and VGG+Huber over T.
R="nbr_mode=spread neighbors=10 lr_rot=3e-3 lr_trans=3e-3 lr_focal=3e-3 levels=8,4,2,1,1 lm_levels=4 samples=1024 impl=analytic lm_impl=analytic lm_solver=dense32 lm_iters=15 lm_samples=4096 lm_stop=2e-3 radial=true radial_max=0 radial_gate=0.01 save=false"
NCC="photo_loss=ncc samples=4096"
VGG="features=vgg photo_loss=huber vgg_fast=true feat_dim=8"
arms = {
    "walk_t50": "iters=50",
    "walk_t800": "iters=800",
    "lrb1e3_t25": "iters=25 lr_bright=1e-3",
    "lrb1e3_t50": "iters=50 lr_bright=1e-3",
    "lrb1e3_t800": "iters=800 lr_bright=1e-3",
    "ncc_t50": f"iters=50 {NCC}",
    "ncc_t200": f"iters=200 {NCC}",
    "vgg_t50": f"iters=50 {VGG}",
    "vgg_t200": f"iters=200 {VGG}",
}
for a, o in arms.items():
    print(f"WANDB_MODE=disabled bash $J/run.sh e2 x1c_{a}_s0 dev 0 {R} {o} seed=0")
