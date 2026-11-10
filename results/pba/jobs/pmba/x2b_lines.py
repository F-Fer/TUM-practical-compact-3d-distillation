# Experiment 2, follow-up: is it the free depth or NCC? NCC with the grid / frozen depth from the ground truth, and the free-depth NCC variants.
G="nbr_mode=spread neighbors=10 lr_rot=3e-3 lr_trans=3e-3 lr_focal=3e-3 samples=1024 levels=1 lm_levels=0 impl=analytic lm_impl=analytic lm_solver=dense lm_iters=20 init_gt=pose_focal track_every=1 save=false photo_loss=ncc"
F="depth_grid=0 lm_free_depth=true lm_fixed=true"
arms = {
    "ncc_frozen": "depth_grid=0 lm_samples=4096 lm_fixed=true",
    "ncc_g8": "depth_grid=8 lm_samples=4096 lm_fixed=true",
    "ncc_g16": "depth_grid=16 lm_samples=4096 lm_fixed=true",
    "ncc_g8_fresh": "depth_grid=8 lm_samples=4096",   # fresh samples, as the recommended LM
    "ncc_nogeo": f"{F} lm_samples=4096 w_free_depth=20 lm_w_geo=0",
    "ncc_w2": f"{F} lm_samples=4096 w_free_depth=2",
    "ncc_w200": f"{F} lm_samples=4096 w_free_depth=200",
    "ncc_m1024": f"{F} lm_samples=1024 w_free_depth=20",
    "ncc_m16384": f"{F} lm_samples=16384 w_free_depth=20",
    "ncc_radial": f"{F} lm_samples=4096 w_free_depth=20 radial=true radial_max=0",
}
for a, o in arms.items():
    print(f"WANDB_MODE=disabled bash $J/run.sh e2 x2b_{a} drift 0 {G} {o}")
