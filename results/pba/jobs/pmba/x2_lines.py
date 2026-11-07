# Experiment 2: free per-sample depth in LM, started at the ground truth. Same setup as pm_B_* (REPORT3 section 3), subset "drift".
G="nbr_mode=spread neighbors=10 lr_rot=3e-3 lr_trans=3e-3 lr_focal=3e-3 samples=1024 levels=1 lm_levels=0 impl=analytic lm_impl=analytic lm_solver=dense lm_iters=20 init_gt=pose_focal track_every=1 save=false"
F="depth_grid=0 lm_free_depth=true lm_fixed=true"
arms = {
    # controls: what fixed samples alone do
    "ctl_frozen": "depth_grid=0 lm_samples=4096 lm_fixed=true",
    "ctl_g8": "depth_grid=8 lm_samples=4096 lm_fixed=true",
    "ctl_g16": "depth_grid=16 lm_samples=4096 lm_fixed=true",
    # prior strength at 4096 samples
    "w2": f"{F} lm_samples=4096 w_free_depth=2",
    "w20": f"{F} lm_samples=4096 w_free_depth=20",
    "w200": f"{F} lm_samples=4096 w_free_depth=200",
    "w2000": f"{F} lm_samples=4096 w_free_depth=2000",
    # number of samples
    "m1024": f"{F} lm_samples=1024 w_free_depth=20",
    "m16384": f"{F} lm_samples=16384 w_free_depth=20",
    # without the geometric row, and patches
    "nogeo": f"{F} lm_samples=4096 w_free_depth=20 lm_w_geo=0",
    "ncc": f"{F} lm_samples=4096 w_free_depth=20 photo_loss=ncc",
    "radial": f"{F} lm_samples=4096 w_free_depth=20 radial=true radial_max=0",
}
for a, o in arms.items():
    print(f"WANDB_MODE=disabled bash $J/run.sh e1 x2_{a} drift 0 {G} {o}")
