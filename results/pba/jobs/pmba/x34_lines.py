# Experiment 3: sweep of the patch stage after Adam 4x50 (one setting per run, 8 alignments x 10 LM). x4: LM on NCC with free depth instead, reading the images.
A="nbr_mode=spread neighbors=10 lr_rot=3e-3 lr_trans=3e-3 lr_focal=3e-3 samples=1024 impl=analytic lm_impl=analytic lm_solver=dense32 radial=true radial_max=0 radial_gate=0.01 save=false iters=50 track_every=1000"
P=f"{A} levels=8,4,2,1 patch=1:8 patch_inner=10"
arms = {
    "x3_c800": "patch_count=800",
    "x3_r2": "patch_radius=2",
    "x3_shape": "patch_info=shape patch_delta=1",
    "x3_geo01": "lm_w_geo=0.1",
    "x3_geo0": "lm_w_geo=0",
    "x3_w2": "w_free_depth=2",
    "x3_w200": "w_free_depth=200",
    "x3_ncc08": "patch_ncc=0.8",
    "x3_lk1": "patch_lk=1",
    "x3_sched": "patch=2:4,1:4",
}
L=f"{A} levels=8,4,2,1,1 lm_levels=4 lm_samples=4096 lm_iters=40 lm_fixed=true"
direct = {
    # LM on NCC after Adam: the control (grid solved, fresh pixels), then fixed pixels, then free depth
    "x4_ctl_ncc_grid": "lm_photo_loss=ncc lm_fixed=false",
    "x4_ctl_ncc_grid_fixed": "lm_photo_loss=ncc",
    "x4_free_ncc": "lm_photo_loss=ncc lm_free_depth=true",
    "x4_free_ncc_nobright": "lm_photo_loss=ncc lm_free_depth=true bright=false",
    "x4_free_ncc_nogeo": "lm_photo_loss=ncc lm_free_depth=true lm_w_geo=0",
    "x4_free_huber": "lm_free_depth=true",
}
for a, o in direct.items():
    print(f"WANDB_MODE=disabled bash $J/run.sh e3 {a} dev 0 {L} {o}")
for a, o in arms.items():
    print(f"WANDB_MODE=disabled bash $J/run.sh e3 {a} dev 0 {P} {o}")
