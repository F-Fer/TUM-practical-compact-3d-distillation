# Experiment 4: no Adam. And experiments 3/direct on the fixed brightness: Adam prefix with bright_rel.
B="nbr_mode=spread neighbors=10 lr_rot=3e-3 lr_trans=3e-3 lr_focal=3e-3 samples=1024 impl=analytic lm_impl=analytic lm_solver=dense32 radial=true radial_max=0 radial_gate=0.01 save=false track_every=1000 bright_rel=true iters=50"
arms = {
    # no Adam at all
    "x6_na_patch": "levels= patch=8:2,4:2,2:2,1:4 patch_inner=10 patch_gate_after=2",
    "x6_na_patch_geo0": "levels= patch=8:2,4:2,2:2,1:4 patch_inner=10 patch_gate_after=2 lm_w_geo=0",
    "x6_na_patch_shape": "levels= patch=8:2,4:2,2:2,1:4 patch_inner=10 patch_gate_after=2 patch_info=shape",
    "x6_na_lmncc": "levels=8,4,2,1 lm_levels=0,1,2,3 lm_iters=10 lm_samples=4096 lm_photo_loss=ncc",
    "x6_na_lmncc_free": "levels=8,4,2,1 lm_levels=0,1,2,3 lm_iters=10 lm_samples=4096 lm_photo_loss=ncc lm_free_depth=true lm_fixed=true",
    # Adam 4x50 with the fix, then: patch stage / LM on NCC with the grid / LM on NCC with free depth
    "x6_a50_patch": "levels=8,4,2,1 patch=1:8 patch_inner=10",
    "x6_a50_lmncc": "levels=8,4,2,1,1 lm_levels=4 lm_iters=40 lm_samples=4096 lm_photo_loss=ncc",
    "x6_a50_lmncc_free": "levels=8,4,2,1,1 lm_levels=4 lm_iters=40 lm_samples=4096 lm_photo_loss=ncc lm_free_depth=true lm_fixed=true",
    # the same from a long Adam: does the second stage still add anything
    "x6_a800_patch": "iters=800 levels=8,4,2,1 patch=1:8 patch_inner=10",
    "x6_a800_lmncc": "iters=800 levels=8,4,2,1,1 lm_levels=4 lm_iters=40 lm_samples=4096 lm_photo_loss=ncc",
    "x6_a800_nolm": "iters=800 levels=8,4,2,1",
}
for a, o in arms.items():
    print(f"WANDB_MODE=disabled bash $J/run.sh e4 {a} dev 0 {B} {o}")
