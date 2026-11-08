# Session 5 configurations. B = REPORT4's "relative" configuration (REPORT3 Recommended + bright_rel=true); expanded at enqueue time.
B="nbr_mode=spread neighbors=10 lr_rot=3e-3 lr_trans=3e-3 lr_focal=3e-3 levels=8,4,2,1,1 lm_levels=4 samples=1024 impl=analytic lm_impl=analytic lm_solver=dense32 lm_iters=15 lm_samples=4096 lm_stop=2e-3 radial=true radial_max=0 radial_gate=0.01 bright_rel=true"
# Ground-truth start (REPORT3 section 3, REPORT4 section 4): one LM level on the sharp images from the GT poses and focal, 20 iterations, tracked
G="nbr_mode=spread neighbors=10 lr_rot=3e-3 lr_trans=3e-3 lr_focal=3e-3 samples=1024 levels=1 lm_levels=0 impl=analytic lm_impl=analytic lm_solver=dense lm_iters=20 init_gt=pose_focal track_every=1 save=false lm_samples=4096"
# REPORT4 section 7: Adam with relative brightness, then LM on NCC of 3x3 patches (grid, fresh pixels, no stop rule); with iters=800 lm_iters=40
N="nbr_mode=spread neighbors=10 lr_rot=3e-3 lr_trans=3e-3 lr_focal=3e-3 samples=1024 impl=analytic lm_impl=analytic lm_solver=dense32 radial=true radial_max=0 radial_gate=0.01 bright_rel=true levels=8,4,2,1,1 lm_levels=4 lm_samples=4096 lm_photo_loss=ncc"
F="$B focal_init=median"  # B' = B with one shared focal length
