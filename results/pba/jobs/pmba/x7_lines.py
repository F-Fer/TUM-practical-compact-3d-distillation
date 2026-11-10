# Adam with the fixed brightness, then LM on NCC (grid, fresh pixels): its held-out test, cheaper T, LM length, and F1.
B="nbr_mode=spread neighbors=10 lr_rot=3e-3 lr_trans=3e-3 lr_focal=3e-3 samples=1024 impl=analytic lm_impl=analytic lm_solver=dense32 radial=true radial_max=0 radial_gate=0.01 bright_rel=true levels=8,4,2,1,1 lm_levels=4 lm_samples=4096 lm_photo_loss=ncc"
arms = [
    ("x7h_t800_n40", "full", "iters=800 lm_iters=40 save=false"),
    ("x7h_t200_n40", "full", "iters=200 lm_iters=40 save=false"),
    ("x7_t200_n40", "dev", "iters=200 lm_iters=40 save=false"),
    ("x7_t400_n40", "dev", "iters=400 lm_iters=40 save=false"),
    ("x7_t800_n20", "dev", "iters=800 lm_iters=20 save=false"),
    ("x7_t800_n80", "dev", "iters=800 lm_iters=80 save=false"),
    ("x7f_t800_n40", "dev", "iters=800 lm_iters=40"),
    ("x7_t800_n40_s1", "dev", "iters=800 lm_iters=40 save=false seed=1"),
]
for a, sub, o in arms:
    print(f"WANDB_MODE=disabled bash $J/run.sh e4 {a} {sub} 0 {B} {o}")
