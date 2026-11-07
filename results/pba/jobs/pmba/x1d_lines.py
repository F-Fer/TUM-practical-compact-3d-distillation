# Brightness that can't collapse: relative gain (bright_rel), over T. And the signed gain of the original form.
R="nbr_mode=spread neighbors=10 lr_rot=3e-3 lr_trans=3e-3 lr_focal=3e-3 levels=8,4,2,1,1 lm_levels=4 samples=1024 impl=analytic lm_impl=analytic lm_solver=dense32 lm_iters=15 lm_samples=4096 lm_stop=2e-3 radial=true radial_max=0 radial_gate=0.01 save=false"
arms = {
    "sign_t50": "iters=50",
    "sign_t800": "iters=800",
    "rel_t25": "iters=25 bright_rel=true",
    "rel_t50": "iters=50 bright_rel=true",
    "rel_t200": "iters=200 bright_rel=true",
    "rel_t400": "iters=400 bright_rel=true",
    "rel_t800": "iters=800 bright_rel=true",
}
for a, o in arms.items():
    print(f"WANDB_MODE=disabled bash $J/run.sh e4 x1d_{a}_s0 dev 0 {R} {o} seed=0")
