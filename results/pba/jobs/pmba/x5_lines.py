# bright_rel: seeds, saturation, the Adam-noise test after the fix, F1 (saved predictions, scored on CPU), held-out (full benchmark).
R="nbr_mode=spread neighbors=10 lr_rot=3e-3 lr_trans=3e-3 lr_focal=3e-3 levels=8,4,2,1,1 lm_levels=4 samples=1024 impl=analytic lm_impl=analytic lm_solver=dense32 lm_iters=15 lm_samples=4096 lm_stop=2e-3 radial=true radial_max=0 radial_gate=0.01 bright_rel=true"
L4="lr_rot=7.5e-4 lr_trans=7.5e-4 lr_focal=7.5e-4 lr_depth=1.25e-3 lr_bright=2.5e-3"
def line(name, subset, opts):
    print(f"WANDB_MODE=disabled bash $J/run.sh e4 {name} {subset} 0 {R} {opts}")
# held-out first: it decides whether the rest matters
line("x5h_t50", "full", "iters=50 save=false")
line("x5h_t800", "full", "iters=800 save=false")
line("x5_t1600_s0", "dev", "iters=1600 save=false")
line("x5_t3200_s0", "dev", "iters=3200 save=false")
line("x5_lr4_t800_s0", "dev", f"iters=800 save=false {L4}")
# F1 needs predictions, about 2 GB per run
for T in (50, 200, 800):
    line(f"x5f_t{T}", "dev", f"iters={T}")
for s in (1, 2):
    for T in (50, 800, 200, 400, 25):
        line(f"x5_t{T}_s{s}", "dev", f"iters={T} save=false seed={s}")
