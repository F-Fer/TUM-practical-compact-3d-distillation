# Follow-up of experiment 1: is it the per-frame brightness? lr_bright was not scaled in x1_lr4; the ncc and vgg arms have no brightness either.
R="nbr_mode=spread neighbors=10 lr_rot=3e-3 lr_trans=3e-3 lr_focal=3e-3 levels=8,4,2,1,1 lm_levels=4 samples=1024 impl=analytic lm_impl=analytic lm_solver=dense32 lm_iters=15 lm_samples=4096 lm_stop=2e-3 radial=true radial_max=0 radial_gate=0.01 save=false"
L4="lr_rot=7.5e-4 lr_trans=7.5e-4 lr_focal=7.5e-4 lr_depth=1.25e-3"
arms = {
    "lr4all": f"iters=800 {L4} lr_bright=2.5e-3",   # really everything / 4
    "brightlr4": "iters=800 lr_bright=2.5e-3",      # only the brightness / 4
    "nobright_lr4": f"iters=800 bright=false {L4}",  # noise of the rest, without brightness
    "nobright_t25": "iters=25 bright=false",
    "nobright_t50": "iters=50 bright=false",
    "nobright_t200": "iters=200 bright=false",
    "nobright_t400": "iters=400 bright=false",
}
for a, o in arms.items():
    print(f"WANDB_MODE=disabled bash $J/run.sh d15 x1b_{a}_s0 dev 0 {R} {o} seed=0")
