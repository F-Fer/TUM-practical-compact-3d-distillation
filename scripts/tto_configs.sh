# Track-based BA configurations, sourced by run_tto_roma_loop.sh and run_tto_dev.sh.
#
# RoMa's tracks rarely agree with DA3's own geometry (its focal length can be 25% off), so no reprojection gate and a robust loss;
# every benchmark scene has one camera, so BA shares it (as run_tto_pba.py does with focal=shared)
ROMA="tracker=roma max_reproj=1000000 loss=cauchy loss_scale=0.25 vis_thresh=0.5 extra_queries=4 shared_camera=true"
Q4="$ROMA roma_setting=turbo query_frames=4"
declare -A CONFIGS=(
    [roma-turbo-q4]="$ROMA roma_setting=turbo query_frames=4"
    [roma-turbo-q8]="$ROMA roma_setting=turbo query_frames=8"
    [roma-fast-q4]="$ROMA roma_setting=fast query_frames=4"
    [roma-base-q4]="$ROMA roma_setting=base query_frames=4"
    [roma-precise-q4]="$ROMA roma_setting=precise query_frames=4"  # 1280 px matching, needs > 10 GB
    [vggsfm-trivial]="tracker=vggsfm max_points=20480"
    [vggsfm-cauchy]="tracker=vggsfm max_reproj=1000000 loss=cauchy loss_scale=0.25 shared_camera=true max_points=20480"
    # fp32 tracker: the old DA3-*-ba runs (before bf16 autocast went in) used it
    [vggsfm-trivial-fp32]="tracker=vggsfm max_points=20480 autocast=false"
    [vggsfm-cauchy-fp32]="tracker=vggsfm max_reproj=1000000 loss=cauchy loss_scale=0.25 shared_camera=true max_points=20480 autocast=false"
    # baseline: DA3's own prediction, scored by the same pipeline
    [init-dev]="tracker=none"
    # seed noise of the main config (seed 0 = roma-turbo-q8), and the same with the query frames shifted
    [roma-turbo-q8-s1]="$ROMA roma_setting=turbo query_frames=8 seed=1"
    [roma-turbo-q8-s2]="$ROMA roma_setting=turbo query_frames=8 seed=2"
    [roma-turbo-q8-off3]="$ROMA roma_setting=turbo query_frames=8 query_offset=3"
    # ablations of roma-turbo-q4, one factor each (later overrides win)
    [abl-loss-trivial]="$Q4 loss=trivial"
    [abl-cauchy-s01]="$Q4 loss_scale=0.1"
    [abl-cauchy-s05]="$Q4 loss_scale=0.5"
    [abl-cauchy-s10]="$Q4 loss_scale=1.0"
    [abl-gate8]="$Q4 max_reproj=8"
    [abl-gate30]="$Q4 max_reproj=30"
    [abl-sepcam]="$Q4 shared_camera=false"
    [abl-fixfocal]="$Q4 refine_focal=false"
    [abl-fast]="$Q4 roma_setting=fast"
    [abl-base]="$Q4 roma_setting=base"
    [abl-q1]="$Q4 query_frames=1"
    [abl-q2]="$Q4 query_frames=2"
    [abl-q12]="$Q4 query_frames=12"
    [abl-noextra]="$Q4 extra_queries=0"
    [abl-pts1024]="$Q4 max_query_pts=1024"
    [abl-pts2048]="$Q4 max_query_pts=2048"
    [abl-vis02]="$Q4 vis_thresh=0.2"
    [abl-vis08]="$Q4 vis_thresh=0.8"
    [abl-pairs10]="$Q4 pairs=10"
    [abl-pairs20]="$Q4 pairs=20"
    [abl-shrink25]="$Q4 shrink=0.25"
    [abl-shrink50]="$Q4 shrink=0.5"
    [abl-shrink75]="$Q4 shrink=0.75"
)
