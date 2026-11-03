# Track-based BA configurations, sourced by run_tto_roma_loop.sh and run_tto_dev.sh.
#
# RoMa's tracks rarely agree with DA3's own geometry (its focal length can be 25% off), so no reprojection gate and a robust loss;
# every benchmark scene has one camera, so BA shares it (as run_tto_pba.py does with focal=shared)
ROMA="tracker=roma max_reproj=1000000 loss=cauchy loss_scale=0.25 vis_thresh=0.5 extra_queries=4 shared_camera=true"
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
)
