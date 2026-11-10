"""One picture of the depth finding: DA3's depth error is smooth over regions plus large errors at edges, not pixel noise.

One frame, its photo faded to grey, overlaid with the signed relative depth error of DA3 (after removing only the scene's
overall scale): blue where DA3 is too near, red where it is too far.

    uv run scripts/plot_depth_error_example.py DA3-SMALL scannetpp fb5a96b1a2 --frame 54 --vmax 20
"""

import argparse
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import analyse_failure_modes as analysis  # noqa: E402

CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("err", ["#1f5fb4", "#2a78d6", "#ffffff", "#d1342f", "#a3201c"])
INK2 = "#52514e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="DA3-SMALL")
    ap.add_argument("dataset", nargs="?", default="scannetpp")
    ap.add_argument("scene", nargs="?", default="fb5a96b1a2")
    ap.add_argument("--frame", type=int, default=54, help="frame number (1-based, in frame order)")
    ap.add_argument("--vmax", type=float, default=20.0, help="colour range in percent")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    scene_dir = args.scene.replace("_", "/", 2) if args.dataset == "hiroom" else args.scene
    data = analysis.load_scene(args.model, args.dataset, scene_dir)
    saved = np.load(os.path.join(analysis.REPO_ROOT, "workspace/failure_modes", args.model, args.dataset, args.scene + ".npz"))
    t = args.frame - 1

    pred, gt, valid = data["pred_depth"][t].astype(np.float64), data["gt_depth"][t], data["valid"][t]
    H, W = pred.shape
    pct = np.zeros((H, W))
    pct[valid] = (np.exp(np.log(pred[valid]) - np.log(gt[valid]) - float(saved["scene_log_scale"])) - 1) * 100

    gray = cv2.cvtColor(cv2.resize(cv2.imread(data["image_files"][t]), (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) / 255.0
    base = np.repeat((0.45 * gray + 0.55)[..., None], 3, axis=2)
    color = CMAP(np.clip(pct / args.vmax * 0.5 + 0.5, 0, 1))[..., :3]
    alpha = (np.clip(np.abs(pct) / args.vmax, 0, 1) ** 0.6 * valid)[..., None]
    img = base * (1 - alpha) + color * alpha

    fig_w = 8.0
    fig = plt.figure(figsize=(fig_w, fig_w * H / W + 0.9))
    ax = fig.add_axes([0, 0.9 / (fig_w * H / W + 0.9), 1, fig_w * H / W / (fig_w * H / W + 0.9)])
    ax.imshow(img, interpolation="lanczos")
    ax.set_axis_off()
    cax = fig.add_axes([0.25, 0.42 / (fig_w * H / W + 0.9), 0.5, 0.12 / (fig_w * H / W + 0.9)])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(-args.vmax, args.vmax))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
    v = int(args.vmax)
    cb.set_ticks([-v, -v // 2, 0, v // 2, v], labels=[f"{-v}%\ntoo near", f"{-v // 2}%", "0", f"+{v // 2}%", f"+{v}%\ntoo far"])
    cb.ax.tick_params(labelsize=9, colors=INK2)
    cb.outline.set_visible(False)
    cb.set_label("DA3-Small depth error", fontsize=10, color=INK2, labelpad=-48)

    out = args.out or os.path.join(analysis.REPO_ROOT, "results/failure_modes", args.model, f"depth_error_{args.dataset}.png")
    fig.savefig(out, dpi=160, facecolor="white")
    print(out, f"median |error| {np.median(np.abs(pct[valid])):.1f}%, > {v}%: {(np.abs(pct[valid]) > v).mean() * 100:.1f}% of pixels")


if __name__ == "__main__":
    main()
