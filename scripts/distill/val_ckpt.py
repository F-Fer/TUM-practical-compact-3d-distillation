"""Fast validation of a trained student on val or train scenes (overfitting check).

    uv run scripts/distill/val_ckpt.py <run name> [train|val] [num_views]"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import bench_subsets, TRAIN_SCENES, VAL_SCENES, WS, load_da3, load_dvlt, predict_fn_da3, predict_fn_dvlt, val_subsets, validate  # noqa: E402

if __name__ == "__main__":
    name, split = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "val")
    nv = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    scenes = TRAIN_SCENES if split == "train" else VAL_SCENES
    subsets = bench_subsets(scenes) if nv == 100 else val_subsets(nv, seed=1, scenes=scenes)
    if name.startswith("da3:") or name.startswith("dvlt:"):
        kind, arg, sd = *name.split(":", 1), None
    else:
        ck = torch.load(os.path.join(WS, "runs", name, "student.pt"), map_location="cpu", weights_only=False)
        kind, arg = ck["student"].split(":", 1)
        sd = ck["state_dict"]
    if kind == "da3":
        net = load_da3(arg)
        fn = predict_fn_da3
    else:
        from common import untie_dvlt

        net = load_dvlt(int(arg.rstrip("u")))
        if arg.endswith("u"):
            untie_dvlt(net, int(arg.rstrip("u")))
        fn = lambda m: predict_fn_dvlt(m, int(arg.rstrip("u")))  # noqa: E731
    if sd is not None:
        net.load_state_dict(sd)
    net = net.cuda().eval()
    mean, _ = validate(fn(net), subsets, "images_dvlt" if kind == "dvlt" else "images")
    print(name, split, nv, {k: round(v, 4) for k, v in mean.items()})
