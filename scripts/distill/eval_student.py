"""Score a trained student (or an untrained DéjàView with K steps) with the official DA3 evaluator
on the held-out ScanNet++ scenes.

    uv run scripts/distill/eval_student.py runs/<name>        -> workspace/distill/eval/<name>/
    uv run scripts/distill/eval_student.py dvlt:4             -> workspace/distill/eval/dvlt-K4/
    [extra evaluator overrides, e.g. "eval.modes=[pose]"]

Goes through scripts/run_bench.py (same timing logs, lowmem patches, KDTree speedup)."""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)  # first: scripts/distill/baselines.py
from common import REPO_ROOT, VAL_SCENES, WS  # noqa: E402

import run_bench  # noqa: E402  (installs da3_lowmem, KDTree patch, allocator config)
import baselines  # noqa: E402
from depth_anything_3.api import DepthAnything3  # noqa: E402


def main():
    target, overrides = sys.argv[1], sys.argv[2:]
    if target.startswith("dvlt:"):
        K = int(target.split(":")[1])
        name, ckpt = f"dvlt-K{K}", None
        student = target
    else:
        run_dir = target if os.path.isabs(target) else os.path.join(WS, target)
        if not os.path.exists(run_dir):
            run_dir = os.path.join(WS, "runs", target)
        name = os.path.basename(run_dir.rstrip("/"))
        ckpt = torch.load(os.path.join(run_dir, "student.pt"), map_location="cpu", weights_only=False)
        student = ckpt["student"]
    model_id = f"distill:{name}"
    kind, _, arg = student.partition(":")

    if kind == "dvlt":
        class Student(baselines.DVLTBaseline):
            def __init__(self, _model_id):
                super().__init__("nvidia/dvlt", inference_steps=int(arg.rstrip("u")))
                if arg.endswith("u"):
                    from common import untie_dvlt

                    untie_dvlt(self.model, int(arg.rstrip("u")))
                if ckpt is not None:
                    self.model.load_state_dict(ckpt["state_dict"])

        baselines.MODELS[model_id] = Student
    else:
        orig = DepthAnything3.from_pretrained

        def from_pretrained(path, *a, **k):
            api = orig(arg)
            api.model.load_state_dict(ckpt["state_dict"])
            return api

        DepthAnything3.from_pretrained = staticmethod(from_pretrained)

    work_dir = os.path.join("workspace", "distill", "eval", name)
    sys.argv = [
        "run_bench.py",
        model_id,
        f"workspace.work_dir={work_dir}",
        "eval.datasets=[scannetpp]",
        f"eval.scenes=[{','.join(VAL_SCENES)}]",
        "eval.modes=[pose,recon_unposed,recon_posed]",
        "inference.num_fusion_workers=2",
        *overrides,
    ]
    run_bench.main()


if __name__ == "__main__":
    main()
