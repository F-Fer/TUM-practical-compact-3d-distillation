# Compact Feed-Forward 3D Reconstruction through Distillation and Test-Time Optimization

TUM, Practical Course: Generative AI - Project 07

## Setup

Requirements: an NVIDIA GPU with driver ≥ 530 (CUDA 12.1 wheels), [uv](https://docs.astral.sh/uv/), and `unzip`.

```sh
git clone --recurse-submodules git@github.com:F-Fer/TUM-practical-compact-3d-distillation.git
cd TUM-practical-compact-3d-distillation
# in an existing clone: git submodule update --init
uv sync    # creates .venv (Python 3.11, torch 2.5.1 cu121, Depth Anything 3 from third_party/)
```

Third-party code lives in git submodules under `third_party/` and isn't modified; our code is in `scripts/`.

Don't upgrade `open3d` past 0.19.0. In 0.20.0, TSDF fusion produces empty meshes, and the reconstruction metrics silently drop to F1 ≈ 1%.

## DA3 benchmark

We evaluate pretrained [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) models ([arXiv:2511.10647](https://arxiv.org/abs/2511.10647)) on its benchmark (HiRoom, ETH3D, DTU, 7-Scenes, ScanNet++). This reproduces the paper's Table 2 (pose AUC@3/AUC@30) and Table 3 (reconstruction F1, or Chamfer distance on DTU, without and with ground-truth poses).

### Download the data

```sh
scripts/download_bench.sh                  # all six: hiroom 7scenes dtu64 dtu scannetpp eth3d
scripts/download_bench.sh hiroom dtu64     # or only some
```

The script downloads the Hugging Face dataset [`depth-anything/DA3-BENCH`](https://huggingface.co/datasets/depth-anything/DA3-BENCH) into `workspace/benchmark_dataset/`. All six are about 41 GB to download and 79 GB extracted. Datasets that are already extracted are skipped, so you can rerun it after an interruption.

### Run

```sh
# full benchmark (all datasets, modes pose, recon_unposed, recon_posed)
uv run scripts/run_bench.py depth-anything/DA3-SMALL

# a subset; extra arguments are passed to the DA3 evaluator as OmegaConf overrides
uv run scripts/run_bench.py depth-anything/DA3-SMALL "eval.datasets=[hiroom]" "eval.modes=[pose]"

# re-score stored predictions without running the model
uv run scripts/run_bench.py depth-anything/DA3-SMALL eval.eval_only=true
```

The models we use are `depth-anything/DA3-SMALL`, `depth-anything/DA3-BASE` and `depth-anything/DA3-LARGE-1.1`. They are downloaded from Hugging Face on first use. `run_bench.py` keeps the backbone's intermediate features in CPU memory (`scripts/da3_lowmem.py`, outputs unchanged), so DA3-LARGE-1.1 at 100 frames needs about 5.5 GiB of GPU memory instead of more than 10.

A full run of one model takes several hours, most of it scoring the reconstructions on the CPU. On a remote machine, start it in tmux so it keeps running after you disconnect:

```sh
tmux new -d -s bench "uv run scripts/run_bench.py depth-anything/DA3-SMALL 2>&1 | tee workspace/bench_small.log"
```

To benchmark several models one after another, use `scripts/run_bench_loop.sh`. It runs DA3-SMALL, DA3-BASE and DA3-LARGE-1.1 in turn and then prints the comparison with the paper. If one model fails, it continues with the next. Any arguments are passed to every run as evaluator overrides, and each model logs to `workspace/bench_<model>.log`.

```sh
scripts/run_bench_loop.sh "eval.datasets=[hiroom]"    # quick check of the whole loop
tmux new -d -s bench scripts/run_bench_loop.sh        # full benchmark of all models in the loop
```

To check on a tmux run, use `tmux attach -t bench` and detach with Ctrl-b d, or follow a log with `tail -f workspace/bench_DA3-BASE.log`.

Results go to `workspace/evaluation/<model>/` (add `workspace.work_dir=<dir>` to choose another folder):

| Path | Content |
|---|---|
| `model_results/<dataset>/<scene>/{unposed,posed}/exports/mini_npz/results.npz` | predictions per scene |
| `metric_results/<dataset>_<mode>.json` | metrics per dataset and mode |
| `timings.jsonl` | forward time and peak GPU memory per inference call |
| `runs.jsonl` | one line per run: host, GPU, versions, wall time |

`workspace/` is gitignored and holds all data, predictions and logs.

### Compare with the paper

```sh
uv run scripts/compare_paper.py            # print ours vs. paper for every run in workspace/evaluation/
uv run scripts/compare_paper.py --save     # also copy metrics, timings and comparison.md to results/da3_bench/<model>/
```

Commit `results/da3_bench/`; it is our record of the baseline numbers.

## References

- **VGGT: Visual Geometry Grounded Transformer** (Wang et al., CVPR 2025): [arXiv:2503.11651](https://arxiv.org/abs/2503.11651), [code](https://github.com/facebookresearch/vggt)
- **DéjàView: Looping Transformers for Multi-View 3D Reconstruction** (NVIDIA): [arXiv:2605.30215](https://arxiv.org/abs/2605.30215)
- **Course kickoff slides** ("Generative AI: From Diffusion to World Models", WS 2026/27): available from the course page

### Distillation
- **OTT3R: Multi-View 3D Reconstruction and Fast Dataset Generation at 1% Compute** (Leblanc et al., 2026): [arXiv:2609.36374](https://arxiv.org/abs/2609.36374)
- **Multi-View 3D Reconstruction using Knowledge Distillation** (Dutt et al., 2026): [arXiv:2412.02039](https://arxiv.org/abs/2412.02039)
- **A survey on knowledge distillation: Recent advancements** (Moslemi et al., 2024): [doi.org/10.1016/j.mlwa.2024.100605](https://doi.org/10.1016/j.mlwa.2024.100605)

### Test-Time Optimization

### Datasets

### Metrics

