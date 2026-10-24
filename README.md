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

### What is saved where

Each run writes to `workspace/evaluation/<model>/` (add `workspace.work_dir=<dir>` to choose another folder). `workspace/` is gitignored and exists only on the machine that ran the benchmark (for our runs: tum1). The predictions take about 16 GB per model.

| Path in `workspace/evaluation/<model>/` | Content |
|---|---|
| `model_results/<dataset>/<scene>/{unposed,posed}/exports/mini_npz/results.npz` | the model's output for one scene: `depth` and `conf` (`[N, H, W]`), `extrinsics` (`[N, 3, 4]`, world-to-camera) and `intrinsics` (`[N, 3, 3]`), all float32, N frames. `unposed` is the run without GT cameras, `posed` the one with. A TTO method can overwrite these files and rescore them with `eval.eval_only=true` |
| `model_results/.../exports/gt_meta.npz` | the ground-truth cameras (`extrinsics` `[N, 4, 4]`, `intrinsics`) and the paths of the N input images |
| `model_results/.../exports/fuse/pcd.ply` | the TSDF-fused point cloud that the recon metrics score (~27 MB per scene) |
| `metric_results/<dataset>_<mode>.json` | metrics per scene and their `mean`. Pose: `auc03`, `auc05`, `auc15`, `auc30`. Recon: `acc`, `comp`, `overall` (distances), `precision`, `recall`, `fscore` |
| `timings.jsonl` | one line per forward pass: scene, posed or not, number of frames, input size, forward time, peak GPU memory. After a rerun, keep only the latest line per scene (we keep the full history in `timings.all.jsonl`) |
| `runs.jsonl` | one line per finished `run_bench.py` call: host, GPU, versions, overrides, wall time. Runs that crashed have no line |

The console logs are `workspace/bench_<model>*.log`. The benchmark data itself is in `workspace/benchmark_dataset/` (see "Download the data").

Committed to git (`results/da3_bench/<model>/`, written by `compare_paper.py --save`): the metric JSONs, `timings.jsonl`, `runs.jsonl` and `comparison.md` (our numbers next to the paper's). The predictions and point clouds are too large and aren't committed. To get them on another machine, rerun the benchmark.

### Compare with the paper

```sh
uv run scripts/compare_paper.py            # print ours vs. paper for every run in workspace/evaluation/
uv run scripts/compare_paper.py --save     # also copy metrics, timings and comparison.md to results/da3_bench/<model>/
```

Commit `results/da3_bench/`; it is our record of the baseline numbers.

### What the benchmark measures

Each scene is evaluated in three modes:

- **Pose** (paper Table 2): the model gets only the images and predicts a camera per frame. After alignment to the ground truth, the relative pose errors between frames are scored as AUC@3° (strict) and AUC@30° (loose), in %.
- **Recon without GT pose** (Table 3): the predicted depth maps and cameras are fused into a mesh (TSDF fusion) and compared with the ground-truth scan. The score is the F1 score in % (precision: are the predicted points close to the true surface; recall: is the true surface covered). DTU reports the Chamfer distance in mm instead (lower is better). This tests depth and cameras together.
- **Recon with GT pose** (Table 3): the same, but the ground-truth cameras are given to the model as input and used for fusion, so it mostly tests the depth maps.

| Dataset | What it is | Scenes, frames per scene | Modes |
|---|---|---|---|
| HiRoom | high-resolution indoor rooms | 30, 10–23 | all three |
| ETH3D | high-resolution multi-view stereo (DSLR, indoor and outdoor, laser-scanned ground truth) | 11, 14–76 | all three |
| DTU | small objects in a lab, robot-arm camera, structured-light scans | DTU-64: 13 scenes for pose; DTU-49: 22 × 49 views for recon | pose / recon |
| 7-Scenes | Microsoft's Kinect RGB-D indoor scenes, made for camera relocalization | 7, 100 | all three |
| ScanNet++ | high-quality indoor scans (laser scanner, DSLR, iPhone), recalibrated by the DA3 authors | 20, 100 | all three |

At most 100 frames per scene are used. One model is 206 forward passes (103 scenes, each without and with GT pose).

### Our results

We ran the full benchmark for DA3-SMALL, DA3-BASE and DA3-LARGE-1.1 (October 2026, RTX 3080). Per-dataset numbers next to the paper's are in `results/da3_bench/<model>/comparison.md`. Most metrics are within ±1.5 points (pose) and ±3 F1 (recon) of the paper, and larger models are better: LARGE-1.1 > BASE > SMALL on every dataset and mode, except 7-Scenes recon with GT pose, where BASE (49.8 F1) beats LARGE-1.1 (48.2). The larger differences are:

- **ScanNet++ pose is lower for all three models**: SMALL AUC@30 −3.6, BASE AUC@3 −4.9 and AUC@30 −2.6, LARGE-1.1 AUC@3 −3.6. Because all sizes are affected, this probably comes from the data or the evaluation, not from one checkpoint. Not investigated yet.
- **LARGE-1.1 differs in both directions**: DTU pose AUC@3 +7.7, ETH3D pose AUC@3 +2.6, 7-Scenes recon without GT pose −5.9 F1.
- Smaller ones inside the tolerance, e.g. BASE HiRoom recon without GT pose −2.5 F1 and SMALL ETH3D recon without GT pose −2.6 F1.

### Why our numbers don't match the paper exactly

- **The released code and data aren't exactly what the paper used.** The DA3 repo's own expected DA3-GIANT results (`docs/BENCHMARK.md` in the submodule) already differ from the paper by up to about 3 F1, so even the authors' setup doesn't reproduce the paper exactly. The ScanNet++ data, for example, was recalibrated.
- **Checkpoints:** the `-1.1` checkpoints were retrained after a training bug fix. The paper's LARGE numbers probably come from the deprecated `DA3-LARGE`, which would explain why LARGE-1.1 deviates more, and in both directions. This is not verified yet (it would need a `DA3-LARGE` run).
- **Hardware and library versions:** inference runs in bf16, and a different GPU or different CUDA/cuDNN kernels change the depth maps slightly. The RANSAC alignment and TSDF fusion can amplify such small changes into visible metric differences, especially AUC@3 and F1 on scenes where the result is borderline.

Our own changes don't cause the differences. The pipeline is deterministic on our machine (repeated forward passes are bit-identical), the evaluation uses fixed seeds (42), and our patches (parallel KDTree queries, one fusion worker, `scripts/da3_lowmem.py`) were checked to give identical outputs.

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

