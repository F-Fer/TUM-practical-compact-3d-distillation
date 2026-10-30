# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

The repository is git@github.com:F-Fer/TUM-practical-compact-3d-distillation.git, and the work is split across branches:

- `main`: `README.md` (reference links), `.gitignore` and this file.
- `proposal`: also holds `proposal/`, the LaTeX project proposal (CVPR template, one file per section in `proposal/sec/`, bibliography in `proposal/main.bib`). It is written in Overleaf, and `preamble.tex` and `cvpr.sty` exist only there, so it doesn't compile locally as-is. The compiled PDF is in `docs/`.
- `benchmarks`: the DA3 benchmark setup described under "Setup and commands" (see also "Plan change" below).

The Beamer slides of the proposal presentation (22.10.26) aren't in any branch. The proposal and slides still describe the original distillation plan.

`docs/` holds reference PDFs (gitignored; each clone downloads them from the links in `README.md`, so the set and the filenames differ between machines) for the TUM research practical course **"Generative AI: From Diffusion to World Models"** (WS 2026/27, run by Umut Kocasari and Robin Borth). On this machine it has:

- `VGGT.pdf`: *VGGT: Visual Geometry Grounded Transformer* (Wang et al., CVPR 2025, arXiv 2503.11651). It is a feed-forward transformer that predicts cameras, point maps, depth maps and point tracks for 1 to hundreds of views in one pass. Code is at https://github.com/facebookresearch/vggt.
- `dejaview.pdf`: *DéjàView: Looping Transformers for Multi-View 3D Reconstruction* (NVIDIA, arXiv 2605.30215). It applies one shared transformer block recurrently for K refinement steps, and K is a compute knob you set at inference time. It matches much larger feed-forward models with far fewer parameters.
- `OTT3R.pdf`: *OTT3R* (Leblanc & Poullis, ACCV 2026, arXiv 2609.36374). It distills π³ (959M) into a 102M student on 2 GPUs, using an offline cache of teacher pseudo-labels. It is purely feed-forward with no test-time optimization. It was the closest prior work for the distillation plan. Code is at https://github.com/TheFourthKaramazov/OTT3R.
- `DA3.pdf`: *Depth Anything 3* (see the section below).
- `Test3R.pdf`: *Test3R: Learning to Reconstruct 3D at Test Time* (Yuan et al., NUS, arXiv 2506.13750). It fine-tunes DUSt3R at test time (prompt tuning) with a self-supervised loss: for an image triplet (I1, I2, I3), the pointmaps of I1 predicted from the pairs (I1, I2) and (I1, I3) must agree. Code is at https://github.com/nopQAQ/Test3R.
- `Compact_Feed_Forward_3D_Reconstruction_through_Distillation_and_Test_Time_Optimization.pdf`: our submitted proposal.

The course kickoff slides ("Generative AI From Diffusion to World Models", covering structure, deadlines and the 8 project topics) aren't in `docs/` on this machine; the course facts below come from them.

### Setup and commands

There are no tests or linters. Third-party code lives in git submodules under `third_party/` (`Depth-Anything-3`, `vggt`, `Pi3` and `dvlt`, all pinned) and is never patched; our code is in `scripts/`. The environment is a uv project (`pyproject.toml` + `uv.lock`, Python 3.11, torch 2.5.1 cu121, so NVIDIA driver ≥ 530). DA3, VGGT and Pi3 are installed editable from the submodules; DéjàView (`dvlt`) isn't installed (its pyproject pins numpy 2.4.4, DA3 needs numpy < 2), `scripts/baselines.py` puts `third_party/dvlt/src` on `sys.path` and our `pyproject.toml` adds its `accelerate` import; DA3's package metadata misses `addict` and `scikit-learn`, which our `pyproject.toml` adds. VGGT's bundle adjustment code (`vggt/dependency/`, used by `demo_colmap.py`) also needs `lightglue` (jytime's fork, pinned by commit), `hydra-core` (imported but unused by the VGGSfM tracker; it caps `omegaconf` below 2.4) and `pycolmap==3.10.0`: VGGT's `np_to_pycolmap.py` uses the 3.10 API (`image.registered`, `cam_from_world` as an attribute), which 3.12 removed with rigs/frames, while DA3 only uses pycolmap for its COLMAP export (written for ≥ 3.12, so broken by the pin, but the benchmark doesn't use it). The tracker downloads its weights (`facebook/VGGSfM` `vggsfm_v2_tracker.pt`) and LightGlue's keypoint weights on first use. Keep `open3d==0.19.0`: in 0.20.0 `ScalableTSDFVolume` produces empty meshes, and DA3's `sample_points_from_mesh` silently replaces those with random points in [-1, 1]³, so recon metrics collapse (F1 ≈ 1%) without any error.

```sh
git submodule update --init                # after cloning
uv sync                                    # creates .venv
scripts/download_bench.sh [hiroom ...]     # DA3-BENCH into workspace/benchmark_dataset/ (~41 GB download, ~79 GB extracted, for all six)
uv run scripts/run_bench.py depth-anything/DA3-SMALL "eval.datasets=[hiroom]" "eval.modes=[pose]"
scripts/run_bench_loop.sh [overrides]      # SMALL, BASE, LARGE-1.1 in turn, then compare_paper.py
uv run scripts/compare_paper.py [--save]   # ours vs DA3 paper Tables 2/3; --save copies into results/da3_bench/
```

- `workspace/` (gitignored) holds everything large and machine-local: benchmark data, predictions and logs. The DA3 evaluator resolves data paths as `workspace/benchmark_dataset/...` relative to the cwd, which is why `run_bench.py` changes to the repo root.
- `run_bench.py <model> [overrides]` runs `depth_anything_3.bench.evaluator` in-process (overrides are its OmegaConf dotlist, e.g. `eval.datasets=[...]`, `eval.modes=[pose,recon_unposed,recon_posed]`, `eval.eval_only=true`). It writes to `workspace/evaluation/<model name>/` and additionally logs per-call forward time and peak GPU memory to `timings.jsonl` there by wrapping `DepthAnything3.forward`/`inference`, plus one line per invocation (host, GPU, versions, wall time) to `runs.jsonl`. `workspace.work_dir=...` overrides the output directory, so runs of the same model (e.g. with and without TTO) don't overwrite each other.
- Predictions are stored per scene as `model_results/<dataset>/<scene>/{unposed,posed}/exports/mini_npz/results.npz`, and `eval.eval_only=true` re-scores them without inference. A TTO method can rewrite these files in place (same format) and reuse the evaluator's metrics unchanged.
- Committed results: `results/da3_bench/<model>/` (metric JSONs, timing logs, `comparison.md`), created by `compare_paper.py --save`. The paper's Table 2/3 values are hard-coded in `compare_paper.py`, which matches a run directory to them by the `-small`/`-base`/`-large` substring of its name.
- Runtime: inference is cheap, but recon scoring dominates: ~95% of it is the scipy KDTree nearest-neighbour query in DA3's `bench/utils.py`, which is single-threaded. `run_bench.py` patches `KDTree.query` to `workers=-1` (identical metrics, ~7x faster on 16 cores; HiRoom scene 54 s → 8 s). TSDF fusion (~8 s per scene, 4 parallel workers) is the next cost. A full benchmark of one model takes hours, so start long runs in their own tmux session (`tmux new -d -s bench "uv run scripts/run_bench.py ..."`), not as a Claude background task, which dies when Claude exits.
- RAM: TSDF fusion runs `inference.num_fusion_workers` scenes in parallel (default 4). On ETH3D `recon_posed` four parallel fusions exceeded the 61 GB of RAM on tum1 (59.7 GB + 7.9 GB swap peak), and the OOM killer took down the whole tmux session, including the loop. On tum1 it is `systemd-oomd` (user slice, acts on memory pressure before the kernel OOM killer; `journalctl --user | grep oomd` shows its kills): on 2026-10-24 it killed the first full LARGE-1.1 run at a 43 GB peak in the first ETH3D `recon_unposed` fusion, while the process still held ~8 GiB of pinned memory (since removed). After a crash in the scoring phase, resume with `eval.eval_only=true`, since all predictions are already saved. `run_bench_loop.sh` therefore passes `inference.num_fusion_workers=1` (identical results); pass it yourself when calling `run_bench.py` on full datasets. Even with one worker, a single ETH3D `recon_posed` fusion can be close to the limit: fusion runs on the full-resolution DSLR images with `ETH3D_MAX_DEPTH = 100000` (no depth truncation), and with DA3-BASE one scene peaked at 58 GB RAM with the 8 GB swap full (SMALL: 27 GB). Don't run anything else memory-heavy on tum1 during ETH3D recon.
- Baselines: `run_bench.py facebook/VGGT-1B | yyfz233/Pi3 | nvidia/dvlt` runs the adapters in `scripts/baselines.py` instead of DA3 (it replaces `DepthAnything3.from_pretrained` in the evaluator). They ignore the GT cameras of posed mode (none of the models is pose-conditioned; the evaluator fuses their depth with GT poses). VGGT and π³ get DA3's own preprocessing (`da3_preprocess`: longest side 504, cv2 `INTER_AREA`, multiple of 14), as the DA3 paper's baselines did: with it their pose AUC matches Table 2 within 0.6 (except VGGT DTU AUC@30, whose 99.8 in the paper is the 49-view DTU value, apparently a copy error), while their native preprocessing was off by up to 3 points (checked 2026-10-30; run dirs `*_da3pre`). Measured on tum1 at 100 × 7-Scenes frames with native preprocessing: VGGT 7.1 GiB peak (bf16 aggregator weights + `scripts/vggt_lowmem.py`; fp32 weights run out of memory even with the offload), π³ 7.5 GiB (bf16 encoder/decoder; 8.9 GiB reserved), DéjàView 3.9 GiB (`decode_chunk_size=8` instead of the checkpoint's 128, which runs out of memory). bf16 weights move depth by ~0.1% mean relative vs fp32 (24 frames). `vggt_lowmem` and the DéjàView chunking are bit-identical (checked 2026-10-26), and the DéjàView adapter matches upstream `predict()` exactly. VGGT and π³ at 100 frames don't fit on the 3070. `compare_paper.py` has VGGT/π³ values from DA3 Tables 2/3 (matched by `vggt`/`pi3` in the run dir name); DéjàView has no reference in that benchmark.
- Recon scores aren't exactly reproducible: Open3D's `sample_points_uniformly` in `bench/utils.py` is unseeded, so re-scoring the same predictions changes F1 by ~1e-4 (pose metrics are bit-identical).
- GPU budget: the only machines are an RTX 3070 (8 GB) and an RTX 3080 (10 GB). Size every benchmark and TTO run for ≤ 8 GB; DA3-Giant and VGGT-1B at 100 frames don't fit. DA3 models up to LARGE-1.1 fit at 100 frames only with `scripts/da3_lowmem.py` (see below).

### GPU memory: `scripts/da3_lowmem.py`

Without it, DA3-LARGE-1.1 runs out of memory on the 3080 already at 18–40 frames (504×504). `run_bench.py` installs the patch for every model and also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (without it, fragmentation made LARGE reserve 9.2 GiB for 7 GiB of tensors). The patch replaces `DinoVisionTransformer.get_intermediate_layers`, a copy of the third-party loop that normalizes each of the 4 stored output layers on the GPU as soon as it is produced and moves it to CPU memory (pageable: with pinned memory, PyTorch's pinned-block cache kept ~8 GiB of RAM locked through the TSDF fusion afterwards), and wraps `DualDPT._forward_impl` so that each head chunk (8 frames) is moved back to the GPU. In the unpatched code these layers are fp32 `[1, S, N, 2048]` tensors (~1 GiB each at 100 frames), held alongside normalized and reshaped copies.

Checks (2026-10-24, on tum1): outputs (depth, conf, extrinsics, intrinsics) are bit-identical (`torch.equal`) to the unpatched model for SMALL and BASE at 100 frames and for LARGE-1.1 at 18 frames, unposed and posed. Peak allocated memory of one forward pass (bf16 autocast):

| Model, input | unpatched | patched |
|---|---|---|
| SMALL, 100 × 378×504 | 3.92 GiB | 1.95 |
| BASE, 100 × 378×504 | 7.21 | 3.61 |
| LARGE-1.1, 40 / 60 / 100 × 378×504 | 7.18 / 8.46 / OOM | 5.04 / 5.13 / 5.46 (6.83 reserved) |
| LARGE-1.1, 40 / 60 / 100 × 504×504 | 8.86 / OOM / OOM | 6.02 / 6.14 / 6.59 (7.65 reserved) |

The benchmark's largest input is 100 × 378×504 (7-Scenes, ScanNet++); HiRoom is ≤ 23 × 504×504. At 100 frames the remaining peak is in the backbone (weights + residual stream + one block's MLP). A smaller head `chunk_size` (2) only helps below ~60 frames, and it changes depth by ~2e-3, so it isn't used. If something still needs less memory (e.g. LARGE on the 3070 together with TTO), the next step is running the block MLPs over chunks of tokens.

### Reading the PDFs

On Linux, `pdftotext` (poppler) is installed: `pdftotext -f 3 -l 5 docs/DA3.pdf -` prints pages 3 to 5 to stdout. On macOS without poppler, use PDFKit through JXA instead (quote the path):

```sh
osascript -l JavaScript -e "ObjC.import('PDFKit'); var d=\$.PDFDocument.alloc.initWithURL(\$.NSURL.fileURLWithPath('$PWD/docs/VGGT.pdf')); var s=''; for (var i=0;i<d.pageCount;i++) s += d.pageAtIndex(i).string.js + '\n'; s"
```

## Course context that shapes the work

- **Goal:** each project runs as a research project that aims for a **SIGGRAPH 2027** submission. Topic descriptions give a direction, not a fixed recipe. The contribution has to be novel and go beyond the state of the art, backed by quantitative metrics, qualitative results, baselines and ablations.
- **Teams:** 2 people.
- **Deliverables:** project proposal (21.10.26, 23:59) → intermediate presentation (10.12.26) → paper discussion (SIGGRAPH deadline − 2 weeks) → final report (SIGGRAPH deadline) → final presentation and poster session (SIGGRAPH deadline + 1 week). The course will announce the exact SIGGRAPH deadline.
- **Proposal format:** 2 pages in CVPR format. The *Why* part has title, abstract, introduction/motivation and related work. The *How* part has problem formulation, method, datasets, evaluation metrics, baselines, milestones/timeline, expected outcomes and references.
- **Weekly meetings:** 5 min presentation + 10 min Q&A per group, with a focus on visuals. Show one change at a time with its effect: loss curves, dataset visualizations, and before/after results for each change. Each member also posts a personal Slack update before the meeting (last week) and after it (next week). Set experiment code up so that every change produces comparable logged metrics and visuals.
- **Grading:** understanding of the state of the art, methodology and technical contribution, own initiative, experimental evaluation, continuous progress, presentation, report quality, quality of the final deliverables.

## Our project: Project 07 (Finn Ferchau and Antonin Granados)

**Compact feed-forward 3D reconstruction through distillation and test-time optimization.** Central question: does high-quality reconstruction need an accurate initial prediction, or can a compact model followed by test-time optimization reach comparable results at lower overall compute?

Milestones from the kickoff slides:
1. Run an existing feed-forward pipeline (VGGT or DéjàView) and establish baselines for quality, runtime and memory.
2. Distill the teacher into progressively smaller students and evaluate the quality of their initial predictions.
3. Apply test-time optimization to refine geometry and camera parameters using multi-view consistency and image-based constraints.
4. Compare quality under matched compute budgets: which initialization errors can optimization correct, and when is a stronger prediction needed?

### Plan change after the proposal presentation (22.10.26)

We don't have enough compute to distill a large teacher properly, so we dropped distillation (milestone 2) and focus on test-time optimization (TTO):

- **Models:** instead of distilling our own students, we use pretrained models of different sizes, most likely the three smallest any-view Depth Anything 3 models (DA3-Small, DA3-Base and DA3-Large, see below).
- **TTO:** we compare several TTO methods on top of each model. The candidates from the presentation refine cameras and depth maps using reprojection error on correspondences (sparse, robust to a poor initialization, e.g. bundle adjustment), photometric consistency (dense, needs a good initialization) or feature-metric consistency (dense, more robust to lighting and viewpoint). Test3R (`docs/Test3R.pdf`) is related work, though it optimizes network parameters rather than the outputs.
- **Question:** at a matched compute budget (one forward pass plus T TTO iterations), which model size, TTO method and T give the best reconstruction? Can a smaller model with TTO match a larger model's feed-forward prediction?

Tasks for the week of 23.10.26:
- **Finn:** set up the DA3 benchmark and run DA3-Small, DA3-Base and DA3-Large without TTO, reproducing the paper's numbers (Tables 2 and 3).
- **Antonin:** review TTO methods and, if time allows, implement the first one.

Reference numbers (DéjàView Table 3, also on kickoff slide 34; 24 views, one A100, averaged over DTU, ETH3D, 7-Scenes, ScanNet++ and nuScenes):

| Model | Params | TFLOPs | Peak mem | IR % | AUC@30 % |
|---|---|---|---|---|---|
| VGGT | 1257M | 190.0 | 14.7 GiB | 66.0 | 84.0 |
| DA3-L | 356M | 71.4 | 7.3 GiB | 58.3 | 79.7 |
| DéjàView | 117M | 75.9 | 4.9 GiB | 80.3 | 91.8 |

VGGT's IMC results show how much optimization adds on top of a feed-forward prediction. Bundle adjustment raises AUC@3 from 39.2 to 66.4 and adds about 1.6 s (0.2 s → 1.8 s).

Evaluation conventions both papers use: predictions are aligned to ground truth with a Sim(3) transform before scoring. Pointmap quality is reported as Rel. L2 and inlier ratio (relative point error < 3%). Pose quality is AUC@3°/AUC@30° (VGGT also uses AUC@10° on IMC).

### Depth Anything 3

*Depth Anything 3: Recovering the Visual Space from Any Views* (arXiv 2511.10647, `docs/DA3.pdf`). Code is at https://github.com/ByteDance-Seed/Depth-Anything-3. These are the any-view checkpoints on Hugging Face, with parameter counts from the repo README:

| Checkpoint | Params | License |
|---|---|---|
| `depth-anything/DA3-SMALL` | 0.08B | Apache 2.0 |
| `depth-anything/DA3-BASE` | 0.12B | Apache 2.0 |
| `depth-anything/DA3-LARGE-1.1` | 0.35B | CC BY-NC 4.0 |
| `depth-anything/DA3-GIANT-1.1` | 1.15B | CC BY-NC 4.0 |

The `-1.1` checkpoints were retrained after a training bug fix and replace the deprecated `DA3-LARGE` and `DA3-GIANT`. It isn't clear which checkpoints produced the paper's numbers, so if DA3-Large doesn't match, try the deprecated `DA3-LARGE` as well. There is a separate monocular series (`DA3MONO-LARGE`, `DA3METRIC-LARGE`), which we don't use.

**Benchmark:** the repo documents it in `docs/BENCHMARK.md`. The preprocessed data is the Hugging Face dataset `depth-anything/DA3-BENCH`, and the dataset paths are set in `src/depth_anything_3/utils/constants.py`. Run it with `python -m depth_anything_3.bench.evaluator model.path=depth-anything/DA3-SMALL`. To run less, add `eval.datasets=[hiroom]` or `eval.modes=[pose,recon_unposed]`. The doc only shows `DA3-GIANT`, but `model.path` also accepts the smaller variants: a DA3-SMALL pilot on HiRoom matched the paper within 0.5 points.

- **Datasets:** HiRoom, ETH3D, DTU (DTU-64 for pose, DTU-49 for reconstruction), 7-Scenes and ScanNet++. At most 100 frames per scene are used (`max_frames`).
- **Metrics:** the paper reports all four model sizes. Pose is reported as AUC@3/AUC@30 (Table 2). Reconstruction is reported as F1 after TSDF fusion, or Chamfer distance in mm on DTU (Table 3), each without and with ground-truth poses as input. These metrics differ from the VGGT/DéjàView conventions above.