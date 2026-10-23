# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

This directory has no code, build system or tests yet. It is a git repository whose `main` branch tracks `origin/main` at git@github.com:F-Fer/TUM-practical-compact-3d-distillation.git. `proposal/` holds the LaTeX project proposal (CVPR template, one file per section in `proposal/sec/`, bibliography in `proposal/main.bib`). It is written in Overleaf, and `preamble.tex` and `cvpr.sty` exist only there, so it doesn't compile locally as-is. `presentation/` holds the Beamer slides (`slides.tex`) of the proposal presentation on 22.10.26, built locally with latexmk. Both still describe the original distillation plan (see "Plan change" below).

`docs/` holds reference PDFs (gitignored; each clone downloads them from the links in `README.md`) for the TUM research practical course **"Generative AI: From Diffusion to World Models"** (WS 2026/27, run by Umut Kocasari and Robin Borth):

- `Generative AI From Diffusion to World Models.pdf`: the course kickoff slides, covering structure, deadlines and the 8 project topics.
- `VGGT.pdf`: *VGGT: Visual Geometry Grounded Transformer* (Wang et al., CVPR 2025, arXiv 2503.11651). It is a feed-forward transformer that predicts cameras, point maps, depth maps and point tracks for 1 to hundreds of views in one pass. Code is at https://github.com/facebookresearch/vggt.
- `Déjà-View.pdf`: *DéjàView: Looping Transformers for Multi-View 3D Reconstruction* (NVIDIA, arXiv 2605.30215). It applies one shared transformer block recurrently for K refinement steps, and K is a compute knob you set at inference time. It matches much larger feed-forward models with far fewer parameters.
- `OTT3R.pdf`: *OTT3R* (Leblanc & Poullis, ACCV 2026, arXiv 2609.36374). It distills π³ (959M) into a 102M student on 2 GPUs, using an offline cache of teacher pseudo-labels. It is purely feed-forward with no test-time optimization, and it is the closest prior work. Code is at https://github.com/TheFourthKaramazov/OTT3R.

When code is added, update this file with the real setup, training and evaluation commands.

### Reading the PDFs

`poppler` isn't installed, so the Read tool cannot render these PDFs, and `pypdf` and `fitz` are missing too. Use macOS PDFKit through JXA to extract text instead (quote the paths, since the filenames contain spaces and non-ASCII characters):

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
- **TTO:** we compare several TTO methods on top of each model. The candidates from the presentation refine cameras and depth maps using reprojection error on correspondences (sparse, robust to a poor initialization, e.g. bundle adjustment), photometric consistency (dense, needs a good initialization) or feature-metric consistency (dense, more robust to lighting and viewpoint). Test3R is related work.
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

*Depth Anything 3: Recovering the Visual Space from Any Views* (arXiv 2511.10647, not in `docs/` yet). Code is at https://github.com/ByteDance-Seed/Depth-Anything-3. These are the any-view checkpoints on Hugging Face, with parameter counts from the repo README:

| Checkpoint | Params | License |
|---|---|---|
| `depth-anything/DA3-SMALL` | 0.08B | Apache 2.0 |
| `depth-anything/DA3-BASE` | 0.12B | Apache 2.0 |
| `depth-anything/DA3-LARGE-1.1` | 0.35B | CC BY-NC 4.0 |
| `depth-anything/DA3-GIANT-1.1` | 1.15B | CC BY-NC 4.0 |

The `-1.1` checkpoints were retrained after a training bug fix and replace the deprecated `DA3-LARGE` and `DA3-GIANT`. It isn't clear which checkpoints produced the paper's numbers, so if DA3-Large doesn't match, try the deprecated `DA3-LARGE` as well. There is a separate monocular series (`DA3MONO-LARGE`, `DA3METRIC-LARGE`), which we don't use.

**Benchmark:** the repo documents it in `docs/BENCHMARK.md`. The preprocessed data is the Hugging Face dataset `depth-anything/DA3-BENCH`, and the dataset paths are set in `src/depth_anything_3/utils/constants.py`. Run it with `python -m depth_anything_3.bench.evaluator model.path=depth-anything/DA3-SMALL`. To run less, add `eval.datasets=[hiroom]` or `eval.modes=[pose,recon_unposed]`. The doc only shows `DA3-GIANT`, so it is untested whether `model.path` accepts the smaller variants.

- **Datasets:** HiRoom, ETH3D, DTU (DTU-64 for pose, DTU-49 for reconstruction), 7-Scenes and ScanNet++. At most 100 frames per scene are used (`max_frames`).
- **Metrics:** the paper reports all four model sizes. Pose is reported as AUC@3/AUC@30 (Table 2). Reconstruction is reported as F1 after TSDF fusion, or Chamfer distance in mm on DTU (Table 3), each without and with ground-truth poses as input. These metrics differ from the VGGT/DéjàView conventions above.
