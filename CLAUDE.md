# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

This directory has no code, build system or tests yet. It is a git repository whose `main` branch tracks `origin/main` at git@github.com:F-Fer/TUM-practical-compact-3d-distillation.git. It holds three reference PDFs for the TUM research practical course **"Generative AI: From Diffusion to World Models"** (WS 2026/27, run by Umut Kocasari and Robin Borth):

- `Generative AI From Diffusion to World Models.pdf`: the course kickoff slides, covering structure, deadlines and the 8 project topics.
- `VGGT.pdf`: *VGGT: Visual Geometry Grounded Transformer* (Wang et al., CVPR 2025, arXiv 2503.11651). It is a feed-forward transformer that predicts cameras, point maps, depth maps and point tracks for 1 to hundreds of views in one pass. Code is at https://github.com/facebookresearch/vggt.
- `Déjà-View.pdf`: *DéjàView: Looping Transformers for Multi-View 3D Reconstruction* (NVIDIA, arXiv 2605.30215). It applies one shared transformer block recurrently for K refinement steps, and K is a compute knob you set at inference time. It matches much larger feed-forward models with far fewer parameters.

When code is added, update this file with the real setup, training and evaluation commands.

### Reading the PDFs

`poppler` isn't installed, so the Read tool cannot render these PDFs, and `pypdf` and `fitz` are missing too. Use macOS PDFKit through JXA to extract text instead (quote the paths, since the filenames contain spaces and non-ASCII characters):

```sh
osascript -l JavaScript -e "ObjC.import('PDFKit'); var d=\$.PDFDocument.alloc.initWithURL(\$.NSURL.fileURLWithPath('$PWD/VGGT.pdf')); var s=''; for (var i=0;i<d.pageCount;i++) s += d.pageAtIndex(i).string.js + '\n'; s"
```

## Course context that shapes the work

- **Goal:** each project runs as a research project that aims for a **SIGGRAPH 2027** submission. Topic descriptions give a direction, not a fixed recipe. The contribution has to be novel and go beyond the state of the art, backed by quantitative metrics, qualitative results, baselines and ablations.
- **Teams:** 2 people.
- **Deliverables:** project proposal (21.10.26, 23:59) → intermediate presentation (10.12.26) → paper discussion (SIGGRAPH deadline − 2 weeks) → final report (SIGGRAPH deadline) → final presentation and poster session (SIGGRAPH deadline + 1 week). The course will announce the exact SIGGRAPH deadline.
- **Proposal format:** 2 pages in CVPR format. The *Why* part has title, abstract, introduction/motivation and related work. The *How* part has problem formulation, method, datasets, evaluation metrics, baselines, milestones/timeline, expected outcomes and references.
- **Weekly meetings:** 5 min presentation + 10 min Q&A per group, with a focus on visuals. Show one change at a time with its effect: loss curves, dataset visualizations, and before/after results for each change. Each member also posts a personal Slack update before the meeting (last week) and after it (next week). Set experiment code up so that every change produces comparable logged metrics and visuals.
- **Grading:** understanding of the state of the art, methodology and technical contribution, own initiative, experimental evaluation, continuous progress, presentation, report quality, quality of the final deliverables.

## Our project: Project 07 (team of 2)

**Compact feed-forward 3D reconstruction through distillation and test-time optimization.** Central question: does high-quality reconstruction need an accurate initial prediction, or can a compact model followed by test-time optimization reach comparable results at lower overall compute?

Milestones from the kickoff slides:
1. Run an existing feed-forward pipeline (VGGT or DéjàView) and establish baselines for quality, runtime and memory.
2. Distill the teacher into progressively smaller students and evaluate the quality of their initial predictions.
3. Apply test-time optimization to refine geometry and camera parameters using multi-view consistency and image-based constraints.
4. Compare quality under matched compute budgets: which initialization errors can optimization correct, and when is a stronger prediction needed?

Reference numbers (DéjàView Table 3, also on kickoff slide 34; 24 views, one A100, averaged over DTU, ETH3D, 7-Scenes, ScanNet++ and nuScenes):

| Model | Params | TFLOPs | Peak mem | IR % | AUC@30 % |
|---|---|---|---|---|---|
| VGGT | 1257M | 190.0 | 14.7 GiB | 66.0 | 84.0 |
| DA3-L | 356M | 71.4 | 7.3 GiB | 58.3 | 79.7 |
| DéjàView | 117M | 75.9 | 4.9 GiB | 80.3 | 91.8 |

VGGT's IMC results show how much optimization adds on top of a feed-forward prediction. Bundle adjustment raises AUC@3 from 39.2 to 66.4 and adds about 1.6 s (0.2 s → 1.8 s).

Evaluation conventions both papers use: predictions are aligned to ground truth with a Sim(3) transform before scoring. Pointmap quality is reported as Rel. L2 and inlier ratio (relative point error < 3%). Pose quality is AUC@3°/AUC@30° (VGGT also uses AUC@10° on IMC).
