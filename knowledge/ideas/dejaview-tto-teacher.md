# Does fine-tuning DéjàView towards TTO-refined cameras/depth ("TTO as teacher") beat plain self-distillation?
Status: rejected (at this budget) · Area: DéjàView training · Updated 2026-11-18
**Answer:** No. On ARKitScenes clips, photometric TTO (T = 25) improves the released model's cameras (ARKit GT
AUC@3 18.9 → 24.4, AUC@30 79.5 → 81.5, 954 clips), but a 1.75 h fine-tune towards those targets (C1) scores 58.8 alone /
71.8 after TTO on all 81 pose scenes (K = 12, fit extrinsics + head intrinsics). That ties the control with the plain
teacher as target (C0: 59.6 / 71.9) and the released model (58.7 / 71.7). All three are within noise.

## Evidence
| date | setting (model, set, seeds, baseline) | change | result | source (report §, run names) |
|---|---|---|---|---|
| 2026-11-18 | released dvlt 504 px K = 16, fit E + head K, 1000 random ARKit train clips (16-32 views, seed 7) | `run_tto_pba.refine_scene` iters = 25 as target generator | vs ARKit GT (slerped lowres_wide.traj): AUC@3 18.9 → 24.4, AUC@30 79.5 → 81.5; TTO focal ≈ GT (~418 px at 504) | `scripts/distill_steps/gen_tto_targets.py`, `workspace/distill_data/tto_targets/` |
| 2026-11-18 | fine-tune released dvlt, 504 px, K_s ∈ {12, 16}, lr 8e-6, 1.75 h on 3080 (937 steps), eval K = 12 fhk, 81 pose scenes, TTO T = 25 | C1: rays / depth / pose_enc targets = teacher + TTO correction (`train_res.py --targets`) | 58.8 alone, **71.8** after TTO (held-out 71.1); HiRoom/7Sc/SN++/DTU/ETH3D alone 76.1/29.1/74.9/78.2/35.8 | `workspace/distill_runs/C1`, `C1_K12_fhk/compare_e.md` |
| 2026-11-18 | same, plain teacher targets (control) | C0 | 59.6 alone, **71.9** after TTO (held-out 71.3); 77.2/30.4/73.0/80.0/37.2 | `workspace/distill_runs/C0`, `C0_K12_fhk` |
| 2026-11-18 | released K = 12 fhk (baseline) | none | 58.7 alone, 71.7 after TTO (held-out 71.0) | `rel_K12_fhk` |

## Open / don't repeat
- The TTO correction on ARKit is small (+5.5 AUC@3 vs GT on clips where the teacher alone is at 18.9). It is also
  partly noise, and most of the gain is focal, which head-K + TTO already recovers at test time. A larger effect needs
  stronger targets (longer NCC/LM TTO, track BA, or ARKit GT itself) and more steps than 937.
- Training against ARKit GT poses directly (they are already downloaded with the frames) is the untried variant.
- C0 ≈ released shows that a short 504 px fine-tune at K_s ∈ {12, 16} does not by itself make K = 12 better.
Related: [dejaview-speed](dejaview-speed.md), [dejaview-few-step-distillation](dejaview-few-step-distillation.md)
