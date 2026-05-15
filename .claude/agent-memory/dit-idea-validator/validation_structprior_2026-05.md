---
name: validation-structprior-2026-05
description: StructPrior (conditioning-geometry → trunk attention sparsity) — Conditional Go ONLY if trunk-attention cost share + training-free trunk mask quality verified; multi-modality and general-DiT-ControlNet differentiators already closed by EVCtrl.
metadata:
  type: project
---

# StructPrior validation (2026-05-15)

## Verdict
🟡 CONDITIONAL GO — strict gates. Lean toward NO-GO at top-tier if either gate fails.

## Decisive prior art
- **EVCtrl (arXiv:2508.10963)** — training-free, multi-modality (Canny/HED/OpenPose/MLSD), works on FLUX/CogVideo/Wan2.1, 1.95-2.13x on Flux. **Closes** the "multi-modality 통합" and "general DiT-ControlNet" novelty claims. **But** modifies only the ControlNet branch (mid-to-late fresh_blocks), not trunk self-attention.
- **IASA (arXiv:2510.02617, SIGGRAPH Asia 2025)** — pose-keypoint correspondence → trunk self-attention mask (global+local). Domain-limited (co-speech video), requires distillation. NOTE: distillation rationale is many-step→few-step acceleration, NOT "naive sparse mask fails quality". The user's lit-check inferred wrong reason.
- **AdaSpa (ICCV 2025)** — training-free adaptive sparse attention for video DiT, query-balanced.
- **SpargeAttn** — training-free generic sparse attention on Flux/SD3.5/Mochi.

## Surviving novelty surface
ONLY: training-free **trunk** self-attention sparsification driven by **conditioning input geometry** (zero-cost prior, no profiling pass needed), evaluated on **image-domain** general DiT-ControlNet. Narrow but real.

## Why "zero-cost" claim is shakier than user thinks
- IASA's distillation is NOT evidence that naive sparse mask fails (it's a few-step distillation). User's lit-check has a misread.
- But that does not mean naive mask works either — image-domain pose mask quality under training-free sparse attention is unverified.
- Naive risk: pose keypoints (10-30 points) are too sparse to define a self-attention mask; depth/segmentation provide denser geometry but dilation strategy is non-trivial. Quality drop is plausible.

## Top 2 blocker gates (must pass before full project commit)
1. **Trunk-cost share gate**: profile FLUX-ControlNet wallclock — what % is trunk self-attention vs ControlNet branch? If <40%, e2e ceiling is too low (Amdahl) and the whole "trunk 1.6-1.8x → e2e 1.3-1.5x" budget collapses. EVCtrl's branch-side targeting suggests the branch may dominate.
2. **Naive mask quality gate**: 1-day pilot — image-domain geometric mask (pose-dilated, depth-thresholded, segmentation-region) vs full attention on FLUX-ControlNet, FID/CLIP/LPIPS delta. If quality drops >2 FID at 50% sparsity, "training-free zero-cost" is dead and falls back to IASA-style distillation (which kills the differentiator).

## Composability question
If trunk-share is ~40-60%, the only viable framing is StructPrior **stacked on EVCtrl** (trunk-side complement). Then contribution becomes "orthogonal trunk-side geometric sparsification, composes with branch-side EVCtrl". Multiplicative speedup story. But this requires reframing — currently positioned as standalone replacement.

## Reviewer kill shot
"Method is essentially IASA's pose-keypoint attention mask, removed from co-speech video domain to image-domain ControlNet. EVCtrl already provides training-free multi-modality acceleration on the same models. Trunk vs branch split changes implementation locus but the core insight — geometry of conditioning predicts attention sparsity pattern — is IASA's. Distillation in IASA is for step-count, not mask quality, so the 'training-free' framing is not a contribution either."

## Strongest version
- Reframe as: **"Trunk-side geometric sparsity for DiT-ControlNet, compositional with branch-side caching (EVCtrl)"**.
- Add: theoretical/empirical analysis of WHEN dense conditioning (depth/seg) vs sparse (pose/scribble) yields valid trunk mask without training. Failure-mode characterization is the contribution.
- Multiplicative speedup demonstration: EVCtrl-only vs StructPrior-only vs stacked on FLUX-ControlNet.
- Drop pose from the headline modality (overlaps IASA most). Lead with depth+segmentation where geometry is dense.

## Minimum publishable bar
- e2e 1.3x+ on FLUX-ControlNet **stacked on top of EVCtrl** (multiplicative), <0.5 FID delta, demonstrated on 3+ control modalities (depth/seg/scribble), fully training-free, on 2+ models (FLUX + SD3 or CogVideoX-Controlnet).
- Without composition story, top-tier is a no.

## Pattern note
This is another instance of [[pattern-combination-as-contribution]]: pose mask (IASA) + multi-modality framework (EVCtrl) + training-free (SpargeAttn/AdaSpa). The narrow surviving differentiator (trunk-side image-domain geometric mask) needs an empirical or theoretical insight, not just relocation.
