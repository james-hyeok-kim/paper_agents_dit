---
name: structprior-novelty-2026-05
description: StructPrior — pose/depth/seg conditioning geometry as zero-cost trunk DiT attention sparsity prior; partial overlap leaning conflict
metadata:
  type: project
---

# StructPrior — Conditioning-Geometry-Derived Trunk Attention Sparsity

Date checked: 2026-05-15
Verdict: PARTIAL OVERLAP leaning CONFLICT

## Core idea
Use ControlNet conditioning input geometry (pose keypoints, depth map, segmentation mask, scribble) as a free spatial prior to derive a sparse self-attention mask in the trunk DiT, training-free. Goal: trunk 1.6-1.8x, e2e 1.3-1.5x.

## Key prior art

1. **IASA — Input-Aware Sparse Attention for Real-Time Co-Speech Video Generation** (arXiv:2510.02617, SIGGRAPH Asia 2025). 🔴 DIRECT for pose modality.
   - Uses pose keypoints to derive sparse attention mask: global mask = K most-similar past frames by pose alignment, local mask = anatomical region (face/hands/arms) constrained inter-frame attention via rigid moving least squares.
   - Mask applied to self-attention.
   - Limitations vs StructPrior: requires distillation training (~16h teacher+student), pose-only, video co-speech only, not zero-cost, not framed as ControlNet trunk acceleration.

2. **EVCtrl** (arXiv:2508.10963, 2025). 🟡 PARTIAL on ControlNet acceleration.
   - Profiles DiT-ControlNet layer responses → partitions into global/local zones, locality-aware caching focuses computation where control signal matters; 2.05-2.16x on CogVideo/Wan2.1-Controlnet.
   - Differs: uses cache-skip not attention sparsity; inferred locality via post-hoc layer profiling, not direct geometry of conditioning input.

3. **Region-Adaptive Sampling (RAS)** (arXiv:2502.10389, ICLR 2025). 🟡 PARTIAL.
   - Region-adaptive sampling rates derived from preceding-step attention/noise metrics, not conditioning geometry. Up to 2.36-2.51x.
   - Differs: temporal continuity from output (not zero-cost from conditioning input), no ControlNet-specific framing.

4. **SGSFormer — Segmentation Guided Sparse Transformer** (arXiv:2403.05906). 🟡 PARTIAL conceptual analog.
   - Uses instance segmentation maps as prior to guide sparse self-attention regions; image restoration domain (under-display camera), not DiT acceleration.

5. **AT-EDM, DaTo, Sparse-vDiT, LiteAttention, Compact Attention, SpargeAttn, FG-Attn, VSA, Radial Attention, CalibAtt, Trainable Dynamic Mask Sparse Attention**: ⬜ all derive sparsity from internal attention statistics, content patterns, or layer-position priors — none from external conditioning geometry.

6. **ZestGuide, FreeControl, Layered Rendering Diffusion**: orthogonal — use attention masks for spatial control of generation, not for compute reduction.

## Overlap analysis

The exact recipe "pose keypoints → trunk DiT self-attention sparsity mask, training-free" is heavily occupied by IASA, even though IASA requires distillation. The conceptual move "external geometric conditioning → trunk attention sparsity" is also covered by SGSFormer (pre-DiT era) and partially by EVCtrl (locality-zone ControlNet caching).

What residual novelty exists:
- IASA needs distillation; StructPrior claims zero-cost (training-free).
- IASA is pose-only and co-speech-domain; StructPrior unifies pose/depth/seg/scribble.
- EVCtrl uses caching, not attention-mask sparsification.
- RAS doesn't tap conditioning input.

Risks:
- Reviewers will treat StructPrior as a training-free generalization of IASA. The novelty delta (training-free + multi-modality unified framework + general DiT-ControlNet) is real but narrow.
- Need calibrated evidence that pose 5-10% trunk attention is sufficient without quality drop — IASA had to distill to recover quality, suggesting naive zero-cost mask may be too aggressive.

## Recommendation

Differentiate. Emphasize:
1. Training-free (no distillation, no fine-tune) vs IASA's 16h distill.
2. Unified multi-modality framework (pose+depth+seg+scribble), not just pose.
3. General DiT-ControlNet pipeline (FLUX/SD3-Controlnet, CogVideoX-Controlnet) not narrow co-speech.
4. Quantitative comparison vs EVCtrl (caching) vs IASA (distilled-pose) on ControlNet workloads.

If quality recovery requires any tuning, immediately becomes redundant with IASA. Validate zero-cost claim early.
