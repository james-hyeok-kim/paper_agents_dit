---
name: idea-structprior
description: Use structural priors of conditioning maps (depth/pose/mask) to derive trunk attention sparsity at zero cost — conditioning structure leaks into trunk computation.
metadata:
  type: project
---

# StructPrior: Conditioning-Structure-Derived Trunk Attention Sparsity

## One-line summary
For ControlNet/T2I-Adapter pipelines with structured conditioning (depth, pose, mask), derive a static spatial sparsity pattern from the conditioning map and apply it to the *trunk* DiT's self-attention — no attention pattern learning, no per-step adaptation, conditioning structure is the prior.

## The single move prior work didn't make
- EVCtrl, OminiControl2: cache the *ControlNet branch* — attacks conditioning compute
- AT-EDM, IBTM, DaTo: prune trunk tokens via cross-attention or merging — pattern is learned per-image online
- Sparse attention (SpargeAttn, Sparse VideoGen): patterns from attention statistics, not from conditioning content
- **No prior work uses the conditioning map's geometric structure (e.g., pose keypoints define ~50 spatial neighborhoods, depth defines smoothness regions) as a *zero-cost prior* for trunk attention sparsity**

## Mechanism (single sentence)
For a given conditioning map C, compute a sparsity mask M(C) once (e.g., dilate pose keypoints to local windows; quantize depth into K planes; mask = identity within plane), then apply M(C) as a fixed attention mask in trunk DiT for all timesteps.

## Discriminating tests (pre-loaded)
1. **Reduce-to test**: M(C) = full attention collapses to standard DiT. M(C) = local windows ignoring C collapses to swin-style local attention (existing). The interesting band is conditioning-shape-aware masks, distinct from both.
2. **Single-mechanism test**: One mechanism — conditioning-map-derived static attention mask. Pass.
3. **Ceiling test**: Pose-conditioning has ~10-30 keypoints → ~5-10% of full attention sufficient. Depth has 5-10 planes → ~10-20% of full attention. Trunk attention is ~40-60% of DiT FLOPs at high resolution. Realistic e2e: 1.4-1.8x for pose, 1.3-1.5x for depth, 1.6-2.0x for sparse mask. Above 1.5x ceiling met.
4. **One-paper-kills-this test**: A "ControlNet-aware sparse attention" paper would kill it. EVCtrl/OminiControl2 attack different surface (conditioning branch, not trunk attention). Need lit-check.

## Why this is robust to "leakage to learned patterns"
- Most learned-sparsity work assumes online discovery → per-sample compute overhead
- Here the mask is **derived** from input data (pose JSON, depth tensor) at zero overhead
- This makes the contribution "we found a free lunch from conditioning structure that prior work missed"

## Modality-specific scope (publishable narrowness)
- Target modalities: pose, depth, segmentation mask, scribble — all have clear geometric structure
- Not target: text (no geometry), image-to-image (full image is condition)
- Narrow scope = faster paper, defendable claims

## Quantitative ceiling
- Pose-controlled FLUX: 1.6-1.8x trunk speedup → 1.3-1.5x e2e
- Depth-controlled SD3: 1.3-1.5x trunk → 1.2-1.3x e2e
- Open-pose video DiT: 2.0x+ possible with temporal mask reuse

## Scores
- Novelty: 4 (using conditioning content as trunk attention prior is genuinely new)
- Impact: 3 (limited to structured-conditioning pipelines, but those are 30%+ of production T2I)
- Feasibility: 4 (mask construction is image-processing; integration with FlashAttention straightforward)
- Publish risk: 3 (hard prompts with fine details may suffer — need careful calibration; risk of degradation in unmasked regions)
- Timeline: 4 (6-8 weeks)

## Risk factors
- Static mask may hurt long-range attention needed for global coherence — mitigation: hybrid local-mask + few-token global anchors
- Mask quality depends on conditioning quality — noisy depth maps could degrade results
- Quality loss in unmasked regions during late steps when fine details emerge

## Recommended next step
Run dit-literature-checker on "StructPrior: conditioning-geometry-derived trunk attention sparsity for ControlNet DiT" to verify against EVCtrl, OminiControl2, ControlNeXt, and 2025-2026 sparse attention work.
