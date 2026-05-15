---
name: idea-mmblockattn
description: A DiT-statistics-aware attention kernel for MM-DiT — exploits empirical low-rank structure of the text-image cross-block, not a hand-coded sparsity pattern. Axis C (DiT-specific kernel).
metadata:
  type: project
---

# MMBlockAttn — A Statistics-Aware Attention Kernel for MM-DiT Joint Attention

**Axis**: C (DiT-specific attention kernel).
**One-line claim**: MM-DiT's joint attention matrix is partitioned into [text-text | text-image; image-text | image-image] sub-blocks; the off-diagonal cross-modal sub-blocks are **empirically near-low-rank** (effective rank ≪ d_head) across timesteps and prompts, while the on-diagonal image-image block is dense. A kernel that fuses **dense image-image attention + low-rank cross-modal attention** in one pass beats FA3 with a generic sparsity mask, because the speedup comes from the *statistical* structure, not a fixed pattern.

## Core Hypothesis
For SD3 / FLUX MM-DiT layers, perform an SVD-style profiling of the cross-modal attention sub-blocks Q_text K_image^T and Q_image K_text^T across a calibration set. Hypothesis: ≥80% of Frobenius energy lies in r ≤ 16 components per layer (typical d_head = 64-128), and r is *stable across prompts* (with small per-layer offline calibration sufficient). Therefore: cross-modal attention can be computed via a fused low-rank projection (cost O(N_text · r + N_image · r)) instead of full O(N_text · N_image · d_head), with quality loss controllable by per-layer rank choice.

## Technical Approach
1. **Offline calibration**: For each MM-DiT layer ℓ and head h, collect Q_text K_image^T over 1k prompts × selected timesteps. Compute SVD; pick r_{ℓ,h} as smallest r with ≥ε energy. Store the right-singular projection matrices (per-layer, per-head, fixed).
2. **Inline kernel** (Triton): Within one fused attention call, decompose Q, K splits into text/image, then:
   - image-image: standard FA3 dense.
   - text-text: standard FA3 dense (small; N_text ≪ N_image).
   - cross terms: project K_image (or K_text) through fixed rank-r basis once, attend in low-rank space, scatter. Both cross blocks reuse the same calibration basis.
3. **Numerical stability**: Use online softmax across all four sub-blocks fused, so logits are normalized correctly without materializing the full attention matrix.
4. **No retraining required**: calibration is post-hoc; the kernel is a drop-in for MM-DiT joint-attention modules.
5. **Validate the statistical claim by ablation**: Replace the *learned* low-rank basis with a *random* rank-r projection. Hypothesis demands that learned basis significantly outperforms random — this is the experiment that distinguishes statistics-driven from generic sparsity.

## Key Innovation (the single move prior work didn't make)
- **FA2 / FA3**: generic dense kernels, modality-agnostic.
- **SpargeAttn / SparseDiT / MoBA**: *pattern-based* sparsity (block-sparse, dilated, hand-designed). Speedup from skipping computation, not from exploiting low-dimensional structure.
- **Linear attention (Performer, Linformer)**: low-rank but with *learned-during-training* projections — requires retraining the entire model and degrades at inference for pretrained models.
- **MMBlockAttn**: the first kernel that exploits *empirically-measured*, *post-hoc-calibrated* low-rank structure of the *cross-modal sub-blocks specifically* — a property unique to MM-DiT's joint-attention design that doesn't exist in single-stream LLMs or non-MM DiTs.

This is precisely the move SpargeAttn didn't make: instead of learning a sparsity mask, calibrate a low-rank basis from real activation statistics on a fixed pre-trained model.

## Pre-hoc Discriminating Test
**Reduction question 1**: "Does this reduce to FA3 + a structured sparsity mask?"
**Breaking experiment**: Compare against (a) FA3 dense (baseline), (b) FA3 + block-sparse cross-modal (any predefined pattern), (c) MMBlockAttn (calibrated low-rank). Claim breaks unless (c) achieves better quality at equal FLOPs than (b) at all sparsity ratios — because (b) cannot capture the statistical structure.

**Reduction question 2**: "Is this just Linformer applied to one sub-block?"
**Breaking experiment**: Linformer requires training the projection. Compare random-projection (Linformer-init), calibrated-projection (ours), and trained-projection (Linformer-trained on a held-out FLUX subset). Hypothesis: calibrated ≈ trained, both ≫ random. If random is competitive, the "statistical structure" claim is false and the idea collapses to "any low-rank works."

**Empirical existence check (pilot, 1 day)**: On 50 prompts, dump cross-modal attention sub-blocks from FLUX, run SVD, plot energy curve. The whole idea is contingent on this curve being *steep* — if it's flat, kill the idea immediately.

## Differentiators from Closest Work
- **SpargeAttn (2025)**: hand-designed sparsity patterns, no calibrated low-rank.
- **Sparse VideoGen**: pattern-based, video-specific.
- **Linformer / Performer**: requires training, loses pretrained weights' value.
- **FlexAttention (PyTorch)**: framework for masks, not a low-rank kernel.
- **Q-DiT / TDQ**: quantization, orthogonal axis.

## Expected Gains
- Cross-modal attention is ~30-50% of MM-DiT attention FLOPs (depending on N_text and N_image). If r=16 vs d=128, that block is 8× cheaper.
- End-to-end attention speedup: 1.4-1.8× on FLUX/SD3 at high resolution.
- End-to-end inference speedup: 1.15-1.3×.
- Quality target: <0.3 FID delta at default ε=0.95.

## Feasibility (3/5)
- **Hardest**: writing a fused Triton kernel for the 4-sub-block joint online-softmax. ~3 weeks for an experienced kernel author.
- Calibration pipeline: ~3 days.
- Could pilot the *statistical* claim in 1 day with PyTorch-only baseline (no kernel needed) by computing what the speedup *would be* if the kernel existed. This decouples idea-validation from kernel-engineering risk.
- A100/H100 access required; FA3 is H100-optimal.

## Publication Target
**MLSys / NeurIPS systems track / ASPLOS**. The kernel-engineering depth and the calibration-vs-pattern empirical contrast suit a systems-flavored ML venue. Could also fit ICLR if pitched as "attention statistics in MM-DiT" methodology paper.

## Risk Factors
- **Highest risk**: the cross-modal sub-blocks may *not* be near-low-rank in FLUX (they may be near-low-rank only in specific layers or only at specific timesteps). Mitigation: per-layer adaptive r is part of the design — degrade gracefully to dense for high-rank layers.
- **Kernel engineering risk**: fusing 4 sub-block patterns with online softmax is non-trivial. Mitigation: target *unfused* prototype first to validate quality, then optimize the kernel.
- A recent paper might have noticed MM-DiT block structure already — pre-flight: search "MM-DiT attention sparsity", "FLUX attention low-rank", "joint attention efficient SD3".

## Pivot Path if Conditional
If "low-rank cross-modal" is already taken: pivot to **per-timestep adaptive rank** — show that r should *grow* through denoising (early steps need less cross-modal info than late detail-binding steps). Keep the kernel; reframe the contribution as the timestep-adaptive rank schedule.

## Scores
- Novelty: 4 (low-rank is old; *calibrated low-rank specifically on MM-DiT cross-blocks* is new)
- Impact: 4 (1.15-1.3× end-to-end on a hot model class)
- Feasibility: 3 (kernel engineering is the gate)
- Publish Risk (higher = safer): 3 (depends on the SVD energy curve being steep; pilot mitigates)
- Timeline (higher = faster): 3 (3-month build for kernel+eval)

## Recommended Next Step
Run dit-literature-checker on "MMBlockAttn" with priority queries:
1. "MM-DiT joint attention sparsity / low-rank"
2. "FLUX SD3 attention acceleration kernel"
3. "calibrated low-rank attention diffusion post-hoc"
4. "cross-modal attention block structure efficient diffusion"
