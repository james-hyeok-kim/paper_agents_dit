---
name: idea-specdit
description: Speculative denoising — shallow draft DiT generates γ future latents, full DiT verifies them in one batched forward, accept/reject per residual. Structural-speculative axis.
metadata:
  type: project
---

# SpecDiT — Speculative Denoising with Shallow Draft Transformer

**Axis**: Structural / speculative (LLM speculative decoding ported to diffusion)
**Status**: Drafted 2026-05-15, not yet checked for novelty

## Hypothesis
A small distilled draft DiT can speculate γ consecutive denoising steps; the target DiT verifies all γ candidates in a single batched forward, accepting until the first velocity-residual violation.

## Method (one-liner)
Half-depth/half-width draft trained by velocity + hidden distillation; γ-step rollout; batched target forward; acceptance criterion `||v_target - v_draft|| / ||v_target|| < τ`; rollback on first reject; γ adapts via EMA of accept rate.

## Differentiator
- vs Progressive distillation / LCM: target quality preserved (verification), not collapsed into draft.
- vs [[idea-trajleap]]: uses neural draft model + parallel verification, not analytical extrapolation. More aggressive but higher training cost.
- vs Predictor-Corrector / Restart sampling: explicit two-model system, mirroring LLM speculative decoding.

## Expected gain
2.0–3.0x wall-clock speedup at γ=3, accept rate ~70%; FID +0.5–1.5.

## Scores
Novelty 5 / Impact 5 / Feasibility 2 / PublishRisk 3 / Timeline 2

## Key risks
- Acceptance rule design IS the contribution — wrong threshold tanks quality.
- Draft training cost (10s–100s GPU-hours).
- Video-DiT extension non-trivial (temporal coherence in draft).

## Next step
Send to dit-literature-checker; verify against AsyncDiff, ParaDiGMS, "speculative diffusion" 2024–2025 papers, and any draft-target diffusion systems.

Related: [[idea-trajleap]] (predicting future latents, weaker mechanism), [[idea-condmask-dit]] (orthogonal, combinable).
