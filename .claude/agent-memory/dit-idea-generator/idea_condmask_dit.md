---
name: idea-condmask-dit
description: Cross-attention-driven token sparsification — use prompt-token attention entropy to drop spatial tokens per timestep, training-free. Spatial axis.
metadata:
  type: project
---

# CondMask-DiT — Cross-Attention-Driven Token Sparsification

**Axis**: Spatial (token pruning) with conditioning signal
**Status**: Drafted 2026-05-15, not yet checked for novelty

## Hypothesis
Spatial tokens with high-entropy cross-attention over prompt tokens are semantically inactive at the current timestep and can be excluded from self-attention without quality loss.

## Method (one-liner)
Per-step importance `s_i = max_j A[i,j] - λ·H(A[i,:])` from cross-attn maps; keep top-(1-r_t)N tokens active; reactivate every 5 steps; identical mask on CFG conditional/unconditional branches.

## Differentiator
- vs ToMe: similarity-based merge, semantic-blind. CondMask uses prompt relevance.
- vs AT-EDM / SparseDM: activation-magnitude based, condition-blind.
- vs PixArt-Σ KV compression: training-time. CondMask is inference-only.

## Expected gain
1.4–1.8x speedup at 50% token retention; FID +0.3–0.8; CLIP score preserved.

## Scores
Novelty 4 / Impact 4 / Feasibility 5 / PublishRisk 4 / Timeline 5

## Key risks
- Dense multi-object prompts → most tokens important → small gain (mitigation: hybrid with ToMe).
- Limited transfer to video DiT where cross-attn is weaker.

## Next step
Send to dit-literature-checker; check against ToMe-SD, AT-EDM, SparseDM, DynamicDiT, and any 2024–2025 cross-attention-guided pruning works.

Related: [[idea-trajleap]] (temporal axis, combinable), [[idea-specdit]] (orthogonal).
