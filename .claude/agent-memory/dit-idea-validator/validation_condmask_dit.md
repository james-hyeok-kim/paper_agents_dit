---
name: validation-condmask-dit
description: Validation record for CondMask-DiT (cross-attention-driven token sparsification). NO-GO at top-tier venues; only 3 ablation-grade differentiators remain after literature overlap. Pivot menu attached.
metadata:
  type: project
---

# CondMask-DiT Validation (2026-05-15)

## Verdict: NO-GO for top-tier (NeurIPS/ICLR/CVPR), CONDITIONAL GO for workshop only

## Scores
- Novelty 1/5 — every sub-component published (AT-EDM CA-WPR, IBTM CA importance, ToPi periodic reactivation, CAT timestep schedule, DaTo CFG-branch alignment).
- Feasibility 5/5 — would obviously work as engineering; no technical blocker.
- Publishability 1/5 at top tier; 3/5 at a workshop.
- Scope 2/5 — narrow + crowded.

## Why "remaining differentiators" don't rescue it
- CFG mask sharing: DaTo (arXiv:2501.00375) already aligns base-token positions across CFG branches. "Systematic comparison" is a half-section, not a paper.
- `max - λH` closed-form analysis: this is analysis of a score function whose information content (max + entropy of attention rows) is already inside G-WPR's row-statistics family. Even if no prior work writes the literal scalar, naming and analyzing a scalar isn't a top-tier insight.
- Text-token axis aggregation ablation: pure ablation. AT-EDM already considered SA vs CA aggregation and chose SA — the user must beat that result, not just ablate.

## Pivot menu (offered to user)
1. Failure-mode pivot: identify regime where AT-EDM/IBTM/ToPi/CAT/DaTo all fail (guidance-distilled DiT, video-DiT temporal coherence, rectified-flow few-step). Build method targeted at that hole.
2. Compositionality pivot: token pruning x feature caching (TeaCache/FORA) — additive, sub-additive, destructive? Genuinely understudied.
3. Theory pivot: bound on FID/CLIP degradation as f(retention rate r_t, attention entropy). None of the cited works have this.
4. Signal pivot: leave cross-attention (mined out). Try per-token loss-gradient norm, cross-timestep token trajectory variance, manifold curvature as importance signal.

See [[pattern-combination-as-contribution]] for the recurring failure shape and [[reference-token-pruning-prior-art]] for verified citation IDs.
