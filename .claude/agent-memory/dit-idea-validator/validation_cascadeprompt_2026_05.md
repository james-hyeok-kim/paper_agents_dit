---
name: validation-cascadeprompt-2026-05
description: CascadePrompt (3-tier prompt-adaptive text encoder cascade for FLUX-schnell) validated 2026-05-15 — Conditional Go pending pilot histogram and wallclock verification.
metadata:
  type: project
---

# CascadePrompt validation (2026-05-15)

Verdict: CONDITIONAL GO. Overall 3.4/5.

## Why the motivation survives despite prior work
The two static-replacement papers actually *strengthen* the cascade motivation when read carefully:
- arXiv:2503.19897 reports FID +1.96, CLIP −1.51, char-accuracy 69.3% vs 76.7%; explicitly excludes text rendering from the parity claim. No per-prompt variance reported.
- NanoFLUX (arXiv:2602.06879): DPG drops 82.6 → 76.3 (−6.3 absolute, large). No per-prompt variance reported.

Both report only averages. Neither analyzes long/compositional/rare-entity prompts. The cascade's defensible contribution is therefore: **"static distillation hides where the gap lives; characterizing the per-prompt heavy tail enables a cascade that Pareto-dominates static distillation."**

## Load-bearing assumptions to verify before scale-out
1. T5-XXL really is 20–25% of FLUX-schnell 4-step wallclock on target hardware. With FP8 / cached encoder / repeat-prompt batching this can collapse to <10%, killing the contribution.
2. Per-prompt CLIP-score (or DPG) delta between T5-base-distilled and T5-XXL on stratified prompts (simple / compositional / rare-entity / long / text-rendering) is heavy-tailed, not a uniform shift. This single histogram is the discriminating pilot.
3. Confidence gate cannot rely on encoder-side signals alone (entropy, projector recon error). Failure modes like attribute binding and text rendering are silent at embedding level. Gate must be trained against a downstream quality delta signal (CLIP / VQA on rendered images).
4. Right target model: FLUX.2 may have changed the encoder stack; verify before committing to FLUX.1-schnell.

## Right vs wrong baseline
- Wrong: cascade vs T5-XXL only (easy win).
- Right: cascade vs static-distilled T5-base at iso-quality. Must show either (a) match T5-XXL quality at >2× speedup, or (b) strictly dominate the static frontier.

## Operating-point sensitivity (must be first-class)
If escalation rate to T5-XXL must exceed ~30% to maintain quality, amortized speedup collapses. Report the latency-quality frontier as a curve, not a single point.

## Common failure pattern check
Echoes [[pattern-combination-as-contribution]] (FrugalGPT + 2503.19897 + RouteT2I). The new empirical insight must be the per-prompt heavy-tail characterization, not "first adaptive cascade for DiT."
