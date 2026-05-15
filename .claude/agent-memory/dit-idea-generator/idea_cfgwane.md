---
name: idea-cfgwane
description: Per-step adaptive Classifier-Free Guidance schedule via online disagreement signal — halve forward cost on late steps without distillation.
metadata:
  type: project
---

# CFGWane: Disagreement-Gated Per-Step CFG Skipping

## One-line summary
Use online cond/uncond disagreement signal to dynamically skip the unconditional forward in late timesteps, eliminating ~50% of CFG cost without retraining.

## The single move prior work didn't make
CFG distillation (Guidance-Distilled, CFG++, APG) bakes guidance into weights at training time. Static CFG-interval schedules (Adaptive Projected Guidance, CFG Scale Rectifier) drop CFG at fixed step ranges. **Nobody runs an online disagreement statistic between cond/uncond branches and uses it as a per-step, per-sample gate to predict when uncond can be linearly extrapolated from prior steps' delta** — keeping the model frozen and the schedule sample-adaptive.

## Mechanism (single sentence)
At each step t, compute a cheap proxy of `||eps_cond - eps_uncond||` from the previous step; if below threshold, skip the uncond forward and reuse the previous step's guidance vector with linear decay.

## Discriminating tests (pre-loaded)
1. **Reduce-to test**: threshold→0 collapses to standard CFG (no skip). Threshold→∞ collapses to CFG-free. Mid-range is genuinely new since the gate is per-sample, not a fixed schedule.
2. **Single-mechanism test**: One mechanism — disagreement-thresholded uncond skipping with linear extrapolation of guidance vector. Pass.
3. **Ceiling test**: CFG doubles every step's forward. Skipping uncond on late half of steps yields ~25% e2e wallclock saving. Skipping 70% of late steps yields ~35%. Above 1.5x feasible only when paired with step reduction; standalone ceiling is ~1.4x. Acceptable but not blockbuster — paper viable, not flagship.
4. **One-paper-kills-this test**: A "online adaptive CFG via disagreement" paper from CVPR 2026 would kill it. Last verified: APG (NeurIPS 2024) and CFG++ (ICLR 2025) use static schedules. Need lit-checker confirmation.

## Why this works in CFG-not-distilled regime
- SD3, FLUX-dev, PixArt-Σ all ship with CFG required at inference
- Distilled variants (FLUX-schnell, SD3-Turbo) are CFG-free, so this targets the ~70% of deployed DiTs that still pay CFG cost
- CFG distillation requires retraining; this is plug-and-play

## Quantitative ceiling
- CFG accounts for exactly 50% of inference FLOPs
- Late-step skipping budget: realistic 40-60% of steps
- Expected e2e: 1.2-1.4x speedup, near-lossless FID (<0.5 degradation)
- If pushed aggressive: 1.5x with measurable but acceptable quality loss

## Scores
- Novelty: 3 (axis is open but adjacent work crowded)
- Impact: 3 (capped at ~1.4x, but applies to majority of deployed DiTs)
- Feasibility: 5 (no training, single forward modification)
- Publish risk: 4 (likely positive results, low risk of negative)
- Timeline: 5 (4-6 weeks to paper)

## Recommended next step
Run dit-literature-checker on "CFGWane: online disagreement-gated CFG skipping" to verify novelty against APG, CFG++, Guidance-Distilled, and any 2025-2026 adaptive CFG papers.
