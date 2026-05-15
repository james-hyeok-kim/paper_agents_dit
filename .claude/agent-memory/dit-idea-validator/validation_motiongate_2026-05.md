---
name: validation-motiongate-2026-05
description: MotionGate (latent-flow region-adaptive Video DiT compute) — NO-GO at top-tier; HSA (arXiv:2605.06892, May 7 2026) preempts the per-token motion-gating + skipped-token reuse mechanism. Conditional GO only via occlusion-aware pivot.
metadata:
  type: project
---

# Validation: MotionGate (Latent-Flow Region-Adaptive Video DiT)

**Date:** 2026-05-15
**Verdict:** NO-GO at CVPR/ICCV/NeurIPS tier. CONDITIONAL GO at workshop tier with the occlusion-aware pivot below.

## Why: decisive prior art

**HSA — "Not All Tokens Need 40 Steps" (Chu & Patel, arXiv:2605.06892, 2026-05-07, ~8 days before this validation)** independently proposes:
- Per-spatial-token gating in Video DiT
- L1-relative velocity-change signal as the "motion proxy" for skip decision
- Skipped-token reuse via cached Euler update (the structural equivalent of MotionGate's warp-and-reuse)
- KV-cache synchronization so active tokens still attend to skipped ones
- Validated on Wan-2.1/2.2 and LTX-2

MotionGate's three pillars — (a) per-token motion gating, (b) skip+reuse, (c) Video DiT focus — are all covered. The only differentiators that survive HSA:
1. Signal source: distilled flow head (~2M params) vs HSA's free self-signal velocity-diff
2. Reuse mechanism: optical-flow warping vs Euler extrapolation
3. Forward-backward consistency for occlusion masking — the **only genuinely new element**

Differentiator (1) is a **disadvantage**, not an advantage: HSA's signal is free; MotionGate adds 2M params + 3D conv head to do the same job.
Differentiator (2) is **fragile**: V-Warper (2025) explicitly shows intermediate diffusion features warp poorly; reliable warping requires query-key features. Distilled flow at 2M params is unlikely to clear that bar reliably.

## How to apply

If the user wants to proceed, push toward this **single salvageable pivot**:

> **Occlusion-aware skipped-token recovery as a drop-in module on top of HSA.**
> - Accept HSA as the baseline; do NOT compete on the gating signal.
> - Use distilled flow ONLY for forward-backward consistency-based occlusion detection (small head justified by this niche role).
> - In occlusion regions, override HSA's skip with full forward (or short re-denoise).
> - Drop warping entirely — keep HSA's Euler extrapolation, which has manifold-validity guarantees the warp does not.
> - Minimum bar for top-tier: dramatic quality gap in occlusion regions (e.g., HSA + occlusion-mask vs HSA: VBench motion-smoothness or PSNR-on-occlusion-tokens +significant delta) at near-equal speedup.
> - Realistic venue: workshop (CVPR-W/ICCV-W). Top-tier requires the gap to be visually striking on standard occlusion-heavy benchmarks (VBench dynamic-degree subset, DAVIS-derived, etc.).

## Other prior art surfaced during this validation (cross-link to [[reference-token-pruning-prior-art]])

- **FLoED (arXiv:2412.00857)** — "flow attention cache" using optical-flow-guided latent warping for diffusion. Inpainting domain but the warping-for-reuse mechanism is published.
- **WAVE (ECCV 2024)** — DDIM inversion feature warping for zero-shot video editing.
- **LatentWarp** — warping latent features across frames using flow.
- **V-Warper (arXiv:2512.12375)** — empirical evidence that intermediate features warp poorly; query-key features are the right targets. Directly weakens MotionGate's warping premise.
- **ADAPTOR (CVPR 2025W)**, **AdaCache+MoReg (ICCV 2025)**, **VGDFR (ICCV 2025)**, **FIS-DiT** — already in user's lit-check.

## Pattern instance

This is another instance of [[pattern-combination-as-contribution]]: distilled flow head + warp-based reuse + FB-consistency = three published mechanisms stitched, with the genuinely novel piece (FB-consistency for occlusion) buried as the third bullet rather than headlined.
