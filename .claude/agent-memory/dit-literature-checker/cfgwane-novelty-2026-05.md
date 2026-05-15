---
name: cfgwane-novelty-2026-05
description: CFGWane (disagreement-gated per-step CFG skipping w/ prior-step guidance extrapolation) — RED conflict; Castillo et al. AAAI'24 AG/LinearAG covers both core mechanisms exactly
metadata:
  type: project
---

# CFGWane novelty check — 2026-05-15

## Idea recap
- Online cond/uncond disagreement statistic as per-sample/per-step gate
- Dynamically skip uncond forward at late timesteps
- When skipped, linearly extrapolate using prior step's guidance vector
- Frozen model, training-free, no distillation
- Target: ~50% CFG cost removal, 1.2-1.4x e2e

## Decisive prior art

### Castillo et al., "Adaptive Guidance: Training-free Acceleration of Conditional Diffusion Models" (arXiv:2312.12487, AAAI 2024)
- **AG variant**: gating signal is cosine similarity gamma_t := <eps_c, eps_u> / (||eps_c|| ||eps_u||) — i.e., disagreement between cond and uncond
- Decision is per-sample per-step — switches from CFG to conditional-only when gamma_t exceeds threshold (skipping uncond forward)
- **Training-free, frozen model**, plug-and-play, ~25% compute reduction
- **LinearAG variant**: replaces uncond network evaluation with linear regression over PAST cond and uncond predictions — eps_hat(x_t, null) = sum beta_i^c eps(x_i, c) + sum beta_i^null eps(x_i, null)
- This is exactly "prior step guidance vector linear extrapolation" with OLS coefficients

This single paper covers BOTH core CFGWane claims:
1. Per-sample disagreement-gated uncond skip → AG with cosine-similarity gate
2. Linear extrapolation from prior-step guidance → LinearAG

### Other strong overlaps
- **OUSAC (arXiv:2512.14096, Dec 2025)**: jointly optimizes which timesteps to skip uncond + guidance scale, eliminates up to 82% of uncond passes; 53% compute savings on DiT-XL/2, 5x on FLUX. Uses evolutionary search (offline schedule), but covers the "skip uncond at many steps" headline.
- **Step AG (arXiv:2506.08351, 2025)**: fixed-schedule CFG-only-in-first-portion, 20-30% speedup. Critiques cosine-similarity gating as not generalizing across models.
- **Dynamic CFG via Online Feedback (arXiv:2509.16131, 2025)**: per-sample per-step CFG **scale** selection via latent evaluators — does not skip uncond, but is the per-sample per-step adaptive CFG paper of 2025.
- **GalaxyDiT (arXiv:2512.03451, Dec 2025)**: per-sample dynamic step skipping for both cond+uncond synchronously with proxy-based gate; explicitly addresses misalignment when uncond is reused asymmetrically (which is exactly the failure mode CFGWane could encounter with linear extrapolation).

## Verdict
RED CONFLICT. Castillo et al. AG (cosine-disagreement-gated per-sample uncond skip) + LinearAG (past-step linear extrapolation of uncond) covers the entire mechanism. CFGWane's two pillars are both already published, in the same paper, training-free, on diffusion models. 1.2-1.4x e2e target is comparable to or below AG's claimed 25% compute reduction.

## What MIGHT salvage residual novelty (slim)
- DiT/MM-DiT specific re-validation (AG was on UNet SD); but trivial extension
- Per-sample online coefficient fitting (vs LinearAG's offline OLS on 200 trajectories) — possibly novel framing but small delta
- Tighter integration with feature caches (AdaCache/TeaCache) — but combinatorial, not a contribution alone

Recommendation: **Abandon** as a standalone idea. Either (a) pivot to a sharper differentiator vs AG/LinearAG (e.g., online per-sample coefficient adaptation with provable error bounds), or (b) generate new ideas via dit-idea-generator.
