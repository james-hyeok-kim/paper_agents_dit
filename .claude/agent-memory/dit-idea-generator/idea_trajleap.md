---
name: idea-trajleap
description: Latent trajectory extrapolation idea — predict next-step latent with small head, verify with uncertainty gate, skip DiT forward when safe. Temporal axis.
metadata:
  type: project
---

# TrajLeap — Latent Trajectory Extrapolation for Denoising Step Skipping

**Axis**: Temporal (denoising trajectory)
**Status**: Drafted 2026-05-15, not yet checked for novelty

## Hypothesis
Denoising trajectory in latent space is piecewise-smooth; a tiny predictor head can extrapolate `ẑ_{t-1}` from past K latents and skip the full DiT forward when an uncertainty gate accepts.

## Method (one-liner)
Small MLP/1D-conv head (<5M params) trained via distillation on frozen DiT trajectories; ensemble-variance + DiT residual norm as verification gate; skip more aggressively in mid/late timesteps (opposite of DeepCache).

## Differentiator
- vs DeepCache/FORA/PAB: prediction+verification, not cache reuse.
- vs DPM-Solver: skips score evaluations entirely, not just reduces them.
- vs Consistency Models / LCM: frozen base model, only small head trained.

## Expected gain
1.6–2.2x end-to-end latency, FID drift <1.0; orthogonal to DPM-Solver.

## Scores
Novelty 4 / Impact 4 / Feasibility 4 / PublishRisk 4 / Timeline 4

## Key risks
- Error accumulation across consecutive skips.
- Detail loss in late timesteps (mitigation: forbid skip in last N steps).

## Next step
Send to dit-literature-checker; specifically verify against PAB, FORA, ∆-DiT, AsyncDiff, and any 2024–2025 "trajectory prediction for diffusion" works.

Related: [[idea-condmask-dit]] (orthogonal axis), [[idea-specdit]] (different prediction mechanism).
