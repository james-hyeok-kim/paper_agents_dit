---
name: convergesense-novelty-2026-05
description: ConvergeSense (per-sample early stop via latent convergence + prompt-cluster threshold) novelty check — partial overlap leaning conflict
metadata:
  type: project
---

# ConvergeSense Novelty Verdict: PARTIAL OVERLAP (leaning CONFLICT)

Idea (Round 4): Per-sample early stop based on latent trajectory relative velocity dropping below a prompt-cluster-calibrated threshold for k consecutive steps. Offline calibration builds a CLIP/T5-cluster -> threshold lookup. No DiT retraining. Target 1.3-1.5x e2e.

## Direct conflicts / strong overlaps found

- **AdaDiff (arXiv 2311.14768)** — "AdaDiff: Adaptive Step Selection for Fast Diffusion Models". Per-prompt instance-specific step selection via learned policy network on text embedding; achieves 33-40% inference reduction. Same problem framing (easy prompts -> fewer steps). Differs only in mechanism (learned policy vs runtime convergence + cluster lookup).
- **AdaDiff (ECCV 2024, arXiv 2309.17074, Tang et al.)** — uncertainty-estimation per timestep used to terminate inference early. 43.6% compute reduction. Per-step adaptive termination, not per-prompt clustering, but same "early-terminate-when-converged" core.
- **StepSaver (arXiv 2408.02054)** — NLP model fine-tuned to predict optimal denoising steps from prompt text. Offline: build prompt -> optimal-step dataset using SSIM-based first-decline criterion. Same offline-calibration-then-prompt-conditioned-step-count idea. Differs in: predictor (NLP head) vs cluster lookup; uses SSIM-on-decoded-image not latent-velocity.
- **Denoising Diffusion Step-aware Models (DDSM, OpenReview c43FGk8Pcg)** — sample-adaptive compute allocation across denoising steps.
- **A Simple Early Exiting Framework (arXiv 2408.05927)** — early-exiting in score network parameters, time-dependent exit schedule.
- **Adapt and Diffuse (arXiv 2309.06642)** — sample-adaptive reverse-step count via latent-space severity encoding (8-10x reduction). Inverse-problem setting, but the "per-sample step count adapts to a latent-derived difficulty signal" core matches.
- **Optimal Stopping in Latent Diffusion Models (arXiv 2510.08409)** — theoretical framework for early-stopping LDMs, latent-dimension-dependent optimal termination.
- **Denoising, Fast and Slow / Patch Forcing (arXiv 2604.19141, CVPR 2026)** — per-patch difficulty head -> adaptive sampler allocates compute where needed.

## What overlaps with ConvergeSense

1. Per-sample adaptive step count: covered by AdaDiff (both), StepSaver, DDSM, Adapt-and-Diffuse, AdaDiff-2311.
2. Prompt-conditioned offline calibration of step count: StepSaver does exactly this (text -> step count via supervised mapping from offline labels). Cluster-vs-regression is a minor implementation detail.
3. Latent-convergence-based runtime stopping: AdaDiff-ECCV (uncertainty as proxy), Adapt-and-Diffuse (severity in latent), Optimal Stopping in LDMs (theoretical) all touch this.
4. No retraining: most of the above (Optimal Stopping, Adapt-and-Diffuse partial, runtime convergence baselines) are training-free or only train a small head.

## Residual novelty angles (thin)

- Combining (a) runtime latent-velocity monitor + (b) prompt-cluster-specific calibrated threshold (vs StepSaver's global regressor or AdaDiff's learned policy) is a specific recipe not exactly published.
- DiT-specific (most prior work is U-Net SD); rectified-flow velocity field interpretation gives a clean signal.
- "k consecutive steps" debounce + offline cluster calibration is a concrete engineering recipe.
- However the conceptual delta from StepSaver + AdaDiff-ECCV combined is small — likely viewed by reviewers as an engineering combination, not a new idea.

## Verdict

**PARTIAL** (leaning CONFLICT for top-tier venue). The conceptual core — per-sample step count adapted from latent/uncertainty signal with offline prompt calibration — is well-covered. The cluster-lookup-table + velocity-threshold + DiT-specific framing is differentiable but reviewers will flag StepSaver and AdaDiff (both versions) as direct prior art.

## Recommendation

DIFFERENTIATE strongly or DEPRIORITIZE. To survive, the idea would need:
- Direct head-to-head against StepSaver and AdaDiff-2311 with clear win on quality-vs-speed Pareto on DiT (FLUX, SD3, PixArt).
- A novel theoretical hook (e.g., velocity-field-based stopping rule for rectified flow specifically, with error bounds).
- Or pivot to compose with caching/distillation (orthogonality story) rather than stand alone.

Otherwise abandon in favor of less-crowded ideas in the round 4 batch.
