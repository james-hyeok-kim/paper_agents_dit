---
name: validation-convergesense-2026-05
description: ConvergeSense (per-sample early stop via latent velocity + cluster threshold) — NO-GO at top-tier; arithmetic gap (1.18x vs claimed 1.3-1.5x), StepSaver-equivalent supervision, full coverage by AdaDiff/AdaFlow/A2S.
metadata:
  type: project
---

# ConvergeSense Validation (2026-05-15)

**Verdict: NO-GO at top-tier (CVPR/NeurIPS); workshop-only conditional pivot possible.**

## Scores
| Dim | Score | Concern |
|---|---|---|
| Novelty | 2/5 | Velocity-as-signal partially overlaps AdaFlow (variance-as-signal) and A2S (acceleration-aware). Cluster-LUT is StepSaver-equivalent supervision. |
| Feasibility | 2/5 | **Internal arithmetic inconsistency**: 0.7*1.0 + 0.3*0.5 = 0.85 → 1.18x e2e, NOT 1.3-1.5x. To hit 1.3x needs 50% prompts at 50% steps; to hit 1.5x needs 30% at 80% saving — far more aggressive than calibrated convergence delivers without quality loss. |
| Publishability | 2/5 | StepSaver/AdaDiff/Adapt-and-Diffuse cover the *task*; AdaFlow/A2S cover the *velocity-derived adaptive sampling* mechanism. Residual = "velocity-norm vs variance-scalar vs UEM-uncertainty" head-to-head — measurable but thin. |
| Scope | 3/5 | Reasonably scoped but degenerates into ablation study without a new claim. |
| **Overall** | **2.25/5** | NO-GO at top-tier |

## Combination-as-Contribution Trigger (per [[pattern-combination-as-contribution]])
Sub-mechanisms inventory:
1. Per-prompt step prediction — StepSaver (2408.02054), AdaDiff (2311.14768), AdaDiff-ECCV (2309.17074)
2. Sample-adaptive step count — Adapt-and-Diffuse (2309.06642, inverse-problem setup, partial fit)
3. Velocity/variance-based adaptive solver — AdaFlow (NeurIPS'24), A2S (acceleration-aware sampling)
4. Cluster-based threshold table — classical
5. k-step debounce — trivial

Five published or trivial pieces. Pattern triggers: burden of proof shifts to "what is empirically true *because* of the combination" — the proposal does not articulate this.

## Top 3 Risks
1. **Arithmetic gap**: claimed 1.3-1.5x is mathematically incompatible with the stated "30% prompts at 50% steps" assumption. Mitigation: re-derive realistic ceiling (~1.18x) or change assumption to 50%@50% / 30%@80%, then *prove* such aggressive savings are achievable on real prompts without FID/CLIP drop.
2. **Supervision-equivalence to StepSaver**: cluster-LUT calibrated on held-out data is information-theoretically a `prompt → step` map identical to StepSaver's NLP head. Reviewers will not credit "no learned head" as contribution. Mitigation: drop the framing; instead position as *signal comparison* paper.
3. **AdaFlow / A2S preemption of velocity-derived signal**: variance-scalar (AdaFlow) and acceleration-decomposition (A2S) already exploit derived velocity-field statistics for per-sample step adaptation in flow models. Mitigation: must show *velocity-norm beats both* in head-to-head, on the *same* per-sample early-stop task, in DiT/RF specifically.

## Devil's Advocate
Hostile reviewer: "This is StepSaver with k-means swapped for fine-tuning, plus AdaFlow's variance signal swapped for raw velocity norm. The 1.3-1.5x speedup claim is not reproducible from the stated 30%/50% assumption. No new insight; no theoretical justification; no head-to-head against AdaFlow."

Killer result: AdaFlow's variance scalar matches or beats velocity-norm at predicting per-sample convergence in head-to-head — kills the only residual claim.

## Strongest Version (Pivot)
**Either** (a) reframe as "Which signal class best predicts per-sample convergence in rectified-flow DiTs?" — head-to-head benchmark of {NLP-head, RL-policy, UEM-uncertainty, AdaFlow-variance, A2S-acceleration, latent-velocity-norm, noise-pred-stability} on FLUX/SD3/CogVideoX. This is a *diagnostic* contribution (workshop-strong, top-tier weak unless the finding is surprising).
**Or** (b) abandon ConvergeSense and pursue a different signal source (e.g., cross-attention-entropy collapse, classifier-free-guidance-disagreement) where prior art is genuinely thinner.

## Minimum Bar (if user insists on pursuing)
- Realistic e2e speedup ≥1.3x with FID delta ≤0.5 AND CLIP delta ≤0.01 on COCO-30k (FLUX) and VBench (CogVideoX)
- Head-to-head beats AdaFlow + AdaDiff + StepSaver at same quality target
- Calibration set for LUT must be ≤1k prompts (else "training-free" claim collapses)
- Show that velocity-norm provides information *not in* CLIP embedding (else cluster-LUT alone suffices and the runtime monitor is redundant)

## Decision
**NO-GO at top-tier.** Recommend the user either (i) pivot to diagnostic study (workshop), or (ii) abandon and rotate to a different efficiency axis. Do NOT proceed to dit-experiment-planner under current framing.
