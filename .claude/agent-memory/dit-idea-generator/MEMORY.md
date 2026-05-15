## Round 1 (2026-05-15) — All hit prior art issues
- [TrajLeap](idea_trajleap.md) — Temporal axis: small head extrapolates next latent, uncertainty gate decides whether to skip DiT forward.
- [CondMask-DiT](idea_condmask_dit.md) — NO-GO: cross-attention token pruning fully covered by AT-EDM/IBTM/ToPi/CAT Pruning/DaTo.
- [SpecDiT](idea_specdit.md) — CONDITIONAL: SpeCa (ACM MM 2025) already did forecast+verify. Trained drafter dismissed by Yoon et al. (ICLR 2025).

## Round 2 (2026-05-15) — Pivot to mined-out-resistant regimes (3 different axes per advisor guidance)
- [CascadePrompt](idea_cascadeprompt.md) — Few-step DiT: confidence-gated text encoder cascade (CLIP→T5-base→T5-XXL). Targets conditioning pathway, not DiT loop. Only meaningful in 1-8 step regime.
- [MotionGate](idea_motiongate.md) — Video DiT: latent-space flow-guided region-adaptive computation. Frame-axis spatial redundancy, orthogonal to PAB's timestep-axis caching.
- [DecoupledVAE](idea_decoupledvae.md) — VAE pathway: mid-trajectory partial decode + learned residual refiner overlapped with remaining DiT steps. New axis entirely outside DiT loop and tokens.

## Round 3 (2026-05-15) — Three orthogonal axes (no video-DiT, single-mechanism, discriminating tests pre-loaded)
- [ControlAsync](idea_controlasync.md) — Axis A: asymmetric timestep schedule for ControlNet/IP-Adapter conditioning branch (decay-curve asymmetry trunk vs. conditioning).
- [LatentPrefix](idea_latentprefix.md) — Axis B: cross-request semantic latent-trajectory prefix cache for diffusion serving (MLSys-target).
- [MMBlockAttn](idea_mmblockattn.md) — Axis C: calibrated low-rank kernel for MM-DiT cross-modal attention sub-blocks (statistics-driven, not pattern-driven).

## Round 4 (2026-05-15) — Four-axis fill to reach 12 total (advisor-guided: A/C/D/E, skip B-solver-theory)
- [CFGWane](idea_cfgwane.md) — Axis A (CFG): online cond/uncond disagreement gate skips uncond branch on late steps; targets ~50% CFG cost without distillation.
- [ConvergeSense](idea_convergesense.md) — Axis C (per-sample stop): latent-velocity convergence + prompt-embedding-clustered threshold calibration for per-sample early termination.
- [TimeWarp](idea_timewarp.md) — Axis D (systems): denoising-loop-aware activation prefetch/offload for single-GPU DiT (MLSys/ASPLOS target, VRAM-throughput Pareto shift).
- [StructPrior](idea_structprior.md) — Axis E (conditioning structure): zero-cost trunk attention sparsity derived from pose/depth/mask conditioning geometry.
