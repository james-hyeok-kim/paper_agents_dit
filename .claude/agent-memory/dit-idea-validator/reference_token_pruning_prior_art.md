---
name: reference-token-pruning-prior-art
description: Verified arXiv IDs and the specific mechanism each prior work contributes for DiT/diffusion token pruning (used to assess novelty overlap).
metadata:
  type: reference
---

# Verified prior art for token-pruning in diffusion / DiT

All arXiv IDs verified by WebSearch/WebFetch on 2026-05-15.

- AT-EDM (arXiv:2405.05252, CVPR 2024) — G-WPR PageRank-style score on attention; explicitly evaluates SA-WPR vs CA-WPR and prefers SA-WPR (CA prunes too many background tokens). DSAP timestep schedule: prune fewer early, more late.
- IBTM (arXiv:2411.16720) — importance = |eps_cond - eps_uncond| (CFG magnitude), training-free token *merging* (not pure pruning). Targets U-Net LDMs, not DiT directly.
- ToPi (arXiv:2602.01609) — DiT-specific, training-free, periodic mask updates at interval ΔT=10, anchor timesteps evenly spaced. Score = sum over layers/heads of ||v|| · attention reception. Does NOT discuss CFG mask sharing.
- CAT Pruning (arXiv:2502.00433) — token-pruning + caching, "noise relative magnitude" score, spatial clustering for distributional balance, 50-60% FLOPs reduction. Staleness-aware reactivation.
- DaTo (arXiv:2501.00375, "Token Pruning for Caching Better") — DiffScore = mean abs diff between consecutive timesteps. Explicitly aligns base-token positions across CFG conditional/unconditional branches. 9x speedup on SD v1.5. U-Net, not DiT.

Quick-lookup table of which sub-mechanism each work owns:
- Cross-attention as importance signal -> AT-EDM CA-WPR, IBTM (via CFG diff), ToPi.
- Periodic reactivation -> ToPi (fixed ΔT).
- Staleness-aware reactivation -> CAT Pruning.
- Timestep-aware retention schedule -> AT-EDM DSAP, CAT Pruning.
- CFG branch mask sharing/alignment -> DaTo.

# Verified prior art for Video DiT acceleration (added 2026-05-15)

- HSA — "Not All Tokens Need 40 Steps" (arXiv:2605.06892, 2026-05-07, Chu & Patel). Per-spatial-token gating using L1-relative velocity-change as the skip signal; cached Euler update propagates skipped tokens; KV-cache sync keeps active tokens attending to inactive ones. Validated on Wan-2.1/2.2 (1.3B and 14B) and LTX-2. **Owns: per-token motion-style gating + skipped-token reuse for Video DiT.**
- FIS-DiT (arXiv:2605.11869) — Frame-Interleaved Sparsity, 2.11–2.41x speedup on video DiT, motion-aware *not* (frame-axis only).
- Sparse VideoGen (arXiv:2502.01776) — 3D attention sparsity classification (Spatial vs Temporal heads), training-free, 2.28–2.33x on CogVideoX-1.5 / HunyuanVideo.
- Sparse-vDiT (arXiv:2506.03065) — sparse attention for video DiT.
- S²Q-VDiT (OpenReview e8pm93koQU) — quantization + sparse-token distillation for video DiT.
- BlockDance (CVPR 2025) — reuses spatio-temporal features in shallow/middle DiT blocks.
- FLoED (arXiv:2412.00857) — flow-attention-cache + optical-flow-guided latent warping. Inpainting domain, but the warping-for-reuse mechanism is published.
- WAVE (ECCV 2024) — DDIM inversion feature warping.
- LatentWarp (OpenReview) — flow-based latent warping across frames.
- V-Warper (arXiv:2512.12375) — empirical: intermediate diffusion features warp poorly, query-key features are the right targets. **Important**: weakens any "warp intermediate features" proposal.
- Foresight — adaptive layer reuse for DiT, shows static reuse fails to preserve critical updates.

Sub-mechanism ownership for Video DiT:
- Per-spatial-token gating + skipped-token cached update -> HSA.
- Frame-axis sparsity -> FIS-DiT, VGDFR.
- Motion-aware caching schedule (per-timestep, not per-token) -> AdaCache+MoReg.
- Per-column temporal cosine-sim compression -> ADAPTOR.
- Spatial-temporal attention head specialization -> Sparse VideoGen.
- Latent warping for reuse -> FLoED, WAVE, LatentWarp.
- Evidence against intermediate-feature warping -> V-Warper.
