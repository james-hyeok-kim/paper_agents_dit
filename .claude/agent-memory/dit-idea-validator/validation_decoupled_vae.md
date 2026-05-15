---
name: validation-decoupled-vae
description: DecoupledVAE (mid-trajectory partial decode + residual refiner + CUDA-stream overlap) — Conditional Go pending three hard pilot gates; top-tier risk dominated by single-GPU bandwidth contention, k-choice / residual-size empirics, and overlap with DiffuseVAE-family refiner prior art.
metadata:
  type: project
---

# DecoupledVAE validation (2026-05)

## Verdict
CONDITIONAL GO with three hard gates. Without all three passing, this is a workshop paper at best, and the combination-as-contribution failure mode (see [[pattern-combination-as-contribution]]) applies.

## Top blockers (must resolve before compute commit)
1. **k = 0.4N is unjustified** — x0-estimate at t≈0.6 is typically still noisy on SD3/Flux/CogVideoX. If k must move to 0.55–0.7N for a usable estimate, the overlap window collapses and the speedup math breaks. Required pilot: LPIPS(D(x̂_0(z_k)), D(x̂_0(z_N))) curve as a function of k on the actual target model.
2. **Single-GPU memory bandwidth contention** — CUDA streams give logical concurrency, not bandwidth. CogVideoX VAE decode is HBM-bound (causal 3D conv >60% of decoder time, OOM-prone on H100). Concurrent DiT-attention + VAE-decode on one GPU typically yields 1.1–1.3× wall-clock, not 1.8×. PipeDiT uses **two GPU groups** specifically to dodge this. Required pilot: microbenchmark concurrent DiT-block + VAE on one H100.
3. **Refiner prior-art overlap** — Hoogeboom et al. and Birodkar et al. already use diffusion to refine autoencoder residuals; DiffuseVAE is the canonical generator→refiner paradigm. The differentiator must be precisely articulated as "intermediate-step partial decode + refiner" not just "refiner-on-VAE-output."

## Secondary risks
- Mixed-modality refiner input: pixel-space partial decode + latent-space residual lack a shared coordinate system. A 30M model re-learning the VAE projection from scratch is optimistic; no published precedent for this asymmetric setup.
- Video temporal consistency: 3D causal VAE does temporal upsampling. Partial decode at k=0.4N can lock in temporally inconsistent motion the refiner cannot cheaply undo. Video DiT is therefore the *harder* case, not the safe haven.
- Fallback path: makes expected latency probabilistic and the fallback path is *worse* than baseline because partial decode is sunk cost. Need fallback-rate estimate and a way to predict fallback before spending the partial decode.

## Hard gates (run in order, kill if fail)
- Gate A (~1 day): single-H100 microbenchmark of concurrent DiT-block + VAE-decode. Need ≥ 1.3× overlap factor vs. serial.
- Gate B (~2 days): LPIPS(D(x̂_0(z_k)), D(x̂_0(z_N))) curve on CogVideoX. Need a k where LPIPS is small (target < ~0.15) AND ≥ 30% of DiT steps remain to overlap.
- Gate C (~1 week): toy 30M refiner on synthetic (partial, full) pairs. Must beat "use a faster VAE" baselines (Flash-VAED 6.16× speedup on Wan 2.1, Turbo-VAED) at fixed latency.

## Novelty status
- Genuinely novel mechanism: intra-prompt mid-trajectory partial decode overlapped with remaining DiT steps + residual refiner. No exact match in literature.
- Closest neighbors:
  - PipeDiT (2511.12056): inter-prompt VAE/DiT pipelining on **two GPU groups** — orthogonal mechanism but owns the "decouple VAE from DiT" framing.
  - DiffusionBrowser (2512.13690): multi-branch decoder for interactive previews from intermediate features. No CUDA-stream overlap, no residual refiner, no end-to-end speedup claim — orthogonal.
  - DiffuseVAE / Hoogeboom / Birodkar: generator→refiner paradigm — refiner pillar is occupied territory; differentiation must be empirical and precise.
  - Flash-VAED / Turbo-VAED: orthogonal "make the VAE itself faster" — these are baselines, not prior art, but they cap the achievable speedup.

## Path to top-tier
Only viable if at least one of:
- New theoretical insight about *why* x0-estimate residuals at intermediate k are structured/low-rank (turns this from systems trick into a finding).
- Single trained refiner generalizes across many DiTs (SD3, Flux, CogVideoX, HunyuanVideo) — generality story.
- Two-GPU extension that beats PipeDiT — but then the differentiator needs to be much sharper than intra-vs-inter prompt.

If gate A passes only marginally (1.1–1.3×) and gate B forces k > 0.55N, accept workshop venue and stop.

## Recurring pattern to log
"Selective partial-decode + refiner" — future ideas in this family must include a measured LPIPS-vs-k curve in the proposal itself, not after-the-fact. See [[pattern-partial-decode-refiner]].
