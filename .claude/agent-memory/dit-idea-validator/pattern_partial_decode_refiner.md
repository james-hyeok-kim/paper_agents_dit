---
name: pattern-partial-decode-refiner
description: Recurring failure mode — proposals that hide VAE/decoder cost by decoding an intermediate x0-estimate and patching it with a small refiner, without first measuring how structured the residual actually is.
metadata:
  type: feedback
---

# Pattern: partial-decode + refiner without residual measurement

When a proposal claims to overlap VAE decode with remaining diffusion steps by decoding an intermediate x0-estimate (or any pre-final latent), require BEFORE scoring:
1. A measured LPIPS(D(x̂_0(z_k)), D(x̂_0(z_N))) curve as a function of k on the actual target model.
2. A single-GPU microbenchmark of concurrent decode + DiT step (CUDA streams give logical concurrency, not HBM bandwidth — typical achieved overlap is 1.1–1.3×, not the implied 1.8–2×).
3. An explicit articulation of the differentiator vs. DiffuseVAE / Hoogeboom-residual-diffusion / Birodkar — the generator→refiner paradigm is owned territory.

**Why:** These proposals consistently overstate the speedup by assuming (a) residuals at the chosen k are small enough for a cheap refiner, and (b) CUDA streams produce near-perfect overlap. Both assumptions usually fail empirically. Without the residual curve and the microbenchmark, the proposal is asking reviewers to trust a load-bearing assumption with no evidence.

**How to apply:** When the idea names a specific k (e.g., 0.4N, 0.5N, "mid-trajectory"), treat it as the most fragile parameter in the proposal. If pushing k later than proposed would close the speedup window, the idea is workshop-tier at best. Bandwidth-bound concurrent kernels on a single GPU rarely overlap better than 1.3× — top-tier claims require either a two-GPU setup (then PipeDiT is the nearest neighbor, see [[validation-decoupled-vae]]) or a new theoretical insight about residual structure.
