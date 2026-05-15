---
name: idea-latentprefix
description: Cross-request semantic latent prefix cache for diffusion serving — approximate retrieval and reuse of low-SNR latent trajectories across semantically similar prompts. Axis B (cross-request memoization, MLSys angle).
metadata:
  type: project
---

# LatentPrefix — Semantic Trajectory Prefix Cache for Diffusion Serving

**Axis**: B (cross-request memoization / serving-layer).
**One-line claim**: The first K denoising steps (high-noise / low-SNR regime) are **approximately prompt-invariant within a semantic neighborhood**, so a serving system can retrieve a cached partial-trajectory from a *similar* (not identical) past prompt and start denoising from step K instead of step 0.

## Core Hypothesis
For two prompts p1, p2 with text-embedding cosine ≥ τ, the latent z_K^{(p1)} (state after K low-SNR denoising steps) is close enough to z_K^{(p2)} that bootstrapping p2's generation from z_K^{(p1)} (with a re-noising correction step) yields outputs whose distribution is statistically indistinguishable from p2 starting at step 0 — for some K(τ) that grows with τ. This is the diffusion analogue of LLM prefix caching, but operating in **semantic latent space** rather than token-prefix space.

## Technical Approach
1. **Per-prompt trajectory log**: At serving time, for each request, log {z_t at t = T, T-Δ, T-2Δ, ...} along with the text-embedding e and the seed. Storage: roughly 4 KB per checkpoint × 10 checkpoints × N requests, indexed in FAISS.
2. **Retrieval**: For a new prompt p_new with embedding e_new, query FAISS for nearest cached e' with cos(e_new, e') ≥ τ_K(K). Pick the largest K for which the similarity threshold holds.
3. **Bootstrap denoising**: Initialize z_K^{(p_new)} ← z_K^{(retrieved)}. Optionally add a small *re-noising* step: z_K ← α·z_K^{(retrieved)} + β·ε, where (α,β) calibrated per K to match the marginal q(z_K).
4. **Continue denoising from step K** with p_new's conditioning. Saves K of the total T steps.
5. **Threshold calibration**: τ_K curves learned offline by sweeping (cos-similarity, K) and measuring FID of bootstrapped vs. fresh generation. For a CFG-50-step pipeline, hypothesize K up to ~20 is achievable when cos ≥ 0.95.

## Key Innovation (the single move prior work didn't make)
- LLM serving (SGLang, vLLM): prefix-KV caching requires *exact token prefix match*.
- Diffusion serving (Nirvana, DiffServe): caches outputs for *exact prompt match* or routes to specialized models.
- **No system retrieves and reuses** *partial latent trajectories* across semantically-near-but-distinct prompts.

The single move: recognize that low-SNR diffusion steps are dominated by the marginal q(z_t | x_0 ∈ semantic class) rather than by the specific prompt, so the early-step trajectory is approximately a *class function*, not a prompt function. This is empirically supported by progressive distillation literature (early steps "shape," late steps "detail") but never operationalized as a serving-layer cache.

## Pre-hoc Discriminating Test
**Reduction question 1**: "Does this reduce to prompt-keyed exact-match caching?"
**Breaking experiment**: Construct a benchmark where every prompt is unique but clusters in semantic space (e.g., LAION-Aesthetic 100k prompts, k-medoids clustered to 1k centroids). Show:
- Exact-match cache: 0% hit rate, 0% speedup.
- LatentPrefix at τ=0.95, K=10: ≥40% hit rate, ≥15% mean latency reduction with FID delta ≤1.0.

**Reduction question 2**: "Is this just running a smaller model for the first K steps?"
**Breaking experiment**: Compare against (a) Würstchen-style small-model warmup, (b) LCM init. LatentPrefix should win on the *amortized* cost across a workload, not single-request — demonstrate amortization curve as cache fills.

**Validation of core hypothesis**: Without retrieval, take 1000 prompt pairs at varying cosine, swap their z_K's, finish denoising, measure CLIP-on-original-prompt. Plot CLIP-degradation vs cos-similarity vs K. The hypothesis predicts a clean Pareto surface with usable τ at K up to ~T/3.

## Differentiators from Closest Work
- **Nirvana (2024)**: caches final outputs by exact prompt — cannot help on novel prompts.
- **DiffServe / SwiftDiffusion**: routing and batching, no trajectory reuse.
- **Approximate Caching for Efficient Image-to-Image (2024)**: caches for img2img workflows, not text-to-image trajectories across prompts.
- **PromptCache (LLM)**: KV cache for prompt prefixes, not latent trajectories.
- **Würstchen / Stable Cascade**: multi-stage *architecture*, not a serving cache.

## Expected Gains
- For a workload with prompt-cluster structure (typical user-facing T2I service), 2-5× throughput improvement at the tail. For purely random prompts: 0% — and that's an honest negative result we report.
- Makes the paper's value proposition workload-dependent, which is normal for systems papers.

## Feasibility (3/5)
- Requires building a serving prototype (FastAPI + FAISS + diffusers): 2 weeks engineering.
- Need a representative workload trace — propose using LAION + Pick-a-Pic prompt distributions as proxies for realistic clustering.
- Re-noising correction calibration is the trickiest step; can fall back to no correction if FID delta is acceptable.

## Publication Target
**MLSys / OSDI / ATC** — this is intentionally a systems paper, not a methods paper. Different reviewer pool (much fewer methods-paper scoopers), and the workload-trace + amortization-curve framing is what those venues reward. Secondary: NSDI, EuroSys.

## Risk Factors
- **Highest risk**: the τ_K Pareto curve may collapse — i.e., need cos ≥ 0.99 to share even K=2 steps, in which case hit rate is too low to matter. Pre-flight: a 2-day pilot with 200 prompt pairs to sketch the curve before committing.
- Workload realism: if reviewers reject the LAION-clustered workload as unrepresentative, paper weakens. Mitigation: contact a hosted T2I service for an anonymized prompt-frequency trace.
- Storage cost concern from reviewers: address upfront with eviction policies (LRU on z_K, keep only e indices for older entries).

## Pivot Path if Conditional
If reviewers say "amortization is too workload-specific": pivot to **batched cross-prompt early-step sharing** — within a single batch of B prompts at one server, find the medoid prompt, run K shared steps once, branch to per-prompt remaining T-K steps. This gives within-batch amortization without needing a workload assumption.

## Scores
- Novelty: 5 (no diffusion-serving paper does semantic-similar trajectory reuse)
- Impact: 4 (workload-dependent but large in the realistic case)
- Feasibility: 3 (systems engineering load is real)
- Publish Risk (higher = safer): 4 (if τ_K pilot is positive)
- Timeline (higher = faster): 3 (build + workload eval = 2-3 months)

## Recommended Next Step
Run dit-literature-checker on "LatentPrefix" with priority queries:
1. "diffusion serving cache cross-prompt trajectory"
2. "approximate caching diffusion latent reuse"
3. "semantic prompt similarity diffusion bootstrap"
