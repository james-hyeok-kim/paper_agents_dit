---
name: idea-timewarp
description: Inter-timestep activation prefetch and offload for single-GPU DiT inference — exploit sequential denoising for memory hierarchy overlap.
metadata:
  type: project
---

# TimeWarp: Inter-Timestep Activation Prefetch for Single-GPU DiT

## One-line summary
Exploit DiT's sequential timestep structure to overlap activation offload (step t) with prefetch (step t+1) on a single GPU, enabling 4K/long-video FLUX/CogVideoX inference under consumer VRAM without throughput loss.

## The single move prior work didn't make
- DistriFusion / PipeFusion: multi-GPU pipeline parallelism — requires ≥2 GPUs
- xDiT, RingAttn for DiT: tensor/sequence parallel multi-GPU
- ZeRO-Inference, FlexGen: LLM-focused, exploits *intra-token* structure, no notion of denoising timestep
- Activation checkpointing: trades compute for memory, no overlap
- **No prior work treats the diffusion timestep loop itself as a prefetch boundary** — the t→t+1 transition is uniquely well-suited because the U-Net/DiT weights and activation shapes are identical across timesteps, so prefetch is perfectly predictable

## Mechanism (single sentence)
Pin a small "active layer window" in VRAM and continuously offload step-t activations to CPU/NVMe while prefetching step-(t+1) weights and KV-cache lines, scheduled by a static graph compiler that knows the denoising loop is fixed-length.

## Discriminating tests (pre-loaded)
1. **Reduce-to test**: window-size = full model collapses to standard inference (no offload). Window-size = 1 collapses to ZeRO-style layer-by-layer execution (no overlap benefit, baseline already exists). Middle band with t→t+1 prefetch overlap is the novel zone.
2. **Single-mechanism test**: One mechanism — denoising-loop-aware double-buffered prefetch. Pass.
3. **Ceiling test**: Currently FLUX-dev needs 24GB VRAM at FP16 for 1024². With this, target 12GB consumer GPU at <1.1x latency penalty (vs. 3-5x penalty for naive offload). Different metric: it is not a speedup, it is a **VRAM-throughput Pareto-frontier shift**. MLSys-style contribution.
4. **One-paper-kills-this test**: An "ZeRO-Inference for diffusion" or "Stable-Swap: pipelined offload for DiT" paper would kill it. Need lit-check; FlexGen-Diffusion or similar is plausible.

## Why it must publish at MLSys-tier
- Not a model contribution — a systems contribution
- Right venue: MLSys, ASPLOS, OSDI, possibly EuroSys
- Standard ML venues will undervalue (no FID improvement)

## Quantitative ceiling
- Memory: 2-3x reduction in peak VRAM (24GB → 8-12GB)
- Latency: 0.95-1.1x (overlap hides offload)
- Enables: FLUX 2K, CogVideoX 5s on RTX 3080/4080 instead of A100

## Scores
- Novelty: 4 (denoising-loop-aware prefetch is genuinely new framing)
- Impact: 5 (unlocks consumer GPU deployment for current SOTA DiTs — large user base)
- Feasibility: 3 (requires CUDA stream scheduling, graph compiler integration; ~3 months engineering)
- Publish risk: 4 (results either work or don't — no ambiguous middle)
- Timeline: 3 (3-4 months, infra-heavy)

## Risk factors
- NVMe bandwidth bottleneck on slower SSDs — must benchmark across storage tiers
- PCIe bandwidth saturation if batch size > 1
- Compiler integration with PyTorch eager mode is non-trivial (might need torch.compile or custom CUDA graph)

## Recommended next step
Run dit-literature-checker on "TimeWarp: denoising-loop-aware activation prefetch for single-GPU DiT" to verify against FlexGen, ZeRO-Inference, Stable-Swap, and any DiffServe/MLSys 2025-2026 systems papers.
