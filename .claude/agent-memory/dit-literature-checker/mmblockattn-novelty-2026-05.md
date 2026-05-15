---
name: mmblockattn-novelty-2026-05
description: MMBlockAttn (post-hoc calibrated low-rank cross-modal attention kernel for MM-DiT) novelty check — partial overlap, residual novelty in low-rank-on-cross-modal-sub-block angle
metadata:
  type: project
---

# MMBlockAttn Novelty Check (2026-05-15)

## Verdict: PARTIAL OVERLAP (yellow), leaning toward conflict on the "post-hoc kernel for MM-DiT" framing but residual novelty if the SVD/low-rank-on-cross-modal-sub-block angle is genuinely the first.

## Most Conflicting Prior Art

### DiTFastAttnV2 (ICCV 2025, arXiv:2503.22796) — biggest threat
- Explicitly identifies the four sub-block partition: visual-visual, visual-text, text-visual, text-text
- Post-training compression for MMDiT (FLUX, SD3)
- Custom fused kernel based on FlashAttention2
- BUT uses arrow attention (sliding window for image-image, full for cross-modal) + caching, NOT SVD low-rank
- Claims 68% FLOP reduction, 1.5x speedup on 2K
- Difference vs MMBlockAttn: DiTFastAttnV2 keeps cross-modal full attention (small N anyway) and sparsifies image-image; MMBlockAttn does the opposite (keep image-image dense, low-rank cross-modal). **These are opposite design choices on the same partition** — could be argued as orthogonal.

### SLA: Sparse-Linear Attention (ICLR 2026 sub, OpenReview eD8IPvNoZB)
- Decomposes attention weights into critical (high-rank, sparse) + marginal (low-rank, linear) + negligible (skip)
- Applied to DiT (Wan2.1-1.3B for video)
- Requires fine-tuning (not post-hoc)
- Doesn't explicitly partition by MM-DiT sub-blocks
- 13.7x attention speedup, 2.2x e2e
- Conceptually closest in "sparse + low-rank decomposition" angle but training-required and not MM-DiT-specific

### LoRA-Sparse (CVPR 2024)
- Low-rank + sparse for multi-modal LLMs (NOT diffusion)
- Different model class

### SpargeAttn (ICML 2025, arXiv:2502.18137)
- Training-free sparse attention for FLUX, SD3.5
- Modality-agnostic two-stage online filter
- Doesn't exploit cross-modal sub-block structure

## Residual Novelty Angles

1. **First to demonstrate empirically that cross-modal sub-blocks (text-image, image-text) in MM-DiT are near-low-rank under SVD analysis** — neither DiTFastAttnV2 (which uses arrow/caching) nor SLA (which doesn't analyze MM-DiT structure) has done this.
2. **Post-hoc calibration via SVD energy curve** — different from DiTFastAttnV2's RSE-based head selection between full/arrow modes
3. **Opposite design choice from DiTFastAttnV2**: dense image-image + low-rank cross-modal, vs DiTFastAttnV2's arrow image-image + full cross-modal — could be complementary or competitive
4. **Custom Triton kernel for the asymmetric (dense + low-rank fused) computation** — appears unbuilt

## Risks
- DiTFastAttnV2 already established the "sub-block partition + post-hoc + fused kernel for MM-DiT" framing
- Reviewers will demand head-to-head against DiTFastAttnV2
- The cross-modal sub-block is small (text tokens ~256, image tokens ~4096+), so low-rank speedup on the smaller block may not yield large e2e gain
- MMDiT joint attention is dominated by image-image FLOPs anyway; speeding up the smaller cross-modal sub-block has limited ceiling

## Recommendation: Differentiate or Pivot

If proceeding:
- Position explicitly as orthogonal/complementary to DiTFastAttnV2 (combine arrow image-image + low-rank cross-modal)
- Emphasize the SVD analysis of MM-DiT cross-modal blocks as a standalone scientific contribution
- Run head-to-head benchmark against DiTFastAttnV2 + SpargeAttn + SLA (after FT)
- Quantify the FLOP ceiling for cross-modal-only speedup before committing

Pivot direction:
- Combine sub-block low-rank with dynamic head routing (SLA-style critical/marginal classification per-sub-block)
- Or focus on the SVD/rank scientific analysis as the contribution and treat the kernel as demonstration

## Key References
- arXiv:2503.22796 (DiTFastAttnV2)
- arXiv:2502.18137 (SpargeAttn)
- OpenReview eD8IPvNoZB (SLA)
- arXiv:2506.07986 (TACA — cross-modal logit rebalancing, different goal)
- arXiv:2502.01776 (Sparse VideoGen)
- arXiv:2006.04768 (Linformer — training-time low-rank baseline)
- arXiv:2411.05007 (SVDQuant — low-rank for quantization, not attention sparsity)

Related ideas in this index: [[condmask_dit_check]] (also attention-sparsity for DiT, different angle)
