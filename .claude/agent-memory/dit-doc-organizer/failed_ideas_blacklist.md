---
name: failed-ideas-blacklist
description: 검증에서 실패한 아이디어 블랙리스트 — 향후 아이디어 생성 시 동일 방향이 재등장하지 않도록 방지
metadata:
  type: project
---

# 실패한 아이디어 블랙리스트

마지막 업데이트: 2026-05-15

이 문서는 dit-idea-generator가 아이디어를 생성하기 전에 반드시 참조해야 하는 블랙리스트다. 아래 방향과 유사한 아이디어는 생성하지 말 것.

---

## 🔴 No-Go 아이디어

### 1. CondMask-DiT (검증일: 2026-05-15)

**방법 요약**: Cross-attention map에서 `s_i = max_j A[i,j] - λ·H(A[i,:])` 로 spatial 토큰 중요도 계산 → self-attention에서 중요도 낮은 토큰 제외. Training-free, inference-only. 매 5스텝 재활성화, CFG branch 동일 마스크, timestep별 retention rate 스케줄링.

**실패 사유**: 모든 컴포넌트가 이미 출판됨
- AT-EDM (CVPR 2024, arXiv:2405.05252): CA-WPR entropy variant = 점수 함수와 의미상 동일, DSAP = timestep 스케줄
- IBTM (arXiv:2411.16720): cross-attention importance signal, training-free DiT
- ToPi (arXiv:2602.01609): DiT 전용 training-free, ΔT=10 주기 재활성화
- CAT Pruning (arXiv:2502.00433): staleness-aware 재활성화, timestep-aware 스케줄
- DaTo (arXiv:2501.00375): CFG branch 마스크 공유

**블랙리스트 규칙**:
- ❌ Cross-attention 기반 token importance scoring + training-free + DiT 조합
- ❌ Timestep retention 스케줄링 단독 contribution
- ❌ CFG branch 마스크 공유 단독 contribution
- ✅ 허용 예외: **few-step (1-8 step) / guidance-distilled DiT** (CFG 없음)에서의 token pruning — 기존 5개 prior work 모두 25-50 step CFG 환경 가정으로 적용 불가

---

## 🟡 Conditional Go (파일럿 필요, 새 아이디어 생성 금지)

### 2. SpecDiT (검증일: 2026-05-15)

**방법 요약**: 절반 크기 draft DiT를 velocity+hidden distillation으로 학습 → γ 스텝 미리 생성 → target DiT가 배치 forward로 검증. 수락 기준: `||v_target - v_draft|| / ||v_target|| < τ`. EMA로 γ 자동 조정.

**실패 사유 (Conditional)**:
- SpeCa (ACM MM 2025, arXiv:2509.11628): "forecast-then-verify" + relative-error acceptance를 동일 backbone에서 이미 검증. Drafter가 TaylorSeer (parameter-free)라는 점만 다름
- Yoon et al. (ICLR 2025, arXiv:2501.05370): trained drafter를 "비실용적"으로 명시적 배제
- γ_effective가 재귀 error 누적으로 2-3에 수렴할 위험
- 이름 충돌: SpecDiff (arXiv:2509.13848) 존재

**현재 상태**: DiT-XL/2 파일럿 실험 대기 중 (γ_effective ≥ 5, FID 손실 <1% 조건)

**블랙리스트 규칙**:
- ❌ Speculative decoding / forecast-then-verify for DiT의 새 아이디어 생성 금지 (SpeCa 선점)
- ❌ Trained separate drafter + acceptance criterion 조합 (SpecDiT로 추적 중)
- ✅ 허용: SpeCa가 실패하는 regime (low-CFG, distilled target, OOD prompt) 특화 방향

---

## 선점된 기법 참조 목록 (아이디어 생성 시 피할 영역)

| 기법 | 핵심 선행 논문 | 비고 |
|------|--------------|------|
| Cross-attention token pruning (DiT) | AT-EDM, IBTM, ToPi, CAT, DaTo | 완전 포화 |
| Timestep caching (DiT) | DeepCache, FORA, PAB, TeaCache, ∇-Cache | 완전 포화 |
| Speculative decoding (diffusion) | SpeCa (2509.11628) | 이미지+비디오 모두 검증 |
| Token merging | ToMe, IBTM | 포화 |
| Step reduction | DDIM, DPM-Solver, LCM, Consistency Models | 포화 |
| Static text encoder 경량화 | NanoFLUX (2602.06879), Scaling Down (2503.19897) | 포화 |
| Video DiT timestep caching | AdaCache, PAB video | 포화 |
| Video DiT frame-axis motion-aware (cosine-sim) | ADAPTOR (CVPR 2025W) | 포화 |
| Video DiT frame-axis (whole-frame) | VGDFR (ICCV 2025), FIS-DiT | 포화 |
| VAE inter-prompt pipelining | PipeDiT (2511.12056) | 포화 |

---

### 5. ControlAsync (검증일: 2026-05-15, 🔴 No-Go)

**방법 요약**: ControlNet/IP-Adapter conditioning branch와 trunk가 비대칭 timestep schedule을 가지도록 분리. Conditioning branch caching interval c >> trunk interval h.

**실패 사유**: EVCtrl + OminiControl2가 정확히 동일한 가설+방법을 이미 발표
- **EVCtrl** (arXiv:2508.10963, Aug 2025): "control branch가 main path보다 redundancy가 높다"는 핵심 가설을 그대로 검증. Denoising Step Skipping(DSS)으로 제어 브랜치를 N step마다 recompute. Flux/CogVideo/Wan-ControlNet 검증 완료
- **OminiControl2** (arXiv:2503.08280, Mar 2025): conditioning feature의 temporal stability를 cosine similarity로 명시적 분석 (Fig. 3). c=∞의 극단 케이스 구현
- **Faster Diffusion** (NeurIPS 2024): ControlNet 2.1× speedup 이미 보고
- **T-GATE**: cross-attention caching의 다른 schedule 원리 이미 확립

**블랙리스트 규칙**:
- ❌ ControlNet/IP-Adapter conditioning branch caching
- ❌ Trunk vs conditioning branch 비대칭 schedule
- ✅ 허용 예외: Multi-adapter (ControlNet + IP-Adapter + LoRA) joint scheduling, Cross-request feature cache for ControlNet serving

---

### 6. LatentPrefix (검증일: 2026-05-15, 🔴 No-Go)

**방법 요약**: Cross-request semantic prefix cache — 첫 K step의 partial latent trajectory를 prompt embedding similarity로 retrieve. LLM의 prefix-cache의 diffusion 버전.

**실패 사유**: NIRVANA가 정확히 동일한 시스템을 NSDI 2024에 발표 (저자가 잘못 기억하고 있던 prior art)
- **NIRVANA** (arXiv:2312.04429, NSDI 2024): CLIP cosine similarity로 partial latent state cache lookup, K∈{5,10,15,20,25}, 유사도 기반 adaptive K, restart denoising, LCBFU eviction. **모든 메커니즘 점령됨**. 21% GPU 절감, 19.8% latency 감소 보고
- **Chorus** (arXiv:2604.04451, 2026): 같은 메커니즘을 video DiT 서빙에 확장. Three-stage adaptive reuse
- **CHAI** (2602.16132): cross-request attention cache for video diffusion
- 원래 제안의 "NIRVANA는 exact prompt match만" 주장이 사실과 다름

**블랙리스트 규칙**:
- ❌ Cross-request partial latent trajectory cache (prompt similarity 기반)
- ❌ Diffusion 서빙에서 semantic neighborhood reuse
- ❌ "LLM prefix-cache의 diffusion 버전" framing
- ✅ 허용 예외: Cross-request KV/attention-state caching for DiT (latent state 아닌 attention state — 별개 idea)

---

### 7. MMBlockAttn (검증일: 2026-05-15, 🔴 No-Go)

**방법 요약**: MM-DiT의 cross-modal sub-block (text-image, image-text)에 calibrated low-rank attention 적용. Image-image dense 유지. Custom Triton kernel.

**실패 사유**:
- DiTFastAttnV2 (ICCV 2025): same 4-way sub-block partition, post-training MMDiT compression, custom kernel — 단 반대 design choice
- E2E ceiling이 너무 낮음: cross-modal은 FLUX-2048에서 5.9% FLOPs → 절약해도 1.04x e2e
- Softmax coupling이 mechanism 자체를 깨뜨림 (joint softmax normalizer가 sub-block 분리를 무효화)
- Triton kernel이 FA2를 못 이길 가능성 높음 (cross-modal 작은 size)

**블랙리스트 규칙**:
- ❌ Cross-modal sub-block을 efficiency target으로 (FLOPs ceiling 너무 낮음)
- ❌ Sub-block partition + post-hoc kernel framing (DiTFastAttnV2)
- ❌ Joint softmax 분해 시도 (수학적으로 깨짐)

---

### 8. CFGWane (검증일: 2026-05-15, 🔴 No-Go)

**방법 요약**: Cond/uncond disagreement를 per-sample/per-step 게이트로 사용해 uncond forward 동적 스킵. Skip 시 prior step guidance vector로 선형 외삽.

**실패 사유**: 단일 논문이 두 메커니즘 모두 선점
- Castillo et al. "Adaptive Guidance" (arXiv:2312.12487, AAAI 2024): cosine sim of cond/uncond을 per-sample per-step 게이트로 정확히 사용 (AG variant). LinearAG variant는 prior step guidance의 선형 결합으로 uncond 대체
- OUSAC (arXiv:2512.14096): uncond pass 82% skip, FLUX 5x speedup — CFGWane 헤드라인 수치 이미 초과
- Step AG (2506.08351): cosine-similarity 게이팅의 일반화 실패 명시 비판
- GalaxyDiT (2512.03451): 비대칭 외삽의 노이즈 mismatch artifact 명시 보고

**블랙리스트 규칙**:
- ❌ Per-sample disagreement-gated CFG skipping (AG)
- ❌ Prior step guidance vector 선형 외삽으로 uncond 대체 (LinearAG)
- ❌ Static schedule CFG skip (Step AG, OUSAC 등 다수)
- ❌ CFG efficiency 단독 contribution 자체가 매우 포화됨

---

### 9. TimeWarp (검증일: 2026-05-15, 🔴 No-Go)

**방법 요약**: DiT denoising timestep 루프를 prefetch boundary로 활용. Single-GPU에서 step t의 activation offload와 step t+1의 prefetch overlap. VRAM 24GB→8-12GB 감축.

**실패 사유**: Production 코드로 이미 출시됨
- vLLM-Omni 공식 문서: "When the last block completes, the first block prefetches for the next denoising step" — 글자 그대로 동일
- SGLang-Diffusion (LMSYS 2026-01): multi-priority CUDA streams, video-DiT default
- HuggingFace Diffusers `apply_group_offloading` (v0.33.0): `use_stream=True`, HunyuanVideo 6.5GB demo
- FastVideo / Lightx2v `dit_layerwise_offload`
- Streamlined Inference (NeurIPS 2024): 동일 venue/positioning 선점

**블랙리스트 규칙**:
- ❌ Single-GPU layerwise offload + CUDA stream prefetch for DiT (production 코드 존재)
- ❌ Denoising-loop를 prefetch boundary로 사용
- ❌ Activation offload + prefetch overlap framing 자체가 포화

---

### 10. ConvergeSense (검증일: 2026-05-15, 🔴 No-Go)

**방법 요약**: Latent trajectory의 relative velocity가 prompt-cluster별 calibrated threshold 이하 k step 지속되면 즉시 decode. CLIP/T5 cluster별 threshold lookup table.

**실패 사유**:
- StepSaver (arXiv:2408.02054): offline prompt → optimal-step + NLP head — cluster-LUT와 정보이론적 등가
- AdaDiff (2311.14768): 학습된 RL policy로 per-prompt step selection (33-40%)
- AdaDiff ECCV (2309.17074): uncertainty 기반 early termination (43.6%)
- Adapt and Diffuse (2309.06642): sample-adaptive reverse step (8-10x)
- AdaFlow (NeurIPS 2024): variance-scalar 신호로 per-sample step adaptation
- A²S (OpenReview 2025): acceleration-aware step skipping
- **수학적 자가모순**: 0.7×1.0 + 0.3×0.5 = 0.85 → e2e 1.18× (1.3-1.5× 불가능)

**블랙리스트 규칙**:
- ❌ Per-sample early stop / latent convergence-based step termination (포화)
- ❌ Prompt-embedding cluster + threshold LUT for step count (StepSaver 등가)
- ❌ Velocity/uncertainty/variance-based per-sample adaptation (모두 발표됨)

---

### 3. MotionGate (검증일: 2026-05-15)

**방법 요약**: Video DiT에서 latent-space optical flow를 2-3 layer 3D conv head로 추정 → motion 작은 token skip + warp 재사용. Forward-backward consistency로 occlusion 검출.

**실패 사유**: HSA (arXiv:2605.06892, 2026-05-07 — 8일 전 preprint)가 핵심 메커니즘 선점
- HSA: per-spatial-token gating + velocity-change motion proxy + cached Euler update로 token reuse → Wan-2.1/2.2, LTX-2에서 검증 완료
- 추가 선행: FLoED (latent warping + flow attention cache), ADAPTOR (CVPR 2025W, cosine-sim per-column gating), AdaCache+MoReg (ICCV 2025, motion-aware timestep schedule), VGDFR (ICCV 2025, motion frequency frame merging)
- Latent warping fragility: V-Warper (2025)가 "intermediate features yield weak alignment" 실증
- 살아있는 유일한 novelty (occlusion mask)는 단독 contribution 불가

**블랙리스트 규칙**:
- ❌ Video DiT per-token motion-based spatial gating (HSA 선점)
- ❌ Latent-space optical flow를 inference gating에 사용 (FLoED 선점)
- ❌ Frame-axis computation skip (FIS-DiT, VGDFR, ADAPTOR 포화)
- ✅ 허용 예외: HSA의 occlusion-heavy prompt failure mode를 정조준한 add-on (HSA 위에서 동작)

---

## 아직 빈 공간 (탐색 권장 방향)

1. **Few-step DiT (1-8 step, CFG 없음)** 특화 모든 기법 — 기존 방법들은 25-50 step 가정
2. **Video DiT: per-token spatial gating + latent optical flow warping** — ADAPTOR는 cosine-sim + hierarchical, warping reuse는 미개척
3. **VAE intra-prompt critical-path 단축** — mid-trajectory x0-estimate parallel decode (DecoupledVAE 추적 중)
4. **이론적 품질 보장** — 어떤 prior work도 FID/CLIP degradation의 formal bound 없음
5. **Conditioning pathway (adaptive)** — Static T5 경량화는 포화, prompt-adaptive cascade는 미개척
