---
name: idea-cascadeprompt
description: Few-step DiT 특화 — T5-XXL 텍스트 인코더를 confidence-gated cascade로 대체. 1-8 step 환경에서 conditioning pathway가 wallclock의 큰 부분이 되는 점 공략.
metadata:
  type: project
---

# CascadePrompt — Confidence-Gated Text Encoder Cascade for Few-Step DiT

**Regime**: Few-step / guidance-distilled DiT (FLUX-schnell 4-step, SDXL-Turbo, SD3-Turbo)
**Axis**: Conditioning pathway (NOT the DiT loop itself)
**Status**: Drafted 2026-05-15

## Core Intuition (왜 작동하는가)

Few-step distilled DiT에서는 step reduction이 이미 극한에 도달 (4 step). 이때 wallclock 분해를 보면 (H100 기준 추정, 실측 필요):
- T5-XXL encoder: ~11B params, 한 번 forward, FLUX-schnell 4-step에서 전체 latency의 ~20-25% 차지 (worst case 40%)
- DiT loop: 4 step × 12B params

**Deployment scope (중요)**: 이 아이디어는 **interactive, diverse-prompt** scenario를 타겟. Production batch serving (같은 prompt로 여러 sample, T5 embedding cache 가능)에서는 T5가 amortize되어 cascade 이득 없음. 따라서 다음 시나리오에 명시적으로 한정:
- Interactive UI (사용자가 prompt 입력 → 1-2 sample 생성)
- Diverse-prompt batch (각 prompt 한 번씩, T5 cache 무효)
- Edge device serving (memory budget tight, T5-XXL load 자체가 부담)

기존 모든 efficiency 연구는 DiT loop만 공격함. 하지만 4-step regime에서는 **conditioning pathway 자체가 병목**. 게다가 대부분의 prompt는 T5-XXL의 full capacity를 필요로 하지 않음 — "a cat sitting on a sofa" 같은 단순 프롬프트는 작은 encoder로 충분하고, "a oil painting in the style of Caravaggio depicting..." 같은 복잡 프롬프트만 큰 encoder가 필요.

핵심 직관: **prompt complexity는 prompt 자체에서 cheap하게 예측 가능**하다. CLIP-text(small) → T5-base → T5-XXL의 cascade를 두고, 각 단계에서 "이 임베딩이 final DiT output에 충분한지"를 가벼운 confidence head로 판단.

## Method

1. **Cascade 구성**: 
   - Tier 1: CLIP-ViT-L text (~123M, 이미 SDXL/SD3에 포함)
   - Tier 2: T5-base (~220M)
   - Tier 3: T5-XXL (~11B, FLUX original)

2. **Embedding alignment**: Tier 1, 2의 embedding을 Tier 3 space로 투영하는 lightweight projector (학습됨, ~10M params)

3. **Confidence gate**: prompt의 token-level entropy + projector reconstruction error를 입력으로 하는 binary classifier. "Tier k embedding으로 만든 이미지가 Tier 3 embedding으로 만든 이미지와 perceptually 동등할 확률" 예측.

4. **학습**:
   - Step A: paired data (prompt → DiT output with full T5-XXL) 수집 (10K prompts)
   - Step B: 각 prompt에 대해 모든 tier로 생성, LPIPS로 동등성 라벨 생성
   - Step C: confidence head 학습 (frozen DiT)

5. **Inference**: 
   - prompt → Tier 1 encode + confidence check → 통과시 stop, 아니면 Tier 2 → ... → Tier 3
   - DiT는 frozen, 단지 conditioning pathway만 변경

## Closest Prior Art + The Move They Didn't Make

**Closest**: 
- DistilCLIP (CLIP knowledge distillation), TinyT5 (T5 distillation) — encoder를 작게 만들기만 함
- ELLA (2024) — LLM을 text encoder로 사용 (방향이 반대; 더 큰 모델로)
- "Speculative encoder" 류 work이 LLM에서는 있지만 (e.g., REST), **diffusion text encoder pipeline에 cascade 적용은 없음**

**The single move they didn't make**: prompt-adaptive encoder selection. 기존 distillation은 모든 prompt에 대해 같은 작은 encoder를 사용 → quality loss가 worst-case에 의해 제약됨. Cascade는 평균 latency를 줄이면서 worst case를 보장.

## Discriminating Test (advisor의 기준)

"Would this give >2x speedup in 50-step CFG model?" → **No**. 50-step에서는 T5-XXL이 전체의 5% 미만 → cascade 이득이 미미. 이 아이디어는 **few-step에서만 의미 있음** ✓

## Expected Gain
- FLUX-schnell 기준 end-to-end latency 1.4-1.8x speedup
- LPIPS 기준 quality drop <0.02 (대부분의 prompt가 Tier 1-2에서 종료)
- 메모리도 절감 (Tier 3 lazy load)

## 실현 가능성 / 위험
- **실현 가능**: frozen DiT, encoder cascade와 confidence head만 학습. 단일 GPU로 가능.
- **위험 1**: T5-XXL embedding과 작은 encoder embedding의 semantic gap이 projector로 안 메워질 수 있음 → ablation 필수
- **위험 2**: complex prompt에서 confidence가 over-confident → calibration 필요
- **위험 3**: SDXL-Turbo는 이미 CLIP만 사용 → 적용 가능 모델 제한 (FLUX, SD3, PixArt-Σ가 주 타겟)

## Scores
Novelty 5 / Impact 4 / Feasibility 4 / PublishRisk 3 / Timeline 4

## Publication Target
NeurIPS 2026 (efficiency track) 또는 ICLR 2027. Few-step efficiency라는 새로운 niche를 개척하는 첫 work이 될 가능성.

## Next Step
**먼저 실측**: H100/A100에서 FLUX-schnell의 T5-XXL forward time vs 4-step DiT forward time을 측정. ~20%면 idea 유효, 5% 미만이면 pivot 필요.

dit-literature-checker로 다음 검증:
- "text encoder cascade diffusion"
- "T5 distillation diffusion"  
- "prompt-adaptive encoder"
- "confidence-gated text encoder"
- "speculative text encoder" (LLM REST의 diffusion port 여부)
- ELLA, SDXL-Turbo, FLUX-schnell의 text encoder ablation 논문
- FLUX-mini, FLUX-lite 류 work이 encoder 부분에서 무엇을 했는지

**Scoring 주의**: 아래 점수는 lit-check 이전 self-estimate. Round 1의 CondMask-DiT도 같은 단계에서 5점이었음.

Related: 별개 axis라 기존 [[idea-trajleap]], [[idea-condmask-dit]], [[idea-specdit]]와 직접 경쟁하지 않음.
