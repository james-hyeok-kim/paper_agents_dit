# CascadePrompt PoC

**Idea**: Few-step DiT의 prompt-adaptive text encoder cascade (CLIP-L → T5-base → T5-XXL).
**Validation score (pre-experiment)**: 3.4/5 (가장 유망)
**원본 idea 명세**: `.claude/agent-memory/dit-idea-generator/idea_cascadeprompt.md`

## Gate 정의

| Gate | 측정 | PASS 기준 |
|---|---|---|
| **1** | FLUX-schnell 4-step에서 T5-XXL wallclock 비중 | ≥ 15% |
| **2** | 다양한 prompt에서 T5-base vs T5-XXL quality gap 분포 | top10%/median gap ratio > 3.0 (heavy-tail) |
| **3** | Confidence gate 학습 가능성 | downstream-supervised classifier가 escalation rate < 30% 에서 quality 보존 |

## Gate 1 결과: 🔴 **FAIL**

### Setup
- **GPU**: NVIDIA B200 (191.5GB)
- **Model**: black-forest-labs/FLUX.1-schnell, bf16
- **Image**: 1024×1024, 4 inference steps
- **Iters**: 2 warmup + 10 measure
- **Prompt**: "A photo of a cat sitting on a sofa, cinematic lighting, high detail"

### 측정값 (mean ± std)

| Component | Time (ms) | E2E share |
|---|---|---|
| T5-XXL encode | **15.0 ± 0.4** | **2.57%** |
| CLIP-L encode | 4.2 ± 0.2 | 0.72% |
| DiT (4 step) | 240.9 ± 0.5 | 41.34% |
| VAE decode | 4.5 ± 0.2 | 0.78% |
| Other (sched/post) | ~318 | 54.59% |
| **E2E** | **582.8 ± 1.4** | 100% |

→ 그래프: [`results/gate1_breakdown.png`](results/gate1_breakdown.png)
→ Raw: [`results/gate1_wallclock.json`](results/gate1_wallclock.json)

### 판정

**T5-XXL share = 2.57% (FAIL, < 10% 임계점)**

CascadePrompt의 motivation은 "T5-XXL이 wallclock의 ~20-25%를 차지한다"는 가정이지만, B200에서 실측한 결과 **T5-XXL은 e2e의 2.57%에 불과**. 4-step DiT loop가 dominant (41%), 나머지는 scheduler/preprocessing overhead.

### 원인 분석

1. **B200의 압도적인 메모리 대역폭**: T5-XXL (11B params)가 단일 batch 1024-token 단일 forward에서 15ms로 완료. 메모리-bound 워크로드가 거의 instant.
2. **DiT 4-step이 여전히 dominant**: 4096 image tokens × 4 step × FLUX.1-dev 12B 파라미터 → DiT가 가장 비싼 컴포넌트로 남음.
3. **"Other" 54%는 측정 오버헤드 + 파이프라인 로직**: hook 기반 sync, scheduler.set_timesteps, prepare_latents, output post-processing 등.

### Sensitivity check

설령 측정 overhead를 제거하고 순수 compute만 본다고 해도:
- 순수 compute ≈ 15 (T5) + 4 (CLIP) + 241 (DiT) + 4.5 (VAE) = 264.5 ms
- T5 share = 15/264.5 = **5.67%** — 여전히 10% 임계점 미달

### Action

🔴 **CascadePrompt 중단**. B200 같은 고성능 GPU에서는 T5-XXL이 더 이상 병목 아님.

**다만**, consumer-class GPU (RTX 4090, A100)에서는 T5-XXL이 메모리 bandwidth-bound라 더 클 가능성. 추후 그 환경에서 재측정할 가치는 있지만 우선순위는 낮음.

### 다음 실험

Conditional Go 두 번째 (점수 3.0/5): **DecoupledVAE Gate A** — 단일 GPU에서 DiT + VAE decode CUDA stream overlap factor 측정.

## 파일 안내

| 파일 | 내용 |
|---|---|
| `gate1_wallclock.py` | Gate 1 측정 스크립트 |
| `make_plot.py` | 결과 그래프 생성 |
| `results/gate1_wallclock.json` | 원본 측정값 |
| `results/gate1_breakdown.png` | 컴포넌트별 wallclock breakdown |
| `results/gate1_run.log` | 전체 실행 로그 |
