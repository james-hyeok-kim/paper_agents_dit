# SpecDiT PoC

**아이디어**: 절반 사이즈 draft DiT가 γ 스텝의 latent를 미리 생성, target DiT가 배치 forward로 검증. 수락 기준 `||v_target − v_draft|| / ||v_target|| < τ`.
**Validation 점수**: 2.6/5 (Conditional Go 중 3위)
**원본 idea 명세**: `.claude/agent-memory/dit-idea-generator/idea_specdit.md`

## 실험 목적

SpecDiT의 핵심 가정은 **"절반 사이즈 drafter가 full DiT의 velocity를 τ=0.1 relative error 안으로 예측할 수 있다"** 입니다. 만약 이 가정이 깨지면 γ_effective가 매우 낮아져 speedup 자체가 안 나옵니다.

본격 SpecDiT는 drafter 학습에 GPU-day 단위 시간이 필요하므로, **quick PoC**는 **layer-skipping을 draft proxy**로 사용합니다. Caveat: layer-skip은 trained drafter보다 strictly worse일 수 있으나, 적어도 "절반 capacity로 velocity가 얼마나 보존되는가"의 upper bound는 제공합니다.

## 게이트 정의

| Gate | 측정 | 판정 |
|---|---|---|
| **Quick (이번 실험)** | 8개 prompt × 4 step에서 layer-skip draft의 rel_err 분포, τ별 acceptance rate | τ=0.10에서 accept ≥ 70%면 BORDERLINE PASS |
| 본격 | DiT-XL/2에서 trained shallow drafter, γ_effective ≥ 5 + FID 손실 <1% | 본격 실험으로만 검증 가능 |

## 결과: 🔴 **FAIL**

### Setup
- **GPU**: NVIDIA B200 (191.5GB), GPU 2
- **Model**: black-forest-labs/FLUX.1-schnell, bf16
- **Resolution**: 1024×1024, 4 step
- **Prompts**: 8개 (cat / 산 풍경 / 우주비행사 / 라면 / 도서관 / 자동차 / 인물 / 도시)
- **Draft proxy**: 19 MMDiT 중 10개 + 38 single 중 19개만 활성화 (≈51% capacity)
- **Skip 방식**: residual passthrough (`return ehs, hs` 변경 없이)

### Per-step rel_err 측정값

| Step | mean | median | max |
|---|---|---|---|
| 0 | 0.8035 | 0.8036 | 0.8665 |
| 1 | 0.9453 | 0.9415 | 1.1094 |
| 2 | 1.0912 | 1.0762 | 1.3679 |
| 3 | 1.1800 | 1.1572 | 1.4857 |
| **전체 (32개)** | **1.0050** | **0.9756** | 1.4857 |

### Acceptance rate

| τ | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|
| Accept rate | 0.0% | **0.0%** | 0.0% | 0.0% | 0.0% |

→ 그래프: [`results/gate_breakdown.png`](results/gate_breakdown.png)
→ Raw: [`results/gate_velocity_agreement.json`](results/gate_velocity_agreement.json)

### 판정

**모든 τ에서 acceptance rate 0%** → 🔴 **FAIL** (layer-skip proxy 기준)

### 해석 및 Caveat

1. **Layer-skip draft proxy는 trained drafter보다 strictly worse일 가능성이 높음**:
   - 50% 블록을 잘라내면 residual passthrough 때문에 velocity 예측이 사실상 무작위에 가까움 (rel_err ≈ 1.0은 "v_draft가 v_target과 거의 직교"라는 의미)
   - Trained shallow drafter는 50% 파라미터로 velocity를 직접 학습하므로 훨씬 정확할 가능성

2. **하지만 이 결과는 여전히 강한 negative signal**:
   - FLUX는 19+38=57개 블록의 deep 구조 — 절반을 단순히 잘라낸 표상은 의미 정보를 완전히 잃음
   - SpecDiT가 동작하려면 drafter가 **target의 hidden representation을 정밀하게 모사**해야 함
   - Yoon et al. (ICLR 2025)이 trained drafter를 "비실용적"이라 명시 배제한 이유와 정합

3. **이 PoC가 줄 수 있는 conclusion**:
   - Naive layer-skip drafter는 SpecDiT의 acceptance criterion을 절대 통과 못 함
   - 그래서 SpecDiT는 trained drafter가 필수 → GPU-day 학습 비용 발생
   - SpeCa(ACM MM 2025)가 parameter-free TaylorSeer로 동일 메커니즘을 이미 점령했으므로, trained drafter의 학습 비용을 정당화하기 매우 어려움

### Action

🔴 **SpecDiT 단독 idea 중단**.

**Pivot 옵션** (validation에서 제안된 것):
- **SpecCache+**: SpeCa의 reject case에서 trained drafter를 fallback으로 (3-tier acceptance)
- **Distillation lineage 명시화**: drafter를 SpeCa의 accepted predictions에서 distill

이 두 pivot은 모두 SpeCa를 baseline으로 인정하고 그 위에 얹는 구조 → workshop tier 가능성은 있으나 top-tier reject 위험 큼.

### 다음 실험

Conditional Go 네 번째 (점수 2.4/5): **StructPrior Gate** — FLUX-ControlNet에서 trunk vs branch wallclock profiling.

## 파일 안내

| 파일 | 내용 |
|---|---|
| `gate_velocity_agreement.py` | Quick gate 측정 스크립트 |
| `make_plot.py` | 그래프 생성 |
| `results/gate_velocity_agreement.json` | 원본 측정값 (per-step rel_err, acceptance rate) |
| `results/gate_breakdown.png` | per-step rel_err boxplot + acceptance rate curve |
| `results/gate_run.log` | 실행 로그 |
