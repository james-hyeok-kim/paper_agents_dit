# StructPrior PoC

**아이디어**: ControlNet의 conditioning input (pose/depth/mask)의 geometric structure를 zero-cost prior로 trunk DiT의 self-attention sparsity mask로 사용.
**Validation 점수**: 2.4/5 (Conditional Go 중 4위, lean No-Go)
**원본 idea 명세**: `.claude/agent-memory/dit-idea-generator/idea_structprior.md`

## 실험 목적

StructPrior의 e2e gain 상한은 **"trunk self-attention이 전체 wallclock에서 차지하는 비중"**에 의해 결정됩니다. 만약 trunk attention이 e2e의 작은 비중이면, sparsity로 그 부분을 100% 제거해도 e2e gain이 적습니다.

본격 Gate는 FLUX-ControlNet을 사용해야 하지만, FLUX-schnell만으로도 **trunk attention share의 상한**을 빠르게 측정 가능 — ControlNet branch가 추가되면 attention share는 더 낮아지므로 이 측정은 idea에 **optimistic upper bound**를 제공합니다.

## 게이트 정의

| Gate | 측정 | 판정 |
|---|---|---|
| **Quick (이번 실험)** | FLUX-schnell trunk self-attention / E2E | ≥40% PASS, 25-40% BORDERLINE, <25% FAIL |
| 본격 | FLUX-ControlNet에서 trunk vs branch wallclock + naive geometric mask quality | 통과 시 본격 실험 진행 |

## 결과: 🔴 **FAIL**

### Setup
- **GPU**: NVIDIA B200 (191.5GB), GPU 3
- **Model**: black-forest-labs/FLUX.1-schnell, bf16
- **Resolution**: 1024×1024, 4 step
- **Hook**: `FluxAttention` 클래스 57개 (19 MMDiT × 1 + 38 single × 1) 모두 CUDA event로 시간 측정
- **반복**: 2 warmup + 8 iters

### 측정값 (mean ± std)

| Component | Time (ms) | Share |
|---|---|---|
| Trunk Attn (57개 FluxAttention) | 125.4 ± 25.3 | 14.71% of E2E / 20.34% of DiT |
| Other DiT (MLP, norm 등) | 491.4 | 57.6% of E2E |
| Other (scheduler/VAE/io) | 235.9 | 27.7% of E2E |
| **DiT total** | **616.8 ± 83.9** | 72.3% of E2E |
| **E2E** | **852.7 ± 119.1** | 100% |

→ 그래프: [`results/gate_breakdown.png`](results/gate_breakdown.png)
→ Raw: [`results/gate_trunk_attn_share.json`](results/gate_trunk_attn_share.json)

### 판정

**Trunk attn / E2E = 14.71% < PASS threshold 40%** → 🔴 **FAIL**

### 원인 분석

1. **B200의 FlashAttention 최적화**:
   - B200은 SDPA / FlashAttention3가 효율적으로 작동
   - 57개 attention 모듈 합쳐 125ms — DiT의 20%에 불과
   - MLP, GeGLU 등 다른 dense 연산이 60%로 더 dominant

2. **Trunk attention만 공격하면 ceiling 너무 낮음**:
   - 100% 제거해도 e2e gain 14.7% (1.17× max)
   - StructPrior expected gain (1.3-1.5×)에 도달 불가능

3. **ControlNet을 추가하면 더 나빠짐**:
   - Branch가 추가 cost를 늘리므로 trunk attention 비중이 더 작아짐
   - 즉 이 측정은 **upper bound** — 실제 ControlNet 설정에서는 더 fail

### Action

🔴 **StructPrior 중단**.

**잠재적 pivot**:
- **2K+ 고해상도**: 해상도 N² 증가 시 self-attention O(N⁴)로 비대 → attention share 50%+ 가능. 단 이 영역은 SpargeAttn/AdaSpa가 이미 점유
- **Video DiT**: temporal attention이 매우 큼 (50-70%) → HSA, FIS-DiT 등 prior art와 정면 충돌
- 따라서 StructPrior 자체는 폐기 권고

### 다음 단계

이번 PoC 라운드의 마지막 실험. **4개 Conditional Go 모두 quick gate 실패**.

## 파일 안내

| 파일 | 내용 |
|---|---|
| `gate_trunk_attn_share.py` | Quick gate 측정 스크립트 |
| `make_plot.py` | 그래프 생성 |
| `results/gate_trunk_attn_share.json` | 원본 측정값 (per-iter attn/dit/e2e) |
| `results/gate_breakdown.png` | E2E composition + threshold 비교 |
| `results/gate_run.log` | 실행 로그 |
