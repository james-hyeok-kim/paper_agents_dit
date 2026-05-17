# DecoupledVAE PoC

**아이디어**: DiT 중간 step에서 x0-estimate로 VAE decode를 별도 stream에서 미리 시작, 남은 DiT step과 overlap. Cheap refiner로 quality 보강하여 expensive VAE decoder를 critical path에서 제거.
**Validation 점수**: 3.0/5 (Conditional Go 중 2위)
**원본 idea 명세**: `.claude/agent-memory/dit-idea-generator/idea_decoupledvae.md`

## 실험 목적

이 idea의 핵심 가정은 **"단일 GPU에서 DiT block forward와 VAE decode를 두 CUDA stream으로 돌리면 wallclock이 줄어든다"**는 것입니다. 만약 HBM bandwidth contention 때문에 사실상 overlap이 안 된다면, idea 자체가 single GPU에서는 무의미해지고 PipeDiT(inter-prompt 2-GPU)와 같은 영역으로 환원됩니다.

## 게이트 정의

| Gate | 측정 | PASS 기준 |
|---|---|---|
| **A** | DiT-block + VAE.decode를 두 stream에서 동시 실행 시 wallclock overlap factor | ≥ 1.30× |
| **B** | (PASS 시) Mid-trajectory x̂_0의 LPIPS curve — 어느 k에서 partial decode quality가 충분한가 | LPIPS < 0.15 구간이 ≥ 30% 남은 step과 공존 |
| **C** | (PASS 시) 30M params refiner가 partial decode + latent residual을 받아 final image를 복원 가능한가 | Flash-VAED baseline 이김 |

## Gate A 결과: 🔴 **FAIL (BORDERLINE)**

### Setup
- **GPU**: NVIDIA B200 (191.5GB), GPU 1 (다른 process와 공유)
- **Model**: black-forest-labs/FLUX.1-schnell, bf16
- **컴포넌트**: `pipe.transformer.forward()` (DiT block) + `pipe.vae.decode()`
- **두 stream**: `torch.cuda.Stream()` 두 개 생성, 각각 호출 후 `synchronize()`
- **반복**: 10 warmup + 50 iters, **median ± trimmed std** (top/bottom 10% 제외)

### 측정값

**1024×1024**:
| 측정 | Median (ms) | std (ms) |
|---|---|---|
| DiT-block only | 122.74 | ±10.91 |
| VAE.decode only | 93.73 | ±3.35 |
| Serial (둘 다 sequential) | 216.56 | ±17.21 |
| Concurrent (두 stream) | 185.88 | ±10.56 |
| **Overlap factor** | **1.165×** | |
| Theoretical max (perfect overlap) | 1.764× | |
| Overlap efficiency | 32.7% | |

**2048×2048**:
| 측정 | Median (ms) | std (ms) |
|---|---|---|
| DiT-block only | 525.71 | ±47.16 |
| VAE.decode only | 400.04 | ±22.22 |
| Serial | 922.91 | ±24.72 |
| Concurrent | 845.03 | ±25.85 |
| **Overlap factor** | **1.092×** | |
| Theoretical max | 1.756× | |
| Overlap efficiency | 19.5% | |

→ 그래프: [`results/gateA_breakdown.png`](results/gateA_breakdown.png)
→ Raw: [`results/gateA_stream_overlap.json`](results/gateA_stream_overlap.json)

### 판정

**최고 overlap factor 1.165× < PASS 임계점 1.30×** → 🔴 **FAIL/BORDERLINE**

### 원인 분석

1. **HBM bandwidth contention이 본질적 병목**:
   - DiT (attention)와 VAE (3D/2D conv) 모두 memory-bound 워크로드
   - 두 kernel이 같은 HBM을 두고 경쟁 → 동시 실행해도 1.7× 이론 최대치 중 19-33% 효율만 달성
   - 1024×1024에서 saving 30.68ms만 (가능한 93.73ms 중 33%)
   - 2048×2048에서 saving 77.88ms만 (가능한 400.04ms 중 19%)

2. **B200처럼 fat GPU일수록 contention 비율 더 높음**:
   - 두 kernel이 SM 점유율 합쳐 100% 넘기지 못함
   - PipeDiT가 **two GPU groups**로 분리한 이유가 정확히 이것 (advisor 사전 경고와 일치)

3. **이 결과의 의미**:
   - DecoupledVAE의 "intra-prompt critical-path 단축" 주장은 단일 GPU에서 1.4× 달성 불가
   - PipeDiT (inter-prompt, 2-GPU)와 직접 경쟁해야 함 → "intra vs inter prompt" 차별화 약함
   - 원래 expected gain (Image 1.15-1.25×, Video 1.4-1.7×)도 이 측정과 부합하지 않음

### Action

🔴 **DecoupledVAE 중단 (단일 GPU framing)**.

**Pivot 옵션** (필요 시):
- **A. Two-GPU framing으로 reframe**: PipeDiT 직접 비교 baseline 필요. 차별화는 "intra-prompt parallelism"인데 PipeDiT는 inter-prompt만 → 새로운 paper 각도 가능. 단, GPU 2장 필요한 솔루션은 deploy 비용 큼
- **B. Video DiT (CogVideoX)에서만 측정**: 3D VAE는 단일 GPU에서도 큰 비중. 같은 실험을 CogVideoX VAE로 재측정 시 다른 결과 가능
- **C. 이론 reframe**: "trajectory geometry finding"으로 systems trick이 아닌 discovery paper로 sell

### 다음 실험

Conditional Go 세 번째 (점수 2.6/5): **SpecDiT Gate** — DiT-XL/2에서 trained shallow drafter의 γ_effective와 acceptance rate 측정.

## 파일 안내

| 파일 | 내용 |
|---|---|
| `gateA_stream_overlap.py` | Gate A 측정 스크립트 |
| `make_plot.py` | 결과 그래프 생성 |
| `results/gateA_stream_overlap.json` | 원본 측정값 (per resolution) |
| `results/gateA_breakdown.png` | Serial vs Concurrent 비교 + overlap factor |
| `results/gateA_run.log` | 전체 실행 로그 |
