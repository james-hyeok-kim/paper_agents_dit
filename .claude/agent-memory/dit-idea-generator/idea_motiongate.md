---
name: idea-motiongate
description: Video DiT 특화 — latent-space optical flow로 attention/MLP를 frame-region별 gating. PAB의 timestep caching과 직교하는 motion-structured computation.
metadata:
  type: project
---

# MotionGate — Latent-Flow-Guided Region-Adaptive Computation for Video DiT

**Regime**: Video DiT (CogVideoX, HunyuanVideo, Open-Sora 1.2/2.0)
**Axis**: Spatial-temporal computation gating via motion structure
**Status**: Drafted 2026-05-15

## Core Intuition (왜 작동하는가)

Video는 본질적으로 motion-structured: 대부분의 frame region은 인접 frame과 유사 (background, slow-moving objects), 일부 region만 빠르게 변화 (moving subjects, scene cuts).

기존 video DiT efficiency work (PAB, ∆-DiT, FORA video variant)는 **timestep 축에서만** redundancy를 찾음:
- PAB: spatial/temporal/cross attention 각각 다른 frequency로 cache across timesteps
- 모두 "이 timestep과 다음 timestep 사이의 attention output이 비슷하다"를 가정

하지만 video latent의 진짜 redundancy는 **frame-spatial axis**에 있음:
- frame t의 region A가 frame t+1의 region A와 거의 같다면, attention/MLP를 다시 계산할 필요 없이 spatial copy
- 이는 timestep 축이 아니라 frame 축의 redundancy

핵심 직관: **diffusion latent의 자체 motion은 cheap하게 추정 가능**. 매우 작은 conv head (~2M params)로 latent의 frame-to-frame displacement field를 추정하면, motion magnitude가 낮은 region은 attention/MLP forward를 skip하고 warped neighbor frame의 결과를 reuse 가능.

## Method

1. **Latent flow estimator**: 
   - 입력: 두 인접 frame의 latent (CogVideoX의 경우 16 frame group의 latent)
   - 출력: per-token displacement vector + magnitude
   - 구조: 2-3 layer 3D conv, ~2M params, frozen DiT의 trajectory에서 distillation

2. **Region-adaptive gating**:
   - magnitude < threshold τ인 token: attention/MLP skip → 인접 frame의 같은 spatial location 결과를 warp으로 가져옴
   - magnitude ≥ τ인 token: full forward
   - threshold는 timestep dependent (early step에서는 더 conservative)

3. **Block-wise application**:
   - Spatial attention, temporal attention, MLP 각각에 다른 gating policy
   - Temporal attention은 motion이 큰 region에서 가장 중요 → motion 큰 곳에만 full computation 

4. **Holes handling**:
   - 빠르게 움직이는 object 뒤에 가려졌다 나타나는 region (occlusion hole)은 skip 불가 → flow의 backward/forward consistency check로 검출

5. **Warp operation**:
   - 단순 bilinear warp이 아닌, learnable feature-space warp (Splatting Network 류) — 단 매우 가벼움

## Closest Prior Art + The Move They Didn't Make

**Closest**:
- **PAB (Pyramid Attention Broadcast, 2024)**: timestep-axis caching, 모든 spatial position 동일 처리. 이미 2-3x speedup.
- **AdaCache (2024)**: timestep-adaptive caching schedule, 하지만 여전히 spatial uniform.
- **Optical flow guided video synthesis** (e.g., RAFT, SPyNet 사용): pixel-space, 추론 시점이 아니라 학습 시점 사용. Latent space에서 추론 중 사용은 안 함.

**The single move they didn't make**: **frame-axis spatial redundancy 활용**. PAB는 timestep redundancy, motion-aware video synthesis는 학습 시 motion 활용. 이 둘의 교집합인 **"inference-time, latent-space, frame-axis motion gating"** 공간이 비어있음.

## Discriminating Test (advisor의 기준)

"Does this reduce to caching attention output across timesteps?" → **No**. Cross-frame, intra-timestep gating ✓
"Does this explicitly model motion structure?" → **Yes**. Latent flow가 핵심 ✓

## Expected Gain
- CogVideoX-5B 기준: 추가 1.6-2.0x speedup, **PAB와 곱셈적 (orthogonal axis)**
- Combined with PAB: 4-5x total speedup vs. baseline
- VBench score drop <2%
- Static scene (low motion)일수록 이득 커짐 → talking head, slow cinematic shot에서 특히 강함

## 실현 가능성 / 위험
- **실현 가능**: 단일 H100 노드에서 학습 가능 (flow head만 학습). 5K video clip이면 충분.
- **위험 1**: 빠른 motion 또는 scene cut에서 quality 급락 → conservative threshold + scene cut detector 필요
- **위험 2**: warp의 미세 misalignment가 누적되면 flicker 발생 → temporal consistency loss로 학습 시 regularize
- **위험 3**: HunyuanVideo의 3D full attention에서는 spatial gating이 attention 구조와 충돌할 수 있음 → 우선 CogVideoX (factorized attention)에서 검증

## Scores
Novelty 5 / Impact 5 / Feasibility 3 / PublishRisk 3 / Timeline 3

## Publication Target
SIGGRAPH 2026 또는 CVPR 2027. Video generation efficiency는 hot한 분야이고, motion-aware라는 직관적 hook이 강함.

## Next Step
dit-literature-checker로 다음 검증 — **inference-time vs training-time 사용 명시적으로 구분**:
- "motion-aware video diffusion **inference**" (학습 시 motion guidance와 구별)
- "latent flow video generation **inference**"
- "frame redundancy video diffusion"
- "region-adaptive computation video"
- 2025년 발표된 PAB 후속작 / AdaCache 변형 / FORA-video 류

**Specific landmines (advisor 지적)** — 각각이 inference-time, latent-flow, region-adaptive gating을 하는지 명확히 확인:
- **TeaCache** (timestep embedding-aware caching, late 2024) + video variants
- **AdaCache** video extensions
- **Sparse-VideoGen** / motion-sparse attention (2025)
- **Object-Centric Video Generation** with motion priors at inference
- **VideoFlow / FlowVid / MotionCtrl** — flow를 conditioning/training용으로 쓰는지, inference gating용인지 확인 (후자라면 collapse risk)
- 특히 CogVideoX 또는 HunyuanVideo 위에서 motion-guided efficiency 시도한 work이 있는지

**Scoring 주의**: 아래 점수는 lit-check 이전 self-estimate. 확정 novelty는 lit-check 후.

Related: [[idea-trajleap]]는 image의 timestep axis, 이건 video의 frame axis. 직교적.
