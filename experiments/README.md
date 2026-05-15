# experiments/

DiT inference efficiency 실제 실험 코드 + 결과 저장소.

## 디렉토리 구조

각 실험은 자체 폴더를 가짐:

```
experiments/
├── README.md
├── {idea_name}_poc/          # PoC / gate 검증 실험
│   ├── README.md             # 실험 목적, 방법, 결과
│   ├── *.py                  # 실험 스크립트
│   └── results/              # 측정값, 로그, 그래프
└── {idea_name}_full/         # 전체 baseline 실험
    └── ...
```

## 진행 예정 실험

Conditional Go 4개 idea의 게이트 실험 (점수 순):

| 점수 | Idea | Gate 실험 | 예상 시간 |
|---|---|---|---|
| 3.4/5 | CascadePrompt | T5-XXL wallclock 비중 + per-prompt heavy-tail histogram | 1-2일 |
| 3.0/5 | DecoupledVAE | CUDA stream overlap microbench + x0-estimate quality curve | 3-4일 |
| 2.6/5 | SpecDiT | DiT-XL/2 drafter γ_effective 측정 | 2주 |
| 2.4/5 | StructPrior | FLUX-ControlNet trunk vs branch profiling + naive mask | 2일 |

## 아이디어 명세 위치

각 아이디어의 상세 명세, 문헌 검토, 종합 검증 결과는:
- `.claude/agent-memory/dit-idea-generator/idea_*.md`
- `.claude/agent-memory/dit-literature-checker/*-novelty-*.md`
- `.claude/agent-memory/dit-idea-validator/validation_*.md`

전체 아이디어 현황 (Conditional Go / No-Go) 요약:
- `.claude/agent-memory/dit-doc-organizer/failed_ideas_blacklist.md`
