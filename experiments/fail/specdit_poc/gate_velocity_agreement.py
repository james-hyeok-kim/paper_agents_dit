"""
SpecDiT Quick Gate: Velocity agreement between full DiT and a "half-sized" draft.

Trained shallow drafter는 시간이 오래 걸리므로, **layer-skipping을 draft proxy**로 사용:
- Full DiT (target): 19 MMDiT + 38 single-stream blocks
- Draft proxy: 절반 블록만 활성화 (every-other-block은 residual passthrough)

Test:
  At each denoising step, run target then draft (same input).
  Compute relative error  e_t = ||v_target - v_draft|| / ||v_target||
  Acceptance rate at threshold τ ∈ {0.05, 0.10, 0.15, 0.20, 0.30}.

Decision rule:
  - τ=0.10에서 acceptance rate ≥ 70% → BORDERLINE PASS (γ_eff ≥ 2.3)
  - τ=0.10에서 acceptance rate ≥ 50% → MARGINAL (γ_eff ≈ 1-2)
  - τ=0.10에서 acceptance rate < 30% → FAIL (compounding error)

(주의: layer-skip은 trained drafter보다 strictly worse 가능. PASS이면 trained drafter는 더 잘할 가능성.)
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import time
import json
import numpy as np
from pathlib import Path
from diffusers import FluxPipeline

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEVICE = "cuda:2"
DTYPE = torch.bfloat16
NUM_INFERENCE_STEPS = 4
WIDTH = HEIGHT = 1024
RESULTS_DIR = Path(__file__).parent / "results"

PROMPTS = [
    "A photo of a cat sitting on a sofa, cinematic lighting",
    "A serene mountain landscape at sunrise with mist",
    "An astronaut riding a horse on Mars, photorealistic",
    "A bowl of ramen with steam rising, top-down view",
    "An ancient library with books floating in the air",
    "A red sports car parked in front of a modern building",
    "A close-up portrait of an old fisherman with weathered face",
    "A futuristic city skyline at night with neon lights",
]
TAUS = [0.05, 0.10, 0.15, 0.20, 0.30]


class SkipController:
    """Monkey-patch each block's forward to optionally short-circuit (residual passthrough)."""
    def __init__(self, transformer):
        self.transformer = transformer
        self.n_mm = len(transformer.transformer_blocks)
        self.n_single = len(transformer.single_transformer_blocks)
        self.mm_skip = [False] * self.n_mm
        self.single_skip = [False] * self.n_single
        self._patch()

    def _patch(self):
        for idx, blk in enumerate(self.transformer.transformer_blocks):
            orig = blk.forward
            skip_list = self.mm_skip
            local_idx = idx
            def make_mm(orig_fn, i):
                def new_forward(*args, **kwargs):
                    if skip_list[i]:
                        hs = kwargs.get("hidden_states", args[0] if args else None)
                        ehs = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
                        return ehs, hs
                    return orig_fn(*args, **kwargs)
                return new_forward
            blk.forward = make_mm(orig, local_idx)

        for idx, blk in enumerate(self.transformer.single_transformer_blocks):
            orig = blk.forward
            skip_list = self.single_skip
            local_idx = idx
            def make_single(orig_fn, i):
                def new_forward(*args, **kwargs):
                    if skip_list[i]:
                        hs = kwargs.get("hidden_states", args[0] if args else None)
                        ehs = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
                        return ehs, hs   # newer diffusers expects tuple
                    return orig_fn(*args, **kwargs)
                return new_forward
            blk.forward = make_single(orig, local_idx)

    def enable_draft(self):
        for i in range(self.n_mm):
            self.mm_skip[i] = (i % 2 == 1)
        for i in range(self.n_single):
            self.single_skip[i] = (i % 2 == 1)

    def enable_full(self):
        for i in range(self.n_mm):
            self.mm_skip[i] = False
        for i in range(self.n_single):
            self.single_skip[i] = False


def capture_velocities(pipe, prompt, n_steps=NUM_INFERENCE_STEPS):
    """Run pipeline, capture velocity prediction at each step."""
    velocities = []
    orig_forward = pipe.transformer.forward
    def hook(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        v = out[0] if isinstance(out, tuple) else out.sample
        velocities.append(v.detach().clone())
        return out
    pipe.transformer.forward = hook
    try:
        _ = pipe(
            prompt=prompt, height=HEIGHT, width=WIDTH,
            num_inference_steps=n_steps, guidance_scale=0.0,
            max_sequence_length=512,
            generator=torch.Generator(device=DEVICE).manual_seed(42),
        ).images[0]
    finally:
        pipe.transformer.forward = orig_forward
    return velocities


def main():
    print(f"=== SpecDiT Quick Gate: Velocity Agreement (Layer-Skip Draft Proxy) ===")
    print(f"GPU: {torch.cuda.get_device_name(int(DEVICE.split(':')[-1]))}")
    print(f"Loading {MODEL_ID} (bf16)...")
    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    pipe = pipe.to(DEVICE)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  MMDiT blocks: {len(pipe.transformer.transformer_blocks)}, Single-stream: {len(pipe.transformer.single_transformer_blocks)}")

    ctrl = SkipController(pipe.transformer)
    print(f"  Draft mode: skip every-other block (~50% active)")

    per_step_errs = {step: [] for step in range(NUM_INFERENCE_STEPS)}

    for p_idx, prompt in enumerate(PROMPTS):
        print(f"\n[{p_idx+1}/{len(PROMPTS)}] {prompt[:60]}")
        ctrl.enable_full()
        v_target = capture_velocities(pipe, prompt)
        ctrl.enable_draft()
        v_draft = capture_velocities(pipe, prompt)
        ctrl.enable_full()

        for step in range(NUM_INFERENCE_STEPS):
            vt, vd = v_target[step].float(), v_draft[step].float()
            rel_err = float((vt - vd).norm() / vt.norm().clamp(min=1e-8))
            per_step_errs[step].append(rel_err)
        msg = "  step rel_errs: " + " ".join(f"{per_step_errs[s][-1]:.3f}" for s in range(NUM_INFERENCE_STEPS))
        print(msg)
        torch.cuda.empty_cache()

    print("\n--- Per-step stats ---")
    all_errs = []
    for step, errs in per_step_errs.items():
        arr = np.array(errs)
        print(f"  step {step}: mean={arr.mean():.4f} median={np.median(arr):.4f} max={arr.max():.4f}")
        all_errs.extend(errs)
    all_errs = np.array(all_errs)
    print(f"\nOverall ({len(all_errs)} measurements):")
    print(f"  mean={all_errs.mean():.4f}  median={np.median(all_errs):.4f}  max={all_errs.max():.4f}")

    print("\n--- Acceptance rates (across all step-prompt pairs) ---")
    accept_rates = {}
    for tau in TAUS:
        rate = float((all_errs <= tau).mean() * 100)
        accept_rates[tau] = rate
        print(f"  τ={tau:.2f}: {rate:5.1f}%")

    p10 = accept_rates[0.10] / 100
    gamma_eff = p10 / (1 - p10) if p10 < 1.0 else float("inf")
    print(f"\n  At τ=0.10: per-step accept = {p10:.3f}  -->  γ_effective ~ {gamma_eff:.2f}")

    if p10 >= 0.70:
        verdict = "BORDERLINE PASS"
        action = "Trained drafter는 layer-skip proxy보다 잘할 가능성. SpecDiT 본격 실험 고려"
    elif p10 >= 0.50:
        verdict = "MARGINAL"
        action = "γ_effective 1-2 수준. SpeCa 대비 head-to-head 어려움. SpecCache+ pivot 고려"
    else:
        verdict = "FAIL"
        action = "compounding error 너무 큼. SpecDiT 중단 또는 새 acceptance criterion 필요"

    print("\n" + "=" * 70)
    print(f"DECISION: per-step accept @ τ=0.10 = {p10*100:.1f}%")
    print(f"  Estimated γ_effective: {gamma_eff:.2f}")
    print(f"  Verdict: {verdict}")
    print(f"  Action: {action}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "gate_velocity_agreement.json", "w") as f:
        json.dump({
            "config": {
                "model": MODEL_ID, "dtype": str(DTYPE), "device": DEVICE,
                "n_steps": NUM_INFERENCE_STEPS, "width": WIDTH, "height": HEIGHT,
                "n_prompts": len(PROMPTS), "taus": TAUS,
                "draft_strategy": "every-other-block-skip (residual passthrough)",
                "gpu": torch.cuda.get_device_name(int(DEVICE.split(':')[-1])),
            },
            "per_step_errs": {str(k): v for k, v in per_step_errs.items()},
            "overall_stats": {
                "mean": float(all_errs.mean()), "median": float(np.median(all_errs)),
                "max": float(all_errs.max()), "min": float(all_errs.min()),
            },
            "acceptance_rates": accept_rates,
            "p10": p10, "gamma_effective_estimate": gamma_eff,
            "verdict": verdict, "action": action,
        }, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR / 'gate_velocity_agreement.json'}")


if __name__ == "__main__":
    main()
