"""
StructPrior Quick Gate: Trunk self-attention share in FLUX-schnell.

The full StructPrior gate (validation 결과) requires FLUX-ControlNet to profile trunk vs branch.
For a quick PoC without ControlNet, we measure the **upper bound** of the achievable speedup:
how much of e2e is taken by the trunk's self-attention modules?

If trunk attention < 30% of e2e → even 100% removal yields <30% e2e gain → idea ceiling too low.

Decision rule:
  - trunk_attn >= 40% of e2e  -> PASS, proceed to Gate 2 (with ControlNet)
  - trunk_attn 25-40%         -> BORDERLINE
  - trunk_attn < 25%          -> FAIL, idea ceiling fundamentally low
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch, time, json
import numpy as np
from pathlib import Path
from collections import defaultdict
from diffusers import FluxPipeline
from diffusers.models.transformers.transformer_flux import FluxAttention

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEVICE = "cuda:3"
DTYPE = torch.bfloat16
N_WARMUP, N_ITERS = 2, 8
WIDTH = HEIGHT = 1024
NUM_INFERENCE_STEPS = 4
PROMPT = "A photo of a cat sitting on a sofa, cinematic lighting, high detail"
RESULTS_DIR = Path(__file__).parent / "results"


def hook_all_attn(model, store):
    """Wrap every Attention module's forward to log CUDA times."""
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, FluxAttention):
            orig = module.forward
            def make_wrap(orig_fn, mod_name):
                def wrapped(*args, **kwargs):
                    torch.cuda.synchronize(DEVICE)
                    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    out = orig_fn(*args, **kwargs)
                    e.record()
                    torch.cuda.synchronize(DEVICE)
                    store[mod_name].append(s.elapsed_time(e))
                    return out
                return wrapped
            module.forward = make_wrap(orig, name)
            handles.append((module, orig))
    return handles


def main():
    print(f"=== StructPrior Quick Gate: Trunk Self-Attention Share ===")
    print(f"GPU: {torch.cuda.get_device_name(int(DEVICE.split(':')[-1]))}")
    print(f"Loading {MODEL_ID} (bf16)...")
    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    pipe = pipe.to(DEVICE)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    per_call = defaultdict(list)
    handles = hook_all_attn(pipe.transformer, per_call)
    n_attn = len(handles)
    print(f"  Hooked {n_attn} Attention modules")

    # Total DiT time hook
    orig_dit = pipe.transformer.forward
    dit_times = []
    def dit_hook(*a, **k):
        torch.cuda.synchronize(DEVICE)
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); out = orig_dit(*a, **k); e.record()
        torch.cuda.synchronize(DEVICE)
        dit_times.append(s.elapsed_time(e))
        return out
    pipe.transformer.forward = dit_hook

    # Warmup
    print(f"\nWarmup ({N_WARMUP})...")
    for _ in range(N_WARMUP):
        per_call.clear(); dit_times.clear()
        _ = pipe(prompt=PROMPT, height=HEIGHT, width=WIDTH,
                 num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=0.0,
                 max_sequence_length=512,
                 generator=torch.Generator(device=DEVICE).manual_seed(0)).images[0]

    print(f"\nMeasuring ({N_ITERS} inferences)...")
    iter_attn_total = []
    iter_dit_total = []
    iter_e2e_total = []

    for i in range(N_ITERS):
        per_call.clear(); dit_times.clear()
        torch.cuda.synchronize(DEVICE)
        t0 = time.perf_counter()
        _ = pipe(prompt=PROMPT, height=HEIGHT, width=WIDTH,
                 num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=0.0,
                 max_sequence_length=512,
                 generator=torch.Generator(device=DEVICE).manual_seed(0)).images[0]
        torch.cuda.synchronize(DEVICE)
        e2e_ms = (time.perf_counter() - t0) * 1000.0

        total_attn = sum(sum(v) for v in per_call.values())
        total_dit = sum(dit_times)
        iter_attn_total.append(total_attn)
        iter_dit_total.append(total_dit)
        iter_e2e_total.append(e2e_ms)
        print(f"  iter {i+1}: attn_total={total_attn:7.1f}  dit_total={total_dit:7.1f}  e2e={e2e_ms:7.1f}  attn/e2e={total_attn/e2e_ms*100:5.2f}%  attn/dit={total_attn/total_dit*100:5.2f}%")

    pipe.transformer.forward = orig_dit
    for mod, orig in handles:
        mod.forward = orig

    a = np.array(iter_attn_total); d = np.array(iter_dit_total); e = np.array(iter_e2e_total)
    a_mean, d_mean, e_mean = a.mean(), d.mean(), e.mean()
    attn_e2e_share = a_mean / e_mean * 100
    attn_dit_share = a_mean / d_mean * 100

    print("\n" + "=" * 70)
    print(f"Summary ({N_ITERS} iterations):")
    print(f"  Total attn time:  {a_mean:7.1f} ± {a.std():5.1f} ms")
    print(f"  Total DiT time:   {d_mean:7.1f} ± {d.std():5.1f} ms")
    print(f"  E2E time:         {e_mean:7.1f} ± {e.std():5.1f} ms")
    print(f"  Attn / E2E:       {attn_e2e_share:5.2f}%   (this is the speedup CEILING if all attn is free)")
    print(f"  Attn / DiT:       {attn_dit_share:5.2f}%")

    if attn_e2e_share >= 40:
        verdict = "PASS"
        action = "Proceed to Gate 2 (ControlNet trunk vs branch + naive geometric mask quality drop)"
    elif attn_e2e_share >= 25:
        verdict = "BORDERLINE"
        action = "Ceiling marginal — Gate 2를 진행하되 e2e 기대치 1.2-1.3x로 낮춰야"
    else:
        verdict = "FAIL"
        action = "Trunk attention share 너무 작아 e2e ceiling 부족. StructPrior 중단 권고"
    print(f"\nDECISION: Trunk attn / E2E = {attn_e2e_share:.2f}%  →  {verdict}")
    print(f"  Action: {action}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "gate_trunk_attn_share.json", "w") as f:
        json.dump({
            "config": {
                "model": MODEL_ID, "device": DEVICE, "dtype": str(DTYPE),
                "width": WIDTH, "height": HEIGHT, "steps": NUM_INFERENCE_STEPS,
                "n_iters": N_ITERS, "n_attn_modules": n_attn,
                "gpu": torch.cuda.get_device_name(int(DEVICE.split(':')[-1])),
            },
            "iter_attn_total_ms": iter_attn_total,
            "iter_dit_total_ms": iter_dit_total,
            "iter_e2e_total_ms": iter_e2e_total,
            "summary": {
                "attn_mean_ms": float(a_mean), "attn_std_ms": float(a.std()),
                "dit_mean_ms":  float(d_mean), "dit_std_ms":  float(d.std()),
                "e2e_mean_ms":  float(e_mean), "e2e_std_ms":  float(e.std()),
                "attn_e2e_share_pct": float(attn_e2e_share),
                "attn_dit_share_pct": float(attn_dit_share),
            },
            "verdict": verdict, "action": action,
        }, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR / 'gate_trunk_attn_share.json'}")


if __name__ == "__main__":
    main()
