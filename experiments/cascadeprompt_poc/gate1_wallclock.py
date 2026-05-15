"""
CascadePrompt Gate 1: Measure T5-XXL wallclock share in FLUX-schnell 4-step inference.

Approach: hook each component's forward to record CUDA time, then run the standard pipeline.

Decision rule:
  - T5-XXL >= 15% of e2e -> PASS, proceed to Gate 2
  - T5-XXL <  10%        -> FAIL, motivation weak, stop
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import time
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from diffusers import FluxPipeline

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEVICE = "cuda:1"   # GPU 1 (less contention)
DTYPE = torch.bfloat16
N_WARMUP = 2
N_ITERS = 10
NUM_INFERENCE_STEPS = 4
WIDTH = HEIGHT = 1024
PROMPT = "A photo of a cat sitting on a sofa, cinematic lighting, high detail"
RESULTS_PATH = Path(__file__).parent / "results" / "gate1_wallclock.json"


class TimedHook:
    """Wrap nn.Module.forward to record CUDA event times per call."""
    def __init__(self, module, name, store):
        self.module = module
        self.name = name
        self.store = store          # dict: name -> list of float (ms)
        self.orig = module.forward

        def wrapped(*args, **kwargs):
            torch.cuda.synchronize(DEVICE)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = self.orig(*args, **kwargs)
            e.record()
            torch.cuda.synchronize(DEVICE)
            self.store[self.name].append(s.elapsed_time(e))
            return out

        module.forward = wrapped

    def restore(self):
        self.module.forward = self.orig


def main():
    print(f"=== CascadePrompt Gate 1 ===")
    print(f"GPU: {torch.cuda.get_device_name(int(DEVICE.split(':')[-1]))}")
    print(f"Loading {MODEL_ID} (bf16)...")
    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    pipe = pipe.to(DEVICE)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Image: {WIDTH}x{HEIGHT}, steps: {NUM_INFERENCE_STEPS}")
    print(f"  Prompt: {PROMPT!r}")

    # Per-iteration storage (cleared each iter)
    per_call = defaultdict(list)

    # Wrap component forwards. Note: pipeline calls vae.decode() not vae(), so hook decode directly.
    class TimedMethod:
        def __init__(self, obj, attr, name, store):
            self.obj, self.attr, self.name, self.store = obj, attr, name, store
            self.orig = getattr(obj, attr)
            def wrapped(*a, **kw):
                torch.cuda.synchronize(DEVICE)
                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record(); out = self.orig(*a, **kw); e.record()
                torch.cuda.synchronize(DEVICE)
                store[name].append(s.elapsed_time(e))
                return out
            setattr(obj, attr, wrapped)
        def restore(self):
            setattr(self.obj, self.attr, self.orig)

    hooks = [
        TimedHook(pipe.text_encoder,    "clip_l",       per_call),
        TimedHook(pipe.text_encoder_2,  "t5_xxl",       per_call),
        TimedHook(pipe.transformer,     "dit_per_step", per_call),
        TimedMethod(pipe.vae, "decode",  "vae_decode",  per_call),
    ]

    # Per-iteration aggregate
    per_iter = defaultdict(list)

    print(f"\nWarmup ({N_WARMUP} iters)...")
    for i in range(N_WARMUP):
        per_call.clear()
        _ = pipe(prompt=PROMPT, height=HEIGHT, width=WIDTH,
                 num_inference_steps=NUM_INFERENCE_STEPS,
                 guidance_scale=0.0, max_sequence_length=512,
                 generator=torch.Generator(device=DEVICE).manual_seed(0)).images[0]

    print(f"\nMeasuring ({N_ITERS} iters)...")
    for i in range(N_ITERS):
        per_call.clear()
        torch.cuda.synchronize(DEVICE)
        e2e_s = torch.cuda.Event(enable_timing=True)
        e2e_e = torch.cuda.Event(enable_timing=True)
        e2e_s.record()
        _ = pipe(prompt=PROMPT, height=HEIGHT, width=WIDTH,
                 num_inference_steps=NUM_INFERENCE_STEPS,
                 guidance_scale=0.0, max_sequence_length=512,
                 generator=torch.Generator(device=DEVICE).manual_seed(0)).images[0]
        e2e_e.record()
        torch.cuda.synchronize(DEVICE)
        e2e_ms = e2e_s.elapsed_time(e2e_e)

        # Aggregate this iteration's per-call timings
        clip_ms = sum(per_call.get("clip_l", []))
        t5_ms   = sum(per_call.get("t5_xxl", []))
        dit_ms  = sum(per_call.get("dit_per_step", []))
        vae_ms  = sum(per_call.get("vae_decode", []))
        per_iter["clip_l"].append(clip_ms)
        per_iter["t5_xxl"].append(t5_ms)
        per_iter["dit_total"].append(dit_ms)
        per_iter["vae_total"].append(vae_ms)
        per_iter["e2e"].append(e2e_ms)

        n_dit_calls = len(per_call.get("dit_per_step", []))
        print(f"  iter {i+1:2d}/{N_ITERS}: "
              f"CLIP={clip_ms:6.1f}  T5={t5_ms:7.1f}  "
              f"DiT[{n_dit_calls}calls]={dit_ms:7.1f}  "
              f"VAE={vae_ms:6.1f}  E2E={e2e_ms:7.1f}  (ms)")

    for h in hooks:
        h.restore()

    # Stats
    stats = {k: {"mean_ms": float(np.mean(v)), "std_ms": float(np.std(v)), "raw": v}
             for k, v in per_iter.items()}
    e2e_mean = stats["e2e"]["mean_ms"]
    pct = {k: stats[k]["mean_ms"] / e2e_mean * 100
           for k in ["t5_xxl", "clip_l", "dit_total", "vae_total"]}
    other_pct = 100 - sum(pct.values())

    print("\n" + "=" * 70)
    print(f"Component breakdown (mean ± std over {N_ITERS} iters):")
    print(f"  T5-XXL encode:  {stats['t5_xxl']['mean_ms']:7.1f} ± {stats['t5_xxl']['std_ms']:5.1f} ms  ({pct['t5_xxl']:5.2f}%)")
    print(f"  CLIP-L encode:  {stats['clip_l']['mean_ms']:7.1f} ± {stats['clip_l']['std_ms']:5.1f} ms  ({pct['clip_l']:5.2f}%)")
    print(f"  DiT (4 step):   {stats['dit_total']['mean_ms']:7.1f} ± {stats['dit_total']['std_ms']:5.1f} ms  ({pct['dit_total']:5.2f}%)")
    print(f"  VAE total:      {stats['vae_total']['mean_ms']:7.1f} ± {stats['vae_total']['std_ms']:5.1f} ms  ({pct['vae_total']:5.2f}%)")
    print(f"  Other (sched/io): {other_pct:5.2f}%")
    print(f"  E2E:            {stats['e2e']['mean_ms']:7.1f} ± {stats['e2e']['std_ms']:5.1f} ms")

    t5_pct = pct["t5_xxl"]
    if t5_pct >= 15:
        verdict = "PASS"
        action = "Proceed to Gate 2 (per-prompt heavy-tail histogram)"
    elif t5_pct < 10:
        verdict = "FAIL"
        action = "Motivation weak, stop CascadePrompt"
    else:
        verdict = "BORDERLINE"
        action = "Consider Gate 2 with reduced expectation"

    print("\n" + "=" * 70)
    print(f"DECISION: T5-XXL share = {t5_pct:.2f}%")
    print(f"  Gate 1 verdict: {verdict}")
    print(f"  Action: {action}")
    print("=" * 70)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "config": {
                "model": MODEL_ID, "dtype": str(DTYPE), "device": DEVICE,
                "width": WIDTH, "height": HEIGHT, "steps": NUM_INFERENCE_STEPS,
                "n_warmup": N_WARMUP, "n_iters": N_ITERS, "prompt": PROMPT,
                "gpu": torch.cuda.get_device_name(int(DEVICE.split(':')[-1])),
            },
            "stats": stats,
            "percentages": pct,
            "other_pct": other_pct,
            "verdict": verdict,
            "action": action,
            "t5_xxl_share_pct": t5_pct,
        }, f, indent=2)
    print(f"\nResults: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
