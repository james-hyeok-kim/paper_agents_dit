"""
DecoupledVAE Gate A: Single-GPU CUDA stream overlap factor for DiT-block + VAE-decode.

Decision rule:
  - overlap factor >= 1.30  -> PASS, proceed to Gate B
  - overlap factor <  1.10  -> FAIL, idea doesn't work on single GPU (pivot to two-GPU)
  - 1.10-1.30              -> BORDERLINE

Setup:
  - Compare serial(DiT_block + VAE.decode) vs concurrent on two streams
  - Test at resolution 1024 and 2048 (VAE share grows quadratically)
  - DiT input = packed latents of one timestep forward (simulates 'remaining DiT step')
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
DEVICE = "cuda:1"
DTYPE = torch.bfloat16
N_WARMUP = 10
N_ITERS = 50
RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = Path("/data/jameskimh/decoupledvae_poc")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def cuda_time_ms(fn, n_warmup=N_WARMUP, n_iters=N_ITERS):
    """Time fn() with perf_counter + cuda sync (works with multi-stream). Returns (mean_ms, std_ms, raw_list)."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize(DEVICE)
    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize(DEVICE)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(DEVICE)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    arr = np.array(times)
    # Robust stats: drop top/bottom 10% as outliers (common pattern for noisy GPU measurements)
    q10, q90 = np.percentile(arr, [10, 90])
    trimmed = arr[(arr >= q10) & (arr <= q90)]
    return float(np.median(trimmed)), float(np.std(trimmed)), times


def prepare_dit_inputs(pipe, height, width, batch_size=1):
    """Prepare realistic DiT inputs at given resolution."""
    H, W = height // pipe.vae_scale_factor // 2, width // pipe.vae_scale_factor // 2
    # Packed latents [B, H*W, 64] (FLUX format)
    latents = torch.randn(batch_size, H * W, 64, device=DEVICE, dtype=DTYPE)
    # text embeddings
    prompt_embeds = torch.randn(batch_size, 512, 4096, device=DEVICE, dtype=DTYPE)
    pooled = torch.randn(batch_size, 768, device=DEVICE, dtype=DTYPE)
    txt_ids = torch.zeros(512, 3, device=DEVICE, dtype=DTYPE)
    img_ids = pipe._prepare_latent_image_ids(batch_size, H, W, DEVICE, DTYPE)
    timestep = torch.tensor([0.5], device=DEVICE, dtype=DTYPE).expand(batch_size)
    guidance = torch.tensor([0.0], device=DEVICE, dtype=DTYPE).expand(batch_size) \
        if pipe.transformer.config.guidance_embeds else None

    # Unpacked latents for VAE [B, 16, H*2, W*2]
    unpacked = torch.randn(batch_size, 16, H * 2, W * 2, device=DEVICE, dtype=DTYPE)

    return {
        "dit": dict(
            hidden_states=latents, encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled, timestep=timestep / 1000,
            txt_ids=txt_ids, img_ids=img_ids, guidance=guidance,
            return_dict=False,
        ),
        "vae_latents": unpacked,
    }


def benchmark_at_resolution(pipe, resolution):
    print(f"\n--- Resolution: {resolution}x{resolution} ---")
    inputs = prepare_dit_inputs(pipe, resolution, resolution)

    # Closures (without sync inside the fn, since cuda_time_ms handles it)
    def fn_dit_only():
        with torch.no_grad():
            _ = pipe.transformer(**inputs["dit"])

    def fn_vae_only():
        with torch.no_grad():
            _ = pipe.vae.decode(inputs["vae_latents"] / pipe.vae.config.scaling_factor
                                + pipe.vae.config.shift_factor, return_dict=False)[0]

    def fn_serial():
        with torch.no_grad():
            _ = pipe.transformer(**inputs["dit"])
            _ = pipe.vae.decode(inputs["vae_latents"] / pipe.vae.config.scaling_factor
                                + pipe.vae.config.shift_factor, return_dict=False)[0]

    # Concurrent: two streams
    stream_dit = torch.cuda.Stream(device=DEVICE)
    stream_vae = torch.cuda.Stream(device=DEVICE)

    def fn_concurrent():
        with torch.no_grad():
            with torch.cuda.stream(stream_dit):
                _ = pipe.transformer(**inputs["dit"])
            with torch.cuda.stream(stream_vae):
                _ = pipe.vae.decode(inputs["vae_latents"] / pipe.vae.config.scaling_factor
                                    + pipe.vae.config.shift_factor, return_dict=False)[0]
            stream_dit.synchronize()
            stream_vae.synchronize()

    # Measure
    dit_mean, dit_std, _ = cuda_time_ms(fn_dit_only)
    vae_mean, vae_std, _ = cuda_time_ms(fn_vae_only)
    ser_mean, ser_std, _ = cuda_time_ms(fn_serial)
    con_mean, con_std, con_raw = cuda_time_ms(fn_concurrent)

    print(f"  DiT-block only:    median {dit_mean:7.2f} ± {dit_std:5.2f} ms (trimmed)")
    print(f"  VAE.decode only:   median {vae_mean:7.2f} ± {vae_std:5.2f} ms (trimmed)")
    print(f"  Serial:            median {ser_mean:7.2f} ± {ser_std:5.2f} ms (trimmed)")
    print(f"  Concurrent:        median {con_mean:7.2f} ± {con_std:5.2f} ms (trimmed)")

    overlap_factor = ser_mean / con_mean
    ideal_max = max(dit_mean, vae_mean)
    overlap_efficiency = (ser_mean - con_mean) / (min(dit_mean, vae_mean)) * 100
    saved_ms = ser_mean - con_mean
    print(f"  -> Overlap factor: {overlap_factor:.3f}x")
    print(f"     Ideal (perfect overlap): {ser_mean/ideal_max:.3f}x")
    print(f"     Overlap efficiency:      {overlap_efficiency:.1f}%  (saved {saved_ms:.2f} / {min(dit_mean, vae_mean):.2f} ms possible)")

    return {
        "resolution": resolution,
        "dit_mean_ms": dit_mean, "dit_std": dit_std,
        "vae_mean_ms": vae_mean, "vae_std": vae_std,
        "serial_mean_ms": ser_mean, "serial_std": ser_std,
        "concurrent_mean_ms": con_mean, "concurrent_std": con_std,
        "concurrent_raw": con_raw,
        "overlap_factor": overlap_factor,
        "ideal_max_factor": ser_mean / ideal_max,
        "overlap_efficiency_pct": overlap_efficiency,
        "saved_ms": saved_ms,
    }


def main():
    print(f"=== DecoupledVAE Gate A: CUDA Stream Overlap Microbench ===")
    print(f"GPU: {torch.cuda.get_device_name(int(DEVICE.split(':')[-1]))}")
    print(f"Loading {MODEL_ID} (bf16)...")
    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    pipe = pipe.to(DEVICE)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    results = []
    for res in [1024, 2048]:
        try:
            results.append(benchmark_at_resolution(pipe, res))
        except torch.cuda.OutOfMemoryError as ex:
            print(f"  OOM at {res}, skipping. {ex}")
            torch.cuda.empty_cache()

    # Decision
    best_overlap = max(r["overlap_factor"] for r in results)
    print("\n" + "=" * 70)
    print(f"Best overlap factor across resolutions: {best_overlap:.3f}x")
    if best_overlap >= 1.30:
        verdict = "PASS"
        action = "Proceed to Gate B (x0-estimate quality curve)"
    elif best_overlap < 1.10:
        verdict = "FAIL"
        action = "Single-GPU overlap insufficient. Pivot to two-GPU (competes with PipeDiT) or abandon"
    else:
        verdict = "BORDERLINE"
        action = "Marginal gain — reconsider expected speedup ceiling"
    print(f"Gate A verdict: {verdict}")
    print(f"Action: {action}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "gateA_stream_overlap.json", "w") as f:
        json.dump({
            "config": {
                "model": MODEL_ID, "dtype": str(DTYPE), "device": DEVICE,
                "n_warmup": N_WARMUP, "n_iters": N_ITERS,
                "gpu": torch.cuda.get_device_name(int(DEVICE.split(':')[-1])),
            },
            "results_per_resolution": results,
            "best_overlap_factor": best_overlap,
            "verdict": verdict, "action": action,
        }, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR / 'gateA_stream_overlap.json'}")


if __name__ == "__main__":
    main()
