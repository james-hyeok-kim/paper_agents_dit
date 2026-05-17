"""Generate breakdown plot for DecoupledVAE Gate A result."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
with open(RESULTS_DIR / "gateA_stream_overlap.json") as f:
    data = json.load(f)

results = data["results_per_resolution"]
resolutions = [f"{r['resolution']}×{r['resolution']}" for r in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: Serial vs Concurrent comparison
x = np.arange(len(resolutions))
w = 0.35
serial   = [r["serial_mean_ms"]      for r in results]
concur   = [r["concurrent_mean_ms"]  for r in results]
ser_err  = [r["serial_std"]          for r in results]
con_err  = [r["concurrent_std"]      for r in results]

b1 = ax1.bar(x - w/2, serial, w, yerr=ser_err, label="Serial (baseline)",
             color="#e74c3c", alpha=0.85, capsize=4)
b2 = ax1.bar(x + w/2, concur, w, yerr=con_err, label="Concurrent (2 streams)",
             color="#27ae60", alpha=0.85, capsize=4)

for bar, v in zip(b1, serial):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 10, f"{v:.0f}ms",
             ha="center", fontsize=9)
for bar, v in zip(b2, concur):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 10, f"{v:.0f}ms",
             ha="center", fontsize=9)

ax1.set_xticks(x)
ax1.set_xticklabels(resolutions)
ax1.set_ylabel("Wallclock (ms)")
ax1.set_title("Serial vs Concurrent (DiT-block + VAE.decode)")
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

# Right: Overlap factor with thresholds
overlap = [r["overlap_factor"] for r in results]
ideal   = [r["ideal_max_factor"] for r in results]

x = np.arange(len(resolutions))
b1 = ax2.bar(x - w/2, ideal, w, label="Theoretical max (perfect overlap)",
             color="#95a5a6", alpha=0.6)
b2 = ax2.bar(x + w/2, overlap, w, label="Measured overlap factor",
             color="#3498db", alpha=0.85)

for bar, v in zip(b1, ideal):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.03, f"{v:.2f}×",
             ha="center", fontsize=9)
for bar, v in zip(b2, overlap):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.03, f"{v:.2f}×",
             ha="center", fontsize=9, fontweight="bold")

# Threshold lines
ax2.axhline(1.30, color="green", linestyle="--", alpha=0.7, label="Gate A PASS (≥1.30×)")
ax2.axhline(1.10, color="red",   linestyle="--", alpha=0.7, label="Gate A FAIL (<1.10×)")
ax2.axhline(1.00, color="black", linestyle="-",  alpha=0.3)

ax2.set_xticks(x)
ax2.set_xticklabels(resolutions)
ax2.set_ylabel("Speedup factor")
ax2.set_title("Stream Overlap Factor vs Threshold")
ax2.legend(fontsize=9, loc="upper right")
ax2.set_ylim(0.9, max(ideal) * 1.15)
ax2.grid(axis="y", alpha=0.3)

plt.suptitle(f"DecoupledVAE Gate A: CUDA Stream Overlap on Single B200\n"
             f"Verdict: {data['verdict']} (best overlap {data['best_overlap_factor']:.3f}× < PASS 1.30×)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
out = RESULTS_DIR / "gateA_breakdown.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved: {out}")
