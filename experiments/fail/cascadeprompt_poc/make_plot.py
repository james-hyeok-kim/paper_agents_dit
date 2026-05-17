"""Generate breakdown plot for CascadePrompt Gate 1 result."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
with open(RESULTS_DIR / "gate1_wallclock.json") as f:
    data = json.load(f)

# Component breakdown
components = ["T5-XXL", "CLIP-L", "DiT (4 step)", "VAE decode", "Other"]
keys = ["t5_xxl", "clip_l", "dit_total", "vae_total"]
means = [data["stats"][k]["mean_ms"] for k in keys]
stds  = [data["stats"][k]["std_ms"]  for k in keys]
e2e_mean = data["stats"]["e2e"]["mean_ms"]
other_mean = e2e_mean - sum(means)
means.append(other_mean)
stds.append(0)

pct = [m / e2e_mean * 100 for m in means]
colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71", "#95a5a6"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Bar chart: absolute ms
bars = ax1.barh(components, means, xerr=stds, color=colors, alpha=0.85, capsize=4)
for bar, m, p in zip(bars, means, pct):
    ax1.text(m + 5, bar.get_y() + bar.get_height()/2,
             f"{m:.1f} ms ({p:.1f}%)", va="center", fontsize=10)
ax1.set_xlabel("Wallclock (ms)")
ax1.set_title(f"FLUX-schnell 4-step Component Wallclock\n(E2E = {e2e_mean:.1f} ms, B200, 1024×1024)")
ax1.set_xlim(0, max(means) * 1.35)
ax1.grid(axis="x", alpha=0.3)
ax1.invert_yaxis()

# Highlight the gate decision
ax1.axvline(e2e_mean * 0.15, color="green", linestyle="--", alpha=0.5,
            label="Gate 1 PASS threshold (15% of E2E)")
ax1.axvline(e2e_mean * 0.10, color="red", linestyle="--", alpha=0.5,
            label="Gate 1 FAIL threshold (10% of E2E)")
ax1.legend(loc="lower right", fontsize=9)

# Pie chart: percentage breakdown
ax2.pie(pct, labels=[f"{c}\n{p:.1f}%" for c, p in zip(components, pct)],
        colors=colors, startangle=90, counterclock=False,
        textprops={"fontsize": 10})
ax2.set_title(f"E2E share — T5-XXL = {pct[0]:.2f}% → Gate 1 FAIL")

plt.suptitle(f"CascadePrompt Gate 1: T5-XXL Wallclock Share\nVerdict: {data['verdict']}",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
out_path = RESULTS_DIR / "gate1_breakdown.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"Saved: {out_path}")
