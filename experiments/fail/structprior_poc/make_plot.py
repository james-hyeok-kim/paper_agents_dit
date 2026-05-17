"""StructPrior gate result plot."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RES = Path(__file__).parent / "results"
with open(RES / "gate_trunk_attn_share.json") as f:
    d = json.load(f)

s = d["summary"]
attn_e2e = s["attn_e2e_share_pct"]
attn_dit = s["attn_dit_share_pct"]
attn_ms = s["attn_mean_ms"]
dit_ms = s["dit_mean_ms"]
e2e_ms = s["e2e_mean_ms"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: stacked horizontal bar — composition of E2E
other = e2e_ms - dit_ms
non_attn_dit = dit_ms - attn_ms
labels = ["Trunk Attn", "Other DiT (MLP+norm)", "Other (sched/io/VAE)"]
vals = [attn_ms, non_attn_dit, other]
colors = ["#e74c3c", "#3498db", "#95a5a6"]
left = 0
for v, c, l in zip(vals, colors, labels):
    ax1.barh([0], [v], left=left, color=c, alpha=0.85, label=f"{l}: {v:.0f}ms ({v/e2e_ms*100:.1f}%)")
    left += v
ax1.set_yticks([])
ax1.set_xlabel("Wallclock (ms)")
ax1.set_title(f"FLUX-schnell 4-step E2E composition\n(B200, 1024×1024, mean over {len(d['iter_e2e_total_ms'])} iters)")
ax1.legend(loc="upper right")

# Right: thresholds
ax2.bar(["Trunk Attn / E2E"], [attn_e2e], color="#e74c3c", alpha=0.85, width=0.5)
ax2.text(0, attn_e2e + 1, f"{attn_e2e:.2f}%", ha="center", fontsize=14, fontweight="bold")
ax2.axhline(40, color="green", linestyle="--", label="PASS threshold (≥40%)")
ax2.axhline(25, color="orange", linestyle="--", label="BORDERLINE (25-40%)")
ax2.set_ylabel("Share of E2E (%)")
ax2.set_title("Trunk attention as fraction of E2E")
ax2.set_ylim(0, 50)
ax2.legend(loc="upper right")
ax2.grid(axis="y", alpha=0.3)

plt.suptitle(f"StructPrior Gate: Trunk Self-Attention Share\n"
             f"Verdict: {d['verdict']} — ceiling {attn_e2e:.1f}% < PASS 40%",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
out = RES / "gate_breakdown.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved: {out}")
