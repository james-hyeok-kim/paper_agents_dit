"""SpecDiT gate result plot."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RES = Path(__file__).parent / "results"
with open(RES / "gate_velocity_agreement.json") as f:
    d = json.load(f)

# Left: per-step rel_err distribution; Right: acceptance rate vs τ
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

steps = sorted(int(k) for k in d["per_step_errs"].keys())
errs_by_step = [d["per_step_errs"][str(s)] for s in steps]

bp = ax1.boxplot(errs_by_step, positions=steps, widths=0.6, patch_artist=True,
                 boxprops={"facecolor": "#3498db", "alpha": 0.7})
ax1.axhline(0.10, color="green", linestyle="--", alpha=0.7, label="τ=0.10 (target accept)")
ax1.axhline(0.30, color="orange", linestyle="--", alpha=0.7, label="τ=0.30 (loose)")
ax1.set_xlabel("Denoising step")
ax1.set_ylabel("Relative velocity error  ||v_target − v_draft|| / ||v_target||")
ax1.set_title("Layer-skip draft velocity error per step\n(across 8 prompts)")
ax1.legend(loc="upper left")
ax1.grid(axis="y", alpha=0.3)

# Acceptance rate vs τ
taus = sorted(float(k) for k in d["acceptance_rates"].keys())
rates = [d["acceptance_rates"][str(t) if str(t) in d["acceptance_rates"] else t] for t in taus]
# json may have float keys serialized as either int-like or float strings - handle both
rates = []
for t in taus:
    # find matching key
    for k, v in d["acceptance_rates"].items():
        if abs(float(k) - t) < 1e-9:
            rates.append(v)
            break

ax2.plot(taus, rates, marker="o", linewidth=2, markersize=10, color="#e74c3c")
for t, r in zip(taus, rates):
    ax2.annotate(f"{r:.1f}%", (t, r), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=10)
ax2.axhline(70, color="green", linestyle="--", alpha=0.7, label="PASS threshold (≥70%)")
ax2.axhline(50, color="orange", linestyle="--", alpha=0.7, label="MARGINAL (≥50%)")
ax2.set_xlabel("Acceptance threshold τ")
ax2.set_ylabel("Acceptance rate (%)")
ax2.set_title("Acceptance rate vs threshold")
ax2.set_ylim(-5, 105)
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle(f"SpecDiT Gate: Layer-Skip Draft Velocity Agreement\n"
             f"Verdict: {d['verdict']} (mean rel_err = {d['overall_stats']['mean']:.3f}, all τ → 0% accept)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
out = RES / "gate_breakdown.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved: {out}")
