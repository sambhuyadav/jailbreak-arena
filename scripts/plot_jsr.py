"""
JSR chart for the video / README — current training progress.

Right now we have:
  - Baseline (real, from baseline_run.txt)
  - Cycle 1 — L1 (estimated from train/reward = -0.356 in W&B run tib83q77;
                  replace with a measured /metrics jailbreak_success_rate
                  once an eval pass is run against arnav-yadav/jailbreak-attacker-l1)

As L2 and L3 finish, append their /metrics readings to STAGES + JSR below
and re-run the script.

  python3 scripts/plot_jsr.py
"""
import os
import matplotlib.pyplot as plt

STAGES = [
    "Baseline\n(untrained)",
    "Cycle 1 — L1\n(GRPO, est.)",
    # "Cycle 1 — L2\n(GRPO)",
    # "Cycle 1 — L3\n(GRPO)",
    # "Self-play\n(defender SFT)",
]
JSR = [
    0.12,   # baseline_run.txt → JSR: 12.5%
    0.28,   # estimated from train/reward = -0.356 (final, step 144); REPLACE with /metrics after eval
    # 0.45,
    # 0.60,
    # 0.25,
]

assert len(STAGES) == len(JSR), "STAGES and JSR must be the same length"

os.makedirs("docs", exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(STAGES, JSR, marker="o", linewidth=2.5, markersize=12, color="#9333ea")
for x, y in zip(STAGES, JSR):
    ax.annotate(f"{y:.0%}", xy=(x, y), xytext=(0, 14),
                textcoords="offset points", ha="center", fontsize=13, fontweight="bold")

ax.set_ylabel("Jailbreak Success Rate", fontsize=12)
ax.set_title("Jailbreak Arena — JSR after GRPO Level 1", fontsize=13)
ax.set_ylim(0, max(max(JSR) * 1.4, 0.4))
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()

out = "docs/jsr_curve.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
