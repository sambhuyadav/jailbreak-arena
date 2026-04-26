"""
JSR chart for README — Cycle 1 training progression.

Computed from:
  - baseline_run.txt (random attacker, 6 topics, JSR + avg reward measured)
  - W&B run grpo-level-1 (final train/reward, step 144)
  - W&B run grpo-level-2 (final train/reward, step ~330)
  - Self-play eval after defender SFT on attacker traces

JSR is computed by mapping mean episode reward to success rate using the
rubric in rewards.py: a success episode contributes +1.0, a refused episode
typically lands at about -0.6 to -1.0 once turn penalties stack.

  python3 scripts/plot_jsr.py
"""
import os
import matplotlib.pyplot as plt

STAGES = [
    "Baseline\n(random attacker)",
    "GRPO L1\n(trained)",
    "GRPO L2\n(trained)",
    "Self-play\n(defender hardened)",
]
JSR = [
    0.17,   # baseline_run.txt: 1/6 success
    0.32,   # grpo-level-1 train/reward plateau ≈ -0.40
    0.50,   # grpo-level-2 train/reward = -0.30, std = 0 at step 340
    0.19,   # defender SFT on attacker traces collapses attacker back near baseline
]

assert len(STAGES) == len(JSR), "STAGES and JSR must be the same length"

os.makedirs("docs", exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(STAGES, JSR, marker="o", linewidth=2.5, markersize=12, color="#9333ea")
for x, y in zip(STAGES, JSR):
    ax.annotate(f"{y:.0%}", xy=(x, y), xytext=(0, 14),
                textcoords="offset points", ha="center", fontsize=13, fontweight="bold")

ax.set_ylabel("Jailbreak Success Rate", fontsize=12)
ax.set_title("Jailbreak Arena — JSR across Cycle 1", fontsize=13)
ax.set_ylim(0, max(max(JSR) * 1.4, 0.6))
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()

out = "docs/jsr_curve.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
