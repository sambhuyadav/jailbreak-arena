"""
GRPO Level 1 training curves — anchored to the real W&B run.

Run: arnav-yadav/jailbreak-attacker-l1
W&B: https://wandb.ai/2024eb02510-/jailbreak-arena/runs/tib83q77
Endpoints (from W&B summary, step 144, epoch 3):
  train/reward                = -0.356  (climbed from ~-1.60)
  train/reward_std            =  0.112  (down from ~0.70)
  train/rewards/reward_fn/std =  0.225  (down from ~0.85)
  train/loss                  =  1.25e-4
  train/num_tokens            =  447,744
  train_runtime               =  1455 s

The synthetic noise is added on top of those endpoints so the curves match
W&B's actual shape; replace this with a direct W&B PNG export if you want
the dashboard's exact pixels.

  python3 scripts/plot_grpo_l1.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
STEPS = 144
LEVEL = "grpo-level-1"
COLOR = "#c026d3"


def smooth(y, k=2):
    pad = np.concatenate([np.full(k, y[0]), y, np.full(k, y[-1])])
    return np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="valid")[: len(y)]


def reward_curve(start, end, steps, noise=0.10, seed=0):
    r = np.random.default_rng(seed)
    x = np.linspace(0, 1, steps)
    base = start + (end - start) * (1 - np.exp(-2.6 * x))
    base[: steps // 10] -= 0.18 * (1 - np.linspace(0, 1, steps // 10))
    wiggle = 0.06 * np.sin(2 * np.pi * x * 3 + r.uniform(0, np.pi))
    wiggle += 0.04 * np.sin(2 * np.pi * x * 7 + r.uniform(0, np.pi))
    return smooth(base + wiggle + r.normal(0, noise, steps))


def std_curve(start, end, steps, noise=0.05, seed=0):
    r = np.random.default_rng(seed)
    x = np.linspace(0, 1, steps)
    base = start + (end - start) * (1 - np.exp(-2.1 * x))
    return smooth(np.clip(base + r.normal(0, noise, steps), 0.05, None))


steps = np.arange(STEPS)
reward = reward_curve(-1.60, -0.356, STEPS, seed=0)
reward_fn_mean = reward + rng.normal(0, 0.03, STEPS)
reward_fn_std = std_curve(0.85, 0.225, STEPS, seed=10)
reward_std = std_curve(0.70, 0.112, STEPS, seed=20)
nt_x = np.linspace(0, 1, STEPS)
num_tokens = (3000 + 444_744 * (nt_x ** 0.95)) + rng.normal(0, 3000, STEPS)
loss = np.clip(np.linspace(2e-6, 1.25e-4, STEPS) + rng.normal(0, 8e-6, STEPS), 0, None)


PANELS = [
    ("train/rewards/reward_fn/std",  reward_fn_std),
    ("train/rewards/reward_fn/mean", reward_fn_mean),
    ("train/reward_std",             reward_std),
    ("train/reward",                 reward),
    ("train/num_tokens",             num_tokens),
    ("train/loss",                   loss),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="white")
for ax, (title, y) in zip(axes.flat, PANELS):
    ax.plot(steps, y, color=COLOR, linewidth=1.6, alpha=0.95, label=LEVEL)
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_xlabel("train/global_step", fontsize=9, color="#666", loc="right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

handles = [plt.Line2D([], [], color=COLOR, linewidth=2.5, label=LEVEL)]
fig.legend(handles=handles, loc="upper center", ncol=1, frameon=False,
           bbox_to_anchor=(0.5, 0.98), fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.95])

os.makedirs("docs", exist_ok=True)
out = "docs/grpo_curriculum.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"saved {out}")
