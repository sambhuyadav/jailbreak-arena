#!/usr/bin/env bash
# End-to-end training for HF Jobs / any clean GPU container.
#
# Runs GRPO at curriculum levels 1 → 2 → 3 sequentially. Each level
# auto-loads the previous level's checkpoint via train.py's
# _resolve_default_model(), so the curriculum compounds.
#
# After each level finishes, the merged checkpoint is pushed to
# <HF_USER>/jailbreak-attacker-l<N> on the Hub so progress survives
# container teardown.
#
# Required env (passed via `hf jobs run --secrets` / --env):
#   HF_TOKEN       write-scoped token (used by `hf upload`)
#   HF_USER        hub namespace for uploads (e.g. "shambhuyadav")
# Optional:
#   WANDB_API_KEY  enables W&B logging
#   PROMPT_REPEATS, NUM_EPOCHS  passed through to train.py
#   LEVELS         space-separated list (default "1 2 3")

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN is required}"
: "${HF_USER:?HF_USER is required}"

export DEFENDER_BACKEND="${DEFENDER_BACKEND:-stub}"
export ENV_BASE_URL="${ENV_BASE_URL:-http://127.0.0.1:7860}"
LEVELS="${LEVELS:-1 2 3}"

echo ">>> installing deps"
pip install -q --no-cache-dir -r requirements.txt
pip install -q --no-cache-dir -r requirements-train.txt
pip install -q --no-cache-dir -U "huggingface_hub[cli]"

echo ">>> starting env server (defender=$DEFENDER_BACKEND)"
uvicorn server:app --host 127.0.0.1 --port 7860 > /tmp/env.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:7860/health > /dev/null; then
    echo ">>> env server up after ${i}s"; break
  fi
  if [ "$i" = "30" ]; then
    echo "!!! env server failed to start; tail of log:"
    tail -50 /tmp/env.log
    exit 1
  fi
  sleep 1
done

upload_checkpoint() {
  local level=$1
  local dir="./checkpoints/jailbreak-attacker-l${level}"
  if [ ! -d "$dir" ]; then
    echo ">>> no checkpoint at $dir, skipping upload"
    return
  fi
  local repo="${HF_USER}/jailbreak-attacker-l${level}"
  echo ">>> uploading $dir → $repo"
  hf upload "$repo" "$dir" --repo-type model --commit-message "level-${level} GRPO checkpoint" \
    || echo "!!! upload failed (continuing)"
}

for level in $LEVELS; do
  echo "================================================================"
  echo ">>> GRPO Level $level"
  echo "================================================================"
  CURRICULUM_LEVEL=$level python train.py
  upload_checkpoint "$level"
done

echo ">>> training complete: levels [$LEVELS]"
