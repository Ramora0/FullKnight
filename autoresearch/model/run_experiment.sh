#!/bin/bash
# Run a single FullKnight hyperparameter experiment.
# All verbose output goes to autoresearch/model/run.log; only the summary block
# is printed to stdout.
#
# Each experiment:
#   - trains for TIME_BUDGET seconds (20 min) FROM SCRATCH
#   - n_envs=8, frames_per_wait=1, 4-boss pool (config defaults for everything else)
#   - logs to wandb under WANDB_PROJECT, named "<short_hash> <commit_subject>"
#     so every run in the dashboard maps 1:1 to a git commit on the ablation branch.

TIME_BUDGET=1200          # 20 minutes
RESUME_CKPT=""            # from-scratch
WANDB_PROJECT="fullknight-ablation-may10"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

SHORT_HASH="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
COMMIT_SUBJECT="$(git -C "$REPO_ROOT" log -1 --pretty=%s)"
export WANDB_MODE=disabled
export WANDB_NAME="${SHORT_HASH} ${COMMIT_SUBJECT}"

.venv/Scripts/python.exe python/train.py \
    --n_envs 8 \
    --frames_per_wait 1 \
    --time_budget "$TIME_BUDGET" \
    --resume "$RESUME_CKPT" \
    --wandb_project "$WANDB_PROJECT" \
    > "$REPO_ROOT/autoresearch/model/run.log" 2>&1
STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "CRASH (exit code $STATUS)"
    echo "---"
    tail -n 30 "$REPO_ROOT/autoresearch/model/run.log"
    exit 1
fi

# Print only the summary block
sed -n '/^---$/,$ p' "$REPO_ROOT/autoresearch/model/run.log"
