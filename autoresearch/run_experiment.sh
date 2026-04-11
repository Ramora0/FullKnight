#!/bin/bash
# Run a single FullKnight hyperparameter experiment.
# All verbose output goes to run.log; only the summary is printed.
#
# Each experiment:
#   - trains for TIME_BUDGET seconds (30 min)
#   - resumes from RESUME_CKPT so short runs build on an already-warm policy
#   - logs to wandb under WANDB_PROJECT, named "<short_hash> <commit_subject>"
#     so every run in the dashboard maps 1:1 to a git commit on the ablation branch.

TIME_BUDGET=1800  # 30 minutes
RESUME_CKPT="models/fullknight_500.pth"
WANDB_PROJECT="fullknight-ablation"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SHORT_HASH="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
COMMIT_SUBJECT="$(git -C "$REPO_ROOT" log -1 --pretty=%s)"
export WANDB_MODE=online
export WANDB_NAME="${SHORT_HASH} ${COMMIT_SUBJECT}"

.venv/Scripts/python.exe python/train.py \
    --time_budget "$TIME_BUDGET" \
    --resume "$RESUME_CKPT" \
    --wandb_project "$WANDB_PROJECT" \
    > "$REPO_ROOT/autoresearch/run.log" 2>&1
STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "CRASH (exit code $STATUS)"
    echo "---"
    tail -n 30 "$REPO_ROOT/autoresearch/run.log"
    exit 1
fi

# Print only the summary block
sed -n '/^---$/,$ p' "$REPO_ROOT/autoresearch/run.log"
