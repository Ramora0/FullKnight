#!/bin/bash
# Run a single FullKnight hyperparameter experiment.
# All verbose output goes to run.log; only the summary is printed.

TIME_BUDGET=1200  # 20 minutes

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/python"

FULLKNIGHT_TIME_BUDGET=$TIME_BUDGET WANDB_MODE=disabled python train.py > "$REPO_ROOT/autoresearch/run.log" 2>&1
STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "CRASH (exit code $STATUS)"
    echo "---"
    tail -n 30 "$REPO_ROOT/autoresearch/run.log"
    exit 1
fi

# Print only the summary block
sed -n '/^---$/,$ p' "$REPO_ROOT/autoresearch/run.log"
