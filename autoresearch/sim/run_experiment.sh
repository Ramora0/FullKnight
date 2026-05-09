#!/bin/bash
# Run a single FullKnight sim-cost experiment.
#
# Each experiment:
#   - rebuilds the C# mod (Debug) so the running game reflects the change
#   - runs python/validate.py for fixed --n-steps at N=1 (quality) and
#     N=THROUGHPUT_N (throughput)
#   - emits a `---` summary block scrapeable by the autoresearch loop
#
# Launch mode is chosen via MODE env var (default "batchmode"):
#   graphical  — full HK window, no flags
#   batchmode  — `-batchmode` (Unity's default headless; render device still
#                initialized, render-path code may still execute)
#   nographics — `-batchmode -nographics` (forces graphicsDeviceType=Null,
#                skips the render thread entirely)
#
# All verbose output goes to autoresearch/sim/run.log; only the summary
# is printed on success.

CHECKPOINT="models/moss_charger_v2.pth"
# Merged single-phase: --merged collapses quality + throughput into one
# N=THROUGHPUT_N run for QUALITY_STEPS steps, computing CIs and
# throughput from the same data. 4096 × 8 = 32768 env-steps yields
# ~23 episodes for tight CIs in ~3.5 min wallclock — half a split run
# while accumulating 5× more episodes for behavior signal.
QUALITY_STEPS=4096
THROUGHPUT_STEPS=1024  # unused under --merged; kept for schema parity
THROUGHPUT_N=8
MODE="${MODE:-nographics}"

case "$MODE" in
    graphical)   GRAPHICAL_FLAG="--graphical" ;;
    batchmode)   GRAPHICAL_FLAG="--headless"  ; export FK_KEEP_GFX=1 ;;
    nographics)  GRAPHICAL_FLAG="--headless"  ;;
    *)
        echo "Unknown MODE='$MODE' (expected: graphical | batchmode | nographics)"
        exit 2
        ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)/.."
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
cd "$REPO_ROOT"

LOG="$REPO_ROOT/autoresearch/sim/run.log"

# Build the C# mod first. Without this the running HK process still has
# the previous mod state and the measurement is meaningless.
echo "Building mod (MODE=$MODE)..." > "$LOG"
dotnet build -c Debug >> "$LOG" 2>&1
BUILD_STATUS=$?
if [ $BUILD_STATUS -ne 0 ]; then
    echo "MOD BUILD FAILED (exit code $BUILD_STATUS)"
    echo "---"
    tail -n 30 "$LOG"
    exit 1
fi

.venv/Scripts/python.exe python/validate.py "$CHECKPOINT" \
    --quality-steps "$QUALITY_STEPS" \
    --throughput-steps "$THROUGHPUT_STEPS" \
    --throughput-n "$THROUGHPUT_N" \
    --merged \
    "$GRAPHICAL_FLAG" \
    >> "$LOG" 2>&1
STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "CRASH (exit code $STATUS, MODE=$MODE)"
    echo "---"
    tail -n 30 "$LOG"
    exit 1
fi

# Print only the `---` summary block.
sed -n '/^---$/,$ p' "$LOG"
