# autoresearch/sim

Aggressively reduce simulation cost (env-steps/sec at training-time N)
without regressing trained-agent quality.

Edit the **C# mod** (`FullKnight.cs`, `Game/`, `Net/`, `Environment/`,
`ProxyController.cs`, etc.) and grades on **throughput**, with quality
as a regression check.

## Setup

To set up a new experiment session:

1. **Agree on a run tag**: propose a tag based on today's date (e.g.
   `may8`). Branch `sim/<tag>` must not exist — fresh run.
2. **Create the branch**: `git checkout -b sim/<tag>` from main.
3. **Read in-scope files** for full context:
   - `CLAUDE.md` — repo overview.
   - `Environment/TrainingEnv.cs` — the per-step coroutine. The frame
     loop, hitbox observation, intro-skip, episode-end logic. The
     hottest path on the C# side.
   - `Environment/HitboxObserver.cs` — hitbox tracking via a per-collider
     MonoBehaviour. Allocations and per-frame iteration cost live here.
   - `Game/SaveFileProxy.cs`, `Game/SceneHooks.cs`, `Game/TimeScale.cs`
     — bootstrap and time control. Touch carefully.
   - `Net/Protocol.cs`, `Net/BinaryProtocol.cs` — wire format.
     Allocations on hot path matter.
   - `python/validate.py` — the harness you grade against.
   - `python/vec_env.py` lines around `_batch_observations` — the
     Python-side step receiver. Sometimes the bottleneck moves here.
4. **Initialize results.tsv**: header is already there; the first run
   establishes the baseline.
5. **Confirm setup**, then launch the loop.

## Experimentation

Each experiment runs `bash autoresearch/sim/run_experiment.sh`, which:
- rebuilds the C# mod (`dotnet build -c Debug`),
- runs `python/validate.py` with `QUALITY_STEPS` (default 8192) at N=1
  for the quality phase and `THROUGHPUT_STEPS` (default 1024) at
  N=`THROUGHPUT_N` (default 8) for the throughput phase,
- captures verbose output to `autoresearch/sim/run.log`,
- prints only the `---` summary block.

The summary block is the signal you record. Keep its keys stable —
both the autoresearch loop and any downstream tooling read by name.

**What you CAN edit**:
- C# mod source: anything under `Environment/`, `Game/`, `Net/`,
  `ProxyController.cs`, `FullKnight.cs`. Game-internal MonoBehaviours
  can be disabled at runtime.
- The HK_managed DLL must remain reachable; set `LocalRefs` correctly
  in `FullKnight.csproj` if needed.

**What you CANNOT edit** (read-only during the loop):
- `python/validate.py`, `python/eval.py`, `python/train.py`,
  `python/model.py`, `python/ppo.py` — the harness must stay constant
  so successive measurements are comparable.
- `autoresearch/sim/run_experiment.sh` — same reason. If you must
  change the harness, end the session, re-baseline, start a new branch.
- The checkpoint path or `--quality-steps` / `--throughput-steps` /
  `--throughput-n` arguments mid-session. They define the comparison.

## Output format

On success the script prints a `---` block:

```
---
dmg_per_ep_mean:    24.3000
dmg_per_ep_lo:      19.5000
dmg_per_ep_hi:      29.1000
hits_per_ep_mean:   2.4000
hits_per_ep_lo:     1.8000
hits_per_ep_hi:     3.0000
quality_episodes:   12
quality_wall_s:     128.4
throughput_eps:     78.5400
throughput_n:       8
throughput_wall_s:  130.5
rtime_mean_ms:      12.3400
rtime_p95_ms:       14.8000
cpu_hk_sum_pct:     520.30
cpu_machine_sat:    65.04
cpu_system_mean:    72.10
cpu_system_peak:    88.50
ram_initial_mb:     1024
ram_final_mb:       1036
ram_growth_mb:      12
gc_heap_mean_mb:    45.20
gc_heap_growth_mb:  1.50
```

On failure: `MOD BUILD FAILED ...` or `CRASH (exit code N)` followed by
the last 30 log lines. Read `autoresearch/sim/run.log` for the full trail.

## Logging results

Append a row to `autoresearch/sim/results.tsv` (tab-separated, NOT csv)
with one column per metric the summary block emits. The description
column is now just the change being tested — no stats stuffed into it.

Columns (21):

| # | column | source |
|---|---|---|
| 1 | `commit` | `git rev-parse --short HEAD` |
| 2 | `status` | `keep` / `discard` / `crash` |
| 3 | `mode` | `headless` / `graphical` (from how the run was invoked) |
| 4 | `throughput_n` | `--throughput-n` value (default 8) |
| 5 | `quality_steps` | `--quality-steps` value (default 8192) |
| 6 | `throughput_steps` | `--throughput-steps` value (default 1024) |
| 7 | `quality_episodes` | summary `quality_episodes` |
| 8 | `dmg_per_ep_mean` | summary `dmg_per_ep_mean` |
| 9 | `dmg_per_ep_lo` | summary `dmg_per_ep_lo` (95% CI low) |
| 10 | `dmg_per_ep_hi` | summary `dmg_per_ep_hi` (95% CI high) |
| 11 | `hits_per_ep_mean` | summary `hits_per_ep_mean` |
| 12 | `hits_per_ep_lo` | summary `hits_per_ep_lo` |
| 13 | `hits_per_ep_hi` | summary `hits_per_ep_hi` |
| 14 | `throughput_eps` | summary `throughput_eps` |
| 15 | `rtime_mean_ms` | summary `rtime_mean_ms` |
| 16 | `rtime_p95_ms` | summary `rtime_p95_ms` |
| 17 | `cpu_machine_sat` | summary `cpu_machine_sat` |
| 18 | `ram_growth_mb` | summary `ram_growth_mb` |
| 19 | `gc_heap_mean_mb` | summary `gc_heap_mean_mb` |
| 20 | `gc_heap_growth_mb` | summary `gc_heap_growth_mb` |
| 21 | `description` | what this experiment changed |

For `crash` rows, fill `0` for the numeric columns and leave config
columns at their attempted value.

Example rows (description trimmed for the README; in the TSV use
descriptions like `baseline`, `disable AudioListener`, `disable
Animator on bosses`):

```
commit  status   mode      n  q_steps  ...  throughput_eps  ...  description
a1b2c3d keep     headless  8  8192     ...  78.54           ...  baseline
b2c3d4e keep     headless  8  8192     ...  95.10           ...  disable AudioListener
c3d4e5f crash    headless  8  8192     ...  0               ...  disable HeroController
d4e5f6g discard  headless  8  8192     ...  102.40          ...  disable Animator on bosses (quality regressed)
```

## The experiment loop

Runs on a dedicated branch (e.g. `sim/may8`).

LOOP FOREVER:

1. Look at the current branch/commit and the last few results.tsv rows.
2. Pick the next idea from `session.md` / `ideas.md` (or generate one
   from the bottleneck signals: high CPU sat → cut compute; high GC
   growth → cut allocations; high rtime p95 → cut variance).
3. Edit the C# mod files.
4. `git commit` (commit message becomes the run label — be descriptive).
5. Run `bash autoresearch/sim/run_experiment.sh`.
6. If it CRASHED: read `tail -n 50 autoresearch/sim/run.log`. If the
   fix is small (typo, missing using-directive), fix and re-run. If the
   idea is structurally broken, log "crash", git reset, move on.
7. Record the row in `results.tsv` (do not commit run.log or
   results.tsv — leave them untracked).
8. Decision rule:
   - **throughput up AND quality CIs straddle baseline**: status=keep,
     advance the branch.
   - **throughput up BUT quality regressed (dmg_per_ep_hi < baseline_lo
     or hits_per_ep_lo > baseline_hi)**: status=discard, git reset.
   - **throughput flat or down**: status=discard, git reset.

**One change at a time**. Combine winners later. Without isolation you
can't tell which change carried the win.

**Follow the gradient**: if disabling AudioListener gave +21% throughput,
the next idea should be the next obvious sound/effect cut, not a
totally different system. Ride the win until you plateau.

**Quality decision rule**: a candidate's `dmg_per_ep` 95% CI must
overlap the baseline's CI to count as quality-preserving. CIs only
overlap → keep. CIs disjoint and the candidate is worse → discard
unless throughput gain is huge (>2x) and you have a hypothesis for why
the regression is recoverable later. Quality variance with small
episode counts is large, so re-run a borderline candidate before
discarding it.

**Bottleneck reading**:
- `cpu_machine_sat` near 100%: you're CPU-bound. Bottlenecks are in
  Unity-thread compute (physics, animation, hitbox iteration).
- `cpu_machine_sat` well under 100% but throughput low: you're not
  CPU-bound. Look at synchronization (frame waits, WebSocket, GC pauses).
- `gc_heap_mean_mb` rising over runs OR `gc_heap_growth_mb >> 0` per
  run: allocations on the hot path. Pool, cache, or remove.
- `rtime_p95_ms` >> `rtime_mean_ms`: long-tail steps. Often a GC pause
  or an off-screen-trigger respawn. Watch the gap.
- `ram_growth_mb` large positive: leak (real, not GC heap). Suspect
  caches that never evict, or per-step allocations that pin GC roots.

**Crashes**: the C# mod uses Hooks. A bad hook can hang the game (no
exit, no log). If `dotnet build` succeeded but the script hangs past
~5x baseline wallclock, kill it, treat as crash, revert.

**Re-baselining**: if the harness changes (validate.py, run_experiment.sh,
checkpoint), re-baseline before continuing — old TSV rows are no longer
comparable. Note re-baseline rows in the description (e.g.
`description=baseline (re-baseline after harness change)`).

**NEVER STOP**: once the loop has begun, do not pause to ask whether to
continue. The user might be away. Keep iterating until manually stopped.
If you run out of ideas, re-read `Environment/TrainingEnv.cs` and
`HitboxObserver.cs` for new angles, scan the HK decompiled source at
`C:\Users\Lee\coding\CSharp\HK\decomp\assembly-csharp\` for
MonoBehaviours that aren't combat-relevant, or look at the run.log's
stderr for noisy log spam (logging is allocation; printf in the inner
loop is real cost).
