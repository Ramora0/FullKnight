# Where the wallclock is going (May 2026, sim/may8 branch)

Branch state when measured: bf69447 effective (terrain_debug Physics2D
queries gated, nographics + merged harness, no uncap, no capture mode).
Checkpoint: `models/moss_charger_v2.pth`. N=8, frames_per_wait=5,
time_scale=3.

## The numbers

```
rtime_mean_ms       14.08      C# step_real_time (HK frame-skip wallclock)
wtime_mean_ms       16.87      Python await env.step() wallclock per env
per_step_wall_ms    47.98      total run wallclock / total env-steps
sim_pct_inner       83.46      rtime / wtime  — matches train.py's perf/sim_pct
sim_pct_total       29.35      rtime / per_step_wall_ms
gtime_mean_s        0.042228   game time per agent decision (what the model sees)
throughput_eps      166.7      env-steps/sec at N=8
cpu_machine_sat     24.28%     76% CPU idle
```

## What this means

`wtime - rtime ≈ 2.8ms` is the WebSocket round-trip + obs parse overhead
**inside** `await env.step()`. Small.

`per_step_wall_ms - wtime_mean_ms ≈ 31ms` is what's spent **outside**
`await env.step()` per env-step. **65% of every step's wallclock.** This
is the big target.

The 31ms is amortized across all env-steps, so it lumps together:
- `agent.collect_action` (PPO net forward pass, batch of N envs)
- Reset wallclock (synchronous in validation; staggered/async in training)
- Per-step Python loop overhead (done flag handling, buffer storage,
  reward computation, async dispatch in `asyncio.gather`)
- Done-recording (one `await env.step` per finished env to drain its
  final obs)

CPU is only 24% saturated. Lots of headroom — running more envs in
parallel would probably help, modulo memory.

## Why training looks like sim is the bottleneck (~80%) but it isn't

Training prints `perf/sim_pct = mean(rtime) / mean(wtime) × 100`. That's
`sim_pct_inner` in the new validate.py — **the fraction of `await
env.step()` time spent in HK**. It's NOT the fraction of total wallclock.

The denominator excludes everything outside `step_all`: the agent forward
pass (`t_fwd`, separately reported), reset wallclock, and Python loop
overhead. Those are ~31ms per env-step here, completely uncounted by
`perf/sim_pct`.

`sim_pct_total` (the new metric) divides by `total_wall / total_steps`,
which counts everything. **29% of total wallclock is HK; 65% is the
agent + resets + Python.** That's the truth.

## Where to dig (in order of likely payoff)

### 1. `agent.collect_action` — PPO forward pass

Look in `python/ppo.py` (`collect_action` at line ~250). It batches all
N envs into one forward pass. CUDA is available on this machine
(`torch.cuda.is_available() == True`); `PPO.__init__` puts the policy on
device automatically. Things to check:
- Is host→device transfer per step expensive? (`from_numpy + .float() +
  .to(device)`)
- Is the GRU sequential dependency forcing a slow forward?
- Are kind/parent embedding lookups being recomputed unnecessarily?
- Is the model itself the bottleneck, or the tensor prep?

`ppo.py` already has CUDA event timers (`h2d_s`, `forward_s`, `d2h_s`,
`tensor_prep_s`, see line 203+). Train.py logs them. Validate.py doesn't
read them — could add to surface inference cost in this harness too.

### 2. Reset wallclock

Validation does **synchronous** resets — when env i hits done, it calls
`env.reset()` and **blocks the next step** until reset returns. Each
reset takes ~1-3 seconds (scene load + intro skip + state init). With
~21 episodes per env over the run, resets add up.

Training runs resets as **async background tasks** via
`vec_env.start_resets` + `await_all_resets`. While reset is in flight,
the env is removed from the active set and the rollout loop continues
on a smaller N. Validation could adopt this pattern, OR the reset cost
itself could be cut on the C# side.

C# reset path (`Environment/TrainingEnv.cs`):
- `KillKnight` (synthetic suicide via real HK damage path) when
  mid-fight; `WaitForSceneChange` on natural end.
- `LoadBossScene` → bounce through GG_Workshop → boss scene transition.
- `WaitForFinishedEnteringScene` x2, `WaitForSeconds(2f)`, `WaitForSeconds(1f)`.
- `RecreateReader + frame`, `InitBossRefs`, hooks.

The `WaitForSeconds(2f)` and `WaitForSeconds(1f)` in `SceneHooks.cs` are
fixed game-time waits to let HK settle. With higher Time.timeScale they
take less wallclock; with `captureDeltaTime` set to fixed 0.05 they'd
take exactly 2 game-sec / 0.05 = 40 frames at uncapped FPS. Lots of
levers if reset becomes the focus.

`LogBossDiag` is called at multiple points in Reset/IntroSkip. Each call
does `GetComponentsInChildren<PlayMakerFSM>(true)` and string-builds a
big diagnostic. Probably a few hundred ms of allocation+iteration per
reset. Gating these behind `_evalMode` would help.

### 3. Per-step Python loop overhead

In `eval.py` extended_eval main loop (line 449+):
- `actions_np, _, _, _, _ = agent.collect_action(obs)` — covered above.
- Building `action_vecs` list of dicts.
- `await vec_env.step_all(action_vecs)`.
- Done handling: per env, drain final obs via another `await env.step`
  for done envs (eval.py:478+).

The done-drain step looks expensive — one extra `await env.step()` per
finished episode. With ~21 eps × 8 envs = 168 done-drain steps over the
run, at ~17ms each = ~2.85s of overhead.

### 4. Don't break dt regime

The agent was trained at **gtime_mean_s ≈ 0.042s per agent step** (per
`step_game_time` summed over 5 frames at variable FPS). Anything that
changes this materially regresses quality. Notes from prior experiments
on this branch:
- **Uncap Unity FPS** (commit e094f23, reverted): +63% throughput but
  shrank gtime ~4× (frames are smaller wallclock under uncap, so dt per
  frame is smaller). Quality CIs disjoint from baseline though metrics
  "improved" on the existing checkpoint. A regime change, not a free
  win.
- **`Time.captureDeltaTime` fixed to 0.0075** (commit a2f7136, reverted):
  matched baseline gtime within 10% but quality cratered (dmg
  88→27). Mechanism unclear — possibly animation events firing on real
  time, or HK FSMs that depend on dt variance pattern. **Capture mode
  is not safe** without further investigation.
- **`Time.captureFramerate=60`** (commit ae23d10, reverted): broke pause
  (Unity ignores `Time.timeScale=0` under captureFramerate).

In short: leave Update timing alone unless explicitly diagnosing the
uncap quality regression. The ~24% throughput improvement we currently
have comes from `-nographics` + harness changes + terrain_debug gating;
all preserve dt and are safe to layer on.

## Useful files / commands

```
autoresearch/sim/run_experiment.sh    # MODE=nographics|batchmode|graphical bash ...
autoresearch/sim/results.tsv          # one row per experiment, mode column distinguishes
python/validate.py                    # --merged single-phase, prints all sim_pct flavors
python/eval.py                        # extended_eval, returns gtime/rtime/wtime samples
python/vec_env.py                     # _timed_op wraps env.step / env.reset with timing
python/ppo.py                         # collect_action + CUDA event timers
python/train.py:402-404               # how training computes its perf/sim_pct
Environment/TrainingEnv.cs            # Reset (slow), Step (with frame skip)
Game/SceneHooks.cs                    # LoadBossScene + bounce
```

## Decision rule (from program.md)

A change to results.tsv:
- **throughput up AND quality CIs straddle baseline**: status=keep.
- **throughput up BUT quality regressed (`dmg_per_ep_hi < baseline_lo`
  or `hits_per_ep_lo > baseline_hi`)**: status=discard.

For the current baseline:
```
dmg_per_ep_lo  = 74.92
dmg_per_ep_hi  = 89.26
hits_per_ep_lo = 7.52
hits_per_ep_hi = 8.71
```
