# autoresearch/model

Autonomous LLM-driven hyperparameter research for the FullKnight policy.

## Setup

To set up a new experiment series, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may10`).
   The branch `ablation/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b ablation/<tag>` from the chosen
   starting commit.
3. **Read the in-scope files** before starting:
   - `CLAUDE.md` — repository context and architecture overview.
   - `python/config.py` — hyperparameters live here. This is the file you
     edit during the loop.
   - `python/train.py` — training loop. Read-only. Understand the metrics.
   - `python/ppo.py` — PPO algorithm. Read-only. Understand what each
     hyperparameter controls.
   - `python/model.py` — network architecture. Read-only.
   - `autoresearch/model/session.md` — current-session priorities.
   - `autoresearch/model/ideas.md` — ablation menu.
4. **Initialize results.tsv**: ensure `autoresearch/model/results.tsv` has only
   the header row. The first run records the baseline.
5. **Confirm and go**.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment trains for a **fixed time budget of 20 minutes** (wall clock)
**from scratch** (no checkpoint resume). The 4-boss pool is the default; the
fpw=1 / n_envs=8 stepping regime is enforced by `run_experiment.sh`.

20 minutes from random init at fpw=1 / n_envs=8 / 1024-step rollouts produces
~500–550 epochs and ~500k env-steps. That's enough to see D climb out of
`D_initial=2.0` and to read off entropy/kl/surrogate steady-state behavior, but
not enough to see late-phase saturation. Pick experiments whose effects are
likely to show in that window — early-policy shape, exploration pressure,
gradient-step efficiency — rather than long-horizon ones.

Launch a run via:

```bash
bash autoresearch/model/run_experiment.sh
```

`run_experiment.sh` handles time budget, wandb config, and run naming. It
forces `--n_envs 8 --frames_per_wait 1` regardless of `config.py` defaults so
the ablation regime is always controlled here, not by code drift. All verbose
output goes to `autoresearch/model/run.log`; the script prints only the summary
block on success.

**Wandb**: runs go to project `fullknight-ablation-may10`, named
`"<short_hash> <commit_subject>"`, so every row in the dashboard maps 1:1 to a
git commit on the ablation branch. (Wandb has been confirmed broken on this
host and runs with `WANDB_MODE=disabled` — the `---` summary block is the
signal.) Write descriptive commit messages — they *are* the run names.

**What you CAN do:**
- Modify `python/config.py` — the only file you edit. Numeric hyperparameters
  are fair game (see `ideas.md` for the menu).
- Tighten or relax CLI overrides in `run_experiment.sh` only if the user asks.

**What you CANNOT do:**
- Modify `train.py`, `ppo.py`, `model.py`, or any other Python/C# source file
  during the loop. If you have a strong hypothesis that requires a small code
  change, surface it to the user first and get explicit approval — but the
  default is config-only.
- Change non-tunable structural fields: `server_host`, `server_port`,
  `n_envs` (forced by script), `frames_per_wait` (forced by script), file
  paths, observation dims (`combat_feature_dim`, `combat_normalized_dims`,
  `terrain_feature_dim`, `terrain_normalized_dims`, `global_state_dim`,
  `n_binary_flags`), action dims (`movement_n`, `direction_n`, `action_n`,
  `jump_n`), or kind/parent vocab dims.
- Change architecture dims (`global_hidden`, `global_output`, `combat_hidden`,
  `combat_output`, `terrain_hidden`, `terrain_output`, `hidden_dim`,
  `gru_dim`, `kind_embed_dim`). These are stable across the ablation series
  for cross-run comparability.
- Change the reward shape constants (`heal_coef`, `D_initial`, `D_min`,
  `D_ema`, `D_window`, `D_max_delta`). The reward is the comparison axis,
  not a tuning knob.
- Install new packages or add dependencies.

**The goal: get the highest `final_D_geomean`** (geomean of per-boss D over
the last 20 epochs of the run). Same time budget + same start state + same
boss pool = experiments are directly comparable. D adapts to the agent's
damage_landed/hits_taken ratio, so a higher cross-boss geomean means the
policy is doing more damage per hit taken on average.

**Multi-boss noise caveat**: 4-boss pool with n_envs=8 means ~2 envs per boss
per epoch. Per-epoch metrics are noisier than single-boss runs; treat
sub-5% deltas as noise. Look at per-boss D values too — sometimes a config
that wins on geomean is dragged up by one boss while regressing on others.

**Domain reminder**: this is a Hollow Knight boss-fighting agent — fast 2D
combat with variable-length hitbox observations, frame-skip, real-time
animations, and hard-commit windows on hold actions (nail charge / focus /
dream nail / super dash). Default PPO hyperparameters come from Atari/MuJoCo
papers and may not suit this environment. Question the defaults.

**The first run**: kick off `run_experiment.sh` unchanged on the chosen
starting commit so the dashboard has a baseline reference for every later
comparison. Record it in `results.tsv` with `status=keep` and
`description=baseline`.

## Output format

The summary block on success looks like:

```
---
curriculum_reward:      0.054093
avg_damage_landed:      0.0625
avg_hits_taken:         0.0085
final_D_geomean:        9.45
final_avg_hits_per_boss: 100.0
final_D/GG_False_Knight: 16.27
final_D/GG_Mega_Moss_Charger: 7.16
final_D/GG_Gruz_Mother: 11.18
final_D/GG_Hornet_1: 7.10
final_entropy:       -2.290000
final_kl:            0.013200
final_surrogate:     0.003400
pct_samples_trained: 100.0
epochs_completed:    520
training_seconds:    1201.3
wall_breakdown:      combat …  intro …  death+exit …  load …  reset-other …
```

If the run crashed, you will see `CRASH (exit code N)` followed by the last
30 lines of the log. The full log lives at `autoresearch/model/run.log`.

## Logging results

When an experiment is done, append a row to `autoresearch/model/results.tsv`
(tab-separated, NOT comma-separated — commas break in descriptions). Do not
commit `results.tsv` or `run.log`; both are gitignored.

The TSV header is:

```
commit	D_geomean	status	description
```

1. git commit hash (short, 7 chars)
2. final_D_geomean achieved (e.g. `9.45`) — use `0.00` for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried (include
   per-boss D + a couple of diagnostic readings if interesting)

Example:

```
commit	D_geomean	status	description
91c4b1b	9.45	keep	may10 baseline (FK 16.27 Moss 7.16 Gruz 11.18 Hornet 7.10, dmg 0.062 ent -2.29, 520 ep)
a1b2c3d	10.20	keep	entropy_coeff 0.02->0.03 (FK 17.5 Moss 7.9 Gruz 11.5 Hornet 8.1)
b2c3d4e	8.12	discard	clip_eps 0.2->0.4 (KL exploded to 0.06+, surrogate thrashed)
```

## The experiment loop

The loop runs on the dedicated branch (e.g. `ablation/may10`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Edit `python/config.py` with a single hyperparameter change.
3. `git commit` with a description that names the change (e.g.
   `entropy_coeff 0.02 -> 0.03`).
4. Run the experiment: `bash autoresearch/model/run_experiment.sh`.
5. If the output says CRASH, read `autoresearch/model/run.log` to diagnose.
   If you can't get things working after a couple of attempts, give up on
   that change.
6. Record the row in `results.tsv` (do NOT commit `results.tsv` or `run.log`).
7. If `final_D_geomean` improved, "advance" — keep the commit on the branch.
8. If it's equal or worse (within ~5%), `git reset --hard HEAD~1` to revert
   so the branch tip stays at the best-known config.

You are an autonomous researcher. Try things, keep wins, discard losses,
iterate.

**One variable at a time.** Prefer changing ONE hyperparameter per experiment.
This is ablation, not random search. After establishing direction for
individual parameters, you can combine winners in later experiments.

**Follow the gradient — don't abandon promising leads.** When a change helps,
keep pushing in that direction until it stops helping. If `entropy_coeff
0.02 → 0.03` improves D_geomean, the next experiment should be `0.04` or
`0.05`, not a totally different parameter. Ride the win to the peak — only
move on once you see a clear plateau or regression. Conversely, when a
change hurts, try the *opposite* direction before giving up: if `lr 3e-4 →
6e-4` regressed, try `1.5e-4` next rather than concluding "lr is fine." A
parameter is only "done" once you've seen both directions underperform or
you've found its peak.

**Use diagnostics**: check `final_entropy`, `final_kl`, `final_surrogate`
in the output, and skim `pct_samples_trained` (low pct => KL early-stop is
firing aggressively). If KL is consistently high (> 0.05), the learning rate
or clip_eps may be too aggressive. If entropy collapses below ~-3.0 over a
20-min run, `entropy_coeff` is too low. Use these signals to pick the next
experiment, not just D_geomean.

**Noise**: RL is noisy. Sub-5% deltas relative to baseline are probably
noise — treat as "equal" and discard. Focus on changes with clear directional
improvement.

**Timeout**: each experiment takes ~20 min plus ~1 min of HK boot/teardown.
If a run exceeds 25 minutes wall clock, kill it and treat as a failure.

**Crashes**: if a run crashes (OOM, bug, etc.), use judgment. If it's
trivially fixable (typo, missing import), fix and re-run. If the idea itself
is fundamentally broken, log "crash" in the tsv and move on.

**NEVER STOP**: once the loop has started, do NOT pause to ask the human
whether to continue. The user expects you to run autonomously until manually
stopped. If you run out of ideas, re-read `ideas.md` for under-explored
directions, try combining previous near-misses, or revisit a parameter at a
wider range. The loop runs until the user interrupts.

The user might leave you running for several hours. At ~20 min per experiment
plus boot/teardown, plan on ~2.5 experiments per hour. A 6-hour overnight
session = ~15 experiments. Use them well.
