# autoresearch

This is an experiment to have the LLM do its own hyperparameter research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr7`). The branch `ablation/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b ablation/<tag>` from current main.
3. **Read the in-scope files**: The Python side is small. Read these files for full context:
   - `CLAUDE.md` — repository context and architecture overview.
   - `python/config.py` — the file you modify. All hyperparameters live here.
   - `python/train.py` — training loop. Do not modify. Understand the metrics.
   - `python/ppo.py` — PPO algorithm. Do not modify. Understand what each hyperparameter controls.
   - `python/model.py` — network architecture. Do not modify.
4. **Initialize results.tsv**: Create `autoresearch/results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment trains for a **fixed time budget of 30 minutes** (wall clock) and **resumes from `models/fullknight_500.pth`** — we're ablating "what's best for continuing this training," not "what's best from scratch." 30 minutes from a warm 500-epoch checkpoint gives a much cleaner signal than 30 minutes from random init (which at ~50 epochs is dominated by noise).

Launch a run via:

```bash
bash autoresearch/run_experiment.sh
```

`run_experiment.sh` handles time budget, resume path, wandb config, and run naming. All verbose output goes to `autoresearch/run.log`; the script prints only the summary block on success.

**Wandb**: runs go to project `fullknight-ablation`, named `"<short_hash> <commit_subject>"`, so every row in the dashboard maps 1:1 to a git commit on the ablation branch. Write descriptive commit messages — they *are* the run names.

**What you CAN do:**
- Modify `python/config.py` — the only file you edit. Numeric hyperparameters are fair game (see "priority ablations" below).

**What you CANNOT do:**
- Modify `train.py`, `ppo.py`, `model.py`, `run_experiment.sh`, or any other file. Read-only during the experiment loop.
- Change non-tunable structural fields: `server_host`, `server_port`, `n_envs`, file paths, observation dims (`combat_feature_dim`, `combat_normalized_dims`, `terrain_feature_dim`, `terrain_normalized_dims`, `global_state_dim`, `n_binary_flags`), action dims (`movement_n`, `direction_n`, `action_n`, `jump_n`), or the kind-vocab dims.
- **Off-limits because we're resuming from a checkpoint** — changing these breaks the warm start:
  - `gamma` — value heads are calibrated to the current discount factor; changing it invalidates V(s).
  - `D_initial`, `D_ema`, `D_window`, `D_min`, `D_max_delta` — curriculum state is restored from the checkpoint, so changing these mid-curriculum produces garbage.
  - Any reward-shape field (there aren't any tunable reward knobs in `Config`, but if you find one, leave it).
  - Architecture dims: `global_hidden`, `global_output`, `combat_hidden`, `combat_output`, `terrain_hidden`, `terrain_output`, `hidden_dim`, `gru_dim`, `kind_embed_dim`. The checkpoint has fixed shapes; changing these forces a fresh init and defeats the point of resuming.
- Install new packages or add dependencies.

**Priority ablations** (things we explicitly want to explore):
- **`entropy_coeff`** — current default 0.01. At epoch 500 the policy may be collapsing; try 0.02, 0.03, 0.05. Or 0.005 if entropy is still high and exploration is wasting steps.
- **`total_steps_per_epoch`** — current default 8192. Longer rollouts = better advantage estimates but fewer gradient updates per wall-clock minute. Try 4096, 16384.
- **`lr`** — current 5e-4. Adam absorbs lr shifts cleanly on resume, so this is a safe knob. Try 2e-4, 1e-3.
- **`clip_eps`** — current 0.2. Tighter (0.1) = more conservative updates from a warm policy; looser (0.3) = more aggressive.
- **`gae_lambda`** — current 0.95. Safe to perturb; trades variance vs bias in the advantage estimate.
- **`batch_size`**, **`chunks_per_batch`**, **`seq_len`**, **`train_iters`** — all safe mid-run. With 8192 steps/epoch and batch_size 128, that's a specific sample-reuse ratio; try wider ranges.
- **`frames_per_wait`** — current 5. Boss attacks are fast; this is on the table if you have a specific hypothesis, but it changes the effective time horizon so watch KL carefully.
- **`max_grad_norm`**, **`target_kl`**, **`value_coeff`**, **`max_value_loss`** — all safe mid-run.

**The goal is simple: get the highest `final_D_geomean`** (geomean of per-boss D over the last 20 epochs). Fixed time budget + same resume point = experiments are directly comparable. D is the adaptive curriculum scale — it converges to damage_landed/hits_taken — so a higher geomean across the boss pool means the policy is doing better per hit taken on average across bosses.

**Multi-boss noise caveat**: the default pool is 4 bosses, so with `n_envs=8` each boss only has ~2 envs collecting data per epoch. Per-epoch metrics are noisier than single-boss ablations. Treat sub-5% deltas as noise.

**This is an early MVP, not a final product.** Keep long-run learning potential in mind, but balance it against measured results:
- A change that *should* help long-run generalization but costs ~5% hit_ratio? Probably worth keeping.
- A change that hurts long-run potential but shows 20%+ improvement? Probably still worth keeping.
- Use judgment. Weigh both.

**Think about the domain.** This is a Hollow Knight boss-fighting agent — a fast-paced 2D action game with variable-length hitbox observations, frame-skip, and real-time combat. Default PPO hyperparameters come from Atari/MuJoCo papers and may not suit this environment. Question the defaults.

**The first run**: Your very first run should establish the baseline at the current config — run the script unchanged so the wandb dashboard has a reference point for every future comparison.

## Output format

Once the script finishes it prints a summary like this:

```
---
hit_ratio:          0.523400
curriculum_reward:   0.124500
avg_damage_landed:   42.30
avg_damage_taken:    18.70
final_D:             2.26
final_entropy:       -3.245000
final_kl:            0.012300
final_surrogate:     0.003400
epochs_completed:    87
training_seconds:    1200.3
```

If the run crashed, you will see `CRASH (exit code N)` followed by the last 30 lines of the log. You can always read the full log at `autoresearch/run.log` for more detail.

## Logging results

When an experiment is done, log it to `autoresearch/results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 4 columns:

```
commit	hit_ratio	status	description
```

1. git commit hash (short, 7 chars)
2. hit_ratio achieved (e.g. 0.523400) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	hit_ratio	status	description
a1b2c3d	0.523400	keep	baseline
b2c3d4e	0.612300	keep	lr 5e-4 (2x baseline)
c3d4e5f	0.487200	discard	lr 1e-3 (too high, KL exploded)
d4e5f6g	0.000000	crash	hidden_dim 1024 (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `ablation/apr7`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Edit `python/config.py` with a hyperparameter change.
3. git commit
4. Run the experiment: `bash autoresearch/run_experiment.sh` (output is already clean — just the summary)
5. If the output says CRASH, read `autoresearch/run.log` via `tail -n 50 autoresearch/run.log` to diagnose. If you can't get things to work after more than a few attempts, give up on that change.
6. Record the results in the tsv (NOTE: do not commit the results.tsv or run.log files, leave them untracked by git)
7. If hit_ratio improved (higher), you "advance" the branch, keeping the git commit
8. If hit_ratio is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**One variable at a time**: Prefer changing ONE hyperparameter per experiment. This is ablation, not random search. After establishing which direction helps for individual parameters, you can combine winners in later experiments.

**Follow the gradient — don't abandon promising leads**: When a change helps, keep pushing in that direction until it stops helping. If `entropy_coeff 0.01 → 0.02` improves hit_ratio, the next experiment should be `0.03` or `0.04`, not a completely different parameter. Ride the win all the way to the peak — only move on once you see a clear plateau or regression. Conversely, when a change hurts, try the *opposite* direction before giving up on that parameter: if `lr 5e-4 → 1e-3` regressed, try `2e-4` next rather than concluding "lr is fine." You've only falsified one side of the knob. A parameter is only "done" once you've seen both directions underperform or you've found its peak. Never declare a promising direction "explored" after a single step — that's how you leave 20% gains on the table.

**Use diagnostics**: Check `final_entropy` and `final_kl` in the output. If KL is consistently high (> 0.05), the learning rate or clip_eps may be too aggressive. If entropy collapses early, `entropy_coeff` is too low. Use these signals to guide your next experiment, not just hit_ratio.

**Noise**: RL is noisy. If a result is within ~5% of baseline, it's probably noise — treat it as "equal" and discard. Focus on changes that show clear directional improvement.

**Timeout**: Each experiment takes ~30 minutes. If a run exceeds 40 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the code for new angles, try combining previous near-misses, try wider ranges of parameters you've already tested. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes ~30 minutes then you can run 2/hour, for a total of about 16 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
