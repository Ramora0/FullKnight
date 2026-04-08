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

Each experiment trains for a **fixed time budget of 20 minutes** (wall clock). Launch it as:

```bash
.venv/Scripts/python.exe python/train.py --time_budget 1200 > autoresearch/run.log 2>&1
```

The `--time_budget` flag sets the wall-clock limit in seconds and auto-disables wandb. All verbose output goes to `autoresearch/run.log`; pipe through `sed -n '/^---$/,$ p' autoresearch/run.log` afterward to extract the summary block.

**What you CAN do:**
- Modify `python/config.py` — this is the only file you edit. Only numeric hyperparameters are fair game: learning rate, discount factor, entropy coefficient, batch size, network dimensions, curriculum parameters, etc.

**What you CANNOT do:**
- Modify `train.py`, `ppo.py`, `model.py`, `run_experiment.sh`, or any other file. They are read-only.
- Change non-tunable structural fields: `server_host`, `server_port`, `n_envs`, file paths, observation dims (`combat_feature_dim`, `terrain_feature_dim`, `global_state_dim`), action dims (`movement_n`, `direction_n`, `action_n`, `jump_n`), or `n_validity_flags`.
- Install new packages or add dependencies.

**The goal is simple: get the highest hit_ratio** (damage_landed / damage_taken, averaged over the last 20 epochs of each run). Since the time budget is fixed, experiments are directly comparable regardless of how many epochs complete. Higher hit_ratio is better.

**This is an early MVP, not a final product.** The model will be trained much longer after this ablation pass. Keep long-run learning potential in mind, but balance it against measured results:
- A change that *should* help long-run generalization but costs ~5% hit_ratio? Probably worth keeping — short-run noise matters less than ceiling.
- A change that hurts long-run potential (e.g. gamma too low, entropy too low, network too small) but shows 20%+ improvement? Probably still worth keeping — a massive empirical win outweighs theoretical concerns.
- Use judgment. Neither short-run metrics nor long-run theory should be an absolute veto. Weigh both.

**Think about the domain.** This is a Hollow Knight boss-fighting agent — a fast-paced 2D action game with variable-length hitbox observations, frame-skip, and real-time combat. Default PPO hyperparameters come from Atari/MuJoCo papers and may not suit this environment at all. Consider:
- **Epoch length** (`total_steps_per_epoch`): 2048 steps at 5-frame skip is only ~570 real-time frames per epoch at 60fps. Is that enough per rollout? Too much? This is absolutely on the table.
- **Frame skip** (`frames_per_wait`): Boss attacks are fast — too much skip and the agent can't react; too little and credit assignment gets harder.
- **Discount factor** (`gamma`): Boss fights have short time horizons compared to e.g. Atari adventure games. 0.99 may or may not be appropriate.
- **Batch size / train iters**: With only 2048 steps per epoch and batch_size 128, that's 16 batches × 4 iters = 64 gradient steps per epoch. Is that enough? Too many reuses of the same data?
- The defaults were guesses. Question all of them.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

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

**Use diagnostics**: Check `final_entropy` and `final_kl` in the output. If KL is consistently high (> 0.05), the learning rate or clip_eps may be too aggressive. If entropy collapses early, `entropy_coeff` is too low. Use these signals to guide your next experiment, not just hit_ratio.

**Noise**: RL is noisy. If a result is within ~5% of baseline, it's probably noise — treat it as "equal" and discard. Focus on changes that show clear directional improvement.

**Timeout**: Each experiment takes ~20 minutes. If a run exceeds 30 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the code for new angles, try combining previous near-misses, try wider ranges of parameters you've already tested. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~20 minutes then you can run 3/hour, for a total of about 24 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
