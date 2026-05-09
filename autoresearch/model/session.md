# Current session — post KL-fix regression

Read this alongside `program.md` (the process) and `ideas.md` (the menu of
ablations). This file is the current session's context and priority ordering.

## What changed since the last session

`train.py`'s KL early-stop semantics were updated. It used to exit the inner
training loop the moment any single minibatch's KL exceeded `target_kl`. It
now exits only when the epoch-mean KL exceeds `target_kl`. This is
empirically correct — per-minibatch KL is noisy and a single outlier isn't a
reliable "we've drifted too far" signal, the mean is.

**But**: the old per-batch early-stop was acting as an *implicit regularizer*.
It was cutting off outlier updates, which capped the policy's per-rollout
drift. Removing that cap is letting the policy drift too far per rollout,
and the model is **regressing rapidly** from the `fullknight_500` warm
start. This is urgent — every experiment started right now will go in the
wrong direction unless we first fix the regression.

## Phase A: stop the regression (do this first)

Use **15-minute experiments**, not 30. The regression signal is extreme —
D_geomean, entropy, and avg_damage_landed will all move noticeably within a
few dozen epochs. No need to wait for a 30-min run to see the signal.

**Before starting phase A**: set `TIME_BUDGET=900` in `autoresearch/run_experiment.sh`
(was 1800). Commit that change on the ablation branch.

### Phase A success criterion

A config is "good enough" to exit phase A and move to phase B when **over a
15-min run, `final_D_geomean` stays at or above the D value recorded in the
`fullknight_500` checkpoint**. (The starting D for Moss Charger at epoch 500
was ~26 per the ablation logs; the 4-boss pool will have per-boss values you
can read from the first epoch's print line.) Bonus signals for a good config:
entropy does not crash (stays above -2.5 or so), and `final_surrogate` moves
steadily rather than thrashing.

### Phase A ideas, ordered by expected impact

All of these directly or indirectly reduce per-rollout policy drift, which is
the root cause now that the per-batch clamp is gone.

1. **`train_iters 4 → 2`**. Biggest single lever — halves the number of
   sequential gradient steps per rollout, directly halving per-rollout drift.
   If you're going to try one thing first, this is it.
2. **`target_kl 0.03 → 0.015`**. Tightens the new (correct) epoch-mean
   ceiling. Partially restores the old "too much drift → stop" cutoff, just
   with a better-statistic version.
3. **`target_kl 0.03 → 0.01`**. If 0.015 helped, push further.
4. **`train_iters 4 → 1`**. Extreme: one iter per rollout. If 2 helped, this
   might help more — or might be over-tight.
5. **`clip_eps 0.2 → 0.1`**. Orthogonal to the above — tighter ratio bound
   means less drift per minibatch *inside* an iter. Combines naturally with
   fewer iters.
6. **`steps_per_epoch 1024 → 512`**. Smaller rollouts = less data to overfit
   per iter. Expected to help but the scaling advantages are diminishing,
   and at 512 the per-epoch variance gets aggressive. Try last.
7. **Combinations** of winners (e.g. `train_iters 2 + target_kl 0.015`)
   once individual winners are established.

### Exit from phase A

Once you've found a config where D doesn't regress, that becomes the new
baseline. Commit it, then move to phase B. Don't combine endlessly — get to
"regression stopped" and move on.

## Phase B: normal hyperparameter ablation

Once phase A has established a non-regressing baseline:

1. Bump `TIME_BUDGET` back to **1800** (30 min) in `run_experiment.sh`.
   Second-order effects need the longer signal.
2. Re-run a 30-min baseline at the new phase-A-winner config so phase B has
   a clean reference number.
3. Work through `ideas.md` from the top. The top-priority items there are
   still valid: `clip_eps`, `entropy_coeff`, `lr`, `target_kl` (if not
   already fully explored in phase A), and `gae_lambda`.
4. **Do NOT re-try the same ideas you already explored in phase A.** If
   `train_iters 2` was the phase A winner, don't ablate `train_iters` again
   in phase B unless you have a new hypothesis.

## Environment assumptions

- Resume from `models/fullknight_500.pth` (unchanged — baseline checkpoint).
- 4-boss pool is the default (`config.boss_levels =
  "GG_False_Knight,GG_Mega_Moss_Charger,GG_Gruz_Mother,GG_Hornet_1"`).
  **Do not switch back to single-boss.** Per-boss variance is higher, so
  treat sub-5% deltas as noise.
- `n_envs=8`, `total_steps_per_epoch=1024` are the current defaults
  (apr10 ablation winners).
- `D_window`, `D_ema`, `D_max_delta` are now auto-rescaled per-sample by
  `train.py` (the printed `D curriculum: step_scale=... window=... ema=...
  max_delta=...` line at startup tells you the effective values for your
  `steps_per_epoch`). Don't try to re-tune these unless the startup print
  shows something obviously wrong.

## Diagnostics to watch every run

From the `---` summary block:
- **`final_D_geomean`** — primary metric. Should hold or climb vs the
  starting D.
- **`final_entropy`** — should not crash below ~-2.5 over a 15-min run. If
  it does, the policy is collapsing and needs higher `entropy_coeff`.
- **`final_surrogate`** — should be small and move smoothly (order
  10^-3 to 10^-2). Wildly oscillating means the trust region is broken.
- **`avg_damage_landed`** and **`avg_hits_taken`** — sanity check that the
  agent is still interacting with bosses. If both drop to ~0, something's
  seriously wrong.

Also skim the per-epoch log lines for the `kl` field — with the new
mean-based semantics, you should see kl values distributed around target_kl
rather than clipped to below it. That's the confirmation the new semantics
are working as intended.

## Non-goals for this session

- Don't retouch the D scaling math. It's done.
- Don't add new Config fields. Only tune existing numeric knobs.
- Don't modify `train.py`, `ppo.py`, or `model.py`. Read-only as per
  `program.md`.
- Don't try to fix `wandb` — it's been confirmed broken on this box and
  runs with `WANDB_MODE=disabled`. The `---` summary block is the signal.
