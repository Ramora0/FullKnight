# Current session — may10 from-scratch ablation

Read this alongside `program.md` (the process) and `ideas.md` (the menu of
ablations). This file is the current session's context and priority ordering.

## What's different from the last session

Compared to the apr10/apr11 ablation series, the regime has changed:

- **From-scratch**, not warm-start. No `models/fullknight_500.pth` resume.
  Every experiment starts at random init. Things calibrated to a specific
  resume point (e.g. lr-on-resume tricks) don't apply here.
- **fpw=1**, not 5. Per-step game-time is still pinned at 0.0424s by the C#
  `captureDeltaTime` calculation, but each agent-step is now 1 Unity frame
  instead of 5. The agent's per-step regime is unchanged; the sim cost
  per env-step dropped ~5×.
- **n_envs=8**, not 16. Forced by `run_experiment.sh`. Per-boss data is
  ~2 envs × ~128 steps per epoch = ~256 samples/boss/epoch on the 4-boss
  pool. Per-epoch variance is high; treat sub-5% deltas as noise.
- **4-boss pool** is the default
  (`GG_False_Knight, GG_Mega_Moss_Charger, GG_Gruz_Mother, GG_Hornet_1`).
  Don't switch to single-boss for ablations — the user wants cross-boss
  generalization.
- **20-min budget**, not 30. Trades off late-phase signal for more
  experiments per hour.
- **Reward shaping reverted to bare per-step**: `δ_attack/D − hits_taken
  + heal_coef·hp_healed`. The proximity / idle / attack-bonus shaping
  added during the training-collapse session was reverted (5a17d56)
  because the proximity moved its own metric without translating to
  damage and the attack bonus failed entirely. Don't re-add reward
  shaping fields without the user's explicit go-ahead.
- **Action-head init bias** is now the load-bearing fix for the early-policy
  collapse:
  - `head_action.bias[0]   = +1.0` (attack_tap)
  - `head_action.bias[1,3,5,6] = −2.0` (the four hold actions:
    nail_charge, focus, dream_nail, super_dash — each triggers a long
    C#-side hard-commit window of 36/12/71/24 agent-steps)
  - `head_action.bias[7]   = −2.0` (none, the catch-all idle bucket)
  This shape isn't a config knob — it's hard-coded in `model.py`. If you
  want to ablate it, surface the idea to the user first; the default
  loop is config-only.

## Live first priorities

The collapse fixes from the prior session got D up from ~6 (unfixed) to ~10
(fixed) on single-boss Moss Charger over 25 min. The 4-boss from-scratch
20-min number is the new baseline you'll measure against — record it on the
**first run** with no config edits.

After the baseline is locked in, work down `ideas.md`. The top of that file
is sequenced for this session. The first three knobs to push, in priority
order:

1. **`entropy_coeff`** — currently 0.02. The init bias shapes the early
   distribution but the entropy term keeps it that way. If entropy is
   crashing under -2.5 in a 20-min run, push to 0.03 / 0.05. If it's
   holding above -2.0 with no progress, try 0.01.
2. **`lr`** — currently 3e-4. The apr11 sweep found 3e-4 best with
   anneal-on; we removed `anneal_lr`, so the constant 3e-4 is what the
   policy actually sees end-to-end now. Try 1.5e-4 (down) and 5e-4 (up).
3. **`train_iters`** — currently 2. Combined with `target_kl=0` (= no
   early-stop), this means a deterministic 2 grad iters per rollout.
   Try 3 (more grad updates) and 1 (less drift per rollout).

Don't combine these until you've found each individual peak. Once each
parameter has a known peak, run the "all-best" combination as a final
experiment.

## Diagnostics to watch every run

From the `---` summary block:
- **`final_D_geomean`** — primary metric.
- **`final_D/<boss>` for each boss** — sanity-check the geomean isn't
  dragged up by one boss while the others regress.
- **`final_entropy`** — should not crash below ~-3.0 in a 20-min run.
  If it does, the policy is collapsing — bump `entropy_coeff`.
- **`final_kl`** — with `target_kl=0` (no clamp), this just reports what
  the policy is actually doing. Spikes above ~0.05 mean the trust region
  is broken; tighten `clip_eps` or drop `lr` / `train_iters`.
- **`final_surrogate`** — should be small and move smoothly (10⁻³ to 10⁻²).
  Wildly oscillating means trust-region trouble.
- **`pct_samples_trained`** — at `target_kl=0` this should be 100%. If it
  drops below ~95%, something is shorting the inner training loop.
- **`avg_damage_landed`** and **`avg_hits_taken`** — sanity check that the
  agent is interacting with bosses. If both fall to ~0 mid-run, that's
  the knight/boss-disappeared glitch firing — the glitch dumper in
  `train.py` should pick it up; also check if `final_avg_hits_per_boss`
  pegged at the 100-hit timeout.

Also skim the per-epoch action-distribution print line:
`pol  | a[atk=… chg=… spl=… foc=… dsh=… drm=… sdsh=… none=…] m[L=… R=… N=…]`
If `none` ramps past ~0.40 or `atk` collapses to 0.00, the policy is
hiding from the boss again — flag it as a regression of the init-bias
fix even if D_geomean looks fine.

## Non-goals for this session

- Don't add reward-shaping fields. If you think the per-step signal is
  inadequate, surface it to the user before editing.
- Don't change `D_initial`, `D_min`, `D_ema`, `D_window`, `D_max_delta`,
  or `heal_coef`. The reward is the comparison axis.
- Don't change architecture dims or the action-head init bias. Both are
  cross-run constants for this series.
- Don't modify `train.py`, `ppo.py`, or `model.py`.
- Don't try to fix wandb — it's known broken on this host. The summary
  block is the signal.
- Don't switch to single-boss training for "easier" ablations.
