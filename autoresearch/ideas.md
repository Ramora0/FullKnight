# Ablation ideas

Running list of hyperparameter experiments to try, prioritized by expected impact
given what we've learned so far. Diagnostic-driven — reorder based on results.

**Read `session.md` first.** This file is the longer-term menu; `session.md`
has the current-session priority. Right now the first priority is stopping a
post-KL-fix regression (phase A in `session.md`), not general parameter
tuning.

## Context from apr10 ablation

- `total_steps_per_epoch` **8192 → 1024** was a huge win (D_geomean 38.9 → 81.6)
  on single-boss Moss Charger. Default is now 1024.
- Even at 1024 steps, we're hitting the KL boundary consistently during training.
  This means we're **gradient-step-limited, not data-limited**: only ~1 of 4
  nominal train_iters actually completes before early-stop fires. Most of the
  measured win came from fresher on-policy data and more D curriculum updates,
  not from "more gradient updates per wallclock."
- 4-boss pool (default now) has more per-epoch variance than single-boss;
  sub-5% deltas should be treated as noise.
- entropy drifted -2.10 → -1.70/-1.81 during ablation — watch for collapse
  over longer runs.

## Top priority — directly addresses the KL-clamp bottleneck

These should all compound with each other and with the steps=1024 win.

### clip_eps 0.2 → 0.1
Single most informative experiment. Tighter ratio bound means less policy
drift per minibatch, which lets more iters complete before early-stop fires.
If we're currently finishing ~1 iter and this gets us to ~3 iters, we triple
the effective gradient updates per rollout. Watch `final_surrogate` — it
should stay nonzero across all iters if the clip is doing its job.

### train_iters 4 → 1 or 2
If we're only completing ~1 iter anyway, dropping the nominal count costs
nothing and saves loop overhead. Also a sanity check: if D_geomean doesn't
move, confirms the KL-clamp hypothesis. If it drops, we were getting more
than 1 iter of value after all.

### lr 5e-4 → 2.5e-4 (and maybe 1e-4)
Smaller per-step moves, more of them before hitting the boundary. Caveat:
Adam optimizer state is restored on resume, so need to verify the lr change
actually takes effect (may require manually overwriting `optimizer.param_groups[0]["lr"]`
after load, or turning off `anneal_lr` since scheduler state restore overrides it).

### target_kl 0.03 → 0.05 (permissive direction)
Opposite of tightening clip: let more updates through by raising the ceiling.
Risk: allows policy to drift further from rollout policy per epoch, so
advantages get staler. Worth trying as an orthogonal lever to clip_eps.

### entropy_coeff 0.01 → 0.02 (→ 0.03/0.05)
Keeps the policy flatter. Flatter distribution → smaller KL per logit change
→ more headroom before early-stop. Also insurance against entropy collapse
over long runs. Direction driven by `final_entropy` diagnostic.

## Second tier — orthogonal knobs

### gae_lambda 0.95 → 0.9 / 0.98
Bias-variance tradeoff in the advantage estimator. Cheap to try.

### chunks_per_batch / seq_len
Effective batch = chunks_per_batch × seq_len = 8 × 16 = 128 samples.
- Smaller (chunks_per_batch 4): higher gradient variance, more updates per iter
- Larger (chunks_per_batch 16): smoother updates, fewer per iter
- seq_len 8 or 32: BPTT window; affects credit assignment through the GRU

### value_coeff 0.5 → 0.25 / 1.0
Critic-vs-actor loss weighting. `max_value_loss` similarly.

### max_grad_norm 0.5 → 1.0 / 0.25
Gradient clipping for numerical stability.

## Lower priority / specific hypotheses

### n_envs (6-12 range, per user direction)
8 is the current default. Try 6 down / 12 up. Not expected to be a huge
signal — the throughput win is already baked in. Per-boss variance changes
with n_envs in the 4-boss setup.

### Hidden-state reset frequency
train.py currently does staggered resets: `envs_per_reset = max(1, n_envs // 4)`
envs get their GRU hidden state reset each epoch. Try:
- More frequent resets (n_envs // 2) — fresher hidden state, less carryover
- Less frequent resets (n_envs // 8 or no staggered reset) — more temporal context
This probably needs either a train.py tweak or a new config knob.

### total_steps_per_epoch 2048 (re-test)
The steps=1024 win was measured on single-boss. On 4-boss, per-boss variance
might favor slightly longer rollouts. Worth re-baselining 1024 vs 2048 on
the 4-boss setup before assuming 1024 is still optimal.

## Risky / large-scope

### frames_per_wait 5 → 3 / 7
Changes the effective time horizon. Forces re-tuning of most other knobs.
Only try with a specific hypothesis from diagnostics (e.g. if we see consistent
step0 latency issues or action timing misses).

## Combinations to try after individual winners

- clip_eps 0.1 + train_iters 2
- clip_eps 0.1 + entropy_coeff 0.02
- lr 2.5e-4 + target_kl 0.05
- All-in: clip_eps 0.1 + train_iters 2 + entropy_coeff 0.02 (the "fully KL-aware" setup)

## Open investigations (not ablations per se)

- **Does lr change actually take effect on resume?** Verify by printing
  `optimizer.param_groups[0]["lr"]` immediately after `load_checkpoint` and
  comparing to config.lr. If not, need to manually override post-load.
- **How many train_iters are actually completing on average?** Instrument
  train.py to log the real iter count per epoch. This would directly confirm
  the KL-clamp hypothesis and let us tune more precisely.
- **Is advantage normalization happening?** Check ppo.py — if not, adding
  per-batch advantage normalization (mean 0, std 1) might stabilize the
  gradient magnitude and reduce KL variance.
