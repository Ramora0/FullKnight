# Ablation ideas

Running list of hyperparameter experiments to try, prioritized by expected
impact under the current may10 regime. Diagnostic-driven — reorder based on
results.

**Read `session.md` first.** This file is the longer-term menu; `session.md`
has the current-session priority ordering and any urgent fixes. The current
defaults are the may10 baseline state (post-training-collapse fixes,
fpw=1, n_envs=8, from-scratch).

## Current baseline config

Knobs that matter for ablation, with their as-of-may10 values:

| Field                     | Value | Range to consider |
|---------------------------|-------|-------------------|
| `lr`                      | 3e-4  | 1e-4 → 1e-3 |
| `gamma`                   | 0.95  | 0.9 / 0.97 / 0.99 |
| `gae_lambda`              | 0.95  | 0.85 / 0.90 / 0.97 |
| `clip_eps`                | 0.2   | 0.1 / 0.3 / 0.4 |
| `value_coeff`             | 0.5   | 0.25 / 1.0 |
| `entropy_coeff`           | 0.02  | 0.005 / 0.01 / 0.03 / 0.05 |
| `max_grad_norm`           | 0.5   | 0.25 / 1.0 |
| `target_kl`               | 0.0   | 0.01 / 0.02 / 0.05 (re-enable early-stop) |
| `total_steps_per_epoch`   | 1024  | 512 / 2048 |
| `batch_size`              | 128   | 64 / 256 |
| `train_iters`             | 2     | 1 / 3 / 4 |
| `chunks_per_batch`        | 8     | 4 / 16 |
| `seq_len`                 | 16    | 8 / 32 |

Off-limits (see `program.md`): structural fields, observation dims, action
dims, architecture dims, the reward-shape constants
(`D_*`, `heal_coef`, `D_initial`, …), and the action-head init bias.

## Top priority — exploration / drift / gradient-step efficiency

These should compound and address the three things most likely to be
limiting from-scratch 20-min performance.

### entropy_coeff 0.02 → 0.03 → 0.05 (or 0.01 / 0.005 if entropy is healthy)
The init bias gives the action distribution a sane shape, but the entropy
term is what keeps it spread out. If `final_entropy < -3.0` we're losing
exploration; bump up. If `final_entropy > -1.5` and D isn't moving, we're
over-exploring; bump down. Read `final_entropy` from the baseline first to
pick direction.

### lr 3e-4 → 1.5e-4 / 5e-4 / 1e-4 / 1e-3
Apr11 found 3e-4 + lr-anneal optimal on warm-start. We dropped the
anneal, so the constant 3e-4 is what the policy gets end-to-end. Worth
re-baselining: from random init, a higher constant lr might help the
early phase enough to outweigh late-phase drift. Watch `final_kl` —
spikes above ~0.05 say the lr is too aggressive; clamp via
`clip_eps`/`target_kl` or drop lr.

### train_iters 2 → 1 / 3 / 4
With `target_kl=0` (no early-stop), this is a deterministic count of
gradient passes per rollout. 2 is the apr11 winner under the warm-start
regime; from-scratch may want more (more updates per fresh batch) or
less (less per-rollout drift while the value head is still fitting).

### target_kl 0.0 → 0.02 / 0.03 / 0.05 (re-enable early-stop)
Setting target_kl > 0 re-enables the epoch-mean-KL early-stop in
`train.py`. With aggressive lrs or larger train_iters this gives a
soft trust region. Cheap to combine with lr / train_iters experiments.

### clip_eps 0.2 → 0.1 / 0.3
Tighter (0.1) reduces per-minibatch policy drift, lets more iters land
useful gradient. Looser (0.3) lets the policy move further per update —
might help early when the random-init policy is far from optimal.

## Second tier — orthogonal knobs

### gae_lambda 0.95 → 0.9 / 0.85 / 0.97
Variance-bias tradeoff in the advantage estimator. With `gamma=0.95` and
short rollouts (T=128 per env), lower lambda focuses credit on closer
events — might help when the per-step damage signal is the main thing
that matters.

### gamma 0.95 → 0.97 / 0.99 / 0.9
Boss attacks chain over many steps; higher gamma weights long-term
returns more. But high gamma + per-step normalized advantage makes the
critic harder to fit. Test cautiously.

### total_steps_per_epoch 1024 → 512 / 2048
Apr11 showed 256-step epochs were best on warm-start with
reset-amortization. From-scratch may need bigger rollouts to overcome
critic noise, or stay small to maximize curriculum updates per minute.

### batch_size 128 → 64 / 256 (effective batch = chunks_per_batch × seq_len)
Smaller = more updates per iter + higher gradient variance. Larger =
smoother but fewer updates.

### chunks_per_batch 8 → 4 / 16, seq_len 16 → 8 / 32
The BPTT chunk shape. Longer seq_len = more credit through the GRU but
a stricter on-policy assumption (the GRU hidden state from chunk-start
becomes increasingly off-policy as the policy updates).

### value_coeff 0.5 → 0.25 / 1.0
Critic-vs-actor loss weighting. From-scratch the critic is initially
useless; pushing value_coeff up may help it catch up sooner.

### max_grad_norm 0.5 → 0.25 / 1.0
Gradient clipping. If `final_kl` is clean and surrogate is stable, this
probably doesn't matter; if you see KL spikes, tighter clipping is
cheap insurance.

## Combinations to try after individual peaks

Only run combinations once each individual knob has a known peak:

- entropy_coeff(peak) + lr(peak)
- train_iters(peak) + target_kl(peak)
- entropy_coeff(peak) + lr(peak) + train_iters(peak) — the "all-best"
  end-of-session run.

Compounding doesn't always work; if the combination is below the best
individual, pick the best individual.

## Lower priority / specific hypotheses

### `hard_restart_every_epochs`
Currently 1200 (~25k epochs at 20min, so basically never). If you see
state-creep over the run (D drift, action-distribution drift), try 200
or 400 to force fresh HK processes mid-run. Cost: ~30s reset per cadence.

## Risky / large-scope (require user approval before trying)

These are NOT default ablations — surface to the user first.

- Modifying the action-head init bias magnitudes (`bias[0]`, `bias[1,3,5,6]`,
  `bias[7]`).
- Re-introducing reward shaping (proximity / idle penalty / attack bonus /
  shaped intermediate signals).
- Architecture-dim sweeps (`hidden_dim`, `gru_dim`, etc.).
- Changing `frames_per_wait` away from 1.
