# Training-collapse investigation — autonomous session

Branch: `fix/training-collapse` (off `sim/may8`).

## Reproduced symptom

Baseline `train.py --n_envs 8 --boss_levels GG_Mega_Moss_Charger` (seed 42)
plateaus around dmg ~0.05–0.07 per step, hits_taken ~0.008–0.010, with
active_steps stuck at 1–2% throughout 200 epochs. Not literally zero
"hits given/taken" but engagement is so low it's effectively non-learning.

D climbs from initial 2.0 → ~6.0 in 200 epochs. The reward
`dmg/D − taken + heal*hp_healed` approaches zero at equilibrium because
D adapts to track the agent's actual ratio.

## Root cause: action[2]="none" collapse

The breakthrough came from adding action-distribution logging
(commit `94c1b75`). The pre-fix policy distribution showed:

| epoch | atk  | chg  | spl  | foc  | dsh  | none |
|-------|------|------|------|------|------|------|
| 0     | 0.09 | 0.16 | 0.10 | 0.07 | 0.10 | 0.47 |
| 30    | 0.09 | 0.11 | 0.10 | 0.07 | 0.04 | 0.57 |
| 60    | 0.11 | 0.07 | 0.07 | 0.04 | 0.04 | 0.67 |
| 90    | 0.08 | 0.12 | 0.02 | 0.06 | 0.02 | 0.69 |

**Mechanism**: action[2]=7 ("none") is the only choice without a
CAN_X validity mask. Whenever attack/dash/cast/etc. get masked out
by their CAN flags (frequent: attack cooldown, no soul, etc.),
"none" picks up renormalized probability mass. The PG correctly
attributes the value of those steps to "none" and the bias drifts
further toward idle. Idle → no hits taken → positive advantage
relative to "do something risky" → bias rises further. Lock-in.

## Fixes (all on `fix/training-collapse`)

1. `13cc035` — **D-jump bug fix**. The curriculum used to set
   `D = D_raw` directly on the first epoch (single-window-entry
   special case), instantly clobbering `D_initial=2.0` with whatever
   noisy ratio the random policy produced. Removed the special case;
   D now EMAs from D_initial with the standard per-epoch clamp.

2. `94c1b75` — **Potential-based proximity shaping**. The naive
   `coef * Φ_t` form (a per-step bonus for being near a target
   hitbox) was wiped out by per-rollout advantage centering — Φ
   varies slowly so the per-step bonus is roughly constant. Switched
   to `F_t = γΦ_{t+1} − Φ_t`, whose mean is naturally zero so
   centering doesn't kill it. Also added the per-epoch action-head
   distribution print line.

3. `0d31391` — **Idle penalty + attack bonus**. Counter-pressure
   to the "none" collapse:
   - `idle_action_penalty=0.05`: per-step penalty for FREELY-chosen
     action[2]=7. Excluded on committed steps (the C# hard-commit FSM's
     post-charge release transition is action[2]=7 forced — not the
     agent's free choice).
   - `attack_action_bonus=0.05`: symmetric bonus for FREELY-chosen
     action[2]=0. Idle penalty alone made the agent prefer hold
     actions (chg, foc) since their committed-FSM steps are
     penalty-free; the attack bonus targets the desired behavior.

## Validated behavior change

With all fixes at default magnitudes (idle 0.05, attack 0.05,
prox_coef 1.0), seed 42, single boss, 25-min budget:

- "none" stays in 0.30–0.50 range (vs ramping to 0.69 unfixed).
- D climbs to ~9.8 over 540 epochs (vs ~6.0 ceiling unfixed).
- hits_taken drops to ~0.006 (vs ~0.010 unfixed).
- dmg ~0.07–0.12 with peaks to 0.15 (vs ~0.05 plateau unfixed).

The dmg/taken **ratio** is materially better (D=10 vs 6 at end of
comparable runs). The plateau is not eliminated but the agent's
hit-ratio is meaningfully higher than baseline.

## What didn't break the plateau

Tried but reverted to defaults:
- Stronger shaping magnitudes (idle 0.2, attack 0.2): pushed the
  agent into a chg+foc-heavy strategy that had similar dmg without
  using attack-tap at all — gaming the reward, not the desired
  behavior.
- Pinning D=1 (no curriculum): made `curr_rew` look better but
  didn't change actual dmg or active_steps; D adapts to be neutral
  at equilibrium so this just rescales the metric.
- LR=1e-3 (3x): some marginal lift, kept along with default.
- entropy_coef=0.05 (2.5x): kept along with default.

## Open issues / next directions

The agent's later-epoch behavior under default shaping is
chg-and-foc dominant (atk ≈ 0). This may or may not be the right
strategy for boss combat — nail arts deal big damage but require
charge time. atk-tap might be underused. Possible next fixes:

- Action-head bias toward attack at init that's harder to erode
  (e.g., bias[0]=3.0 instead of 1.0).
- Direct constant subtractor in `_mask_logits` to discourage
  action[2]=7 at the model layer (similar to how invalid actions
  are masked, but as a soft prior).
- Curiosity-driven exploration for the genuinely sparse-reward
  approach phase.
- Architectural: GRU may be too slow to learn temporal patterns
  with only ~200k samples.

## Files

- `python/config.py` — added proximity_coef, proximity_scale,
  attack_action_bonus, idle_action_penalty fields.
- `python/ppo.py` — get_advantages takes proximity, idle_mask,
  attack_mask; train_on_rollout builds the masks from actions_arr
  and committed_arr.
- `python/train.py` — _compute_proximity helper; per-epoch
  proximity buffer; action-dist diagnostic print.

Logs from the 12 experiment runs are in `autoresearch/sim/fix*.log`
(untracked).
