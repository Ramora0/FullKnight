import asyncio
import os
import time
from collections import deque
import numpy as np
import torch
import wandb

from config import Config
from vec_env import VecEnv
from ppo import PPO
from instance_manager import InstanceManager


def merge_padded(old, new, indices, fill=0.0):
    """Overwrite rows in old with rows from new, expanding padding dims if needed.

    `indices` is a list of row positions in `old` to overwrite; `new[k]` goes
    into `old[indices[k]]`.
    """
    if old.shape[1:] != new.shape[1:]:
        pad = [(0, 0)] + [(0, max(0, n - o)) for o, n in zip(old.shape[1:], new.shape[1:])]
        old = np.pad(old, pad, constant_values=fill)
    for local_i, env_i in enumerate(indices):
        old[env_i] = fill
        idx = tuple([env_i] + [slice(0, s) for s in new.shape[1:]])
        old[idx] = new[local_i]
    return old


def merge_obs_padded(dst, src, indices):
    """Scatter rows of batched Observation `src` into `dst` at `indices`.
    Handles padding mismatch per-field via merge_padded. global_state has no
    padding so it's a plain row copy.
    """
    out = dst.replace(
        combat_hb=merge_padded(dst.combat_hb, src.combat_hb, indices),
        combat_mask=merge_padded(dst.combat_mask, src.combat_mask, indices),
        combat_kind_ids=merge_padded(dst.combat_kind_ids, src.combat_kind_ids, indices),
        combat_parent_ids=merge_padded(dst.combat_parent_ids, src.combat_parent_ids, indices),
        terrain_hb=merge_padded(dst.terrain_hb, src.terrain_hb, indices),
        terrain_mask=merge_padded(dst.terrain_mask, src.terrain_mask, indices),
    )
    for local_i, env_i in enumerate(indices):
        out.global_state[env_i] = src.global_state[local_i]
    return out


def slice_obs(obs, indices):
    """Return a new Observation containing only the given env rows (in order)."""
    idx = np.asarray(indices, dtype=np.int64)
    return obs.replace(
        combat_hb=obs.combat_hb[idx],
        combat_mask=obs.combat_mask[idx],
        combat_kind_ids=obs.combat_kind_ids[idx],
        combat_parent_ids=obs.combat_parent_ids[idx],
        terrain_hb=obs.terrain_hb[idx],
        terrain_mask=obs.terrain_mask[idx],
        global_state=obs.global_state[idx],
    )


def seed_everything(seed: int):
    """Seed all RNGs for deterministic model init and action sampling."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


async def train(config: Config):
    if config.seed:
        seed_everything(config.seed)
        print(f"Seeded all RNGs with {config.seed}")

    vis = None
    if config.visualize:
        from visualizer import Visualizer
        vis = Visualizer()  # vocab attached after vec_env init
    # Launch game instances
    mgr = None
    if config.hk_path and os.path.exists(config.hk_path):
        print(f"Spawning {config.n_envs} HK instance(s)...")
        mgr = InstanceManager(config.hk_path, config.hk_data_dir)
        mgr.spawn_n(config.n_envs)
        mgr.start_all(graphical=(config.n_envs == 1))
    else:
        print(f"hk_path not found ({config.hk_path}) — launch Hollow Knight manually.")

    try:
        # Start vectorized environment server
        vec_env = VecEnv(config)
        await vec_env.start_server()
        if vis is not None:
            vis.vocab = vec_env.vocab

        bosses = config.boss_levels_list
        assert len(bosses) > 0, "config.boss_levels must list at least one scene"

        # D curriculum knobs (D_window, D_ema, D_max_delta) were tuned for
        # total_steps_per_epoch = 8192. Rescale them per-sample so behavior
        # stays consistent as we vary rollout length: a 1024-step run does
        # 8x more epochs per unit-data, so its per-epoch clamp should be 8x
        # smaller, its EMA decay 8x slower, and its window 8x wider.
        D_BASELINE_STEPS = 8192
        D_step_scale = config.total_steps_per_epoch / D_BASELINE_STEPS
        D_window_eff = max(1, int(round(config.D_window / D_step_scale)))
        D_ema_eff = config.D_ema ** D_step_scale
        D_max_delta_eff = config.D_max_delta * D_step_scale
        print(
            f"D curriculum: step_scale={D_step_scale:.4f} "
            f"window={D_window_eff} ema={D_ema_eff:.4f} "
            f"max_delta={D_max_delta_eff:.4f} (from config "
            f"{config.D_window}/{config.D_ema}/{config.D_max_delta})"
        )

        boss_state = {b: {
            "D": config.D_initial,
            "landed_window": deque(maxlen=D_window_eff),
            "taken_window":  deque(maxlen=D_window_eff),
        } for b in bosses}
        rng = np.random.default_rng(config.seed or None)
        env_boss = [bosses[int(rng.integers(len(bosses)))] for _ in range(config.n_envs)]
        print(f"Boss pool: {bosses}")
        print(f"Initial env_boss: {env_boss}")

        agent = PPO(config)
        start_env_steps = 0
        if config.resume:
            start_env_steps = agent.load_checkpoint(
                config.resume, vocab=vec_env.vocab, boss_state=boss_state
            )
            print(f"Resumed from: {config.resume}")
        print(f"Using device: {agent.device}")
        print(f"Model parameters: {sum(p.numel() for p in agent.policy.parameters()):,}")

        os.makedirs(os.path.dirname(config.save_path) or ".", exist_ok=True)
        # Time-budgeted runs default to wandb-off (quick local experiments),
        # but let the caller opt back in by setting WANDB_MODE=online explicitly.
        if config.time_budget and os.environ.get("WANDB_MODE") is None:
            os.environ["WANDB_MODE"] = "disabled"
        wandb.init(project=config.wandb_project, config=vars(config))

        time_budget = config.time_budget
        t_start = time.perf_counter()
        recent = deque(maxlen=20)

        # Slow-step bookkeeping: any per-env step whose wall time exceeds
        # `slow_step_threshold_s` is counted against the (env, boss) pair it
        # happened on. Lets us tell if slowness tracks a specific boss, a
        # specific env slot, or is scattered uniformly.
        slow_step_threshold_s = 2.0
        slow_count_by_boss = {b: 0 for b in bosses}
        slow_count_by_env = [0] * config.n_envs

        # First epoch: full reset to load boss scenes
        obs_full = await vec_env.reset_all(levels=env_boss)
        agent.reset_hidden(config.n_envs)
        active_envs = list(range(config.n_envs))

        # Staggered reset: every `steps_per_reset` accumulated env-steps,
        # schedule a reset for `envs_per_reset` envs round-robin. Resets run
        # as background tasks overlapping the next rollout; scheduled envs sit
        # out of the active set until their reset completes.
        envs_per_reset = max(1, config.n_envs // config.envs_per_reset_div)
        steps_since_last_reset = 0
        next_reset_env = 0  # round-robin cursor, advances only when scheduling

        # Step-driven training: total_env_steps bounds the run, save cadence
        # and LR annealing are both keyed on env_steps_collected (not epoch
        # count) so they stay meaningful as rollout size varies.
        env_steps_collected = start_env_steps
        last_save_step = start_env_steps
        epoch = -1  # local counter purely for logging
        while env_steps_collected < config.total_env_steps:
            epoch += 1

            # Reap any background resets that have completed since we kicked
            # them off at the end of the prior epoch. Splice new obs into
            # obs_full, zero their hidden state, and readd to active_envs.
            reaped = vec_env.reap_completed_resets()
            if reaped:
                reaped_indices = [env_i for env_i, _ in reaped]
                reaped_obs_batch = vec_env._batch_observations(
                    [raw for _, raw in reaped]
                )
                obs_full = merge_obs_padded(obs_full, reaped_obs_batch, reaped_indices)
                agent.reset_hidden_for(reaped_indices)
                active_envs = sorted(set(active_envs) | set(reaped_indices))

            # Rollout runs over the currently-active subset. Buffers are
            # (T, N_active) shaped; N_active may be < n_envs if a reset from
            # the previous epoch hasn't finished yet.
            N_active = len(active_envs)
            active_set = set(active_envs)
            active_boss = [env_boss[i] for i in active_envs]

            # Rollout buffers
            buf_obs = []  # list of per-step Observations
            buf_actions = {k: [] for k in ["movement", "direction", "action", "jump"]}
            buf_log_probs = []
            buf_values_atk = []
            buf_values_def = []
            buf_damage_landed = []
            buf_hits_taken = []
            buf_hx = []
            buf_step_game_times = []
            buf_step_real_times = []
            buf_step_wall_times = []

            t_rollout_start = time.perf_counter()

            # Slice the active-env view out of obs_full for the first step.
            obs = slice_obs(obs_full, active_envs)

            for t in range(config.rollout_len):
                buf_hx.append(agent.get_hx_snapshot(env_indices=active_envs))
                actions_np, log_probs, values_atk, values_def = agent.collect_action(
                    obs, env_indices=active_envs
                )

                action_vecs = [
                    [
                        int(actions_np["movement"][i]),
                        int(actions_np["direction"][i]),
                        int(actions_np["action"][i]),
                        int(actions_np["jump"][i]),
                    ]
                    for i in range(N_active)
                ]

                t_step = time.perf_counter()
                (next_obs, damage_landed, hits_taken, step_game_times, step_real_times,
                 step_wall_per_env) = await vec_env.step_all(
                    action_vecs, active_indices=active_envs
                )
                wall_dt = time.perf_counter() - t_step

                buf_obs.append(obs)
                for k in buf_actions:
                    buf_actions[k].append(actions_np[k])
                buf_log_probs.append(log_probs)
                buf_values_atk.append(values_atk)
                buf_values_def.append(values_def)
                buf_damage_landed.append(damage_landed)
                buf_hits_taken.append(hits_taken)
                buf_step_game_times.append(step_game_times)
                buf_step_real_times.append(step_real_times)
                buf_step_wall_times.append(step_wall_per_env)

                obs = next_obs

                if vis is not None:
                    vis.update(obs)

            # Bootstrap final values
            _, _, final_vatk, final_vdef = agent.collect_action(
                obs, env_indices=active_envs
            )
            buf_values_atk.append(final_vatk)
            buf_values_def.append(final_vdef)

            # Scatter the final obs back into obs_full so the next epoch has a
            # consistent per-env canonical state to splice reaped resets into.
            obs_full = merge_obs_padded(obs_full, obs, active_envs)

            # Stack buffers: (T, N)
            damage_landed_arr = np.stack(buf_damage_landed)
            hits_taken_arr = np.stack(buf_hits_taken)
            log_probs_arr = np.stack(buf_log_probs)
            values_atk_arr = np.stack(buf_values_atk)
            values_def_arr = np.stack(buf_values_def)
            actions_arr = {k: np.stack(v) for k, v in buf_actions.items()}
            buf_hx_arr = np.stack(buf_hx)  # (T, N, hidden_dim)

            # Diagnostic: first combat event per env, step timing
            any_event = (damage_landed_arr > 0) | (hits_taken_arr > 0)  # (T, N_active)
            active_steps = int(any_event.sum())
            total_steps_epoch = damage_landed_arr.shape[0] * damage_landed_arr.shape[1]
            first_event_steps = []
            for local_i in range(damage_landed_arr.shape[1]):
                col = any_event[:, local_i]
                idxs = np.where(col)[0]
                first_event_steps.append(int(idxs[0]) if len(idxs) > 0 else damage_landed_arr.shape[0])
            wall_time_arr = np.stack(buf_step_wall_times)  # (T, N_active)
            real_time_arr = np.stack(buf_step_real_times)  # (T, N_active) — C# unscaled sim time
            # First step may include intro skip — show it separately
            step0_ms = wall_time_arr[0].mean() * 1000 if wall_time_arr.shape[0] > 0 else 0
            avg_wall_ms = wall_time_arr[1:].mean() * 1000 if wall_time_arr.shape[0] > 1 else 0
            # Per-env slow-step inventory (skip step 0 which includes intro-skip).
            # For each env, report the max step time this epoch and every step
            # over `slow_step_threshold_s` gets counted against the boss.
            post_first = wall_time_arr[1:] if wall_time_arr.shape[0] > 1 else wall_time_arr
            real_post = real_time_arr[1:] if real_time_arr.shape[0] > 1 else real_time_arr
            per_env_max = post_first.max(axis=0) if post_first.shape[0] > 0 else np.zeros(N_active)
            per_env_slow_count = (post_first > slow_step_threshold_s).sum(axis=0)

            # === PERF DIAGNOSTICS ===
            # 1) Per-step spread across envs: how much wall time is wasted
            #    waiting on the slowest env? Big P99 vs P50 → straggler problem
            #    → queue/async stepping would help.
            if post_first.size:
                spread_s = post_first.max(axis=1) - post_first.min(axis=1)  # (T,)
                spread_p50_ms = float(np.percentile(spread_s, 50)) * 1000
                spread_p90_ms = float(np.percentile(spread_s, 90)) * 1000
                spread_p99_ms = float(np.percentile(spread_s, 99)) * 1000
                spread_max_ms = float(spread_s.max()) * 1000
            else:
                spread_p50_ms = spread_p90_ms = spread_p99_ms = spread_max_ms = 0.0

            # 2) C# real_dt vs Python wall_dt: how much of HK time is sim
            #    vs IPC/idle? If real_avg ≈ wall_avg the bottleneck is the
            #    game itself; if real_avg << wall_avg it's IPC/Python.
            #    real_dt is per-env; we average it over (T-1, N).
            real_avg_ms = float(real_post.mean()) * 1000 if real_post.size else 0.0
            overhead_ms = max(avg_wall_ms - real_avg_ms, 0.0)
            sim_pct = 100 * real_avg_ms / avg_wall_ms if avg_wall_ms > 0 else 0

            # 3) Per-boss avg step time: which bosses are slow stragglers?
            #    Helps decide which bosses to drop or load-balance.
            per_boss_step_ms = {}
            if post_first.size:
                for boss in set(active_boss):
                    env_mask = np.array([b == boss for b in active_boss])
                    if env_mask.any():
                        per_boss_step_ms[boss] = float(post_first[:, env_mask].mean()) * 1000
            slow_events_epoch = []
            for local_i, env_i in enumerate(active_envs):
                cnt = int(per_env_slow_count[local_i])
                if cnt > 0:
                    boss = env_boss[env_i]
                    slow_count_by_boss[boss] = slow_count_by_boss.get(boss, 0) + cnt
                    slow_count_by_env[env_i] += cnt
                    slow_events_epoch.append(
                        f"env{env_i}({boss.replace('GG_', '')}):{cnt}×max{per_env_max[local_i]:.1f}s"
                    )
            slow_str = " ".join(slow_events_epoch) if slow_events_epoch else "none"
            # Step-driven reset scheduling: accumulate env-steps collected this
            # epoch, fire a reset batch each time we cross `steps_per_reset`.
            # Cadence is independent of rollout wall time, so the offline pool
            # stays bounded even when rollout << reset.
            steps_since_last_reset += total_steps_epoch
            reset_indices = []
            while steps_since_last_reset >= config.steps_per_reset:
                steps_since_last_reset -= config.steps_per_reset
                for _ in range(envs_per_reset):
                    reset_indices.append(next_reset_env)
                    next_reset_env = (next_reset_env + 1) % config.n_envs
            print(
                f"  diag | active_envs {N_active}/{config.n_envs} | "
                f"active_steps {active_steps}/{total_steps_epoch} "
                f"({100*active_steps/total_steps_epoch:.1f}%) | "
                f"first_event {first_event_steps} | "
                f"step0 {step0_ms:.0f}ms | avg_step {avg_wall_ms:.1f}ms | "
                f"reset_budget {steps_since_last_reset}/{config.steps_per_reset} | "
                f"reset_envs {reset_indices}"
            )
            print(
                f"  perf | spread P50/P90/P99/max "
                f"{spread_p50_ms:.0f}/{spread_p90_ms:.0f}/{spread_p99_ms:.0f}/{spread_max_ms:.0f}ms | "
                f"sim {real_avg_ms:.1f}ms ({sim_pct:.0f}%) overhead {overhead_ms:.1f}ms"
            )
            if per_boss_step_ms:
                boss_perf_str = " ".join(
                    f"{b.replace('GG_','')}:{per_boss_step_ms[b]:.0f}ms"
                    for b in sorted(per_boss_step_ms, key=lambda b: -per_boss_step_ms[b])
                )
                print(f"  perf | per_boss_step {boss_perf_str}")
            if slow_events_epoch:
                cum_boss = " ".join(
                    f"{b.replace('GG_', '')}:{slow_count_by_boss[b]}"
                    for b in bosses if slow_count_by_boss.get(b, 0) > 0
                )
                print(
                    f"  slow | this_epoch: {slow_str} | "
                    f"cum_by_boss: {cum_boss} | cum_by_env: {slow_count_by_env}"
                )

            # Per-boss adaptive D update. Only bosses represented in the
            # currently-active envs this epoch contribute; bosses with no
            # active envs (e.g. only assigned to a currently-resetting env)
            # are left untouched.
            for boss in set(active_boss):
                env_mask = np.array([b == boss for b in active_boss])
                landed_b = float(damage_landed_arr[:, env_mask].sum())
                taken_b = float(hits_taken_arr[:, env_mask].sum())
                bs = boss_state[boss]
                bs["landed_window"].append(landed_b)
                bs["taken_window"].append(taken_b)
                window_landed = sum(bs["landed_window"])
                window_taken = sum(bs["taken_window"])

                if window_landed > 0 and window_taken > 0:
                    # Normal case: EMA toward the raw ratio, clamped.
                    D_raw = max(window_landed / window_taken, config.D_min)
                    if len(bs["landed_window"]) == 1:
                        bs["D"] = D_raw
                    else:
                        D_new = D_ema_eff * bs["D"] + (1 - D_ema_eff) * D_raw
                        bs["D"] = float(np.clip(
                            D_new,
                            bs["D"] * (1 - D_max_delta_eff),
                            bs["D"] * (1 + D_max_delta_eff),
                        ))
                elif window_landed == 0 and window_taken > 0:
                    # Strong signal: policy is taking hits but landing nothing.
                    # Curriculum is too hard — drop D aggressively (2x clamp rate).
                    bs["D"] = float(max(
                        bs["D"] * (1 - 2 * D_max_delta_eff),
                        config.D_min,
                    ))
                elif window_landed > 0 and window_taken == 0:
                    # Weaker signal: policy is landing damage without getting hit.
                    # Push D up at the normal clamp rate. No upper ceiling — D
                    # grows unbounded as the agent improves.
                    bs["D"] = float(bs["D"] * (1 + D_max_delta_eff))
                # else: both zero — no knight/boss interaction at all. Leave D
                # alone; this usually means the arena is broken, not a signal.

            D_per_env = np.array([boss_state[b]["D"] for b in active_boss], dtype=np.float32)

            # Pause game during training. Only pause active envs — resetting
            # envs are mid-scene-load and must not be paused.
            await asyncio.gather(*[
                vec_env.envs[i].pause() for i in active_envs
            ])

            t_rollout = time.perf_counter() - t_rollout_start

            torch.cuda.synchronize()
            inf_timing = agent.report_timing()

            t0 = time.perf_counter()
            metrics = agent.train_on_rollout(
                buf_obs, actions_arr, log_probs_arr,
                damage_landed_arr, hits_taken_arr,
                values_atk_arr, values_def_arr, D_per_env, buf_hx_arr,
            )
            torch.cuda.synchronize()
            t_train = time.perf_counter() - t0

            t_total = t_rollout + t_train
            pct = lambda t: 100 * t / t_total if t_total > 0 else 0
            inf = inf_timing or {}
            t_fwd = inf.get('forward_s', 0)
            t_norm = inf.get('normalize_s', 0)
            t_prep = inf.get('tensor_prep_s', 0)
            t_h2d = inf.get('h2d_s', 0)
            t_d2h = inf.get('d2h_s', 0)
            t_collect = t_norm + t_prep + t_h2d + t_fwd + t_d2h
            t_hk = t_rollout - t_collect
            print(
                f"  timing | rollout {t_rollout:.2f}s | "
                f"hk {t_hk:.2f}s | collect {t_collect:.2f}s | "
                f"train {t_train:.2f}s | total {t_total:.2f}s"
            )
            print(
                f"  collect | norm {t_norm*1000:.0f}ms | prep {t_prep*1000:.0f}ms | "
                f"h2d {t_h2d*1000:.0f}ms | fwd {t_fwd*1000:.0f}ms | d2h {t_d2h*1000:.0f}ms"
            )

            # Step-based linear LR annealing. Progress is measured in
            # env-steps collected (not epochs), so variable rollout sizes
            # and dropped-env epochs decay LR at the same rate per unit work.
            env_steps_collected += total_steps_epoch
            if config.anneal_lr:
                progress = min(1.0, env_steps_collected / config.total_env_steps)
                agent.set_lr(config.lr * (1.0 - progress))

            # Staggered reset: kick off resets for a subset of envs as
            # background tasks, resume everyone else synchronously, and drop
            # the reset envs from active_envs so the next rollout skips them.
            # Only schedule envs that are actually currently active — an env
            # still mid-reset from a prior epoch cannot be reset again.
            reset_indices = [i for i in reset_indices if i in active_set]
            new_bosses = [bosses[int(rng.integers(len(bosses)))] for _ in reset_indices]
            for env_i, b in zip(reset_indices, new_bosses):
                env_boss[env_i] = b
            resume_indices = [i for i in active_envs if i not in set(reset_indices)]
            await vec_env.start_resets(
                reset_indices, levels=new_bosses, resume_indices=resume_indices
            )
            active_envs = [i for i in active_envs if i not in set(reset_indices)]

            # Logging — per-env curriculum reward uses per-env D.
            curriculum_reward = float(
                (damage_landed_arr / D_per_env[None, :] - hits_taken_arr).mean()
            )
            total_steps = env_steps_collected
            Ds = np.array([boss_state[b]["D"] for b in bosses], dtype=np.float64)
            D_geomean = float(np.exp(np.log(np.maximum(Ds, 1e-6)).mean()))
            # Harmonic mean of D expressed in hits units. Dominated by the worst
            # boss — distinct from D_geomean (AM ≥ GM ≥ HM).
            avg_hits_per_boss = float((100.0 / np.maximum(Ds, 1e-6)).mean())

            # Balanced sample means: per-boss mean first, then average across
            # represented bosses. Weights each boss equally regardless of how
            # many envs happened to be assigned to it this epoch. Uses
            # active_boss (captured pre-reset) so the mask aligns with the
            # (T, N_active) rollout arrays.
            per_boss_landed_mean = []
            per_boss_taken_mean = []
            for boss in set(active_boss):
                env_mask = np.array([b == boss for b in active_boss])
                per_boss_landed_mean.append(float(damage_landed_arr[:, env_mask].mean()))
                per_boss_taken_mean.append(float(hits_taken_arr[:, env_mask].mean()))
            balanced_landed = float(np.mean(per_boss_landed_mean))
            balanced_taken = float(np.mean(per_boss_taken_mean))
            log = {
                "loss/surrogate": metrics["surrogate"],
                "loss/value_atk": metrics["value_atk"],
                "loss/value_def": metrics["value_def"],
                "metrics/ev_atk": metrics["ev_atk"],
                "metrics/ev_def": metrics["ev_def"],
                "metrics/pass_frac": metrics["pass_frac"],
                "loss/entropy": metrics["entropy"],
                "metrics/kl": metrics["kl"],
                "metrics/lr": agent.optimizer.param_groups[0]["lr"],
                "curriculum/D_geomean": D_geomean,
                "curriculum/avg_hits_per_boss": avg_hits_per_boss,
                "rollout/curriculum_reward": curriculum_reward,
                "rollout/damage_landed": balanced_landed,
                "rollout/hits_taken": balanced_taken,
                "diag/active_step_pct": 100 * active_steps / total_steps_epoch,
                "diag/first_event_avg": np.mean(first_event_steps),
                "diag/step0_ms": step0_ms,
                "diag/avg_step_ms": avg_wall_ms,
                "diag/max_step_ms": float(post_first.max()) * 1000 if post_first.size else 0,
                "diag/slow_steps_epoch": int(per_env_slow_count.sum()),
                "diag/gru_norm": metrics["gru_norm"],
                # Perf diagnostics — see "PERF DIAGNOSTICS" block above for meaning.
                "perf/spread_p50_ms": spread_p50_ms,
                "perf/spread_p90_ms": spread_p90_ms,
                "perf/spread_p99_ms": spread_p99_ms,
                "perf/spread_max_ms": spread_max_ms,
                "perf/sim_ms": real_avg_ms,
                "perf/overhead_ms": overhead_ms,
                "perf/sim_pct": sim_pct,
                "perf/collect_norm_ms": t_norm * 1000,
                "perf/collect_prep_ms": t_prep * 1000,
                "perf/collect_h2d_ms": t_h2d * 1000,
                "perf/collect_fwd_ms": t_fwd * 1000,
                "perf/collect_d2h_ms": t_d2h * 1000,
                "epoch": epoch,
            }
            for boss, ms in per_boss_step_ms.items():
                log[f"perf/per_boss_step_ms/{boss}"] = ms
            for boss in bosses:
                log[f"curriculum/D/{boss}"] = boss_state[boss]["D"]
                log[f"diag/slow_cum/{boss}"] = slow_count_by_boss.get(boss, 0)
            for env_i in range(config.n_envs):
                log[f"diag/slow_cum_env/{env_i}"] = slow_count_by_env[env_i]
            wandb.log(log, step=total_steps)

            boss_ds = " ".join(f"{b.split('_')[-1]}:{boss_state[b]['D']:.2f}" for b in bosses)
            print(
                f"epoch {epoch:4d} | "
                f"steps {total_steps:8d} | "
                f"D[{boss_ds}] | "
                f"D_geo {D_geomean:6.2f} | "
                f"hits/boss {avg_hits_per_boss:6.1f} | "
                f"curr_rew {curriculum_reward:7.4f} | "
                f"dmg {balanced_landed:6.3f} | "
                f"taken {balanced_taken:6.3f} | "
                f"surr {metrics['surrogate']:7.4f} | "
                f"kl {metrics['kl']:6.4f}"
            )

            recent.append({
                'curriculum_reward': curriculum_reward,
                'damage_landed': balanced_landed,
                'hits_taken': balanced_taken,
                'entropy': metrics['entropy'],
                'kl': metrics['kl'],
                'D_geomean': D_geomean,
                'avg_hits_per_boss': avg_hits_per_boss,
                'surrogate': metrics['surrogate'],
            })

            if time_budget and (time.perf_counter() - t_start) >= time_budget:
                print(f"Time budget ({time_budget}s) reached after {epoch + 1} epochs")
                break

            if env_steps_collected - last_save_step >= config.save_every_steps:
                path = f"{config.save_path}_{env_steps_collected}.pth"
                agent.save_checkpoint(
                    path, vocab=vec_env.vocab, boss_state=boss_state,
                    env_steps=env_steps_collected,
                )
                print(f"  Saved checkpoint: {path}")
                last_save_step = env_steps_collected

        # Print summary (used by autoresearch pipeline)
        if recent:
            n = len(recent)
            avg = {k: sum(m[k] for m in recent) / n for k in recent[0]}
            elapsed = time.perf_counter() - t_start
            print("\n---")
            print(f"curriculum_reward:      {avg['curriculum_reward']:.6f}")
            print(f"avg_damage_landed:      {avg['damage_landed']:.4f}")
            print(f"avg_hits_taken:         {avg['hits_taken']:.4f}")
            print(f"final_D_geomean:        {avg['D_geomean']:.2f}")
            print(f"final_avg_hits_per_boss: {avg['avg_hits_per_boss']:.1f}")
            for boss in bosses:
                print(f"final_D/{boss}: {boss_state[boss]['D']:.2f}")
            print(f"final_entropy:       {avg['entropy']:.6f}")
            print(f"final_kl:            {avg['kl']:.6f}")
            print(f"final_surrogate:     {avg['surrogate']:.6f}")
            print(f"epochs_completed:    {epoch + 1}")
            print(f"training_seconds:    {elapsed:.1f}")

        agent.save_checkpoint(
            f"{config.save_path}_final.pth", vocab=vec_env.vocab,
            boss_state=boss_state, env_steps=env_steps_collected,
        )
        wandb.finish()

        if vis is not None:
            vis.close()
    finally:
        if mgr:
            print("Cleaning up instances...")
            mgr.stop_all()


def main():
    config = Config.from_cli()
    asyncio.run(train(config))


if __name__ == "__main__":
    main()
