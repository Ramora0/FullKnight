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
    """Overwrite rows in old with rows from new, expanding padding dims if needed."""
    if old.shape[1:] != new.shape[1:]:
        pad = [(0, 0)] + [(0, max(0, n - o)) for o, n in zip(old.shape[1:], new.shape[1:])]
        old = np.pad(old, pad, constant_values=fill)
    for local_i, env_i in enumerate(indices):
        old[env_i] = fill
        idx = tuple([env_i] + [slice(0, s) for s in new.shape[1:]])
        old[idx] = new[local_i]
    return old


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
        boss_state = {b: {
            "D": config.D_initial,
            "landed_window": deque(maxlen=config.D_window),
            "taken_window":  deque(maxlen=config.D_window),
        } for b in bosses}
        rng = np.random.default_rng(config.seed or None)
        env_boss = [bosses[int(rng.integers(len(bosses)))] for _ in range(config.n_envs)]
        print(f"Boss pool: {bosses}")
        print(f"Initial env_boss: {env_boss}")

        agent = PPO(config)
        start_epoch = 0
        if config.resume:
            start_epoch = agent.load_checkpoint(
                config.resume, vocab=vec_env.vocab, boss_state=boss_state
            )
            print(f"Resumed from: {config.resume}")
        print(f"Using device: {agent.device}")
        print(f"Model parameters: {sum(p.numel() for p in agent.policy.parameters()):,}")

        os.makedirs(os.path.dirname(config.save_path) or ".", exist_ok=True)
        if config.time_budget:
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
        obs = await vec_env.reset_all(levels=env_boss)
        agent.reset_hidden(config.n_envs)

        # Staggered reset: cycle through envs, resetting n_envs/4 per epoch
        envs_per_reset = max(1, config.n_envs // 4)

        # Initialized so the post-loop final save and summary have a defined
        # `epoch` even if the loop never runs (e.g. resuming from a completed run).
        epoch = start_epoch - 1
        for epoch in range(start_epoch, config.epochs):

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

            for t in range(config.rollout_len):
                buf_hx.append(agent.get_hx_snapshot())
                actions_np, log_probs, values_atk, values_def = agent.collect_action(obs)

                action_vecs = [
                    [
                        int(actions_np["movement"][i]),
                        int(actions_np["direction"][i]),
                        int(actions_np["action"][i]),
                        int(actions_np["jump"][i]),
                    ]
                    for i in range(config.n_envs)
                ]

                t_step = time.perf_counter()
                (next_obs, damage_landed, hits_taken, step_game_times, step_real_times,
                 step_wall_per_env) = await vec_env.step_all(action_vecs)
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
            _, _, final_vatk, final_vdef = agent.collect_action(obs)
            buf_values_atk.append(final_vatk)
            buf_values_def.append(final_vdef)

            # Stack buffers: (T, N)
            damage_landed_arr = np.stack(buf_damage_landed)
            hits_taken_arr = np.stack(buf_hits_taken)
            log_probs_arr = np.stack(buf_log_probs)
            values_atk_arr = np.stack(buf_values_atk)
            values_def_arr = np.stack(buf_values_def)
            actions_arr = {k: np.stack(v) for k, v in buf_actions.items()}
            buf_hx_arr = np.stack(buf_hx)  # (T, N, hidden_dim)

            # Diagnostic: first combat event per env, step timing
            any_event = (damage_landed_arr > 0) | (hits_taken_arr > 0)  # (T, N)
            active_steps = int(any_event.sum())
            total_steps_epoch = damage_landed_arr.shape[0] * damage_landed_arr.shape[1]
            first_event_steps = []
            for env_i in range(damage_landed_arr.shape[1]):
                col = any_event[:, env_i]
                idxs = np.where(col)[0]
                first_event_steps.append(int(idxs[0]) if len(idxs) > 0 else damage_landed_arr.shape[0])
            wall_time_arr = np.stack(buf_step_wall_times)  # (T, N)
            # First step may include intro skip — show it separately
            step0_ms = wall_time_arr[0].mean() * 1000 if wall_time_arr.shape[0] > 0 else 0
            avg_wall_ms = wall_time_arr[1:].mean() * 1000 if wall_time_arr.shape[0] > 1 else 0
            # Per-env slow-step inventory (skip step 0 which includes intro-skip).
            # For each env, report the max step time this epoch and every step
            # over `slow_step_threshold_s` gets counted against the boss.
            post_first = wall_time_arr[1:] if wall_time_arr.shape[0] > 1 else wall_time_arr
            per_env_max = post_first.max(axis=0) if post_first.shape[0] > 0 else np.zeros(config.n_envs)
            per_env_slow_count = (post_first > slow_step_threshold_s).sum(axis=0)
            slow_events_epoch = []
            for env_i in range(config.n_envs):
                cnt = int(per_env_slow_count[env_i])
                if cnt > 0:
                    boss = env_boss[env_i]
                    slow_count_by_boss[boss] = slow_count_by_boss.get(boss, 0) + cnt
                    slow_count_by_env[env_i] += cnt
                    slow_events_epoch.append(
                        f"env{env_i}({boss.replace('GG_', '')}):{cnt}×max{per_env_max[env_i]:.1f}s"
                    )
            slow_str = " ".join(slow_events_epoch) if slow_events_epoch else "none"
            # Which envs will be reset after this epoch's training
            reset_offset = (epoch * envs_per_reset) % config.n_envs
            reset_indices = list(range(reset_offset, reset_offset + envs_per_reset))
            print(
                f"  diag | active {active_steps}/{total_steps_epoch} "
                f"({100*active_steps/total_steps_epoch:.1f}%) | "
                f"first_event {first_event_steps} | "
                f"step0 {step0_ms:.0f}ms | avg_step {avg_wall_ms:.1f}ms | "
                f"reset_envs {reset_indices}"
            )
            if slow_events_epoch:
                cum_boss = " ".join(
                    f"{b.replace('GG_', '')}:{slow_count_by_boss[b]}"
                    for b in bosses if slow_count_by_boss.get(b, 0) > 0
                )
                print(
                    f"  slow | this_epoch: {slow_str} | "
                    f"cum_by_boss: {cum_boss} | cum_by_env: {slow_count_by_env}"
                )

            # Per-boss adaptive D update. Only bosses represented in env_boss
            # this epoch contribute; bosses with no envs are left untouched.
            for boss in set(env_boss):
                env_mask = np.array([b == boss for b in env_boss])
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
                        D_new = config.D_ema * bs["D"] + (1 - config.D_ema) * D_raw
                        bs["D"] = float(np.clip(
                            D_new,
                            bs["D"] * (1 - config.D_max_delta),
                            bs["D"] * (1 + config.D_max_delta),
                        ))
                elif window_landed == 0 and window_taken > 0:
                    # Strong signal: policy is taking hits but landing nothing.
                    # Curriculum is too hard — drop D aggressively (2x clamp rate).
                    bs["D"] = float(max(
                        bs["D"] * (1 - 2 * config.D_max_delta),
                        config.D_min,
                    ))
                elif window_landed > 0 and window_taken == 0:
                    # Weaker signal: policy is landing damage without getting hit.
                    # Push D up at the normal clamp rate.
                    bs["D"] = float(min(
                        bs["D"] * (1 + config.D_max_delta),
                        config.D_max,
                    ))
                # else: both zero — no knight/boss interaction at all. Leave D
                # alone; this usually means the arena is broken, not a signal.

            D_per_env = np.array([boss_state[b]["D"] for b in env_boss], dtype=np.float32)

            # Pause game during training
            await vec_env.pause_all()

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
            t_xfer = inf.get('transfer_s', 0)
            t_hk = t_rollout - t_fwd - t_norm - t_xfer
            print(
                f"  timing | rollout {t_rollout:.2f}s | "
                f"fwd {t_fwd:.2f}s | norm {t_norm:.2f}s | xfer {t_xfer:.2f}s | "
                f"hk {t_hk:.2f}s | train {t_train:.2f}s | total {t_total:.2f}s"
            )

            if config.anneal_lr:
                agent.scheduler.step()

            # Staggered reset: reset a subset of envs (randomly reassigning
            # their boss from the pool) and resume the rest.
            new_bosses = [bosses[int(rng.integers(len(bosses)))] for _ in reset_indices]
            for env_i, b in zip(reset_indices, new_bosses):
                env_boss[env_i] = b
            _, reset_obs = await vec_env.reset_and_resume(reset_indices, levels=new_bosses)

            # Merge reset obs into carried-over obs (handle padding mismatch).
            # Each field gets the same merge_padded treatment driven by reset_indices.
            obs = obs.replace(
                combat_hb=merge_padded(obs.combat_hb, reset_obs.combat_hb, reset_indices),
                combat_mask=merge_padded(obs.combat_mask, reset_obs.combat_mask, reset_indices),
                combat_kind_ids=merge_padded(obs.combat_kind_ids, reset_obs.combat_kind_ids, reset_indices),
                combat_parent_ids=merge_padded(obs.combat_parent_ids, reset_obs.combat_parent_ids, reset_indices),
                terrain_hb=merge_padded(obs.terrain_hb, reset_obs.terrain_hb, reset_indices),
                terrain_mask=merge_padded(obs.terrain_mask, reset_obs.terrain_mask, reset_indices),
            )
            for local_i, env_i in enumerate(reset_indices):
                obs.global_state[env_i] = reset_obs.global_state[local_i]
            agent.reset_hidden_for(reset_indices)

            # Logging — per-env curriculum reward uses per-env D.
            curriculum_reward = float(
                (damage_landed_arr / D_per_env[None, :] - hits_taken_arr).mean()
            )
            total_steps = (epoch + 1) * config.n_envs * config.rollout_len
            Ds = np.array([boss_state[b]["D"] for b in bosses], dtype=np.float64)
            D_geomean = float(np.exp(np.log(np.maximum(Ds, 1e-6)).mean()))
            # Harmonic mean of D expressed in hits units. Dominated by the worst
            # boss — distinct from D_geomean (AM ≥ GM ≥ HM).
            avg_hits_per_boss = float((100.0 / np.maximum(Ds, 1e-6)).mean())

            # Balanced sample means: per-boss mean first, then average across
            # represented bosses. Weights each boss equally regardless of how
            # many envs happened to be assigned to it this epoch.
            per_boss_landed_mean = []
            per_boss_taken_mean = []
            for boss in set(env_boss):
                env_mask = np.array([b == boss for b in env_boss])
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
                "epoch": epoch,
            }
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

            if epoch % config.save_every == 0:
                path = f"{config.save_path}_{epoch}.pth"
                agent.save_checkpoint(path, vocab=vec_env.vocab, boss_state=boss_state, epoch=epoch)
                print(f"  Saved checkpoint: {path}")

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

        agent.save_checkpoint(f"{config.save_path}_final.pth", vocab=vec_env.vocab, boss_state=boss_state, epoch=epoch)
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
