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
        vis = Visualizer()
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

        agent = PPO(config)
        if config.resume:
            agent.load_checkpoint(config.resume)
            print(f"Resumed from: {config.resume}")
        print(f"Using device: {agent.device}")
        print(f"Model parameters: {sum(p.numel() for p in agent.policy.parameters()):,}")

        os.makedirs(os.path.dirname(config.save_path) or ".", exist_ok=True)
        if config.time_budget:
            os.environ["WANDB_MODE"] = "disabled"
        wandb.init(project=config.wandb_project, config=vars(config))

        D = config.D_initial
        landed_window = deque(maxlen=config.D_window)
        taken_window = deque(maxlen=config.D_window)
        time_budget = config.time_budget
        t_start = time.perf_counter()
        recent = deque(maxlen=20)

        # First epoch: full reset to load boss scenes
        combat_hb, combat_mask, terrain_hb, terrain_mask, global_state = \
            await vec_env.reset_all()
        agent.reset_hidden(config.n_envs)

        # Staggered reset: cycle through envs, resetting n_envs/4 per epoch
        envs_per_reset = max(1, config.n_envs // 4)

        for epoch in range(config.epochs):

            # Rollout buffers
            buf_combat_hb = []
            buf_combat_mask = []
            buf_terrain_hb = []
            buf_terrain_mask = []
            buf_global = []
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
                actions_np, log_probs, values_atk, values_def = agent.collect_action(
                    combat_hb, combat_mask, terrain_hb, terrain_mask, global_state
                )

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
                next_chb, next_cm, next_thb, next_tm, next_gs, damage_landed, hits_taken, step_game_times, step_real_times = \
                    await vec_env.step_all(action_vecs)
                wall_dt = time.perf_counter() - t_step

                buf_combat_hb.append(combat_hb)
                buf_combat_mask.append(combat_mask)
                buf_terrain_hb.append(terrain_hb)
                buf_terrain_mask.append(terrain_mask)
                buf_global.append(global_state)
                for k in buf_actions:
                    buf_actions[k].append(actions_np[k])
                buf_log_probs.append(log_probs)
                buf_values_atk.append(values_atk)
                buf_values_def.append(values_def)
                buf_damage_landed.append(damage_landed)
                buf_hits_taken.append(hits_taken)
                buf_step_game_times.append(step_game_times)
                buf_step_real_times.append(step_real_times)
                buf_step_wall_times.append(wall_dt)

                combat_hb, combat_mask = next_chb, next_cm
                terrain_hb, terrain_mask = next_thb, next_tm
                global_state = next_gs

                if vis is not None:
                    vis.update(combat_hb, combat_mask, terrain_hb, terrain_mask, global_state)

            # Bootstrap final values
            _, _, final_vatk, final_vdef = agent.collect_action(
                combat_hb, combat_mask, terrain_hb, terrain_mask, global_state
            )
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
            wall_time_arr = np.array(buf_step_wall_times)  # (T,)
            # First step may include intro skip — show it separately
            step0_ms = wall_time_arr[0] * 1000 if len(wall_time_arr) > 0 else 0
            avg_wall_ms = wall_time_arr[1:].mean() * 1000 if len(wall_time_arr) > 1 else 0
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

            # Compute adaptive D from rolling window of epochs
            total_landed = damage_landed_arr.sum()
            total_taken = hits_taken_arr.sum()
            landed_window.append(total_landed)
            taken_window.append(total_taken)
            window_landed = sum(landed_window)
            window_taken = sum(taken_window)
            if window_taken > 0 and window_landed > 0:
                D_raw = window_landed / window_taken
                D_raw = np.clip(D_raw, config.D_min, config.D_max)
                if epoch == 0:
                    D = D_raw
                else:
                    D_new = config.D_ema * D + (1 - config.D_ema) * D_raw
                    D = np.clip(D_new, D * (1 - config.D_max_delta), D * (1 + config.D_max_delta))

            # Pause game during training
            await vec_env.pause_all()

            t_rollout = time.perf_counter() - t_rollout_start

            torch.cuda.synchronize()
            inf_timing = agent.report_timing()

            t0 = time.perf_counter()
            metrics = agent.train_on_rollout(
                buf_combat_hb, buf_combat_mask, buf_terrain_hb, buf_terrain_mask,
                buf_global, actions_arr, log_probs_arr,
                damage_landed_arr, hits_taken_arr,
                values_atk_arr, values_def_arr, D, buf_hx_arr,
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

            # Staggered reset: reset a subset of envs, resume the rest
            _, reset_obs = await vec_env.reset_and_resume(reset_indices)
            r_chb, r_cm, r_thb, r_tm, r_gs = reset_obs

            # Merge reset obs into carried-over obs (handle padding mismatch)
            combat_hb = merge_padded(combat_hb, r_chb, reset_indices)
            combat_mask = merge_padded(combat_mask, r_cm, reset_indices)
            terrain_hb = merge_padded(terrain_hb, r_thb, reset_indices)
            terrain_mask = merge_padded(terrain_mask, r_tm, reset_indices)
            for local_i, env_i in enumerate(reset_indices):
                global_state[env_i] = r_gs[local_i]
            agent.reset_hidden_for(reset_indices)

            # Logging
            curriculum_reward = (damage_landed_arr / D - hits_taken_arr).mean()
            total_steps = (epoch + 1) * config.n_envs * config.rollout_len
            wandb.log({
                "loss/surrogate": metrics["surrogate"],
                "loss/value_atk": metrics["value_atk"],
                "loss/value_def": metrics["value_def"],
                "loss/entropy": metrics["entropy"],
                "metrics/kl": metrics["kl"],
                "metrics/lr": agent.optimizer.param_groups[0]["lr"],
                "curriculum/D": D,
                "rollout/curriculum_reward": curriculum_reward,
                "rollout/total_damage_landed": total_landed,
                "rollout/total_hits_taken": total_taken,
                "rollout/hit_ratio": total_landed / max(total_taken, 1),
                "diag/active_step_pct": 100 * active_steps / total_steps_epoch,
                "diag/first_event_avg": np.mean(first_event_steps),
                "diag/step0_ms": step0_ms,
                "diag/avg_step_ms": avg_wall_ms,
                "diag/gru_norm": metrics["gru_norm"],
                "epoch": epoch,
            }, step=total_steps)

            print(
                f"epoch {epoch:4d} | "
                f"steps {total_steps:8d} | "
                f"D {D:7.1f} | "
                f"curr_rew {curriculum_reward:7.4f} | "
                f"landed {total_landed:5.0f} | "
                f"taken {total_taken:5.0f} | "
                f"surr {metrics['surrogate']:7.4f} | "
                f"kl {metrics['kl']:6.4f}"
            )

            recent.append({
                'hit_ratio': float(total_landed / max(total_taken, 1)),
                'curriculum_reward': float(curriculum_reward),
                'damage_landed': float(total_landed),
                'damage_taken': float(total_taken),
                'entropy': metrics['entropy'],
                'kl': metrics['kl'],
                'D': float(D),
                'surrogate': metrics['surrogate'],
            })

            if time_budget and (time.perf_counter() - t_start) >= time_budget:
                print(f"Time budget ({time_budget}s) reached after {epoch + 1} epochs")
                break

            if epoch % config.save_every == 0:
                path = f"{config.save_path}_{epoch}.pth"
                agent.save_checkpoint(path)
                print(f"  Saved checkpoint: {path}")

        # Print summary (used by autoresearch pipeline)
        if recent:
            n = len(recent)
            avg = {k: sum(m[k] for m in recent) / n for k in recent[0]}
            elapsed = time.perf_counter() - t_start
            print("\n---")
            print(f"hit_ratio:          {avg['hit_ratio']:.6f}")
            print(f"curriculum_reward:   {avg['curriculum_reward']:.6f}")
            print(f"avg_damage_landed:   {avg['damage_landed']:.2f}")
            print(f"avg_damage_taken:    {avg['damage_taken']:.2f}")
            print(f"final_D:             {avg['D']:.2f}")
            print(f"final_entropy:       {avg['entropy']:.6f}")
            print(f"final_kl:            {avg['kl']:.6f}")
            print(f"final_surrogate:     {avg['surrogate']:.6f}")
            print(f"epochs_completed:    {epoch + 1}")
            print(f"training_seconds:    {elapsed:.1f}")

        agent.save_checkpoint(f"{config.save_path}_final.pth")
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
