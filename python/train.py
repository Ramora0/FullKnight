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
        mgr.start_all()
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
        time_budget = config.time_budget
        t_start = time.perf_counter()
        recent = deque(maxlen=20)

        for epoch in range(config.epochs):
            # Reset scene each epoch so the knight doesn't get stuck
            combat_hb, combat_mask, terrain_hb, terrain_mask, global_state = \
                await vec_env.reset_all()
            agent.reset_hidden(config.n_envs)

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

                next_chb, next_cm, next_thb, next_tm, next_gs, damage_landed, hits_taken = \
                    await vec_env.step_all(action_vecs)

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

            # Compute adaptive D from this rollout
            total_landed = damage_landed_arr.sum()
            total_taken = hits_taken_arr.sum()
            if total_taken > 0 and total_landed > 0:
                D_raw = total_landed / total_taken
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

            await vec_env.resume_all()

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
