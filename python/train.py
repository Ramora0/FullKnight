import argparse
import asyncio
import os
import numpy as np
import torch
import wandb

from config import Config
from vec_env import VecEnv
from ppo import PPO
from instance_manager import InstanceManager


async def train(config: Config):
    vis = None
    if config.visualize:
        from visualizer import Visualizer
        vis = Visualizer()
    # Optionally launch game instances
    mgr = None
    if config.n_envs > 1 and config.hk_path and os.path.exists(config.hk_path):
        print(f"Spawning {config.n_envs} HK instances...")
        mgr = InstanceManager(config.hk_path, config.hk_data_dir)
        mgr.spawn_n(config.n_envs)
        mgr.start_all()
    else:
        print("Single instance mode — launch Hollow Knight manually.")

    # Start vectorized environment server
    vec_env = VecEnv(config)
    await vec_env.start_server()

    agent = PPO(config)
    print(f"Using device: {agent.device}")
    print(f"Model parameters: {sum(p.numel() for p in agent.policy.parameters()):,}")

    os.makedirs(os.path.dirname(config.save_path) or ".", exist_ok=True)
    wandb.init(project=config.wandb_project, config=vars(config))

    D = config.D_initial

    for epoch in range(config.epochs):
        # Reset scene each epoch so the knight doesn't get stuck
        combat_hb, combat_mask, terrain_hb, terrain_mask, global_state = \
            await vec_env.reset_all()

        # Rollout buffers
        buf_combat_hb = []
        buf_combat_mask = []
        buf_terrain_hb = []
        buf_terrain_mask = []
        buf_global = []
        buf_actions = {k: [] for k in ["movement", "direction", "action", "jump"]}
        buf_log_probs = []
        buf_values = []
        buf_damage_dealt = []
        buf_damage_taken = []

        for t in range(config.rollout_len):
            actions_np, log_probs, values = agent.collect_action(
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

            next_chb, next_cm, next_thb, next_tm, next_gs, dmg_dealt, dmg_taken = \
                await vec_env.step_all(action_vecs)

            buf_combat_hb.append(combat_hb)
            buf_combat_mask.append(combat_mask)
            buf_terrain_hb.append(terrain_hb)
            buf_terrain_mask.append(terrain_mask)
            buf_global.append(global_state)
            for k in buf_actions:
                buf_actions[k].append(actions_np[k])
            buf_log_probs.append(log_probs)
            buf_values.append(values)
            buf_damage_dealt.append(dmg_dealt)
            buf_damage_taken.append(dmg_taken)

            combat_hb, combat_mask = next_chb, next_cm
            terrain_hb, terrain_mask = next_thb, next_tm
            global_state = next_gs

            if vis is not None:
                vis.update(combat_hb, combat_mask, terrain_hb, terrain_mask, global_state)

        # Bootstrap final value
        _, _, final_values = agent.collect_action(
            combat_hb, combat_mask, terrain_hb, terrain_mask, global_state
        )
        buf_values.append(final_values)

        # Stack buffers: (T, N)
        damage_dealt_arr = np.stack(buf_damage_dealt)
        damage_taken_arr = np.stack(buf_damage_taken)
        log_probs_arr = np.stack(buf_log_probs)
        values_arr = np.stack(buf_values)
        actions_arr = {k: np.stack(v) for k, v in buf_actions.items()}

        # Compute adaptive D from this rollout (EMA-smoothed)
        total_dealt = damage_dealt_arr.sum()
        total_taken = damage_taken_arr.sum()
        if total_taken > 0 and total_dealt > 0:
            D_raw = total_dealt * config.knight_max_hp / total_taken
            D_raw = np.clip(D_raw, config.D_min, config.D_max)
            D = config.D_ema * D + (1 - config.D_ema) * D_raw

        # Compute rewards retroactively using D
        rewards_arr = damage_dealt_arr / D - damage_taken_arr / config.knight_max_hp

        # Pause game during training
        await vec_env.pause_all()

        metrics = agent.train_on_rollout(
            buf_combat_hb, buf_combat_mask, buf_terrain_hb, buf_terrain_mask,
            buf_global, actions_arr, log_probs_arr, rewards_arr, values_arr,
        )

        if config.anneal_lr:
            agent.scheduler.step()

        await vec_env.resume_all()

        # Logging
        avg_reward = rewards_arr.mean()
        total_steps = (epoch + 1) * config.n_envs * config.rollout_len
        wandb.log({
            "loss/surrogate": metrics["surrogate"],
            "loss/value": metrics["value"],
            "loss/entropy": metrics["entropy"],
            "metrics/kl": metrics["kl"],
            "metrics/lr": agent.optimizer.param_groups[0]["lr"],
            "curriculum/D": D,
            "rollout/avg_reward": avg_reward,
            "rollout/total_damage_dealt": total_dealt,
            "rollout/total_damage_taken": total_taken,
            "rollout/hit_ratio": total_dealt / max(total_taken, 1),
        }, step=total_steps)

        print(
            f"epoch {epoch:4d} | "
            f"steps {total_steps:8d} | "
            f"D {D:7.1f} | "
            f"avg_rew {avg_reward:7.4f} | "
            f"dealt {total_dealt:5.0f} | "
            f"taken {total_taken:5.0f} | "
            f"surr {metrics['surrogate']:7.4f} | "
            f"kl {metrics['kl']:6.4f}"
        )

        if epoch % config.save_every == 0:
            path = f"{config.save_path}_{epoch}.pth"
            agent.save_checkpoint(path)
            print(f"  Saved checkpoint: {path}")

    agent.save_checkpoint(f"{config.save_path}_final.pth")
    wandb.finish()

    if vis is not None:
        vis.close()

    if mgr:
        mgr.stop_all()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="Live-render model observations")
    args = parser.parse_args()

    config = Config(visualize=args.visualize)
    asyncio.run(train(config))


if __name__ == "__main__":
    main()
