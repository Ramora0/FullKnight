import asyncio
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import Config
from vec_env import VecEnv
from ppo import PPO
from instance_manager import InstanceManager


async def train(config: Config):
    # Optionally launch game instances
    mgr = None
    if config.hk_path and os.path.exists(config.hk_path):
        print(f"Spawning {config.n_envs} HK instances...")
        mgr = InstanceManager(config.hk_path, config.hk_data_dir)
        mgr.spawn_n(config.n_envs)
        mgr.start_all()

    # Start vectorized environment server
    vec_env = VecEnv(config)
    await vec_env.start_server()

    agent = PPO(config)
    print(f"Using device: {agent.device}")
    print(f"Model parameters: {sum(p.numel() for p in agent.policy.parameters()):,}")

    os.makedirs(os.path.dirname(config.save_path) or ".", exist_ok=True)
    writer = SummaryWriter(config.log_path)

    # Log hyperparameters
    writer.add_text("config", str(config), 0)

    for epoch in range(config.epochs):
        # Reset all envs
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
        buf_rewards = []
        buf_dones = []
        buf_values = []

        ep_rewards = np.zeros(config.n_envs)
        ep_lengths = np.zeros(config.n_envs)
        completed_ep_rewards = []
        completed_ep_lengths = []

        for t in range(config.rollout_len):
            actions_np, log_probs, values = agent.collect_action(
                combat_hb, combat_mask, terrain_hb, terrain_mask, global_state
            )

            # Convert to list-of-lists for C# side
            action_vecs = [
                [
                    int(actions_np["movement"][i]),
                    int(actions_np["direction"][i]),
                    int(actions_np["action"][i]),
                    int(actions_np["jump"][i]),
                ]
                for i in range(config.n_envs)
            ]

            # Step all envs
            next_chb, next_cm, next_thb, next_tm, next_gs, rewards, dones = \
                await vec_env.step_all(action_vecs)

            # Store in buffers
            buf_combat_hb.append(combat_hb)
            buf_combat_mask.append(combat_mask)
            buf_terrain_hb.append(terrain_hb)
            buf_terrain_mask.append(terrain_mask)
            buf_global.append(global_state)
            for k in buf_actions:
                buf_actions[k].append(actions_np[k])
            buf_log_probs.append(log_probs)
            buf_rewards.append(rewards)
            buf_dones.append(dones)
            buf_values.append(values)

            # Track episode metrics
            ep_rewards += rewards
            ep_lengths += 1
            for i in range(config.n_envs):
                if dones[i]:
                    completed_ep_rewards.append(ep_rewards[i])
                    completed_ep_lengths.append(ep_lengths[i])
                    ep_rewards[i] = 0
                    ep_lengths[i] = 0

            combat_hb, combat_mask = next_chb, next_cm
            terrain_hb, terrain_mask = next_thb, next_tm
            global_state = next_gs

        # Bootstrap final value
        _, _, final_values = agent.collect_action(
            combat_hb, combat_mask, terrain_hb, terrain_mask, global_state
        )
        buf_values.append(final_values)

        # Stack buffers: (T, N, ...)
        rewards_arr = np.stack(buf_rewards)
        dones_arr = np.stack(buf_dones)
        log_probs_arr = np.stack(buf_log_probs)
        values_arr = np.stack(buf_values)
        actions_arr = {k: np.stack(v) for k, v in buf_actions.items()}

        # Pause game during training
        await vec_env.pause_all()

        # Train
        metrics = agent.train_on_rollout(
            buf_combat_hb, buf_combat_mask, buf_terrain_hb, buf_terrain_mask,
            buf_global, actions_arr, log_probs_arr, rewards_arr, dones_arr, values_arr,
        )

        if config.anneal_lr:
            agent.scheduler.step()

        # Resume game
        await vec_env.resume_all()

        # Logging
        # Include partial episodes in averages
        all_ep_rewards = completed_ep_rewards + list(ep_rewards)
        all_ep_lengths = completed_ep_lengths + list(ep_lengths)

        total_steps = (epoch + 1) * config.n_envs * config.rollout_len
        writer.add_scalar("loss/surrogate", metrics["surrogate"], total_steps)
        writer.add_scalar("loss/value", metrics["value"], total_steps)
        writer.add_scalar("loss/entropy", metrics["entropy"], total_steps)
        writer.add_scalar("metrics/kl", metrics["kl"], total_steps)
        writer.add_scalar("metrics/lr", agent.optimizer.param_groups[0]["lr"], total_steps)
        writer.add_scalar("rollout/avg_ep_reward", np.mean(all_ep_rewards) if all_ep_rewards else 0, total_steps)
        writer.add_scalar("rollout/avg_ep_length", np.mean(all_ep_lengths) if all_ep_lengths else 0, total_steps)
        writer.add_scalar("rollout/completed_episodes", len(completed_ep_rewards), total_steps)

        avg_rew = np.mean(all_ep_rewards) if all_ep_rewards else 0
        print(
            f"epoch {epoch:4d} | "
            f"steps {total_steps:8d} | "
            f"avg_rew {avg_rew:7.3f} | "
            f"surr {metrics['surrogate']:7.4f} | "
            f"val {metrics['value']:7.4f} | "
            f"ent {metrics['entropy']:7.4f} | "
            f"kl {metrics['kl']:6.4f}"
        )

        if epoch % config.save_every == 0:
            path = f"{config.save_path}_{epoch}.pth"
            agent.save_checkpoint(path)
            print(f"  Saved checkpoint: {path}")

    # Final save
    agent.save_checkpoint(f"{config.save_path}_final.pth")
    writer.close()

    if mgr:
        mgr.stop_all()


def main():
    config = Config()
    asyncio.run(train(config))


if __name__ == "__main__":
    main()
