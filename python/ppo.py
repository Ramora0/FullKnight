import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import FullKnightActorCritic


class PPO:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.policy = FullKnightActorCritic(config).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.lr
        )

        if config.anneal_lr:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: 1.0 - epoch / config.epochs,
            )

    def get_advantages(self, rewards, dones, values):
        """GAE computation."""
        advantages = np.empty_like(rewards, dtype=np.float32)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = (
                rewards[t]
                + self.config.gamma * values[t + 1] * nonterminal
                - values[t]
            )
            lastgaelam = (
                delta
                + self.config.gamma * self.config.gae_lambda * nonterminal * lastgaelam
            )
            advantages[t] = lastgaelam
        return advantages

    @torch.no_grad()
    def collect_action(self, combat_hb, combat_mask, terrain_hb, terrain_mask, global_state):
        """Get actions for a batch of observations during rollout collection.
        All inputs are numpy arrays. Returns numpy arrays.
        """
        chb = torch.from_numpy(combat_hb).float().to(self.device)
        cm = torch.from_numpy(combat_mask).float().to(self.device)
        thb = torch.from_numpy(terrain_hb).float().to(self.device)
        tm = torch.from_numpy(terrain_mask).float().to(self.device)
        gs = torch.from_numpy(global_state).float().to(self.device)

        actions, log_prob, _, value = self.policy.get_action_and_value(
            chb, cm, thb, tm, gs
        )

        actions_np = {k: v.cpu().numpy() for k, v in actions.items()}
        return actions_np, log_prob.cpu().numpy(), value.cpu().numpy()

    def train_on_rollout(self, buf_combat_hb, buf_combat_mask, buf_terrain_hb,
                         buf_terrain_mask, buf_global, actions_arr,
                         log_probs_arr, rewards_arr, dones_arr, values_arr):
        """Train on a collected rollout. Processes per-env then averages.

        buf_*: lists of length T, each element is (N, ...) numpy array
        actions_arr: dict of (T, N) numpy arrays
        log_probs_arr: (T, N)
        rewards_arr: (T, N)
        dones_arr: (T, N)
        values_arr: (T+1, N)
        """
        T, N = rewards_arr.shape
        cfg = self.config

        total_metrics = {"surrogate": 0, "value": 0, "entropy": 0, "kl": 0}

        for env_i in range(N):
            # Per-env slices
            env_rewards = rewards_arr[:, env_i]
            env_dones = dones_arr[:, env_i]
            env_values = values_arr[:, env_i]  # length T+1

            advantages = self.get_advantages(env_rewards, env_dones, env_values)
            returns = advantages + env_values[:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            env_log_probs = log_probs_arr[:, env_i]
            env_actions = {k: actions_arr[k][:, env_i] for k in actions_arr}

            # Convert to tensors
            adv_t = torch.from_numpy(advantages).float().to(self.device)
            ret_t = torch.from_numpy(returns).float().to(self.device)
            old_lp_t = torch.from_numpy(env_log_probs).float().to(self.device)
            act_t = {k: torch.from_numpy(v).long().to(self.device) for k, v in env_actions.items()}

            # Re-pad hitboxes for this env across all timesteps
            combat_hbs = [buf_combat_hb[t][env_i] for t in range(T)]
            combat_masks = [buf_combat_mask[t][env_i] for t in range(T)]
            terrain_hbs = [buf_terrain_hb[t][env_i] for t in range(T)]
            terrain_masks = [buf_terrain_mask[t][env_i] for t in range(T)]
            gs_list = [buf_global[t][env_i] for t in range(T)]

            # Find max hitbox counts for this trajectory
            max_combat = max(max(int(m.sum()) for m in combat_masks), 1)
            max_terrain = max(max(int(m.sum()) for m in terrain_masks), 1)

            chb_t = torch.zeros(T, max_combat, cfg.hitbox_feature_dim, device=self.device)
            cm_t = torch.zeros(T, max_combat, device=self.device)
            thb_t = torch.zeros(T, max_terrain, cfg.hitbox_feature_dim, device=self.device)
            tm_t = torch.zeros(T, max_terrain, device=self.device)
            gs_t = torch.from_numpy(np.stack(gs_list)).float().to(self.device)

            for t_idx in range(T):
                nc = min(int(combat_masks[t_idx].sum()), max_combat)
                if nc > 0:
                    chb_t[t_idx, :nc] = torch.from_numpy(combat_hbs[t_idx][:nc]).float()
                    cm_t[t_idx, :nc] = torch.from_numpy(combat_masks[t_idx][:nc]).float()

                nt = min(int(terrain_masks[t_idx].sum()), max_terrain)
                if nt > 0:
                    thb_t[t_idx, :nt] = torch.from_numpy(terrain_hbs[t_idx][:nt]).float()
                    tm_t[t_idx, :nt] = torch.from_numpy(terrain_masks[t_idx][:nt]).float()

            kl = 0.0
            for _ in range(cfg.train_iters):
                _, new_lp, entropy, values = self.policy.get_action_and_value(
                    chb_t, cm_t, thb_t, tm_t, gs_t, actions=act_t
                )

                log_ratio = new_lp - old_lp_t
                ratio = torch.exp(log_ratio)
                clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
                surrogate = -torch.min(ratio * adv_t, clipped * adv_t).mean()
                value_loss = F.mse_loss(values, ret_t)
                entropy_loss = -entropy.mean()

                loss = (
                    surrogate
                    + cfg.value_coeff * value_loss
                    + cfg.entropy_coeff * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean().item()
                    if cfg.target_kl and kl > cfg.target_kl:
                        break

            total_metrics["surrogate"] += surrogate.item()
            total_metrics["value"] += value_loss.item()
            total_metrics["entropy"] += entropy_loss.item()
            total_metrics["kl"] += kl

        return {k: v / N for k, v in total_metrics.items()}

    def save_checkpoint(self, path):
        torch.save(
            {
                "model": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.config.anneal_lr else None,
            },
            path,
        )

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.config.anneal_lr and ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
