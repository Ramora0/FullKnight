import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import FullKnightActorCritic


class RunningNormalizer:
    """Welford online normalizer for observation vectors."""

    def __init__(self, shape, clip=5.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, batch):
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, x):
        return np.clip(
            (x - self.mean.astype(np.float32)) / np.sqrt(self.var.astype(np.float32) + 1e-8),
            -self.clip, self.clip,
        ).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state):
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


class PPO:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.policy = FullKnightActorCritic(config).to(self.device)
        self.obs_normalizer = RunningNormalizer(config.global_state_dim)

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
        self.obs_normalizer.update(global_state)
        gs_norm = self.obs_normalizer.normalize(global_state)

        chb = torch.from_numpy(combat_hb).float().to(self.device)
        cm = torch.from_numpy(combat_mask).float().to(self.device)
        thb = torch.from_numpy(terrain_hb).float().to(self.device)
        tm = torch.from_numpy(terrain_mask).float().to(self.device)
        gs = torch.from_numpy(gs_norm).float().to(self.device)

        actions, log_prob, _, value = self.policy.get_action_and_value(
            chb, cm, thb, tm, gs
        )

        actions_np = {k: v.cpu().numpy() for k, v in actions.items()}
        return actions_np, log_prob.cpu().numpy(), value.cpu().numpy()

    def train_on_rollout(self, buf_combat_hb, buf_combat_mask, buf_terrain_hb,
                         buf_terrain_mask, buf_global, actions_arr,
                         log_probs_arr, rewards_arr, dones_arr, values_arr):
        """Train on a collected rollout with shuffled minibatches.

        buf_*: lists of length T, each element is (N, ...) numpy array
        actions_arr: dict of (T, N) numpy arrays
        log_probs_arr: (T, N)
        rewards_arr: (T, N)
        dones_arr: (T, N)
        values_arr: (T+1, N)
        """
        T, N = rewards_arr.shape
        cfg = self.config
        total_samples = T * N

        # Compute GAE per-env, then flatten
        all_advantages = np.empty_like(rewards_arr)
        for env_i in range(N):
            all_advantages[:, env_i] = self.get_advantages(
                rewards_arr[:, env_i], dones_arr[:, env_i], values_arr[:, env_i]
            )
        all_returns = all_advantages + values_arr[:-1]

        flat_adv = all_advantages.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        flat_ret = all_returns.reshape(-1)
        flat_lp = log_probs_arr.reshape(-1)
        flat_act = {k: actions_arr[k].reshape(-1) for k in actions_arr}

        # Flatten observations with global-max padding
        feat_dim = cfg.hitbox_feature_dim
        max_combat = max(buf_combat_hb[t].shape[1] for t in range(T))
        max_terrain = max(buf_terrain_hb[t].shape[1] for t in range(T))
        max_combat = max(max_combat, 1)
        max_terrain = max(max_terrain, 1)

        flat_chb = np.zeros((total_samples, max_combat, feat_dim), dtype=np.float32)
        flat_cm = np.zeros((total_samples, max_combat), dtype=np.float32)
        flat_thb = np.zeros((total_samples, max_terrain, feat_dim), dtype=np.float32)
        flat_tm = np.zeros((total_samples, max_terrain), dtype=np.float32)
        flat_gs = np.zeros((total_samples, cfg.global_state_dim), dtype=np.float32)

        for t in range(T):
            s, e = t * N, (t + 1) * N
            nc = buf_combat_hb[t].shape[1]
            flat_chb[s:e, :nc] = buf_combat_hb[t]
            flat_cm[s:e, :nc] = buf_combat_mask[t]
            nt = buf_terrain_hb[t].shape[1]
            flat_thb[s:e, :nt] = buf_terrain_hb[t]
            flat_tm[s:e, :nt] = buf_terrain_mask[t]
            flat_gs[s:e] = buf_global[t]

        # Normalize global state
        flat_gs = self.obs_normalizer.normalize(flat_gs)

        # Move to device
        adv_t = torch.from_numpy(flat_adv).float().to(self.device)
        ret_t = torch.from_numpy(flat_ret).float().to(self.device)
        old_lp_t = torch.from_numpy(flat_lp).float().to(self.device)
        act_t = {k: torch.from_numpy(v).long().to(self.device) for k, v in flat_act.items()}
        chb_t = torch.from_numpy(flat_chb).to(self.device)
        cm_t = torch.from_numpy(flat_cm).to(self.device)
        thb_t = torch.from_numpy(flat_thb).to(self.device)
        tm_t = torch.from_numpy(flat_tm).to(self.device)
        gs_t = torch.from_numpy(flat_gs).to(self.device)

        total_metrics = {"surrogate": 0, "value": 0, "entropy": 0, "kl": 0}
        n_updates = 0

        for _ in range(cfg.train_iters):
            indices = np.random.permutation(total_samples)
            early_stop = False

            for start in range(0, total_samples, cfg.batch_size):
                idx = indices[start:start + cfg.batch_size]

                _, new_lp, entropy, values = self.policy.get_action_and_value(
                    chb_t[idx], cm_t[idx], thb_t[idx], tm_t[idx], gs_t[idx],
                    actions={k: v[idx] for k, v in act_t.items()},
                )

                log_ratio = new_lp - old_lp_t[idx]
                ratio = torch.exp(log_ratio)
                clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
                surrogate = -torch.min(ratio * adv_t[idx], clipped * adv_t[idx]).mean()
                value_loss = F.mse_loss(values, ret_t[idx])
                clipped_vloss = torch.clamp(value_loss, max=cfg.max_value_loss)
                entropy_loss = -entropy.mean()

                loss = (
                    surrogate
                    + cfg.value_coeff * clipped_vloss
                    + cfg.entropy_coeff * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                n_updates += 1
                total_metrics["surrogate"] += surrogate.item()
                total_metrics["value"] += value_loss.item()
                total_metrics["entropy"] += entropy_loss.item()

                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean().item()
                    total_metrics["kl"] += kl
                    if cfg.target_kl and kl > cfg.target_kl:
                        early_stop = True
                        break

            if early_stop:
                break

        return {k: v / max(n_updates, 1) for k, v in total_metrics.items()}

    def save_checkpoint(self, path):
        torch.save(
            {
                "model": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.config.anneal_lr else None,
                "obs_normalizer": self.obs_normalizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.config.anneal_lr and ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("obs_normalizer"):
            self.obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
