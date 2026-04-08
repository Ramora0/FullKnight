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
        # Only normalize continuous features (indices 0-7), not binary validity flags (8-13)
        self.obs_normalizer = RunningNormalizer(config.global_state_dim - config.n_validity_flags)
        self.combat_normalizer = RunningNormalizer(config.combat_feature_dim)
        self.terrain_normalizer = RunningNormalizer(config.terrain_feature_dim)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.lr
        )

        if config.anneal_lr:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: 1.0 - epoch / config.epochs,
            )

    def get_advantages(self, damage_landed, hits_taken, values_atk, values_def, D):
        """GAE with decomposed value heads and curriculum scaling.

        Values are trained on stationary rewards, D scales at advantage time.
        damage_landed is normalized to nail-hit equivalents (1.0 = one nail hit).
        δ_t = δ_attack_t / D - δ_defense_t
        """
        T = len(damage_landed)
        gamma = self.config.gamma
        lam = self.config.gae_lambda

        advantages = np.empty(T, dtype=np.float32)
        atk_returns = np.empty(T, dtype=np.float32)
        def_returns = np.empty(T, dtype=np.float32)
        lastgaelam = 0
        lastgaelam_atk = 0
        lastgaelam_def = 0

        for t in reversed(range(T)):
            # Stationary TD errors for each head
            delta_atk = damage_landed[t] + gamma * values_atk[t + 1] - values_atk[t]
            delta_def = hits_taken[t] + gamma * values_def[t + 1] - values_def[t]

            # Curriculum-scaled advantage
            delta = delta_atk / D - delta_def
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages[t] = lastgaelam

            # Stationary returns for value loss (no D)
            lastgaelam_atk = delta_atk + gamma * lam * lastgaelam_atk
            atk_returns[t] = lastgaelam_atk + values_atk[t]
            lastgaelam_def = delta_def + gamma * lam * lastgaelam_def
            def_returns[t] = lastgaelam_def + values_def[t]

        return advantages, atk_returns, def_returns

    def _normalize_global_state(self, global_state):
        """Normalize continuous features (0:8), pass binary validity flags (8:14) through raw."""
        n_cont = self.config.global_state_dim - self.config.n_validity_flags
        gs_norm = np.empty_like(global_state)
        gs_norm[..., :n_cont] = self.obs_normalizer.normalize(global_state[..., :n_cont])
        gs_norm[..., n_cont:] = global_state[..., n_cont:]
        return gs_norm

    def _normalize_hitboxes(self, hitboxes, mask, normalizer):
        """Normalize hitbox features, respecting the padding mask."""
        flat = hitboxes.reshape(-1, hitboxes.shape[-1])
        flat_mask = mask.reshape(-1)
        real = flat[flat_mask > 0]
        if len(real) > 0:
            normalizer.update(real)
        normed = hitboxes.copy()
        for i in range(hitboxes.shape[0]):
            n_real = int(mask[i].sum())
            if n_real > 0:
                normed[i, :n_real] = normalizer.normalize(hitboxes[i, :n_real])
        return normed

    def _ensure_event_log(self):
        if not hasattr(self, '_event_log'):
            self._event_log = []
            self._norm_total = 0.0

    def report_timing(self):
        """Call once per epoch after cuda.synchronize(). Returns timing dict."""
        if not hasattr(self, '_event_log') or not self._event_log:
            return None
        fwd_ms = sum(s.elapsed_time(e) for s, e, _ in self._event_log)
        xfer_ms = sum(s.elapsed_time(e) for _, _, (s, e) in self._event_log)
        c = len(self._event_log)
        result = {
            'normalize_s': self._norm_total,
            'forward_s': fwd_ms / 1000,
            'transfer_s': xfer_ms / 1000,
            'count': c,
        }
        self._event_log.clear()
        self._norm_total = 0.0
        return result

    @torch.no_grad()
    def collect_action(self, combat_hb, combat_mask, terrain_hb, terrain_mask, global_state):
        """Get actions for a batch of observations during rollout collection.
        All inputs are numpy arrays. Returns numpy arrays.
        """
        import time as _time
        self._ensure_event_log()

        t0 = _time.perf_counter()
        n_cont = self.config.global_state_dim - self.config.n_validity_flags
        self.obs_normalizer.update(global_state[..., :n_cont])
        gs_norm = self._normalize_global_state(global_state)
        chb_norm = self._normalize_hitboxes(combat_hb, combat_mask, self.combat_normalizer)
        thb_norm = self._normalize_hitboxes(terrain_hb, terrain_mask, self.terrain_normalizer)
        self._norm_total += _time.perf_counter() - t0

        chb = torch.from_numpy(chb_norm).float().to(self.device)
        cm = torch.from_numpy(combat_mask).float().to(self.device)
        thb = torch.from_numpy(thb_norm).float().to(self.device)
        tm = torch.from_numpy(terrain_mask).float().to(self.device)
        gs = torch.from_numpy(gs_norm).float().to(self.device)

        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        xfer_start = torch.cuda.Event(enable_timing=True)
        xfer_end = torch.cuda.Event(enable_timing=True)

        fwd_start.record()
        actions, log_prob, _, value_atk, value_def = self.policy.get_action_and_value(
            chb, cm, thb, tm, gs
        )
        fwd_end.record()

        xfer_start.record()
        actions_np = {k: v.cpu().numpy() for k, v in actions.items()}
        result = actions_np, log_prob.cpu().numpy(), value_atk.cpu().numpy(), value_def.cpu().numpy()
        xfer_end.record()

        self._event_log.append((fwd_start, fwd_end, (xfer_start, xfer_end)))
        return result

    def train_on_rollout(self, buf_combat_hb, buf_combat_mask, buf_terrain_hb,
                         buf_terrain_mask, buf_global, actions_arr,
                         log_probs_arr, damage_landed_arr, hits_taken_arr,
                         values_atk_arr, values_def_arr, D):
        """Train on a collected rollout with shuffled minibatches.

        buf_*: lists of length T, each element is (N, ...) numpy array
        actions_arr: dict of (T, N) numpy arrays
        log_probs_arr: (T, N)
        damage_landed_arr, hits_taken_arr: (T, N)
        values_atk_arr, values_def_arr: (T+1, N)
        D: current curriculum scaling factor
        """
        T, N = damage_landed_arr.shape
        cfg = self.config
        total_samples = T * N

        # Compute decomposed GAE per-env, then flatten
        all_advantages = np.empty((T, N), dtype=np.float32)
        all_atk_returns = np.empty((T, N), dtype=np.float32)
        all_def_returns = np.empty((T, N), dtype=np.float32)
        for env_i in range(N):
            adv, atk_ret, def_ret = self.get_advantages(
                damage_landed_arr[:, env_i], hits_taken_arr[:, env_i],
                values_atk_arr[:, env_i], values_def_arr[:, env_i], D,
            )
            all_advantages[:, env_i] = adv
            all_atk_returns[:, env_i] = atk_ret
            all_def_returns[:, env_i] = def_ret

        flat_adv = all_advantages.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        flat_atk_ret = all_atk_returns.reshape(-1)
        flat_def_ret = all_def_returns.reshape(-1)
        flat_lp = log_probs_arr.reshape(-1)
        flat_act = {k: actions_arr[k].reshape(-1) for k in actions_arr}

        # Flatten observations with global-max padding
        max_combat = max(buf_combat_hb[t].shape[1] for t in range(T))
        max_terrain = max(buf_terrain_hb[t].shape[1] for t in range(T))
        max_combat = max(max_combat, 1)
        max_terrain = max(max_terrain, 1)

        flat_chb = np.zeros((total_samples, max_combat, cfg.combat_feature_dim), dtype=np.float32)
        flat_cm = np.zeros((total_samples, max_combat), dtype=np.float32)
        flat_thb = np.zeros((total_samples, max_terrain, cfg.terrain_feature_dim), dtype=np.float32)
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

        # Normalize global state (continuous only) and hitboxes
        flat_gs = self._normalize_global_state(flat_gs)
        for i in range(total_samples):
            nc = int(flat_cm[i].sum())
            if nc > 0:
                flat_chb[i, :nc] = self.combat_normalizer.normalize(flat_chb[i, :nc])
            nt = int(flat_tm[i].sum())
            if nt > 0:
                flat_thb[i, :nt] = self.terrain_normalizer.normalize(flat_thb[i, :nt])

        # Move to device
        adv_t = torch.from_numpy(flat_adv).float().to(self.device)
        atk_ret_t = torch.from_numpy(flat_atk_ret).float().to(self.device)
        def_ret_t = torch.from_numpy(flat_def_ret).float().to(self.device)
        old_lp_t = torch.from_numpy(flat_lp).float().to(self.device)
        act_t = {k: torch.from_numpy(v).long().to(self.device) for k, v in flat_act.items()}
        chb_t = torch.from_numpy(flat_chb).to(self.device)
        cm_t = torch.from_numpy(flat_cm).to(self.device)
        thb_t = torch.from_numpy(flat_thb).to(self.device)
        tm_t = torch.from_numpy(flat_tm).to(self.device)
        gs_t = torch.from_numpy(flat_gs).to(self.device)

        total_metrics = {"surrogate": 0, "value_atk": 0, "value_def": 0, "entropy": 0, "kl": 0}
        n_updates = 0

        for _ in range(cfg.train_iters):
            indices = np.random.permutation(total_samples)
            early_stop = False

            for start in range(0, total_samples, cfg.batch_size):
                idx = indices[start:start + cfg.batch_size]

                _, new_lp, entropy, v_atk, v_def = self.policy.get_action_and_value(
                    chb_t[idx], cm_t[idx], thb_t[idx], tm_t[idx], gs_t[idx],
                    actions={k: v[idx] for k, v in act_t.items()},
                )

                log_ratio = new_lp - old_lp_t[idx]
                ratio = torch.exp(log_ratio)
                clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
                surrogate = -torch.min(ratio * adv_t[idx], clipped * adv_t[idx]).mean()

                # Per-element value clamp (preserves gradient)
                atk_vloss = (v_atk - atk_ret_t[idx]).pow(2)
                def_vloss = (v_def - def_ret_t[idx]).pow(2)
                value_loss = (
                    torch.clamp(atk_vloss, max=cfg.max_value_loss).mean()
                    + torch.clamp(def_vloss, max=cfg.max_value_loss).mean()
                )

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

                n_updates += 1
                total_metrics["surrogate"] += surrogate.item()
                total_metrics["value_atk"] += atk_vloss.mean().item()
                total_metrics["value_def"] += def_vloss.mean().item()
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
                "combat_normalizer": self.combat_normalizer.state_dict(),
                "terrain_normalizer": self.terrain_normalizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.config.anneal_lr and ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("obs_normalizer"):
            self.obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
        if ckpt.get("combat_normalizer"):
            self.combat_normalizer.load_state_dict(ckpt["combat_normalizer"])
        if ckpt.get("terrain_normalizer"):
            self.terrain_normalizer.load_state_dict(ckpt["terrain_normalizer"])
