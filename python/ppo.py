import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

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
        self.hx = None  # GRU hidden state, shape (N, hidden_dim) during rollout

        # Only normalize continuous features (indices 0-7), not binary validity flags (8-13)
        self.obs_normalizer = RunningNormalizer(config.global_state_dim - config.n_binary_flags)
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
        damage_landed is % of boss max HP dealt (1.0 = 1% of boss HP).
        D is % boss HP we deal per hit taken against us.
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
        """Normalize continuous features (0:7), pass binary flags (7:19) through raw."""
        n_cont = self.config.global_state_dim - self.config.n_binary_flags
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

    def reset_hidden(self, n_envs):
        """Zero the GRU hidden state. Call at epoch start."""
        self.hx = np.zeros((n_envs, self.config.gru_dim), dtype=np.float32)

    def reset_hidden_for(self, indices):
        """Zero the GRU hidden state for specific envs only."""
        for i in indices:
            self.hx[i] = 0.0

    def get_hx_snapshot(self):
        """Return a copy of the current hidden state for buffering."""
        return self.hx.copy()

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
        n_cont = self.config.global_state_dim - self.config.n_binary_flags
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

        hx_t = torch.from_numpy(self.hx).float().to(self.device)

        fwd_start.record()
        actions, log_prob, _, value_atk, value_def, hx_new = self.policy.get_action_and_value(
            chb, cm, thb, tm, gs, hx=hx_t
        )
        fwd_end.record()

        self.hx = hx_new.cpu().numpy()

        xfer_start.record()
        actions_np = {k: v.cpu().numpy() for k, v in actions.items()}
        result = actions_np, log_prob.cpu().numpy(), value_atk.cpu().numpy(), value_def.cpu().numpy()
        xfer_end.record()

        self._event_log.append((fwd_start, fwd_end, (xfer_start, xfer_end)))
        return result

    def train_on_rollout(self, buf_combat_hb, buf_combat_mask, buf_terrain_hb,
                         buf_terrain_mask, buf_global, actions_arr,
                         log_probs_arr, damage_landed_arr, hits_taken_arr,
                         values_atk_arr, values_def_arr, D, buf_hx):
        """Train on a collected rollout with chunked truncated BPTT.

        buf_*: lists of length T, each element is (N, ...) numpy array
        actions_arr: dict of (T, N) numpy arrays
        log_probs_arr: (T, N)
        damage_landed_arr, hits_taken_arr: (T, N)
        values_atk_arr, values_def_arr: (T+1, N)
        D: current curriculum scaling factor
        buf_hx: (T, N, hidden_dim) GRU hidden states at each timestep
        """
        T, N = damage_landed_arr.shape
        cfg = self.config
        L = cfg.seq_len
        n_chunks_per_env = T // L
        T_used = n_chunks_per_env * L
        total_chunks = n_chunks_per_env * N
        max_combat_dim = max(buf_combat_hb[t].shape[1] for t in range(T))
        max_terrain_dim = max(buf_terrain_hb[t].shape[1] for t in range(T))
        rollout_samples = T_used * N
        total_passes = rollout_samples * cfg.train_iters
        print(
            f"  train | T={T} N={N} L={L} chunks={total_chunks} "
            f"chunks/batch={cfg.chunks_per_batch} iters={cfg.train_iters} "
            f"| max combat hb={max_combat_dim} terrain hb={max_terrain_dim} "
            f"| samples={rollout_samples:,} × iters={cfg.train_iters} = {total_passes:,} passes",
            flush=True,
        )

        # Compute decomposed GAE per-env
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

        # --- Chunk (T, N) arrays into (total_chunks, L) ---
        def chunk_tn(arr):
            """(T, N) -> (total_chunks, L): group by chunk then env."""
            return arr[:T_used].reshape(n_chunks_per_env, L, N).transpose(0, 2, 1).reshape(-1, L)

        adv_chunks = chunk_tn(all_advantages)
        atk_ret_chunks = chunk_tn(all_atk_returns)
        def_ret_chunks = chunk_tn(all_def_returns)
        lp_chunks = chunk_tn(log_probs_arr)
        act_chunks = {k: chunk_tn(actions_arr[k]) for k in actions_arr}

        # Hidden states at chunk boundaries
        chunk_starts = np.arange(n_chunks_per_env) * L
        hx_at_starts = buf_hx[chunk_starts]  # (n_chunks_per_env, N, gru_dim)
        hx_chunks = hx_at_starts.reshape(-1, cfg.gru_dim)  # (total_chunks, gru_dim)

        # --- Build (T_used, N, ...) observation arrays with global-max padding ---
        max_combat = max(buf_combat_hb[t].shape[1] for t in range(T))
        max_terrain = max(buf_terrain_hb[t].shape[1] for t in range(T))
        max_combat = max(max_combat, 1)
        max_terrain = max(max_terrain, 1)

        flat_chb = np.zeros((T_used, N, max_combat, cfg.combat_feature_dim), dtype=np.float32)
        flat_cm = np.zeros((T_used, N, max_combat), dtype=np.float32)
        flat_thb = np.zeros((T_used, N, max_terrain, cfg.terrain_feature_dim), dtype=np.float32)
        flat_tm = np.zeros((T_used, N, max_terrain), dtype=np.float32)
        flat_gs = np.zeros((T_used, N, cfg.global_state_dim), dtype=np.float32)

        for t in range(T_used):
            nc = buf_combat_hb[t].shape[1]
            flat_chb[t, :, :nc] = buf_combat_hb[t]
            flat_cm[t, :, :nc] = buf_combat_mask[t]
            nt = buf_terrain_hb[t].shape[1]
            flat_thb[t, :, :nt] = buf_terrain_hb[t]
            flat_tm[t, :, :nt] = buf_terrain_mask[t]
            flat_gs[t] = buf_global[t]

        # Normalize (flatten to 2D for normalizers, then reshape back)
        total_samples = T_used * N
        flat_gs_2d = flat_gs.reshape(total_samples, cfg.global_state_dim)
        flat_gs_2d = self._normalize_global_state(flat_gs_2d)
        flat_gs = flat_gs_2d.reshape(T_used, N, cfg.global_state_dim)

        flat_chb_2d = flat_chb.reshape(total_samples, max_combat, cfg.combat_feature_dim)
        flat_cm_2d = flat_cm.reshape(total_samples, max_combat)
        for i in range(total_samples):
            nc = int(flat_cm_2d[i].sum())
            if nc > 0:
                flat_chb_2d[i, :nc] = self.combat_normalizer.normalize(flat_chb_2d[i, :nc])
        flat_chb = flat_chb_2d.reshape(T_used, N, max_combat, cfg.combat_feature_dim)

        flat_thb_2d = flat_thb.reshape(total_samples, max_terrain, cfg.terrain_feature_dim)
        flat_tm_2d = flat_tm.reshape(total_samples, max_terrain)
        for i in range(total_samples):
            nt = int(flat_tm_2d[i].sum())
            if nt > 0:
                flat_thb_2d[i, :nt] = self.terrain_normalizer.normalize(flat_thb_2d[i, :nt])
        flat_thb = flat_thb_2d.reshape(T_used, N, max_terrain, cfg.terrain_feature_dim)

        # --- Chunk observations: (T_used, N, ...) -> (total_chunks, L, ...) ---
        def chunk_obs(arr):
            rest = arr.shape[2:]
            x = arr.reshape(n_chunks_per_env, L, N, *rest)
            x = np.moveaxis(x, 2, 1)  # (n_chunks, N, L, ...)
            return x.reshape(total_chunks, L, *rest)

        chb_chunks = chunk_obs(flat_chb)
        cm_chunks = chunk_obs(flat_cm)
        thb_chunks = chunk_obs(flat_thb)
        tm_chunks = chunk_obs(flat_tm)
        gs_chunks = chunk_obs(flat_gs)

        # Normalize advantages
        flat_adv = adv_chunks.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        adv_chunks = flat_adv.reshape(total_chunks, L)

        # Move to device
        adv_t = torch.from_numpy(adv_chunks).float().to(self.device)
        atk_ret_t = torch.from_numpy(atk_ret_chunks).float().to(self.device)
        def_ret_t = torch.from_numpy(def_ret_chunks).float().to(self.device)
        old_lp_t = torch.from_numpy(lp_chunks).float().to(self.device)
        act_t = {k: torch.from_numpy(v).long().to(self.device) for k, v in act_chunks.items()}
        chb_t = torch.from_numpy(chb_chunks).to(self.device)
        cm_t = torch.from_numpy(cm_chunks).to(self.device)
        thb_t = torch.from_numpy(thb_chunks).to(self.device)
        tm_t = torch.from_numpy(tm_chunks).to(self.device)
        gs_t = torch.from_numpy(gs_chunks).to(self.device)
        hx_t = torch.from_numpy(hx_chunks).float().to(self.device)

        # --- Training loop: shuffle chunks, process in minibatches ---
        CPB = cfg.chunks_per_batch
        total_metrics = {"surrogate": 0, "value_atk": 0, "value_def": 0, "entropy": 0, "kl": 0,
                         "gru_norm": 0}
        n_updates = 0

        pbar = tqdm(total=total_passes, unit="pass", unit_scale=True,
                    desc="  train", leave=False, dynamic_ncols=True)
        for _ in range(cfg.train_iters):
            chunk_indices = np.random.permutation(total_chunks)
            early_stop = False

            for start in range(0, total_chunks, CPB):
                idx = chunk_indices[start:start + CPB]

                hx_mb = hx_t[idx].detach()

                new_lp, entropy, v_atk, v_def, gru_info = self.policy.forward_sequence(
                    chb_t[idx], cm_t[idx], thb_t[idx], tm_t[idx], gs_t[idx],
                    hx_mb, {k: v[idx] for k, v in act_t.items()},
                )

                # Flatten (B, L) -> (B*L,) for loss
                new_lp_flat = new_lp.reshape(-1)
                entropy_flat = entropy.reshape(-1)
                v_atk_flat = v_atk.reshape(-1)
                v_def_flat = v_def.reshape(-1)
                adv_flat = adv_t[idx].reshape(-1)
                atk_ret_flat = atk_ret_t[idx].reshape(-1)
                def_ret_flat = def_ret_t[idx].reshape(-1)
                old_lp_flat = old_lp_t[idx].reshape(-1)

                log_ratio = new_lp_flat - old_lp_flat
                ratio = torch.exp(log_ratio)
                clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
                surrogate = -torch.min(ratio * adv_flat, clipped * adv_flat).mean()

                atk_vloss = (v_atk_flat - atk_ret_flat).pow(2)
                def_vloss = (v_def_flat - def_ret_flat).pow(2)
                value_loss = (
                    torch.clamp(atk_vloss, max=cfg.max_value_loss).mean()
                    + torch.clamp(def_vloss, max=cfg.max_value_loss).mean()
                )

                entropy_loss = -entropy_flat.mean()

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
                total_metrics["gru_norm"] += gru_info["gru_norm"]

                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean().item()
                    total_metrics["kl"] += kl

                pbar.update(len(idx) * L)
                pbar.set_postfix_str(f"surr={surrogate.item():+.3f} kl={kl:.3f}")

                if cfg.target_kl and kl > cfg.target_kl:
                    early_stop = True
                    break

            if early_stop:
                break

        pbar.close()
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
                "hx": self.hx,
            },
            path,
        )

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt["model"]
        # Remap old nn.GRUCell parameter names → new nn.GRU names so checkpoints
        # saved before the GRUCell→GRU swap still load. Same shapes, same convention.
        gru_remap = {
            "gru.weight_ih": "gru.weight_ih_l0",
            "gru.weight_hh": "gru.weight_hh_l0",
            "gru.bias_ih":   "gru.bias_ih_l0",
            "gru.bias_hh":   "gru.bias_hh_l0",
        }
        for old, new in gru_remap.items():
            if old in state and new not in state:
                state[new] = state.pop(old)
        missing, unexpected = self.policy.load_state_dict(state, strict=False)
        if missing:
            print(f"  Checkpoint missing keys (using init): {missing}")
        if unexpected:
            print(f"  Checkpoint unexpected keys (ignored): {unexpected}")
        # If GRU keys were missing (pre-GRU checkpoint), zero out LayerNorm
        # weight so the residual connection is a true no-op.
        if any("gru" in k for k in missing):
            with torch.no_grad():
                self.policy.gru_ln.weight.zero_()
                self.policy.gru_ln.bias.zero_()
        try:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError:
            pass  # param group mismatch (e.g. pre-GRU checkpoint); fine for eval
        if self.config.anneal_lr and ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("obs_normalizer"):
            self.obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
        if ckpt.get("combat_normalizer"):
            self.combat_normalizer.load_state_dict(ckpt["combat_normalizer"])
        if ckpt.get("terrain_normalizer"):
            self.terrain_normalizer.load_state_dict(ckpt["terrain_normalizer"])
        if ckpt.get("hx") is not None:
            self.hx = ckpt["hx"]
