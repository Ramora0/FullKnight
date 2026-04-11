import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from model import FullKnightActorCritic
from observation import Observation, CB


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


def _load_normalizer_compat(normalizer, state, label):
    """Load running-normalizer stats, truncating leading columns if the checkpoint
    was saved with more dims than the current normalizer tracks. Used for the
    combat (8 -> 4) and terrain (5 -> 4) shrink when binary cols stopped being
    running-normalized. Column order is preserved, so the first N stats are still
    valid for the same physical features."""
    ckpt_mean = np.asarray(state["mean"])
    cur_n = normalizer.mean.shape[0]
    ckpt_n = ckpt_mean.shape[0]
    if ckpt_n == cur_n:
        normalizer.load_state_dict(state)
        return
    if ckpt_n > cur_n:
        normalizer.mean = ckpt_mean[:cur_n].astype(np.float64)
        normalizer.var = np.asarray(state["var"])[:cur_n].astype(np.float64)
        normalizer.count = state["count"]
        print(f"  [compat] {label}_normalizer: truncated {ckpt_n}->{cur_n} dims from checkpoint")
    else:
        print(f"  [compat] {label}_normalizer: checkpoint has {ckpt_n} dims, current expects {cur_n} — skipping load")


class PPO:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.policy = FullKnightActorCritic(config).to(self.device)
        self.hx = None  # GRU hidden state, shape (N, hidden_dim) during rollout

        # Only normalize continuous features (indices 0:n_cont), not binary validity flags
        self.obs_normalizer = RunningNormalizer(config.global_state_dim - config.n_binary_flags)
        # Combat normalizer covers cols [0, combat_normalized_dims). The trailing
        # hp_raw / hp_max_raw columns get log1p compression instead of z-scoring,
        # via _log_compress_combat_tail — see that helper for the rationale.
        self.combat_normalizer = RunningNormalizer(config.combat_normalized_dims)
        self.terrain_normalizer = RunningNormalizer(config.terrain_normalized_dims)

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

    @staticmethod
    def _log_compress_combat_hp(combat_hb):
        """In-place log1p compression of the hp_raw / hp_max_raw columns
        (CB.HP_RAW, CB.HP_MAX_RAW). Padded slots stay zero (log1p(0)=0).
        Values are clamped to >=0 first to be defensive against negative HP
        from edge cases. See config.combat_feature_dim docstring for the
        magnitude rationale."""
        if combat_hb.shape[-1] <= CB.HP_RAW:
            return
        hp = combat_hb[..., CB.HP_RAW:CB.HP_MAX_RAW + 1]
        np.maximum(hp, 0, out=hp)
        np.log1p(hp, out=hp)

    def _normalize_hitboxes(self, hitboxes, mask, normalizer):
        """Normalize the first `normalizer.mean.shape[0]` columns; pass the rest through.
        Combat hp columns (hp_raw/hp_max_raw) are log1p-compressed by the caller
        via _log_compress_combat_hp."""
        n_norm = normalizer.mean.shape[0]
        flat = hitboxes.reshape(-1, hitboxes.shape[-1])
        flat_mask = mask.reshape(-1)
        real = flat[flat_mask > 0, :n_norm]
        if len(real) > 0:
            normalizer.update(real)
        normed = hitboxes.copy()
        for i in range(hitboxes.shape[0]):
            n_real = int(mask[i].sum())
            if n_real > 0:
                normed[i, :n_real, :n_norm] = normalizer.normalize(hitboxes[i, :n_real, :n_norm])
        return normed

    def reset_hidden(self, n_envs):
        """Zero the GRU hidden state. Call at epoch start."""
        self.hx = np.zeros((n_envs, self.config.gru_dim), dtype=np.float32)

    def reset_hidden_for(self, indices):
        """Zero the GRU hidden state for specific envs only."""
        for i in indices:
            self.hx[i] = 0.0

    def get_hx_snapshot(self, env_indices=None):
        """Return a copy of the current hidden state for buffering.

        If env_indices is provided, returns only those rows (in order)."""
        if env_indices is None:
            return self.hx.copy()
        return self.hx[env_indices].copy()

    def _ensure_event_log(self):
        if not hasattr(self, '_event_log'):
            # Each entry: (h2d_start, h2d_end, fwd_start, fwd_end, d2h_start, d2h_end)
            self._event_log = []
            self._norm_total = 0.0
            self._tensor_prep_total = 0.0  # CPU-side from_numpy/.float() before .to()

    def report_timing(self):
        """Call once per epoch after cuda.synchronize(). Returns timing dict.

        Components of one collect_action call:
          - normalize_s:    CPU-side numpy normalization
          - tensor_prep_s:  CPU-side from_numpy/.float() (before async .to())
          - h2d_s:          GPU-side host->device transfer
          - forward_s:      GPU-side model forward + sampling
          - d2h_s:          GPU-side device->host transfer of actions/values
        """
        if not hasattr(self, '_event_log') or not self._event_log:
            return None
        h2d_ms = sum(s.elapsed_time(e) for s, e, _, _, _, _ in self._event_log)
        fwd_ms = sum(s.elapsed_time(e) for _, _, s, e, _, _ in self._event_log)
        d2h_ms = sum(s.elapsed_time(e) for _, _, _, _, s, e in self._event_log)
        c = len(self._event_log)
        result = {
            'normalize_s':   self._norm_total,
            'tensor_prep_s': self._tensor_prep_total,
            'h2d_s':         h2d_ms / 1000,
            'forward_s':     fwd_ms / 1000,
            'd2h_s':         d2h_ms / 1000,
            'count': c,
        }
        self._event_log.clear()
        self._norm_total = 0.0
        self._tensor_prep_total = 0.0
        return result

    @torch.no_grad()
    def collect_action(self, obs: Observation, env_indices=None):
        """Get actions for a batch of observations during rollout collection.
        Input is a numpy-backed Observation. Returns numpy arrays.

        If env_indices is provided, the batch in `obs` corresponds to those
        specific env slots. Only the matching rows of self.hx are read and
        written; self.hx is kept full-sized (n_envs). Rows for envs not in
        env_indices are left untouched.
        """
        import time as _time
        self._ensure_event_log()

        t0 = _time.perf_counter()
        n_cont = self.config.global_state_dim - self.config.n_binary_flags
        self.obs_normalizer.update(obs.global_state[..., :n_cont])
        gs_norm = self._normalize_global_state(obs.global_state)
        chb_norm = self._normalize_hitboxes(obs.combat_hb, obs.combat_mask, self.combat_normalizer)
        self._log_compress_combat_hp(chb_norm)
        thb_norm = self._normalize_hitboxes(obs.terrain_hb, obs.terrain_mask, self.terrain_normalizer)
        self._norm_total += _time.perf_counter() - t0

        # CPU-side tensor prep (from_numpy + .float()) — async .to() is timed
        # separately via cuda events. We measure CPU prep with wall time and
        # bracket the .to() calls with cuda events for the actual transfer.
        h2d_start = torch.cuda.Event(enable_timing=True)
        h2d_end = torch.cuda.Event(enable_timing=True)
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        d2h_start = torch.cuda.Event(enable_timing=True)
        d2h_end = torch.cuda.Event(enable_timing=True)

        t1 = _time.perf_counter()
        chb_t = torch.from_numpy(chb_norm).float()
        cm_t = torch.from_numpy(obs.combat_mask).float()
        ckid_t = torch.from_numpy(obs.combat_kind_ids).long()
        cpid_t = torch.from_numpy(obs.combat_parent_ids).long()
        thb_t = torch.from_numpy(thb_norm).float()
        tm_t = torch.from_numpy(obs.terrain_mask).float()
        gs_t = torch.from_numpy(gs_norm).float()
        hx_slice = self.hx[env_indices] if env_indices is not None else self.hx
        hx_pinned = torch.from_numpy(hx_slice).float()
        self._tensor_prep_total += _time.perf_counter() - t1

        h2d_start.record()
        gpu_obs = Observation(
            combat_hb=chb_t.to(self.device),
            combat_mask=cm_t.to(self.device),
            combat_kind_ids=ckid_t.to(self.device),
            combat_parent_ids=cpid_t.to(self.device),
            terrain_hb=thb_t.to(self.device),
            terrain_mask=tm_t.to(self.device),
            global_state=gs_t.to(self.device),
        )
        hx_t = hx_pinned.to(self.device)
        h2d_end.record()

        fwd_start.record()
        actions, log_prob, _, value_atk, value_def, hx_new = self.policy.get_action_and_value(
            gpu_obs, hx=hx_t
        )
        fwd_end.record()

        hx_new_np = hx_new.cpu().numpy()
        if env_indices is None:
            self.hx = hx_new_np
        else:
            self.hx[env_indices] = hx_new_np

        d2h_start.record()
        actions_np = {k: v.cpu().numpy() for k, v in actions.items()}
        result = actions_np, log_prob.cpu().numpy(), value_atk.cpu().numpy(), value_def.cpu().numpy()
        d2h_end.record()

        self._event_log.append((h2d_start, h2d_end, fwd_start, fwd_end, d2h_start, d2h_end))
        return result

    def train_on_rollout(self, obs_buf, actions_arr, log_probs_arr,
                         damage_landed_arr, hits_taken_arr,
                         values_atk_arr, values_def_arr, D_per_env, buf_hx):
        """Train on a collected rollout with chunked truncated BPTT.

        obs_buf: list of length T, each element a per-step Observation with
                 leading dim (N, ...). Combined into (T, N, ...) here.
        actions_arr: dict of (T, N) numpy arrays
        log_probs_arr: (T, N)
        damage_landed_arr, hits_taken_arr: (T, N)
        values_atk_arr, values_def_arr: (T+1, N)
        D_per_env: (N,) per-env curriculum scaling factor (one D per boss assignment)
        buf_hx: (T, N, hidden_dim) GRU hidden states at each timestep
        """
        T, N = damage_landed_arr.shape
        cfg = self.config
        L = cfg.seq_len
        n_chunks_per_env = T // L
        T_used = n_chunks_per_env * L
        total_chunks = n_chunks_per_env * N
        max_combat_dim = max(o.combat_hb.shape[1] for o in obs_buf)
        max_terrain_dim = max(o.terrain_hb.shape[1] for o in obs_buf)
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
                values_atk_arr[:, env_i], values_def_arr[:, env_i],
                float(D_per_env[env_i]),
            )
            all_advantages[:, env_i] = adv
            all_atk_returns[:, env_i] = atk_ret
            all_def_returns[:, env_i] = def_ret

        # Explained variance at rollout time: how well did the critic predict
        # returns using the values it produced during collection. Scale-free,
        # so atk and def are directly comparable (unlike raw MSE).
        def _ev(returns, values):
            var = returns.var()
            return float(1.0 - (returns - values).var() / var) if var > 1e-8 else 0.0
        ev_atk = _ev(all_atk_returns, values_atk_arr[:T])
        ev_def = _ev(all_def_returns, values_def_arr[:T])

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

        # --- Stack the per-step observations into one (T_used, N, ...) Observation
        # via the dataclass helper (handles global-max repad automatically). ---
        stacked = Observation.stack(obs_buf[:T_used])
        max_combat = stacked.combat_hb.shape[2]
        max_terrain = stacked.terrain_hb.shape[2]

        # Normalize (flatten to 2D for normalizers, then reshape back)
        total_samples = T_used * N
        flat_gs_2d = stacked.global_state.reshape(total_samples, cfg.global_state_dim)
        flat_gs_2d = self._normalize_global_state(flat_gs_2d)
        flat_gs = flat_gs_2d.reshape(T_used, N, cfg.global_state_dim)

        flat_chb_2d = stacked.combat_hb.reshape(total_samples, max_combat, cfg.combat_feature_dim)
        flat_cm_2d = stacked.combat_mask.reshape(total_samples, max_combat)
        n_norm_c = cfg.combat_normalized_dims
        for i in range(total_samples):
            nc = int(flat_cm_2d[i].sum())
            if nc > 0:
                flat_chb_2d[i, :nc, :n_norm_c] = self.combat_normalizer.normalize(
                    flat_chb_2d[i, :nc, :n_norm_c])
        # Log1p the hp columns over the whole buffer at once. Padded rows are
        # zero so log1p(0)=0 keeps them clean.
        self._log_compress_combat_hp(flat_chb_2d)
        flat_chb = flat_chb_2d.reshape(T_used, N, max_combat, cfg.combat_feature_dim)

        flat_thb_2d = stacked.terrain_hb.reshape(total_samples, max_terrain, cfg.terrain_feature_dim)
        flat_tm_2d = stacked.terrain_mask.reshape(total_samples, max_terrain)
        n_norm_t = cfg.terrain_normalized_dims
        for i in range(total_samples):
            nt = int(flat_tm_2d[i].sum())
            if nt > 0:
                flat_thb_2d[i, :nt, :n_norm_t] = self.terrain_normalizer.normalize(
                    flat_thb_2d[i, :nt, :n_norm_t])
        flat_thb = flat_thb_2d.reshape(T_used, N, max_terrain, cfg.terrain_feature_dim)

        flat_ckid = stacked.combat_kind_ids
        flat_cpid = stacked.combat_parent_ids
        flat_cm = stacked.combat_mask
        flat_tm = stacked.terrain_mask

        # --- Chunk observations: (T_used, N, ...) -> (total_chunks, L, ...) ---
        def chunk_obs(arr):
            rest = arr.shape[2:]
            x = arr.reshape(n_chunks_per_env, L, N, *rest)
            x = np.moveaxis(x, 2, 1)  # (n_chunks, N, L, ...)
            return x.reshape(total_chunks, L, *rest)

        chb_chunks = chunk_obs(flat_chb)
        cm_chunks = chunk_obs(flat_cm)
        ckid_chunks = chunk_obs(flat_ckid)
        cpid_chunks = chunk_obs(flat_cpid)
        thb_chunks = chunk_obs(flat_thb)
        tm_chunks = chunk_obs(flat_tm)
        gs_chunks = chunk_obs(flat_gs)

        # Normalize advantages
        flat_adv = adv_chunks.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        adv_chunks = flat_adv.reshape(total_chunks, L)

        # Move to device — bundle into a single (total_chunks, L, ...) Observation
        # so the inner training loop can index obs_t[idx] in one shot.
        obs_t = Observation(
            combat_hb=torch.from_numpy(chb_chunks).to(self.device),
            combat_mask=torch.from_numpy(cm_chunks).to(self.device),
            combat_kind_ids=torch.from_numpy(ckid_chunks).long().to(self.device),
            combat_parent_ids=torch.from_numpy(cpid_chunks).long().to(self.device),
            terrain_hb=torch.from_numpy(thb_chunks).to(self.device),
            terrain_mask=torch.from_numpy(tm_chunks).to(self.device),
            global_state=torch.from_numpy(gs_chunks).to(self.device),
        )
        adv_t = torch.from_numpy(adv_chunks).float().to(self.device)
        atk_ret_t = torch.from_numpy(atk_ret_chunks).float().to(self.device)
        def_ret_t = torch.from_numpy(def_ret_chunks).float().to(self.device)
        old_lp_t = torch.from_numpy(lp_chunks).float().to(self.device)
        act_t = {k: torch.from_numpy(v).long().to(self.device) for k, v in act_chunks.items()}
        hx_t = torch.from_numpy(hx_chunks).float().to(self.device)

        # --- Training loop: shuffle chunks, process in minibatches ---
        CPB = cfg.chunks_per_batch
        total_metrics = {"surrogate": 0, "value_atk": 0, "value_def": 0, "entropy": 0, "kl": 0,
                         "gru_norm": 0}
        n_updates = 0
        passes_done = 0

        pbar = tqdm(total=total_passes, unit="pass", unit_scale=True,
                    desc="  train", leave=False, dynamic_ncols=True)
        stop_training = False
        for _ in range(cfg.train_iters):
            if stop_training:
                break
            chunk_indices = np.random.permutation(total_chunks)
            iter_kl_sum = 0.0
            iter_kl_n = 0

            for start in range(0, total_chunks, CPB):
                idx = chunk_indices[start:start + CPB]

                hx_mb = hx_t[idx].detach()

                obs_mb = Observation(
                    combat_hb=obs_t.combat_hb[idx],
                    combat_mask=obs_t.combat_mask[idx],
                    combat_kind_ids=obs_t.combat_kind_ids[idx],
                    combat_parent_ids=obs_t.combat_parent_ids[idx],
                    terrain_hb=obs_t.terrain_hb[idx],
                    terrain_mask=obs_t.terrain_mask[idx],
                    global_state=obs_t.global_state[idx],
                )
                new_lp, entropy, v_atk, v_def, gru_info = self.policy.forward_sequence(
                    obs_mb, hx_mb, {k: v[idx] for k, v in act_t.items()},
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
                    iter_kl_sum += kl
                    iter_kl_n += 1

                passes_done += len(idx) * L
                pbar.update(len(idx) * L)
                pbar.set_postfix_str(f"surr={surrogate.item():+.3f} kl={kl:.3f}")

                # Mid-iter running-mean halt with 2-minibatch warmup: catches
                # cumulative drift as soon as it exceeds target, but avoids the
                # "first outlier kills the epoch" jankiness of per-batch halt.
                if (
                    cfg.target_kl
                    and iter_kl_n >= 2
                    and (iter_kl_sum / iter_kl_n) > cfg.target_kl
                ):
                    stop_training = True
                    break

        pbar.close()
        out = {k: v / max(n_updates, 1) for k, v in total_metrics.items()}
        out["ev_atk"] = ev_atk
        out["ev_def"] = ev_def
        out["pass_frac"] = passes_done / max(total_passes, 1)
        return out

    def save_checkpoint(self, path, vocab=None, boss_state=None, epoch=None):
        # Serialize per-boss curriculum state: D plus the raw rolling windows
        # so resume can continue the EMA without a warm-up gap.
        ckpt_boss = None
        if boss_state is not None:
            ckpt_boss = {
                b: {
                    "D": float(s["D"]),
                    "landed_window": list(s["landed_window"]),
                    "taken_window": list(s["taken_window"]),
                }
                for b, s in boss_state.items()
            }
        torch.save(
            {
                "model": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.config.anneal_lr else None,
                "obs_normalizer": self.obs_normalizer.state_dict(),
                "combat_normalizer": self.combat_normalizer.state_dict(),
                "terrain_normalizer": self.terrain_normalizer.state_dict(),
                "hx": self.hx,
                "kind_vocab": vocab.state_dict() if vocab is not None else None,
                "boss_state": ckpt_boss,
                "epoch": epoch,
            },
            path,
        )

    def load_checkpoint(self, path, vocab=None, boss_state=None):
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
            _load_normalizer_compat(self.combat_normalizer, ckpt["combat_normalizer"], "combat")
        if ckpt.get("terrain_normalizer"):
            _load_normalizer_compat(self.terrain_normalizer, ckpt["terrain_normalizer"], "terrain")
        if ckpt.get("hx") is not None:
            self.hx = ckpt["hx"]
        if vocab is not None and ckpt.get("kind_vocab") is not None:
            vocab.load_state_dict(ckpt["kind_vocab"])
            print(f"  Loaded kind vocab: {len(vocab)} entries")
        start_epoch = 0
        ckpt_epoch = ckpt.get("epoch")
        if ckpt_epoch is not None:
            start_epoch = int(ckpt_epoch) + 1
            print(f"  Resuming at epoch {start_epoch} (checkpoint saved at epoch {ckpt_epoch})")
        if boss_state is not None and ckpt.get("boss_state") is not None:
            ckpt_boss = ckpt["boss_state"]
            restored, new, dropped = [], [], []
            for b, s in ckpt_boss.items():
                if b in boss_state:
                    boss_state[b]["D"] = float(s["D"])
                    boss_state[b]["landed_window"].clear()
                    boss_state[b]["landed_window"].extend(s["landed_window"])
                    boss_state[b]["taken_window"].clear()
                    boss_state[b]["taken_window"].extend(s["taken_window"])
                    restored.append(f"{b}={s['D']:.2f}")
                else:
                    dropped.append(b)
            for b in boss_state:
                if b not in ckpt_boss:
                    new.append(b)
            if restored:
                print(f"  Restored per-boss D: {', '.join(restored)}")
            if new:
                print(f"  New bosses (using D_initial): {new}")
            if dropped:
                print(f"  Checkpoint bosses not in current pool (skipped): {dropped}")
        return start_epoch
