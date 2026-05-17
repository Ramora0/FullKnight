import time as _time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from model import FullKnightActorCritic
from observation import Observation, CB, mirror_observation, mirror_movement
from graph_runner import BucketedGraphRunner
from train_graph_runner import BucketedTrainGraphRunner


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
        self.hx = None  # GRU hidden state, shape (N, gru_dim) during rollout

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
        # LR annealing is now step-based and driven from train.py via
        # set_lr(); no torch LR scheduler needed.

        # Lazy-initialized on first collect_action() call once we know n_envs.
        self._graph_runner = None
        # Lazy-initialized on first train_on_rollout() call.
        self._train_graph_runner = None

    def get_advantages(self, damage_landed, hits_taken, hp_healed, values_atk, values_def, D, heal_coef, dones=None):
        """GAE with decomposed value heads and curriculum scaling.

        Values are trained on stationary rewards, D scales at advantage time.
        damage_landed is % of boss max HP dealt (1.0 = 1% of boss HP).
        D is % boss HP we deal per hit taken against us.
        hp_healed is raw HP restored this step.
        heal_coef scales hp_healed (unscaled by D, like defense).
        dones[t] = True means episode ended at step t; bootstrap to 0.
        δ_t = δ_attack_t / D - δ_defense_t + heal_coef * hp_healed_t
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
            # Terminal state: bootstrap to 0
            if dones is not None and dones[t]:
                next_vatk = 0
                next_vdef = 0
                lastgaelam = 0
                lastgaelam_atk = 0
                lastgaelam_def = 0
            else:
                next_vatk = values_atk[t + 1]
                next_vdef = values_def[t + 1]

            # Stationary TD errors for each head
            delta_atk = damage_landed[t] + gamma * next_vatk - values_atk[t]
            delta_def = hits_taken[t] + gamma * next_vdef - values_def[t]

            # Curriculum-scaled advantage with heal reward
            delta = delta_atk / D - delta_def + heal_coef * hp_healed[t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages[t] = lastgaelam

            # Stationary returns for value loss (no D, no heal — value heads track raw signals)
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

    def _ensure_train_graph_runner(self):
        """Lazy-init the training CUDA graph runner on first train_on_rollout."""
        if self._train_graph_runner is not None:
            return
        if not getattr(self.config, "use_train_cuda_graphs", False):
            return
        if not torch.cuda.is_available():
            return
        # Reuse the rollout buckets — same bounds work for training.
        combat_buckets = [int(x) for x in self.config.graph_combat_buckets.split(",") if x.strip()]
        terrain_buckets = [int(x) for x in self.config.graph_terrain_buckets.split(",") if x.strip()]
        print(
            f"  [train_cuda_graphs] capturing CPB={self.config.chunks_per_batch} "
            f"L={self.config.seq_len} combat={combat_buckets} terrain={terrain_buckets}",
            flush=True,
        )
        self._train_graph_runner = BucketedTrainGraphRunner(
            self.policy, self.optimizer, self.config, self.device,
            combat_buckets, terrain_buckets,
        )

    def _ensure_graph_runner(self):
        """Lazy-init the bucketed CUDA graph runner on first collect_action."""
        if self._graph_runner is not None:
            return
        if not self.config.use_cuda_graphs or not torch.cuda.is_available():
            return
        n_envs = self.hx.shape[0]
        combat_buckets = [int(x) for x in self.config.graph_combat_buckets.split(",") if x.strip()]
        terrain_buckets = [int(x) for x in self.config.graph_terrain_buckets.split(",") if x.strip()]
        print(f"  [cuda_graphs] capturing B={n_envs} "
              f"combat={combat_buckets} terrain={terrain_buckets}...", flush=True)
        self._graph_runner = BucketedGraphRunner(
            self.policy, B=n_envs,
            combat_buckets=combat_buckets,
            terrain_buckets=terrain_buckets,
            cfg=self.config, device=self.device,
        )

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
        self._ensure_graph_runner()

        t0 = _time.perf_counter()
        n_cont = self.config.global_state_dim - self.config.n_binary_flags
        self.obs_normalizer.update(obs.global_state[..., :n_cont])
        gs_norm = self._normalize_global_state(obs.global_state)
        chb_norm = self._normalize_hitboxes(obs.combat_hb, obs.combat_mask, self.combat_normalizer)
        self._log_compress_combat_hp(chb_norm)
        thb_norm = self._normalize_hitboxes(obs.terrain_hb, obs.terrain_mask, self.terrain_normalizer)
        self._norm_total += _time.perf_counter() - t0

        if self._graph_runner is not None:
            return self._collect_via_graph(
                obs, gs_norm, chb_norm, thb_norm, env_indices
            )

        # --- Eager path (unchanged) ---
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
        (actions, log_prob, _, value_atk, value_def, hx_new,
         log_prob_action, _) = self.policy.get_action_and_value(gpu_obs, hx=hx_t)
        fwd_end.record()

        hx_new_np = hx_new.cpu().numpy()
        if env_indices is None:
            self.hx = hx_new_np
        else:
            self.hx[env_indices] = hx_new_np

        d2h_start.record()
        actions_np = {k: v.cpu().numpy() for k, v in actions.items()}
        result = (actions_np, log_prob.cpu().numpy(),
                  log_prob_action.cpu().numpy(),
                  value_atk.cpu().numpy(), value_def.cpu().numpy())
        d2h_end.record()

        self._event_log.append((h2d_start, h2d_end, fwd_start, fwd_end, d2h_start, d2h_end))
        return result

    def _collect_via_graph(self, obs, gs_norm, chb_norm, thb_norm, env_indices):
        """CUDA-graph path: pad inputs to full n_envs, replay the captured
        graph, slice outputs back to the active env_indices.

        We always run the graph at the captured B (= n_envs); inactive slots
        are zero-masked so masked attention ignores them. The wasted compute
        on inactive rows is overshadowed by the per-call launch-overhead
        savings (graph replay is ~0.5 ms vs ~6 ms eager).
        """
        runner = self._graph_runner
        n_envs = runner.B
        cfg = self.config
        active = (
            list(range(n_envs)) if env_indices is None else list(env_indices)
        )
        n_combat_in = obs.combat_hb.shape[1]
        n_terrain_in = obs.terrain_hb.shape[1]

        # Build full-B padded inputs. Inactive slots are zero (mask=0 makes
        # them no-ops through masked attention).
        full_obs = Observation(
            combat_hb=np.zeros((n_envs, n_combat_in, cfg.combat_feature_dim), dtype=np.float32),
            combat_mask=np.zeros((n_envs, n_combat_in), dtype=np.float32),
            combat_kind_ids=np.zeros((n_envs, n_combat_in), dtype=np.int64),
            combat_parent_ids=np.zeros((n_envs, n_combat_in), dtype=np.int64),
            terrain_hb=np.zeros((n_envs, n_terrain_in, cfg.terrain_feature_dim), dtype=np.float32),
            terrain_mask=np.zeros((n_envs, n_terrain_in), dtype=np.float32),
            global_state=np.zeros((n_envs, cfg.global_state_dim), dtype=np.float32),
        )
        full_obs.combat_hb[active] = chb_norm
        full_obs.combat_mask[active] = obs.combat_mask
        full_obs.combat_kind_ids[active] = obs.combat_kind_ids
        full_obs.combat_parent_ids[active] = obs.combat_parent_ids
        full_obs.terrain_hb[active] = thb_norm
        full_obs.terrain_mask[active] = obs.terrain_mask
        full_obs.global_state[active] = gs_norm

        # hx is already full-size; pass it through as-is. Active rows hold
        # real state; inactive rows hold whatever was last written (doesn't
        # matter — we only read back hx_new for active rows).
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        fwd_start.record()
        out, _bucket = runner.run_numpy(full_obs, self.hx)
        fwd_end.record()

        # Update hx for active envs only.
        self.hx[active] = out["hx_new"][active].copy()

        actions_np = {
            "movement": out["act_movement"][active].copy(),
            "direction": out["act_direction"][active].copy(),
            "action": out["act_action"][active].copy(),
            "jump": out["act_jump"][active].copy(),
        }
        result = (
            actions_np,
            out["log_prob"][active].copy(),
            out["log_prob_action"][active].copy(),
            out["value_atk"][active].copy(),
            out["value_def"][active].copy(),
        )

        # Log timing — bundle h2d+fwd+d2h into one "fwd" segment, h2d/d2h
        # log to zero so report_timing's split still works syntactically.
        zero_evt_a = torch.cuda.Event(enable_timing=True)
        zero_evt_b = torch.cuda.Event(enable_timing=True)
        zero_evt_a.record(); zero_evt_b.record()
        self._event_log.append((zero_evt_a, zero_evt_b, fwd_start, fwd_end, zero_evt_a, zero_evt_b))
        return result

    def train_on_rollout(self, obs_buf, actions_arr, log_probs_arr,
                         log_probs_action_arr,
                         damage_landed_arr, hits_taken_arr, hp_healed_arr,
                         values_atk_arr, values_def_arr, D_per_env, buf_hx,
                         dones_arr=None, valid_arr=None, committed_arr=None,
                         boss_per_env=None, value_var_state=None):
        """Train on a collected rollout with chunked truncated BPTT.

        obs_buf: list of length T, each element a per-step Observation with
                 leading dim (N, ...). Combined into (T, N, ...) here.
        actions_arr: dict of (T, N) numpy arrays
        log_probs_arr: (T, N) summed log_prob over all 4 action heads.
        log_probs_action_arr: (T, N) action-head log_prob alone (subtracted
                              out on hard-commit steps so the action-head
                              gradient is zero — the agent didn't freely choose
                              that action).
        damage_landed_arr, hits_taken_arr, hp_healed_arr: (T, N)
        values_atk_arr, values_def_arr: (T+1, N)
        D_per_env: (N,) per-env curriculum scaling factor (one D per boss assignment)
        buf_hx: (T, N, gru_dim) GRU hidden states at each timestep
        dones_arr: (T, N) boolean array, True if episode ended at that step
        valid_arr: (T, N) boolean array, True for steps that should contribute
                   to the loss. Post-death filler steps (frozen all-zero obs
                   between death and end-of-rollout reset) are masked out.
                   None = all steps valid.
        committed_arr: (T, N) boolean array, True iff action[2] was overridden
                       by the C# hard-commit state machine on that step. The
                       action head's policy-loss and entropy contributions are
                       masked to zero on committed steps; movement/direction/
                       jump heads remain free and contribute normally.
                       None = no commits this rollout.
        boss_per_env: list of length N giving the boss name for each env.
                      When provided, enables per-boss advantage normalization
                      (per-rollout mean/std per boss) and per-boss value-loss
                      variance normalization (EMA-tracked across rollouts via
                      value_var_state). When None, falls back to rollout-wide
                      stats — single-boss equivalent behavior.
        value_var_state: dict {boss: {"atk_var_ema": float|None,
                                      "def_var_ema": float|None}}.
                         Mutated in place after this rollout's variance is
                         folded in. Required when boss_per_env is provided.
                         EMA decay is β ** (boss_samples / fair_share) so
                         heavier-represented bosses move the EMA more; a boss
                         with no valid samples this rollout is left untouched.
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

        # Diagnostic phase wallclocks (seconds). Consumed by train.py's
        # TimingTracker — train/* buckets get their fine-grained split from
        # this. CPU phases use perf_counter; GPU phases (forward/backward)
        # use cuda events recorded inline and read once at the end.
        train_phase_t = {}
        _t_phase = _time.perf_counter()

        # Compute decomposed GAE per-env
        all_advantages = np.empty((T, N), dtype=np.float32)
        all_atk_returns = np.empty((T, N), dtype=np.float32)
        all_def_returns = np.empty((T, N), dtype=np.float32)
        heal_coef = cfg.heal_coef
        for env_i in range(N):
            env_dones = dones_arr[:, env_i] if dones_arr is not None else None
            adv, atk_ret, def_ret = self.get_advantages(
                damage_landed_arr[:, env_i], hits_taken_arr[:, env_i],
                hp_healed_arr[:, env_i],
                values_atk_arr[:, env_i], values_def_arr[:, env_i],
                float(D_per_env[env_i]), heal_coef, env_dones,
            )
            all_advantages[:, env_i] = adv
            all_atk_returns[:, env_i] = atk_ret
            all_def_returns[:, env_i] = def_ret

        train_phase_t["gae"] = _time.perf_counter() - _t_phase
        _t_phase = _time.perf_counter()

        if valid_arr is None:
            valid_arr = np.ones((T, N), dtype=bool)
        valid_bool = valid_arr.astype(bool)

        # Explained variance at rollout time: how well did the critic predict
        # returns using the values it produced during collection. Scale-free,
        # so atk and def are directly comparable (unlike raw MSE). Masked to
        # exclude post-death zero-obs samples that bias both numerator and
        # denominator toward 0.
        def _ev(returns, values, mask):
            r = returns[mask]
            v = values[mask]
            if r.size == 0:
                return 0.0
            var = r.var()
            return float(1.0 - (r - v).var() / var) if var > 1e-8 else 0.0
        ev_atk = _ev(all_atk_returns, values_atk_arr[:T], valid_bool)
        ev_def = _ev(all_def_returns, values_def_arr[:T], valid_bool)

        # --- Chunk (T, N) arrays into (total_chunks, L) ---
        def chunk_tn(arr):
            """(T, N) -> (total_chunks, L): group by chunk then env."""
            return arr[:T_used].reshape(n_chunks_per_env, L, N).transpose(0, 2, 1).reshape(-1, L)

        adv_chunks = chunk_tn(all_advantages)
        atk_ret_chunks = chunk_tn(all_atk_returns)
        def_ret_chunks = chunk_tn(all_def_returns)
        lp_chunks = chunk_tn(log_probs_arr)
        lp_a_chunks = chunk_tn(log_probs_action_arr)
        act_chunks = {k: chunk_tn(actions_arr[k]) for k in actions_arr}
        valid_chunks = chunk_tn(valid_arr.astype(np.float32))
        if committed_arr is None:
            committed_arr = np.zeros((T, N), dtype=bool)
        committed_chunks = chunk_tn(committed_arr.astype(np.float32))

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

        # --- Per-boss advantage normalization + per-boss value-loss variance.
        # When boss_per_env is provided we normalize each sample by its own
        # boss's stats (per-rollout for advantage, EMA-tracked for value var)
        # so mixing bosses with different reward scales doesn't dilute either
        # signal. Falls back to rollout-wide stats when boss_per_env is None.
        # Chunk layout from chunk_tn: chunk_idx = chunk_block * N + env_i, so
        # env-per-chunk is chunk_idx % N. All L steps in a chunk share an env
        # (and therefore a boss).
        env_per_chunk = np.arange(total_chunks) % N
        if boss_per_env is not None:
            assert len(boss_per_env) == N, (
                f"boss_per_env length {len(boss_per_env)} != N={N}")
            boss_per_chunk = np.array([boss_per_env[e] for e in env_per_chunk])
            unique_bosses = list(dict.fromkeys(boss_per_env))  # preserves order
        else:
            boss_per_chunk = np.full(total_chunks, "__all__", dtype=object)
            unique_bosses = ["__all__"]

        # Per-boss advantage normalization (per-rollout, no EMA — PPO trust
        # region wants current-batch scaling). Per-sample mean/std lookup is
        # built per-chunk and broadcast over L.
        adv_mean_per_chunk = np.zeros(total_chunks, dtype=np.float32)
        adv_std_per_chunk = np.ones(total_chunks, dtype=np.float32)
        # Per-rollout per-boss adv std for diagnostics; also feeds adv_std_raw
        # as the sample-weighted mean across bosses (keeps the metric meaningful
        # when the dashboard already plots a single scalar).
        adv_std_by_boss = {}
        sample_weighted_std_num = 0.0
        sample_weighted_std_den = 0.0
        for boss in unique_bosses:
            chunk_mask = (boss_per_chunk == boss)
            if not chunk_mask.any():
                continue
            boss_adv = adv_chunks[chunk_mask].reshape(-1)
            boss_valid = valid_chunks[chunk_mask].reshape(-1)
            n_valid = float(boss_valid.sum())
            if n_valid > 1:
                m = float((boss_adv * boss_valid).sum() / n_valid)
                v = float(((boss_adv - m) ** 2 * boss_valid).sum() / n_valid)
                s = float(np.sqrt(v))
                adv_mean_per_chunk[chunk_mask] = m
                adv_std_per_chunk[chunk_mask] = s
                adv_std_by_boss[boss] = s
                sample_weighted_std_num += s * n_valid
                sample_weighted_std_den += n_valid
        adv_std_raw = (sample_weighted_std_num / sample_weighted_std_den
                       if sample_weighted_std_den > 0 else 0.0)
        adv_chunks = (
            (adv_chunks - adv_mean_per_chunk[:, None])
            / (adv_std_per_chunk[:, None] + 1e-8)
        )

        # Per-boss return-variance EMA for value-loss normalization (PopArt-lite).
        # Compute this rollout's per-boss variance over valid samples, fold into
        # the EMA at rate β ** (boss_samples / fair_share). Bosses with no valid
        # samples this rollout keep their existing EMA. Fair share is total_valid
        # / n_active_bosses so the decay is calibrated to balanced rollouts.
        beta_base = float(cfg.value_var_ema)
        if boss_per_env is not None:
            boss_per_env_arr = np.array(boss_per_env)
        else:
            boss_per_env_arr = np.full(N, "__all__", dtype=object)

        atk_var_per_env = np.zeros(N, dtype=np.float32)
        def_var_per_env = np.zeros(N, dtype=np.float32)
        atk_var_by_boss = {}
        def_var_by_boss = {}
        total_valid_samples = int(valid_bool.sum()) if valid_bool.any() else 0
        n_active_bosses = max(1, sum(
            1 for b in unique_bosses
            if (boss_per_env_arr == b).any()
            and valid_bool[:, boss_per_env_arr == b].any()
        ))
        fair_share = max(1.0, total_valid_samples / n_active_bosses)
        for boss in unique_bosses:
            env_mask = (boss_per_env_arr == boss)
            if not env_mask.any():
                continue
            sub_valid = valid_bool[:, env_mask]
            if not sub_valid.any():
                # Boss had no valid samples this rollout — leave EMA untouched,
                # but seed per-env denominators from existing EMA (or fallback).
                if value_var_state is not None and boss in value_var_state:
                    prev = value_var_state[boss]
                    if prev.get("atk_var_ema") is not None:
                        atk_var_per_env[env_mask] = float(prev["atk_var_ema"])
                    if prev.get("def_var_ema") is not None:
                        def_var_per_env[env_mask] = float(prev["def_var_ema"])
                continue
            atk_v = float(all_atk_returns[:, env_mask][sub_valid].var())
            def_v = float(all_def_returns[:, env_mask][sub_valid].var())
            atk_var_by_boss[boss] = atk_v
            def_var_by_boss[boss] = def_v
            n_samples = float(sub_valid.sum())
            if value_var_state is not None and boss in value_var_state:
                slot = value_var_state[boss]
                if slot.get("atk_var_ema") is None or slot.get("def_var_ema") is None:
                    # First observation — seed without smoothing.
                    slot["atk_var_ema"] = atk_v
                    slot["def_var_ema"] = def_v
                else:
                    beta_eff = beta_base ** (n_samples / fair_share)
                    slot["atk_var_ema"] = float(
                        beta_eff * slot["atk_var_ema"] + (1.0 - beta_eff) * atk_v)
                    slot["def_var_ema"] = float(
                        beta_eff * slot["def_var_ema"] + (1.0 - beta_eff) * def_v)
                atk_var_per_env[env_mask] = float(slot["atk_var_ema"])
                def_var_per_env[env_mask] = float(slot["def_var_ema"])
            else:
                # No EMA state — use this rollout's per-boss variance directly.
                atk_var_per_env[env_mask] = atk_v
                def_var_per_env[env_mask] = def_v

        # Aggregate scalars retained for the metrics dict (dashboard continuity).
        if valid_bool.any():
            atk_var = float(all_atk_returns[valid_bool].var())
            def_var = float(all_def_returns[valid_bool].var())
        else:
            atk_var = def_var = 0.0

        # Per-sample variance denominators, broadcast through chunk layout.
        # Floor with 1e-3 like the old atk_var_eff/def_var_eff.
        atk_var_per_chunk = np.maximum(atk_var_per_env[env_per_chunk], 0.0) + 1e-3
        def_var_per_chunk = np.maximum(def_var_per_env[env_per_chunk], 0.0) + 1e-3

        train_phase_t["normalize"] = _time.perf_counter() - _t_phase
        _t_phase = _time.perf_counter()

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
        old_lp_a_t = torch.from_numpy(lp_a_chunks).float().to(self.device)
        act_t = {k: torch.from_numpy(v).long().to(self.device) for k, v in act_chunks.items()}
        hx_t = torch.from_numpy(hx_chunks).float().to(self.device)
        valid_t = torch.from_numpy(valid_chunks).float().to(self.device)
        committed_t = torch.from_numpy(committed_chunks).float().to(self.device)
        atk_var_t = torch.from_numpy(atk_var_per_chunk).float().to(self.device)
        def_var_t = torch.from_numpy(def_var_per_chunk).float().to(self.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_phase_t["h2d"] = _time.perf_counter() - _t_phase
        _t_phase = _time.perf_counter()

        # Accumulate inner-loop GPU phase timings via cuda events: pair (start,
        # end) recorded around each forward_sequence and bwd/clip/step block,
        # summed once after the loop. Read elapsed_time after a final sync at
        # the end so the GPU has finished by the time we query.
        _fwd_evt_pairs = []
        _bwd_evt_pairs = []
        _cuda_ok = torch.cuda.is_available()

        # --- Training loop: shuffle chunks, process in minibatches ---
        CPB = cfg.chunks_per_batch
        total_metrics = {"surrogate": 0, "value_atk": 0, "value_def": 0, "entropy": 0, "kl": 0,
                         "gru_norm": 0}
        n_updates = 0
        passes_done = 0

        pbar = tqdm(total=total_passes, unit="pass", unit_scale=True,
                    desc="  train", leave=False, dynamic_ncols=True)
        stop_training = False

        # Lazy-init the training CUDA graph runner. After init, run() returns
        # None for any minibatch whose combat/terrain dims exceed the captured
        # buckets — caller falls back to the eager path below.
        self._ensure_train_graph_runner()
        train_runner = self._train_graph_runner

        for _ in range(cfg.train_iters):
            if stop_training:
                break
            chunk_indices = np.random.permutation(total_chunks)
            iter_kl_sum = 0.0
            iter_kl_n = 0

            for start in range(0, total_chunks, CPB):
                idx = chunk_indices[start:start + CPB]
                # The captured graph requires a full CPB-sized minibatch. In
                # our standard configs total_chunks is always a multiple of
                # CPB so partial minibatches don't fire; if they ever do
                # (n_envs=1 debug runs) use_graph stays False and we fall
                # through to the eager path.
                use_graph = train_runner is not None and len(idx) == CPB

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
                act_mb = {k: v[idx] for k, v in act_t.items()}

                # Horizontal-mirror augmentation (50% per minibatch). Flips
                # x-axis obs fields and swaps movement left↔right; old log_probs
                # are reused. The IS-ratio is exact when π_old is mirror-
                # equivariant — and the augmentation itself drives the policy
                # toward that equilibrium, so the approximation tightens during
                # training. GRU initial hidden state is left un-mirrored (we
                # don't have an equivariant permutation for it); the chunk
                # length L absorbs that initial-state imperfection.
                # Mirror runs OUTSIDE the captured graph (the conditional and
                # in-place tensor ops would be variable across replays).
                if np.random.rand() < 0.5:
                    obs_mb = mirror_observation(obs_mb)
                    act_mb["movement"] = mirror_movement(act_mb["movement"])

                # ---- Try captured-graph fast path. -------------------
                graph_out = None
                if use_graph:
                    graph_out = train_runner.run(
                        obs_mb, hx_mb, act_mb,
                        adv_t[idx], atk_ret_t[idx], def_ret_t[idx],
                        old_lp_t[idx], old_lp_a_t[idx],
                        valid_t[idx], committed_t[idx],
                        atk_var_t[idx], def_var_t[idx],
                    )

                if graph_out is not None:
                    # Forward + loss + backward + clip + step all happened
                    # inside the replay + eager optim.step. Read scalar
                    # outputs; one implicit sync per minibatch via .item().
                    surrogate_val = graph_out["surrogate"].item()
                    value_atk_val = graph_out["value_atk"].item()
                    value_def_val = graph_out["value_def"].item()
                    entropy_val = graph_out["entropy"].item()
                    kl_val = graph_out["kl"].item()
                    gru_norm_val = graph_out["gru_norm"].item()

                    total_metrics["surrogate"] += surrogate_val
                    total_metrics["value_atk"] += value_atk_val
                    total_metrics["value_def"] += value_def_val
                    total_metrics["entropy"] += entropy_val
                    total_metrics["gru_norm"] += gru_norm_val
                    total_metrics["kl"] += kl_val
                    iter_kl_sum += kl_val
                    iter_kl_n += 1
                    n_updates += 1

                    passes_done += len(idx) * L
                    pbar.update(len(idx) * L)
                    pbar.set_postfix_str(f"surr={surrogate_val:+.3f} kl={kl_val:.3f}")

                    if (
                        cfg.target_kl
                        and iter_kl_n >= 2
                        and (iter_kl_sum / iter_kl_n) > cfg.target_kl
                    ):
                        stop_training = True
                        break
                    continue
                # ---- End graph fast path; eager fallback below. -------

                if _cuda_ok:
                    _fwd_s = torch.cuda.Event(enable_timing=True)
                    _fwd_e = torch.cuda.Event(enable_timing=True)
                    _fwd_s.record()
                (new_lp, entropy, v_atk, v_def, gru_info,
                 new_lp_a, ent_a) = self.policy.forward_sequence(
                    obs_mb, hx_mb, act_mb,
                )
                if _cuda_ok:
                    _fwd_e.record()
                    _fwd_evt_pairs.append((_fwd_s, _fwd_e))

                # Flatten (B, L) -> (B*L,) for loss
                new_lp_flat = new_lp.reshape(-1)
                new_lp_a_flat = new_lp_a.reshape(-1)
                entropy_flat = entropy.reshape(-1)
                ent_a_flat = ent_a.reshape(-1)
                v_atk_flat = v_atk.reshape(-1)
                v_def_flat = v_def.reshape(-1)
                adv_flat = adv_t[idx].reshape(-1)
                atk_ret_flat = atk_ret_t[idx].reshape(-1)
                def_ret_flat = def_ret_t[idx].reshape(-1)
                old_lp_flat = old_lp_t[idx].reshape(-1)
                old_lp_a_flat = old_lp_a_t[idx].reshape(-1)
                valid_flat = valid_t[idx].reshape(-1)
                committed_flat = committed_t[idx].reshape(-1)
                valid_sum = valid_flat.sum().clamp(min=1.0)
                # Per-sample value-loss variance denominators. All L steps in
                # a chunk share a boss → atk_var_t/def_var_t are (total_chunks,)
                # and broadcast over L.
                atk_var_flat = atk_var_t[idx].unsqueeze(-1).expand(-1, v_atk.shape[-1]).reshape(-1)
                def_var_flat = def_var_t[idx].unsqueeze(-1).expand(-1, v_def.shape[-1]).reshape(-1)

                # Hard-commit masking: on committed steps, the action head's
                # log_prob and entropy contributions are subtracted out so the
                # PPO ratio and entropy bonus are computed over m + d + j only.
                # Movement/direction/jump heads stay free and get normal gradient.
                new_lp_eff = new_lp_flat - committed_flat * new_lp_a_flat
                old_lp_eff = old_lp_flat - committed_flat * old_lp_a_flat
                entropy_eff = entropy_flat - committed_flat * ent_a_flat

                log_ratio = new_lp_eff - old_lp_eff
                ratio = torch.exp(log_ratio)
                clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
                surrogate_per = -torch.min(ratio * adv_flat, clipped * adv_flat)
                surrogate = (surrogate_per * valid_flat).sum() / valid_sum

                # Raw squared error for logging (dashboard continuity); the
                # actual loss divides per-sample by that sample's boss-specific
                # EMA return variance so atk/def — and per-boss return scales —
                # all contribute comparable magnitudes to the gradient.
                atk_vloss = (v_atk_flat - atk_ret_flat).pow(2)
                def_vloss = (v_def_flat - def_ret_flat).pow(2)
                value_loss = (
                    (atk_vloss * valid_flat / atk_var_flat).sum() / valid_sum
                    + (def_vloss * valid_flat / def_var_flat).sum() / valid_sum
                )

                entropy_loss = -(entropy_eff * valid_flat).sum() / valid_sum

                loss = (
                    surrogate
                    + cfg.value_coeff * value_loss
                    + cfg.entropy_coeff * entropy_loss
                )

                if _cuda_ok:
                    _bwd_s = torch.cuda.Event(enable_timing=True)
                    _bwd_e = torch.cuda.Event(enable_timing=True)
                    _bwd_s.record()
                # set_to_none=False so grad tensor addresses stay stable
                # across replays of the captured training graph (when the
                # eager path runs interleaved with graph replays).
                self.optimizer.zero_grad(set_to_none=False)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                if _cuda_ok:
                    _bwd_e.record()
                    _bwd_evt_pairs.append((_bwd_s, _bwd_e))

                n_updates += 1
                total_metrics["surrogate"] += surrogate.item()
                total_metrics["value_atk"] += ((atk_vloss * valid_flat).sum() / valid_sum).item()
                total_metrics["value_def"] += ((def_vloss * valid_flat).sum() / valid_sum).item()
                total_metrics["entropy"] += entropy_loss.item()
                total_metrics["gru_norm"] += gru_info["gru_norm"]

                with torch.no_grad():
                    kl = (((ratio - 1) - log_ratio) * valid_flat).sum() / valid_sum
                    kl_val = kl.item()
                    total_metrics["kl"] += kl_val
                    iter_kl_sum += kl_val
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
        train_phase_t["train_loop"] = _time.perf_counter() - _t_phase
        # cuda.Event.elapsed_time returns ms and requires the recorded events
        # to have completed — sync once here so the summation below is safe
        # whether or not the caller syncs.
        if _cuda_ok and (_fwd_evt_pairs or _bwd_evt_pairs):
            torch.cuda.synchronize()
            train_phase_t["forward_seq"] = sum(
                s.elapsed_time(e) for s, e in _fwd_evt_pairs) / 1000.0
            train_phase_t["backward_optim"] = sum(
                s.elapsed_time(e) for s, e in _bwd_evt_pairs) / 1000.0
        else:
            train_phase_t["forward_seq"] = 0.0
            train_phase_t["backward_optim"] = 0.0

        out = {k: v / max(n_updates, 1) for k, v in total_metrics.items()}
        out["ev_atk"] = ev_atk
        out["ev_def"] = ev_def
        out["pass_frac"] = passes_done / max(total_passes, 1)
        out["adv_std_raw"] = adv_std_raw
        out["atk_return_var"] = atk_var
        out["def_return_var"] = def_var
        out["train_phase_t"] = train_phase_t
        return out

    def set_lr(self, lr: float):
        """Manually set the optimizer LR. Used by train.py for step-based
        linear annealing (replacing the old LambdaLR scheduler)."""
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def save_checkpoint(self, path, vocab=None, boss_state=None, env_steps=None):
        # Serialize per-boss curriculum state: D plus the raw rolling windows
        # so resume can continue the EMA without a warm-up gap.
        ckpt_boss = None
        if boss_state is not None:
            ckpt_boss = {
                b: {
                    "D": float(s["D"]),
                    "landed_window": list(s["landed_window"]),
                    "taken_window": list(s["taken_window"]),
                    "atk_var_ema": (None if s.get("atk_var_ema") is None
                                    else float(s["atk_var_ema"])),
                    "def_var_ema": (None if s.get("def_var_ema") is None
                                    else float(s["def_var_ema"])),
                }
                for b, s in boss_state.items()
            }
        torch.save(
            {
                "model": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "obs_normalizer": self.obs_normalizer.state_dict(),
                "combat_normalizer": self.combat_normalizer.state_dict(),
                "terrain_normalizer": self.terrain_normalizer.state_dict(),
                "hx": self.hx,
                "kind_vocab": vocab.state_dict() if vocab is not None else None,
                "boss_state": ckpt_boss,
                "env_steps": env_steps,
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
        start_env_steps = 0
        ckpt_env_steps = ckpt.get("env_steps")
        if ckpt_env_steps is not None:
            start_env_steps = int(ckpt_env_steps)
            print(f"  Resuming at env_steps={start_env_steps}")
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
                    # var EMA fields may be missing on pre-PopArt checkpoints; leave None.
                    if s.get("atk_var_ema") is not None:
                        boss_state[b]["atk_var_ema"] = float(s["atk_var_ema"])
                    if s.get("def_var_ema") is not None:
                        boss_state[b]["def_var_ema"] = float(s["def_var_ema"])
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
        return start_env_steps
