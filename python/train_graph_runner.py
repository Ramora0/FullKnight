"""Bucketed CUDA-graph runner for the PPO training inner loop.

Captures forward_sequence + loss + backward + clip_grad_norm + optimizer.step
as a single CUDA graph per (combat_bucket, terrain_bucket) pair. Replay
costs ~kernel launch * 1 instead of ~kernel launch * O(layers) per minibatch
in eager mode.

Why this works:
  - The training pass has fixed shapes: (CPB, L, ...) per minibatch.
  - Adam with capturable=True keeps step state on GPU and is replay-safe.
  - clip_grad_norm_ uses _foreach_* ops; graph-compatible.
  - All Categorical distributions in the policy already pass
    validate_args=False (required: capture rejects host-syncing ops).
  - gru_info["gru_norm"] is now a tensor (model.py forward_sequence) so
    we can copy it into the static output buffer instead of .item()ing.

Caveats:
  - LR is baked into the captured optim.step(). A `set_lr` between replays
    has NO effect on the captured graph. Caller should be aware (this
    project's training script anneals LR ~5% over 800 epochs which is
    noise-level; we accept the bake-in).
  - Mirror augmentation MUST happen on the source minibatch tensors before
    we copy them into the static buffers — the mirror op itself is not
    inside the graph.
  - Combat / terrain dims are ceiling-padded to the next available bucket
    and zero-filled; padded slots have mask=0 so attention ignores them.
"""
import torch
import torch.nn as nn

from observation import Observation


class BucketedTrainGraphRunner:
    def __init__(self, policy, optimizer, config, device,
                 combat_buckets, terrain_buckets, warmup=10):
        self.policy = policy
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.combat_buckets = sorted(combat_buckets)
        self.terrain_buckets = sorted(terrain_buckets)
        self.gru_dim = policy.gru.hidden_size
        self.graphs = {}
        for nc in self.combat_buckets:
            for nt in self.terrain_buckets:
                self.graphs[(nc, nt)] = self._capture(nc, nt, warmup)

    def _make_input_buffers(self, n_combat, n_terrain):
        cfg, device = self.config, self.device
        CPB, L = cfg.chunks_per_batch, cfg.seq_len
        f32 = torch.float32
        i64 = torch.int64
        z = lambda shape, dtype=f32: torch.zeros(shape, dtype=dtype, device=device)
        return {
            "combat_hb":         z((CPB, L, n_combat, cfg.combat_feature_dim)),
            "combat_mask":       z((CPB, L, n_combat)),
            "combat_kind_ids":   z((CPB, L, n_combat), i64),
            "combat_parent_ids": z((CPB, L, n_combat), i64),
            "terrain_hb":        z((CPB, L, n_terrain, cfg.terrain_feature_dim)),
            "terrain_mask":      z((CPB, L, n_terrain)),
            "global_state":      z((CPB, L, cfg.global_state_dim)),
            "hx":                z((CPB, self.gru_dim)),
            "act_movement":      z((CPB, L), i64),
            "act_direction":     z((CPB, L), i64),
            "act_action":        z((CPB, L), i64),
            "act_jump":          z((CPB, L), i64),
            "adv":               z((CPB, L)),
            "atk_ret":           z((CPB, L)),
            "def_ret":           z((CPB, L)),
            "old_lp":            z((CPB, L)),
            "old_lp_a":          z((CPB, L)),
            "valid":             z((CPB, L)),
            "committed":         z((CPB, L)),
            # Per-rollout scalars passed in as 0-d GPU tensors; updated
            # in-place via .fill_() before each replay so capture sees a
            # tensor read, not a Python float.
            "atk_var_eff":       torch.ones((), dtype=f32, device=device),
            "def_var_eff":       torch.ones((), dtype=f32, device=device),
        }

    @staticmethod
    def _make_output_buffers(device):
        z = lambda: torch.zeros((), dtype=torch.float32, device=device)
        return {
            "surrogate":   z(),
            "value_atk":   z(),
            "value_def":   z(),
            "entropy":     z(),
            "kl":          z(),
            "gru_norm":    z(),
        }

    def _step_fn(self, d_in, d_out):
        cfg = self.config

        def step():
            obs = Observation(
                combat_hb=d_in["combat_hb"],
                combat_mask=d_in["combat_mask"],
                combat_kind_ids=d_in["combat_kind_ids"],
                combat_parent_ids=d_in["combat_parent_ids"],
                terrain_hb=d_in["terrain_hb"],
                terrain_mask=d_in["terrain_mask"],
                global_state=d_in["global_state"],
            )
            actions = {
                "movement":  d_in["act_movement"],
                "direction": d_in["act_direction"],
                "action":    d_in["act_action"],
                "jump":      d_in["act_jump"],
            }

            (new_lp, entropy, v_atk, v_def, gru_info,
             new_lp_a, ent_a) = self.policy.forward_sequence(
                obs, d_in["hx"], actions,
            )

            new_lp_flat = new_lp.reshape(-1)
            new_lp_a_flat = new_lp_a.reshape(-1)
            entropy_flat = entropy.reshape(-1)
            ent_a_flat = ent_a.reshape(-1)
            v_atk_flat = v_atk.reshape(-1)
            v_def_flat = v_def.reshape(-1)
            adv_flat = d_in["adv"].reshape(-1)
            atk_ret_flat = d_in["atk_ret"].reshape(-1)
            def_ret_flat = d_in["def_ret"].reshape(-1)
            old_lp_flat = d_in["old_lp"].reshape(-1)
            old_lp_a_flat = d_in["old_lp_a"].reshape(-1)
            valid_flat = d_in["valid"].reshape(-1)
            committed_flat = d_in["committed"].reshape(-1)
            valid_sum = valid_flat.sum().clamp(min=1.0)

            new_lp_eff = new_lp_flat - committed_flat * new_lp_a_flat
            old_lp_eff = old_lp_flat - committed_flat * old_lp_a_flat
            entropy_eff = entropy_flat - committed_flat * ent_a_flat

            log_ratio = new_lp_eff - old_lp_eff
            ratio = torch.exp(log_ratio)
            clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
            surrogate_per = -torch.min(ratio * adv_flat, clipped * adv_flat)
            surrogate = (surrogate_per * valid_flat).sum() / valid_sum

            atk_vloss = (v_atk_flat - atk_ret_flat).pow(2)
            def_vloss = (v_def_flat - def_ret_flat).pow(2)
            value_loss = (
                (atk_vloss * valid_flat).sum() / (valid_sum * d_in["atk_var_eff"])
                + (def_vloss * valid_flat).sum() / (valid_sum * d_in["def_var_eff"])
            )

            entropy_loss = -(entropy_eff * valid_flat).sum() / valid_sum

            loss = (
                surrogate
                + cfg.value_coeff * value_loss
                + cfg.entropy_coeff * entropy_loss
            )

            # set_to_none=False keeps the gradient tensors alive across
            # replays — required for graph capture, since we need the same
            # tensor addresses each time. set_to_none=True would deallocate
            # them, forcing backward to reallocate (variable addresses).
            self.optimizer.zero_grad(set_to_none=False)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
            self.optimizer.step()

            with torch.no_grad():
                kl = (((ratio - 1) - log_ratio) * valid_flat).sum() / valid_sum
                d_out["surrogate"].copy_(surrogate.detach())
                d_out["value_atk"].copy_(((atk_vloss * valid_flat).sum() / valid_sum).detach())
                d_out["value_def"].copy_(((def_vloss * valid_flat).sum() / valid_sum).detach())
                d_out["entropy"].copy_(entropy_loss.detach())
                d_out["kl"].copy_(kl.detach())
                d_out["gru_norm"].copy_(gru_info["gru_norm"].detach())

        return step

    def _capture(self, n_combat, n_terrain, warmup):
        d_in = self._make_input_buffers(n_combat, n_terrain)
        d_out = self._make_output_buffers(self.device)
        step = self._step_fn(d_in, d_out)

        # Warmup on a side stream so cuDNN/cuBLAS heuristics stabilize and
        # Adam state buffers (m, v, step) get allocated before capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()

        return {
            "graph": g,
            "d_in": d_in,
            "d_out": d_out,
            "n_combat": n_combat,
            "n_terrain": n_terrain,
        }

    def _pick_bucket(self, n_combat, n_terrain):
        c = next((b for b in self.combat_buckets if b >= n_combat), None)
        t = next((b for b in self.terrain_buckets if b >= n_terrain), None)
        if c is None or t is None:
            return None
        return (c, t)

    def run(self, obs_mb, hx_mb, act_mb,
            adv_mb, atk_ret_mb, def_ret_mb,
            old_lp_mb, old_lp_a_mb, valid_mb, committed_mb,
            atk_var_eff, def_var_eff):
        """Replay the captured graph for one minibatch.

        All inputs are GPU tensors (already on self.device). Returns the
        d_out scalar dict; caller can .item() any of them after sync.
        Returns None if the input doesn't fit any bucket — caller falls
        back to eager.
        """
        n_combat = obs_mb.combat_hb.shape[2]
        n_terrain = obs_mb.terrain_hb.shape[2]
        bucket = self._pick_bucket(n_combat, n_terrain)
        if bucket is None:
            return None
        slot = self.graphs[bucket]
        d_in = slot["d_in"]

        # Fill static input buffers. We zero the combat/terrain padding
        # slots first because masks are also copied — copy_ on slices
        # leaves the rest of the static tensor with stale data from prior
        # replays, which would survive the mask zero-out and corrupt
        # attention's softmax through the kind-embed lookup if e.g.
        # combat_kind_ids has stale nonzero ids in pad slots.
        d_in["combat_hb"].zero_()
        d_in["combat_mask"].zero_()
        d_in["combat_kind_ids"].zero_()
        d_in["combat_parent_ids"].zero_()
        d_in["terrain_hb"].zero_()
        d_in["terrain_mask"].zero_()

        d_in["combat_hb"][:, :, :n_combat].copy_(obs_mb.combat_hb)
        d_in["combat_mask"][:, :, :n_combat].copy_(obs_mb.combat_mask)
        d_in["combat_kind_ids"][:, :, :n_combat].copy_(obs_mb.combat_kind_ids)
        d_in["combat_parent_ids"][:, :, :n_combat].copy_(obs_mb.combat_parent_ids)
        d_in["terrain_hb"][:, :, :n_terrain].copy_(obs_mb.terrain_hb)
        d_in["terrain_mask"][:, :, :n_terrain].copy_(obs_mb.terrain_mask)
        d_in["global_state"].copy_(obs_mb.global_state)
        d_in["hx"].copy_(hx_mb)
        d_in["act_movement"].copy_(act_mb["movement"])
        d_in["act_direction"].copy_(act_mb["direction"])
        d_in["act_action"].copy_(act_mb["action"])
        d_in["act_jump"].copy_(act_mb["jump"])
        d_in["adv"].copy_(adv_mb)
        d_in["atk_ret"].copy_(atk_ret_mb)
        d_in["def_ret"].copy_(def_ret_mb)
        d_in["old_lp"].copy_(old_lp_mb)
        d_in["old_lp_a"].copy_(old_lp_a_mb)
        d_in["valid"].copy_(valid_mb)
        d_in["committed"].copy_(committed_mb)
        d_in["atk_var_eff"].fill_(atk_var_eff)
        d_in["def_var_eff"].fill_(def_var_eff)

        slot["graph"].replay()
        return slot["d_out"]
