"""Bucketed CUDA-graph runner for collect_action.

Captures one CUDA graph per (max_combat, max_terrain) bucket. Each graph
bundles three things:
  1) pinned-host -> GPU h2d copies
  2) get_action_and_value forward pass
  3) GPU -> pinned-host d2h copies of all 10 outputs

Caller hands in numpy arrays + numpy hx; receives numpy outputs. Every
captured graph reads from a fixed set of pinned host buffers, so the
caller fills those buffers (CPU memcpy, microseconds) and replays.

Padding semantics: combat/terrain dims are ceiling-padded to the next
bucket and zero-filled. Mask columns for pad slots are 0 so masked
attention ignores them. kind_ids/parent_ids 0 == kind_embed.padding_idx
which already maps to a zero embedding.

Why this works with continuous training:
  - graph captures *kernel launches with addresses*, not values
  - optimizer.step() mutates Parameter.data IN PLACE (Adam uses
    addcdiv_ etc), so the data pointer stays the same
  - Categorical.sample's CUDA RNG advances per replay automatically
"""
import torch
import numpy as np

from env.observation import Observation


class BucketedGraphRunner:
    """Bucketed CUDA-graph runner with pinned-host I/O.

    Captures one CUDA graph per (max_combat, max_terrain) bucket.
    `run_numpy(np_obs, np_hx)` returns numpy outputs in ~0.5 ms
    (vs ~6 ms eager) for a B=8 model on a 4080.

    Constraints:
      - The *numpy batch dim* (B) must equal the captured B at construction
        time. Pad with zero-mask slots for envs you want to ignore.
      - Inputs combat dim <= max(combat_buckets), same for terrain.
      - validate_args=False on Categorical (already done in model.py) is
        REQUIRED — capture rejects host-syncing ops.
    """

    def __init__(self, model, B, combat_buckets, terrain_buckets,
                 cfg, device, warmup=20):
        self.model = model
        self.B = B
        self.combat_buckets = sorted(combat_buckets)
        self.terrain_buckets = sorted(terrain_buckets)
        self.cfg = cfg
        self.device = device
        self.gru_dim = model.gru.hidden_size
        self.graphs = {}
        for nc in self.combat_buckets:
            for nt in self.terrain_buckets:
                self.graphs[(nc, nt)] = self._capture_one(nc, nt, warmup)

    @staticmethod
    def _pin(shape, dtype=torch.float32):
        return torch.zeros(shape, dtype=dtype, pin_memory=True)

    def _capture_one(self, n_combat, n_terrain, warmup):
        device, B, cfg = self.device, self.B, self.cfg
        h_in = {
            "combat_hb": self._pin((B, n_combat, cfg.combat_feature_dim)),
            "combat_mask": self._pin((B, n_combat)),
            "combat_kind_ids": self._pin((B, n_combat), torch.int64),
            "combat_parent_ids": self._pin((B, n_combat), torch.int64),
            "terrain_hb": self._pin((B, n_terrain, cfg.terrain_feature_dim)),
            "terrain_mask": self._pin((B, n_terrain)),
            "global_state": self._pin((B, cfg.global_state_dim)),
            "hx": self._pin((B, self.gru_dim)),
        }
        d_in = {k: torch.zeros_like(v, device=device) for k, v in h_in.items()}

        def gpu_obs_view():
            return Observation(
                combat_hb=d_in["combat_hb"],
                combat_mask=d_in["combat_mask"],
                combat_kind_ids=d_in["combat_kind_ids"],
                combat_parent_ids=d_in["combat_parent_ids"],
                terrain_hb=d_in["terrain_hb"],
                terrain_mask=d_in["terrain_mask"],
                global_state=d_in["global_state"],
            )

        # Warmup on side stream so cuDNN/cuBLAS algo selection settles
        # before capture. Reuse last warmup output to learn output shapes.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.no_grad():
                for _ in range(warmup):
                    out = self.model.get_action_and_value(gpu_obs_view(), hx=d_in["hx"])
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        actions_d, lp, _ent, v_atk, v_def, hx_new, lp_a, ent_a = out
        out_shapes = {
            "act_movement": (actions_d["movement"].shape, actions_d["movement"].dtype),
            "act_direction": (actions_d["direction"].shape, actions_d["direction"].dtype),
            "act_action": (actions_d["action"].shape, actions_d["action"].dtype),
            "act_jump": (actions_d["jump"].shape, actions_d["jump"].dtype),
            "log_prob": (lp.shape, lp.dtype),
            "value_atk": (v_atk.shape, v_atk.dtype),
            "value_def": (v_def.shape, v_def.dtype),
            "hx_new": (hx_new.shape, hx_new.dtype),
            "log_prob_action": (lp_a.shape, lp_a.dtype),
            "entropy_action": (ent_a.shape, ent_a.dtype),
        }
        h_out = {
            k: torch.empty(shape, dtype=dtype, pin_memory=True)
            for k, (shape, dtype) in out_shapes.items()
        }

        g = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(g):
                for k, host in h_in.items():
                    d_in[k].copy_(host, non_blocking=True)
                actions_d, lp, _ent, v_atk, v_def, hx_new, lp_a, ent_a = (
                    self.model.get_action_and_value(gpu_obs_view(), hx=d_in["hx"])
                )
                h_out["act_movement"].copy_(actions_d["movement"], non_blocking=True)
                h_out["act_direction"].copy_(actions_d["direction"], non_blocking=True)
                h_out["act_action"].copy_(actions_d["action"], non_blocking=True)
                h_out["act_jump"].copy_(actions_d["jump"], non_blocking=True)
                h_out["log_prob"].copy_(lp, non_blocking=True)
                h_out["value_atk"].copy_(v_atk, non_blocking=True)
                h_out["value_def"].copy_(v_def, non_blocking=True)
                h_out["hx_new"].copy_(hx_new, non_blocking=True)
                h_out["log_prob_action"].copy_(lp_a, non_blocking=True)
                h_out["entropy_action"].copy_(ent_a, non_blocking=True)

        return {
            "graph": g,
            "h_in": h_in,
            "h_out": h_out,
            "n_combat": n_combat,
            "n_terrain": n_terrain,
        }

    def _pick_bucket(self, n_combat, n_terrain):
        c = next((b for b in self.combat_buckets if b >= n_combat), None)
        t = next((b for b in self.terrain_buckets if b >= n_terrain), None)
        if c is None or t is None:
            raise ValueError(
                f"input combat={n_combat} terrain={n_terrain} exceeds biggest "
                f"bucket ({self.combat_buckets[-1]}, {self.terrain_buckets[-1]}). "
                f"Widen graph_terrain_buckets / graph_combat_buckets in Config."
            )
        return c, t

    def run_numpy(self, np_obs, np_hx):
        """Run a step from numpy inputs, returning numpy outputs.

        np_obs.* leading dim must equal self.B; if you have fewer real envs,
        pad with zero-masked rows and slice the output yourself.

        Returns (out_dict, bucket) where out_dict has ALL outputs as numpy
        arrays. Returned arrays alias the underlying pinned buffers; copy
        them out before the next run_numpy() call if you need persistence.
        """
        n_combat_in = np_obs.combat_hb.shape[1]
        n_terrain_in = np_obs.terrain_hb.shape[1]
        c, t = self._pick_bucket(n_combat_in, n_terrain_in)
        slot = self.graphs[(c, t)]
        h_in = slot["h_in"]

        # CPU memcpy numpy -> pinned host. We use tensor.numpy() to get a
        # zero-copy view that supports slice assignment.
        h_in["combat_hb"].numpy()[:, :n_combat_in] = np_obs.combat_hb
        h_in["combat_mask"].numpy()[:, :n_combat_in] = np_obs.combat_mask
        h_in["combat_kind_ids"].numpy()[:, :n_combat_in] = np_obs.combat_kind_ids
        h_in["combat_parent_ids"].numpy()[:, :n_combat_in] = np_obs.combat_parent_ids
        h_in["terrain_hb"].numpy()[:, :n_terrain_in] = np_obs.terrain_hb
        h_in["terrain_mask"].numpy()[:, :n_terrain_in] = np_obs.terrain_mask
        h_in["global_state"].numpy()[:] = np_obs.global_state
        h_in["hx"].numpy()[:] = np_hx
        if n_combat_in < c:
            h_in["combat_hb"].numpy()[:, n_combat_in:] = 0
            h_in["combat_mask"].numpy()[:, n_combat_in:] = 0
            h_in["combat_kind_ids"].numpy()[:, n_combat_in:] = 0
            h_in["combat_parent_ids"].numpy()[:, n_combat_in:] = 0
        if n_terrain_in < t:
            h_in["terrain_hb"].numpy()[:, n_terrain_in:] = 0
            h_in["terrain_mask"].numpy()[:, n_terrain_in:] = 0

        slot["graph"].replay()
        torch.cuda.synchronize()

        h_out = slot["h_out"]
        return {k: v.numpy() for k, v in h_out.items()}, (c, t)
