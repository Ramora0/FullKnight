"""Standalone forward pass benchmark — diagnose why PPO forward is slow.

Sweeps batch size to distinguish launch-overhead-dominated (forward at B=8 ~=
forward at B=64) from compute-dominated (linear scaling with B).

Tests several variants:
  - pure forward (CUDA event, no .cpu(), no h2d, no normalize)
  - inference_mode vs no_grad
  - full round-trip (numpy->gpu->forward->.cpu()->numpy) like collect_action
  - autocast (mixed precision) at the production batch

Skips torch.compile entirely — not supported on Windows + CUDA.
"""
import time
import numpy as np
import torch

from config import Config
from model import FullKnightActorCritic
from observation import Observation, GS


def make_obs(B, n_combat, n_terrain, cfg, device):
    """Build a synthetic GPU Observation."""
    obs = Observation(
        combat_hb=torch.randn(B, n_combat, cfg.combat_feature_dim, device=device),
        combat_mask=torch.ones(B, n_combat, device=device),
        combat_kind_ids=torch.zeros(B, n_combat, dtype=torch.long, device=device),
        combat_parent_ids=torch.zeros(B, n_combat, dtype=torch.long, device=device),
        terrain_hb=torch.randn(B, n_terrain, cfg.terrain_feature_dim, device=device),
        terrain_mask=torch.ones(B, n_terrain, device=device),
        global_state=torch.randn(B, cfg.global_state_dim, device=device),
    )
    obs.global_state[:, GS.HAS_DASH:] = (obs.global_state[:, GS.HAS_DASH:] > 0).float()
    return obs


def time_pure_forward(model, obs, n_iters, warmup, ctx_factory):
    """CUDA event timing of get_action_and_value. ctx_factory is no_grad or inference_mode."""
    for _ in range(warmup):
        with ctx_factory():
            model.get_action_and_value(obs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with ctx_factory():
        for _ in range(n_iters):
            model.get_action_and_value(obs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters


def time_round_trip(model, np_obs, device, n_iters, warmup):
    """Wallclock for full numpy->gpu->forward->.cpu()->numpy cycle (like collect_action)."""
    def one_call():
        gpu_obs = Observation(
            combat_hb=torch.from_numpy(np_obs.combat_hb).float().to(device),
            combat_mask=torch.from_numpy(np_obs.combat_mask).float().to(device),
            combat_kind_ids=torch.from_numpy(np_obs.combat_kind_ids).long().to(device),
            combat_parent_ids=torch.from_numpy(np_obs.combat_parent_ids).long().to(device),
            terrain_hb=torch.from_numpy(np_obs.terrain_hb).float().to(device),
            terrain_mask=torch.from_numpy(np_obs.terrain_mask).float().to(device),
            global_state=torch.from_numpy(np_obs.global_state).float().to(device),
        )
        actions, lp, _, va, vd, _, _, _ = model.get_action_and_value(gpu_obs)
        _ = {k: v.cpu().numpy() for k, v in actions.items()}
        _ = lp.cpu().numpy()
        _ = va.cpu().numpy()
        _ = vd.cpu().numpy()

    with torch.no_grad():
        for _ in range(warmup):
            one_call()
    torch.cuda.synchronize()

    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(n_iters):
            one_call()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iters * 1000  # ms


def time_cuda_graph_stage1(model, obs, n_iters, warmup):
    """Stage 1: capture get_action_and_value at one fixed shape, replay it.

    Goal: validate that the forward (encoders + masked attention + GRU + heads
    + Categorical.sample) can be captured at all on this Win+CUDA setup, and
    confirm the launch-overhead cost actually disappears on replay.

    Returns dict with replay ms/iter, max value-head diff vs eager, and the
    captured graph + static input handles for downstream stages.
    """
    device = obs.combat_hb.device
    B = obs.global_state.shape[0]
    gru_dim = model.gru.hidden_size

    # Static input buffers — replays read the *same memory addresses* each
    # time, so production code copies new inputs into these tensors with
    # tensor.copy_(...) before calling g.replay().
    static_obs = Observation(
        combat_hb=obs.combat_hb.clone(),
        combat_mask=obs.combat_mask.clone(),
        combat_kind_ids=obs.combat_kind_ids.clone(),
        combat_parent_ids=obs.combat_parent_ids.clone(),
        terrain_hb=obs.terrain_hb.clone(),
        terrain_mask=obs.terrain_mask.clone(),
        global_state=obs.global_state.clone(),
    )
    static_hx = torch.zeros(B, gru_dim, device=device)

    # Warmup is REQUIRED before capture: cuDNN/cuBLAS pick algos on first call
    # and we don't want algo selection inside the captured graph. Run on a
    # side stream so any allocator quirks are flushed before main stream
    # capture. (PyTorch capture protocol.)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.no_grad():
            for _ in range(warmup):
                model.get_action_and_value(static_obs, hx=static_hx)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Capture the forward.
    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g):
            static_out = model.get_action_and_value(static_obs, hx=static_hx)
    # static_out tensors live in the graph's reserved memory pool — they
    # get overwritten by each replay.
    actions_d, lp, ent, v_atk, v_def, hx_new, lp_a, ent_a = static_out

    # Sanity-replay once before timing.
    g.replay()
    torch.cuda.synchronize()

    # Time replay.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    ms_replay = start.elapsed_time(end) / n_iters

    # Verify outputs match eager mode within fp tolerance, for the
    # *deterministic* outputs only. Sampling advances the CUDA RNG so the
    # sampled actions and their log_probs / entropy diverge between calls
    # — that's expected, not a bug.
    g.replay()
    torch.cuda.synchronize()
    v_atk_replay = v_atk.clone()
    v_def_replay = v_def.clone()
    hx_new_replay = hx_new.clone()

    with torch.no_grad():
        eager_out = model.get_action_and_value(static_obs, hx=static_hx)
    _, _, _, v_atk_eager, v_def_eager, hx_new_eager, _, _ = eager_out

    diff_v_atk = (v_atk_replay - v_atk_eager).abs().max().item()
    diff_v_def = (v_def_replay - v_def_eager).abs().max().item()
    diff_hx = (hx_new_replay - hx_new_eager).abs().max().item()

    return {
        "ms_replay": ms_replay,
        "diff_v_atk": diff_v_atk,
        "diff_v_def": diff_v_def,
        "diff_hx": diff_hx,
        "graph": g,
        "static_obs": static_obs,
        "static_hx": static_hx,
        "static_out": static_out,
    }


def stage2_check_rng(graph, static_out, n_replays=2000):
    """Stage 2: replay the captured graph many times, ensure Categorical.sample
    advances the CUDA RNG so we get fresh draws each replay (not a frozen
    sample baked into the graph).

    Returns (n_unique_action_tuples, expected_min_unique).
    With movement_n=3, direction_n=3, action_n=8, jump_n=2 = 144 joint outcomes
    per sample, B=8 independent samples, 2000 replays we should see hundreds
    of unique tuples; a frozen RNG would give exactly 1.
    """
    actions_d = static_out[0]
    seen = set()
    for _ in range(n_replays):
        graph.replay()
        # Read a per-replay action signature off the static output buffers.
        # CPU-fetch is unavoidable to enumerate uniqueness, but we only do it
        # for this validation, not in production.
        sig = (
            tuple(actions_d["movement"].cpu().tolist())
            + tuple(actions_d["direction"].cpu().tolist())
            + tuple(actions_d["action"].cpu().tolist())
            + tuple(actions_d["jump"].cpu().tolist())
        )
        seen.add(sig)
    return len(seen), n_replays


def stage3_check_weight_update(model, graph, static_out, static_obs, static_hx):
    """Stage 3: mutate weights in-place after capture, ensure replay reflects
    the new weights. Validates that PPO's optimizer.step() (which is
    in-place) keeps the captured graph valid.

    Returns dict with diff_after_perturb (should be > 0) and
    diff_after_restore (should be 0).
    """
    v_atk = static_out[3]

    # Snapshot output with original weights.
    graph.replay()
    torch.cuda.synchronize()
    v_orig = v_atk.clone()

    # Mutate a weight in-place (mimics optimizer.step()'s param.data.add_).
    # critic_attack's first Linear directly drives v_atk, so we'll see the
    # diff cleanly. Adam mutates param.data in place via addcdiv_; the
    # tensor *address* doesn't change, only its values, so the captured
    # graph reads the new values on next replay.
    target = model.critic_attack[0].weight
    snapshot = target.data.clone()
    target.data.add_(torch.randn_like(target.data) * 0.1)

    graph.replay()
    torch.cuda.synchronize()
    v_perturbed = v_atk.clone()
    diff_after_perturb = (v_perturbed - v_orig).abs().max().item()

    # Restore weights, replay again — should match original.
    target.data.copy_(snapshot)
    graph.replay()
    torch.cuda.synchronize()
    v_restored = v_atk.clone()
    diff_after_restore = (v_restored - v_orig).abs().max().item()

    return {
        "diff_after_perturb": diff_after_perturb,
        "diff_after_restore": diff_after_restore,
    }


class BucketedGraphRunner:
    """Bucketed CUDA-graph runner with pinned-host I/O.

    Per bucket (max_combat, max_terrain) we capture a single graph that
    bundles three things:
      1) pinned host input -> GPU static input   (h2d copy)
      2) get_action_and_value forward
      3) GPU output       -> pinned host output  (d2h copy)

    Padding semantics: combat/terrain dims are zero-extended; mask
    columns for pad slots are 0 so masked attention ignores them.
    Kind/parent ids are 0 (== kind_embed.padding_idx) → zero embedding.

    Why pinned: pageable→GPU and GPU→pageable copies aren't safely
    capturable, but pinned↔GPU is. Pinning also lets the CPU memcpy
    we do BEFORE replay run at full bandwidth (~16 GB/s).
    """

    # Output keys we read back from each replay (and their owning index in
    # the get_action_and_value tuple):
    #   actions (movement, direction, action, jump), log_prob, value_atk,
    #   value_def, hx_new, log_prob_action, entropy_action.
    # entropy (idx 2) is unused by collect_action so we skip d2h on it.

    def __init__(self, model, B, combat_buckets, terrain_buckets, cfg, device, warmup=30):
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

        # Pinned host input buffers — addresses fixed for the lifetime of
        # this bucket. Caller writes into these (CPU memcpy) before replay.
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

        # GPU static input buffers — captured graph reads from these.
        d_in = {
            "combat_hb": torch.zeros_like(h_in["combat_hb"], device=device),
            "combat_mask": torch.zeros_like(h_in["combat_mask"], device=device),
            "combat_kind_ids": torch.zeros_like(h_in["combat_kind_ids"], device=device),
            "combat_parent_ids": torch.zeros_like(h_in["combat_parent_ids"], device=device),
            "terrain_hb": torch.zeros_like(h_in["terrain_hb"], device=device),
            "terrain_mask": torch.zeros_like(h_in["terrain_mask"], device=device),
            "global_state": torch.zeros_like(h_in["global_state"], device=device),
            "hx": torch.zeros_like(h_in["hx"], device=device),
        }

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

        # Warmup on a side stream — required before capture so cuDNN/cuBLAS
        # algo selection is settled. Reuse the final warmup output to learn
        # output shapes for pre-allocating pinned host out-bufs (cannot
        # allocate pinned memory mid-capture).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.no_grad():
                for _ in range(warmup):
                    out = self.model.get_action_and_value(gpu_obs_view(), hx=d_in["hx"])
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        actions_d, lp, ent, v_atk, v_def, hx_new, lp_a, ent_a = out
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
                # 1) h2d: pinned host -> GPU static input.
                for k, host in h_in.items():
                    d_in[k].copy_(host, non_blocking=True)

                # 2) forward.
                actions_d, lp, ent, v_atk, v_def, hx_new, lp_a, ent_a = (
                    self.model.get_action_and_value(gpu_obs_view(), hx=d_in["hx"])
                )

                # 3) d2h: GPU outputs -> pre-allocated pinned host bufs.
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
                f"bucket ({self.combat_buckets[-1]}, {self.terrain_buckets[-1]})"
            )
        return c, t

    def run_numpy(self, np_obs, np_hx):
        """Run a step from numpy inputs, returning numpy outputs.

        np_obs: Observation with numpy fields, leading dim (B, ...).
        np_hx:  (B, gru_dim) float32 numpy array.

        Returns dict with all 10 outputs as numpy arrays. After return, the
        underlying pinned buffers are reused on the next call — copy out if
        you need persistence.
        """
        n_combat_in = np_obs.combat_hb.shape[1]
        n_terrain_in = np_obs.terrain_hb.shape[1]
        c, t = self._pick_bucket(n_combat_in, n_terrain_in)
        slot = self.graphs[(c, t)]
        h_in = slot["h_in"]

        # Stage data into pinned host bufs. Use slice assignment so unfilled
        # pad slots stay zero (they were allocated zero, and we only ever
        # write into [:, :n_*_in]; previous-call data in pad region is fine
        # because mask is also zero there).
        # NOTE: numpy <-> pinned-tensor memcpy avoids any Python-side
        # autograd/dtype overhead since we use tensor.numpy() views.
        h_in["combat_hb"].numpy()[:, :n_combat_in] = np_obs.combat_hb
        h_in["combat_mask"].numpy()[:, :n_combat_in] = np_obs.combat_mask
        h_in["combat_kind_ids"].numpy()[:, :n_combat_in] = np_obs.combat_kind_ids
        h_in["combat_parent_ids"].numpy()[:, :n_combat_in] = np_obs.combat_parent_ids
        h_in["terrain_hb"].numpy()[:, :n_terrain_in] = np_obs.terrain_hb
        h_in["terrain_mask"].numpy()[:, :n_terrain_in] = np_obs.terrain_mask
        h_in["global_state"].numpy()[:] = np_obs.global_state
        h_in["hx"].numpy()[:] = np_hx
        # Zero the pad region in case prior call's pad shape was larger
        # (defensive — mask is zero so it shouldn't matter).
        if n_combat_in < c:
            h_in["combat_hb"].numpy()[:, n_combat_in:] = 0
            h_in["combat_mask"].numpy()[:, n_combat_in:] = 0
            h_in["combat_kind_ids"].numpy()[:, n_combat_in:] = 0
            h_in["combat_parent_ids"].numpy()[:, n_combat_in:] = 0
        if n_terrain_in < t:
            h_in["terrain_hb"].numpy()[:, n_terrain_in:] = 0
            h_in["terrain_mask"].numpy()[:, n_terrain_in:] = 0

        slot["graph"].replay()
        torch.cuda.synchronize()  # wait for d2h to land in pinned host

        h_out = slot["h_out"]
        return {k: v.numpy() for k, v in h_out.items()}, (c, t)


def stage4_validate(runner, model, cfg, device, n_iters=200):
    """Stage 4 validation: for a few real-shaped numpy inputs, dispatch
    through the bucketed runner. Compare deterministic outputs against an
    eager round-trip baseline (numpy in, numpy out — same path as
    production). Wallclock-time the full round trip end to end.
    """
    B = runner.B
    test_shapes = [(1, 8), (5, 32), (12, 70), (20, 50)]
    eager_total = 0.0
    graph_total = 0.0
    max_diff = 0.0

    for n_combat, n_terrain in test_shapes:
        try:
            c, t = runner._pick_bucket(n_combat, n_terrain)
        except ValueError:
            continue

        obs_gpu = make_obs(B, n_combat, n_terrain, cfg, device)
        np_obs = to_np_obs(obs_gpu)
        np_hx = np.zeros((B, runner.gru_dim), dtype=np.float32)

        # Eager numpy round trip (mimics current ppo.collect_action).
        def eager_step():
            gpu_obs = Observation(
                combat_hb=torch.from_numpy(np_obs.combat_hb).to(device),
                combat_mask=torch.from_numpy(np_obs.combat_mask).to(device),
                combat_kind_ids=torch.from_numpy(np_obs.combat_kind_ids).to(device),
                combat_parent_ids=torch.from_numpy(np_obs.combat_parent_ids).to(device),
                terrain_hb=torch.from_numpy(np_obs.terrain_hb).to(device),
                terrain_mask=torch.from_numpy(np_obs.terrain_mask).to(device),
                global_state=torch.from_numpy(np_obs.global_state).to(device),
            )
            gpu_hx = torch.from_numpy(np_hx).to(device)
            with torch.no_grad():
                actions, lp, _, v_atk, v_def, hx_new, lp_a, _ = (
                    model.get_action_and_value(gpu_obs, hx=gpu_hx)
                )
            return {
                "value_atk": v_atk.cpu().numpy(),
                "value_def": v_def.cpu().numpy(),
                "hx_new": hx_new.cpu().numpy(),
            }

        for _ in range(5):
            eager_step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            eager_step()
        torch.cuda.synchronize()
        eager_ms = (time.perf_counter() - t0) / n_iters * 1000

        # Bucketed graph numpy round trip.
        for _ in range(5):
            runner.run_numpy(np_obs, np_hx)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            runner.run_numpy(np_obs, np_hx)
        torch.cuda.synchronize()
        graph_ms = (time.perf_counter() - t0) / n_iters * 1000

        # Correctness: deterministic outputs match eager.
        e = eager_step()
        out, bucket = runner.run_numpy(np_obs, np_hx)
        diff_v_atk = float(np.abs(out["value_atk"] - e["value_atk"]).max())
        diff_v_def = float(np.abs(out["value_def"] - e["value_def"]).max())
        diff_hx = float(np.abs(out["hx_new"] - e["hx_new"]).max())
        local_max = max(diff_v_atk, diff_v_def, diff_hx)
        max_diff = max(max_diff, local_max)

        eager_total += eager_ms
        graph_total += graph_ms
        print(f"  c={n_combat:>2} t={n_terrain:>2} -> bucket {bucket}: "
              f"eager {eager_ms:.2f}ms / graph {graph_ms:.3f}ms "
              f"({eager_ms/graph_ms:.1f}x)  diff={local_max:.1e}")

    return {
        "eager_avg_ms": eager_total / len(test_shapes),
        "graph_avg_ms": graph_total / len(test_shapes),
        "max_diff": max_diff,
    }


def to_np_obs(obs):
    return Observation(
        combat_hb=obs.combat_hb.cpu().numpy(),
        combat_mask=obs.combat_mask.cpu().numpy(),
        combat_kind_ids=obs.combat_kind_ids.cpu().numpy(),
        combat_parent_ids=obs.combat_parent_ids.cpu().numpy(),
        terrain_hb=obs.terrain_hb.cpu().numpy(),
        terrain_mask=obs.terrain_mask.cpu().numpy(),
        global_state=obs.global_state.cpu().numpy(),
    )


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullKnightActorCritic(cfg).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"{gpu} torch={torch.__version__} params={n_params:,}")

    N_COMBAT, N_TERRAIN, WARMUP, ITERS = 12, 70, 30, 200
    obs8 = make_obs(8, N_COMBAT, N_TERRAIN, cfg, device)
    eager_ms = time_pure_forward(model, obs8, ITERS, WARMUP, torch.no_grad)

    # --- Stage 1: capture and replay at one fixed shape ---
    try:
        r = time_cuda_graph_stage1(model, obs8, ITERS, WARMUP)
        max_diff = max(r['diff_v_atk'], r['diff_v_def'], r['diff_hx'])
        verdict = "PASS" if max_diff < 1e-4 else f"WARN diff={max_diff:.1e}"
        print(f"S1 capture/replay B=8 c={N_COMBAT} t={N_TERRAIN}: "
              f"eager {eager_ms:.2f}ms -> replay {r['ms_replay']:.3f}ms "
              f"({eager_ms/r['ms_replay']:.1f}x)  [{verdict}]")
    except Exception as e:
        print(f"S1 FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return

    # --- Stage 2: RNG advances per replay (Categorical.sample) ---
    n_unique, n_replays = stage2_check_rng(r['graph'], r['static_out'], n_replays=2000)
    s2_pass = n_unique > 50  # frozen RNG -> 1; healthy -> hundreds
    print(f"S2 RNG advance: {n_unique}/{n_replays} unique action tuples  "
          f"[{'PASS' if s2_pass else 'FAIL — RNG frozen in graph'}]")

    # --- Stage 3: in-place weight update reflected in replay output ---
    s3 = stage3_check_weight_update(model, r['graph'], r['static_out'],
                                     r['static_obs'], r['static_hx'])
    s3_pass = s3['diff_after_perturb'] > 1e-3 and s3['diff_after_restore'] < 1e-5
    print(f"S3 weight update: diff(perturbed)={s3['diff_after_perturb']:.3e} "
          f"diff(restored)={s3['diff_after_restore']:.3e}  "
          f"[{'PASS' if s3_pass else 'FAIL — weights baked in'}]")

    # --- Stage 4: bucketed dispatcher across multiple combat sizes ---
    print(f"S4 capturing buckets...", end=" ", flush=True)
    t0 = time.perf_counter()
    runner = BucketedGraphRunner(
        model, B=8,
        combat_buckets=(4, 8, 16, 32),
        terrain_buckets=(72,),  # one bucket pads everything <= 72; widen later
        cfg=cfg, device=device, warmup=20,
    )
    capture_s = time.perf_counter() - t0
    print(f"{capture_s:.1f}s for {len(runner.graphs)} graphs")
    s4 = stage4_validate(runner, model, cfg, device, n_iters=200)
    s4_pass = s4['max_diff'] < 1e-4
    print(f"S4 dispatcher: avg eager {s4['eager_avg_ms']:.2f}ms -> "
          f"graph {s4['graph_avg_ms']:.3f}ms ({s4['eager_avg_ms']/s4['graph_avg_ms']:.1f}x), "
          f"max diff {s4['max_diff']:.1e}  "
          f"[{'PASS' if s4_pass else 'FAIL'}]")


if __name__ == "__main__":
    main()
