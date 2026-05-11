"""Stage 5 integration smoke test: PPO.collect_action with and without
CUDA graphs on synthetic observations. Compares deterministic outputs and
times both paths.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch

from config import Config
from ppo import PPO
from env.observation import Observation


def make_np_obs(n_envs, n_combat, n_terrain, cfg):
    rng = np.random.default_rng(0)
    return Observation(
        combat_hb=rng.standard_normal((n_envs, n_combat, cfg.combat_feature_dim)).astype(np.float32),
        combat_mask=np.ones((n_envs, n_combat), dtype=np.float32),
        combat_kind_ids=rng.integers(0, 32, (n_envs, n_combat)).astype(np.int64),
        combat_parent_ids=rng.integers(0, 32, (n_envs, n_combat)).astype(np.int64),
        terrain_hb=rng.standard_normal((n_envs, n_terrain, cfg.terrain_feature_dim)).astype(np.float32),
        terrain_mask=np.ones((n_envs, n_terrain), dtype=np.float32),
        global_state=(rng.standard_normal((n_envs, cfg.global_state_dim)) * 0.5).astype(np.float32),
    )


def main():
    n_envs = 8
    n_combat, n_terrain = 12, 70

    # Build two PPOs with the SAME initial weights — copy state dict from
    # one to the other so the comparison is apples-to-apples.
    cfg = Config()

    ppo_eager = PPO(cfg)
    ppo_eager.reset_hidden(n_envs)

    cfg_graph = Config()
    cfg_graph.use_cuda_graphs = True
    ppo_graph = PPO(cfg_graph)
    ppo_graph.policy.load_state_dict(ppo_eager.policy.state_dict())
    ppo_graph.reset_hidden(n_envs)

    obs_a = make_np_obs(n_envs, n_combat, n_terrain, cfg)

    # First call: eager
    actions_e, lp_e, lp_a_e, va_e, vd_e = ppo_eager.collect_action(obs_a)
    # First call: graph (also captures graphs lazily)
    actions_g, lp_g, lp_a_g, va_g, vd_g = ppo_graph.collect_action(obs_a)

    # Deterministic outputs should match within fp tolerance.
    diff_va = float(np.abs(va_e - va_g).max())
    diff_vd = float(np.abs(vd_e - vd_g).max())
    diff_hx = float(np.abs(ppo_eager.hx - ppo_graph.hx).max())
    print(f"  full-batch (B={n_envs}) diffs: v_atk={diff_va:.2e} "
          f"v_def={diff_vd:.2e} hx={diff_hx:.2e}")

    # Subset call (mimicking active_envs in async resets).
    obs_b_full = make_np_obs(n_envs, n_combat, n_terrain, cfg)
    active = [0, 2, 5, 7]  # 4 of 8 envs active
    obs_sub = Observation(
        combat_hb=obs_b_full.combat_hb[active],
        combat_mask=obs_b_full.combat_mask[active],
        combat_kind_ids=obs_b_full.combat_kind_ids[active],
        combat_parent_ids=obs_b_full.combat_parent_ids[active],
        terrain_hb=obs_b_full.terrain_hb[active],
        terrain_mask=obs_b_full.terrain_mask[active],
        global_state=obs_b_full.global_state[active],
    )

    # Snapshot hx so both PPOs see the same starting state.
    hx_pre = ppo_eager.hx.copy()
    ppo_graph.hx = hx_pre.copy()

    res_e = ppo_eager.collect_action(obs_sub, env_indices=active)
    res_g = ppo_graph.collect_action(obs_sub, env_indices=active)

    diff_va_s = float(np.abs(res_e[3] - res_g[3]).max())
    diff_vd_s = float(np.abs(res_e[4] - res_g[4]).max())
    # hx for active envs should match.
    diff_hx_s = float(np.abs(ppo_eager.hx[active] - ppo_graph.hx[active]).max())
    print(f"  subset (B_active={len(active)} of {n_envs}) diffs: "
          f"v_atk={diff_va_s:.2e} v_def={diff_vd_s:.2e} hx={diff_hx_s:.2e}")

    # Timing: 200 calls each.
    N_ITERS = 200
    obs_t = make_np_obs(n_envs, n_combat, n_terrain, cfg)

    for _ in range(20):
        ppo_eager.collect_action(obs_t)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        ppo_eager.collect_action(obs_t)
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) / N_ITERS * 1000

    for _ in range(20):
        ppo_graph.collect_action(obs_t)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        ppo_graph.collect_action(obs_t)
    torch.cuda.synchronize()
    graph_ms = (time.perf_counter() - t0) / N_ITERS * 1000

    pass_corr = max(diff_va, diff_vd, diff_hx, diff_va_s, diff_vd_s, diff_hx_s) < 1e-4
    print(f"S5 PPO.collect_action: eager {eager_ms:.2f}ms -> graph "
          f"{graph_ms:.3f}ms ({eager_ms/graph_ms:.1f}x)  "
          f"[{'PASS' if pass_corr else 'FAIL — outputs diverge'}]")


if __name__ == "__main__":
    main()
