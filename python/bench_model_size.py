"""Benchmark CUDA-graph inference time vs model size.

Captures one graph per config at (combat_bucket=8, terrain_bucket=32) and
times 500 replays at fixed input (n_combat=8, n_terrain=30) on a 4080 Super.

Sweeps (one axis at a time, baseline elsewhere):
  - hidden_dim    (trunk + critic width)
  - encoder dim   (combat/terrain/global output_dim, hidden_dim together)
  - gru_dim       (recurrent backbone width)
  - batch size B  (n_envs at capture)

Plus baseline, "big" (all bumped), "huge" (all maxed). Outputs a table to
stdout and saves a scatter chart to python/bench_model_size.png.
"""
import gc
import sys
import time

import numpy as np
import torch

from config import Config
from model import FullKnightActorCritic
from observation import Observation
from graph_runner import BucketedGraphRunner

ITERS = 500
WARMUP = 30
N_COMBAT = 8
N_TERRAIN = 30


def make_inputs(cfg, B):
    rng = np.random.default_rng(0)
    combat_hb = rng.standard_normal((B, N_COMBAT, cfg.combat_feature_dim)).astype(np.float32)
    combat_mask = np.ones((B, N_COMBAT), dtype=np.float32)
    combat_kind = rng.integers(0, cfg.kind_vocab_size, (B, N_COMBAT)).astype(np.int64)
    combat_parent = rng.integers(0, cfg.kind_vocab_size, (B, N_COMBAT)).astype(np.int64)
    terrain_hb = rng.standard_normal((B, N_TERRAIN, cfg.terrain_feature_dim)).astype(np.float32)
    terrain_mask = np.ones((B, N_TERRAIN), dtype=np.float32)
    gs = rng.standard_normal((B, cfg.global_state_dim)).astype(np.float32)
    gs[:, cfg.global_state_dim - cfg.n_binary_flags:] = 1.0
    hx = np.zeros((B, cfg.gru_dim), dtype=np.float32)
    obs = Observation(
        combat_hb=combat_hb, combat_mask=combat_mask,
        combat_kind_ids=combat_kind, combat_parent_ids=combat_parent,
        combat_kind_ids=combat_kind, combat_anim_ids=combat_parent,
        terrain_hb=terrain_hb, terrain_mask=terrain_mask, global_state=gs,
    )
    return obs, hx


def bench_one(cfg, B):
    device = torch.device("cuda")
    model = FullKnightActorCritic(cfg).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    obs, hx = make_inputs(cfg, B)
    runner = BucketedGraphRunner(
        model, B=B, combat_buckets=[8], terrain_buckets=[32],
        cfg=cfg, device=device, warmup=10,
    )
    for _ in range(WARMUP):
        runner.run_numpy(obs, hx)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        runner.run_numpy(obs, hx)
    torch.cuda.synchronize()
    ms_per = (time.perf_counter() - t0) * 1000 / ITERS

    del model, runner, obs, hx
    gc.collect()
    torch.cuda.empty_cache()
    return n_params, ms_per


def enc_kwargs(d):
    """Bundle all encoder widths (input projection + attention) together."""
    return dict(
        combat_output=d, terrain_output=d, global_output=d,
        combat_hidden=d, terrain_hidden=d, global_hidden=d,
    )


def main():
    configs = []
    configs.append(("baseline (256/64/64, B=16)", Config(n_envs=16), 16, "baseline"))

    for h in [256, 512, 1024, 2048]:
        configs.append((f"hidden_dim={h}", Config(n_envs=16, hidden_dim=h), 16, "hidden_dim"))

    for d in [64, 128, 256, 512]:
        configs.append((f"enc_out={d}", Config(n_envs=16, **enc_kwargs(d)), 16, "encoder_out"))

    for g in [64, 256, 512, 1024]:
        configs.append((f"gru_dim={g}", Config(n_envs=16, gru_dim=g), 16, "gru_dim"))

    for b in [8, 16, 32, 64]:
        configs.append((f"B={b}", Config(n_envs=b), b, "batch"))

    cfg_big = Config(n_envs=16, hidden_dim=1024, gru_dim=256, **enc_kwargs(128))
    configs.append(("big (1024/128/256)", cfg_big, 16, "combined"))

    cfg_huge = Config(n_envs=16, hidden_dim=2048, gru_dim=512, **enc_kwargs(256))
    configs.append(("huge (2048/256/512)", cfg_huge, 16, "combined"))

    cfg_30m = Config(n_envs=16, hidden_dim=2048, gru_dim=1024, **enc_kwargs(512))
    configs.append(("~30M target", cfg_30m, 16, "combined"))

    print(f"{'name':<32} {'params':>10} {'ms/call':>10} {'rel':>7}")
    print("-" * 65)
    results = []
    baseline_ms = None
    for name, cfg, B, group in configs:
        print(f"{name:<32} ", end="", flush=True)
        n_params, ms = bench_one(cfg, B)
        if baseline_ms is None:
            baseline_ms = ms
        rel = ms / baseline_ms
        print(f"{n_params/1e6:>8.2f}M  {ms:>8.3f}ms  {rel:>6.2f}x")
        results.append((name, n_params, ms, group, B))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping chart)")
        return results

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {
        "baseline": "black",
        "hidden_dim": "tab:blue",
        "encoder_out": "tab:orange",
        "gru_dim": "tab:green",
        "batch": "tab:purple",
        "combined": "tab:red",
    }
    markers = {
        "baseline": "*",
        "hidden_dim": "o",
        "encoder_out": "s",
        "gru_dim": "^",
        "batch": "D",
        "combined": "X",
    }
    plotted_groups = set()
    for name, n_params, ms, group, B in results:
        label = group if group not in plotted_groups else None
        plotted_groups.add(group)
        ax.scatter(
            n_params / 1e6, ms,
            c=colors[group], marker=markers[group],
            s=140 if group in ("baseline", "combined") else 80,
            label=label, edgecolors="black", linewidths=0.5, zorder=3,
        )
        ax.annotate(name, (n_params / 1e6, ms),
                    fontsize=7, xytext=(5, 5), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("model params (M)")
    ax.set_ylabel("ms / inference call (graph replay, B at capture)")
    ax.set_title("FullKnight inference latency vs model size (CUDA graphs, 4080 Super)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    out = "python/bench_model_size.png"
    plt.savefig(out, dpi=120)
    print(f"\nchart saved to {out}")
    return results


if __name__ == "__main__":
    main()
