"""Compare new model (HEAD) vs old model (main) on rollout + training paths.

Cases per model (8): {rollout, train} x {eager, cuda-graph} x {fp32, bf16}.

Methodology: cuda.Event timing, 200-iter warmup (cuBLAS algo selection +
graph capture both need this), median + p95 over 1000 timed iters, fixed
input tensors so RNG is out of the picture. Reports peak GPU mem and
param count per (model, case).

Shapes match Config buckets at B=8 (n_envs), combat=32, terrain=96 — the
captured CUDA-graph bucket the production loop usually hits. Train uses
B*L=128 with seq_len=16 to match `chunks_per_batch * seq_len`.

Setup notes:
  - Old model from main is loaded by:
      git show main:python/model.py > /tmp/model_old.py
    The script auto-applies that on first run and patches the import path
    (`env.observation` -> `observation`).
  - The new model's `forward_sequence` has a `.item()` inside that breaks
    CUDA graph capture; we load a patched copy in /tmp that drops it.
  - Old model expects `cfg.attn_n_heads`, `cfg.hold_action_init_bias`;
    those defaults from main are injected here so its __init__ works.
"""
import argparse
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.gettempdir())

from config import Config
from observation import Observation, GS

DEVICE = torch.device("cuda")
B = 8
L = 16
N_COMBAT = 32
N_TERRAIN = 96
WARMUP = 200
ITERS = 1000


# ---------------------------------------------------------------------------
# Module loading: pull old model from main, patched, and re-import new model
# with the .item() stripped so graph capture works.
# ---------------------------------------------------------------------------

def _load_patched_module(name, src_text, replacements):
    out = src_text
    for old, new in replacements:
        out = out.replace(old, new)
    tmp = TMP / f"{name}.py"
    tmp.write_text(out)
    spec = importlib.util.spec_from_file_location(name, tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_old_model_class():
    src = subprocess.check_output(
        ["git", "show", "main:python/model.py"], cwd=ROOT.parent, text=True
    )
    mod = _load_patched_module(
        "model_old",
        src,
        [("from env.observation", "from observation")],
    )
    return mod.FullKnightActorCritic


def load_new_model_class():
    src = (ROOT / "model.py").read_text()
    mod = _load_patched_module(
        "model_new_bench",
        src,
        [(".mean().item()}", ".mean()}")],
    )
    return mod.FullKnightActorCritic


def make_old_config():
    cfg = Config()
    cfg.__dict__["attn_n_heads"] = 4
    cfg.__dict__["hold_action_init_bias"] = -2.0
    return cfg


# ---------------------------------------------------------------------------
# Input builders. Tensors are fixed across iters (compute, not RNG).
# ---------------------------------------------------------------------------

def make_rollout_inputs(cfg):
    obs = Observation(
        combat_hb=torch.randn(B, N_COMBAT, cfg.combat_feature_dim, device=DEVICE),
        combat_mask=torch.ones(B, N_COMBAT, device=DEVICE),
        combat_kind_ids=torch.zeros(B, N_COMBAT, dtype=torch.long, device=DEVICE),
        combat_parent_ids=torch.zeros(B, N_COMBAT, dtype=torch.long, device=DEVICE),
        terrain_hb=torch.randn(B, N_TERRAIN, cfg.terrain_feature_dim, device=DEVICE),
        terrain_mask=torch.ones(B, N_TERRAIN, device=DEVICE),
        global_state=torch.randn(B, cfg.global_state_dim, device=DEVICE),
    )
    obs.global_state[:, GS.HAS_DASH:] = (obs.global_state[:, GS.HAS_DASH:] > 0).float()
    hx = torch.zeros(B, cfg.gru_dim, device=DEVICE)
    return obs, hx


def make_train_inputs(cfg):
    obs = Observation(
        combat_hb=torch.randn(B, L, N_COMBAT, cfg.combat_feature_dim, device=DEVICE),
        combat_mask=torch.ones(B, L, N_COMBAT, device=DEVICE),
        combat_kind_ids=torch.zeros(B, L, N_COMBAT, dtype=torch.long, device=DEVICE),
        combat_parent_ids=torch.zeros(B, L, N_COMBAT, dtype=torch.long, device=DEVICE),
        terrain_hb=torch.randn(B, L, N_TERRAIN, cfg.terrain_feature_dim, device=DEVICE),
        terrain_mask=torch.ones(B, L, N_TERRAIN, device=DEVICE),
        global_state=torch.randn(B, L, cfg.global_state_dim, device=DEVICE),
    )
    obs.global_state[..., GS.HAS_DASH:] = (obs.global_state[..., GS.HAS_DASH:] > 0).float()
    hx = torch.zeros(B, cfg.gru_dim, device=DEVICE)
    actions = {
        "movement": torch.zeros(B, L, dtype=torch.long, device=DEVICE),
        "direction": torch.zeros(B, L, dtype=torch.long, device=DEVICE),
        "action": torch.zeros(B, L, dtype=torch.long, device=DEVICE),
        "jump": torch.zeros(B, L, dtype=torch.long, device=DEVICE),
    }
    target_atk = torch.zeros(B, L, device=DEVICE)
    target_def = torch.zeros(B, L, device=DEVICE)
    return obs, hx, actions, target_atk, target_def


# ---------------------------------------------------------------------------
# Timing helpers.
# ---------------------------------------------------------------------------

def time_iters(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    samples = np.array([starts[i].elapsed_time(ends[i]) for i in range(iters)])
    return float(np.median(samples)), float(np.percentile(samples, 95))


def autocast_ctx(dtype):
    if dtype is None:
        # No-op context.
        class _Null:
            def __enter__(self): return self
            def __exit__(self, *_): return False
        return _Null()
    return torch.autocast(device_type="cuda", dtype=dtype)


# ---------------------------------------------------------------------------
# Rollout: eager.
# ---------------------------------------------------------------------------

def bench_rollout_eager(model, obs, hx, dtype):
    def step():
        with torch.no_grad():
            with autocast_ctx(dtype):
                model.get_action_and_value(obs, hx=hx)
    return time_iters(step)


# ---------------------------------------------------------------------------
# Rollout: cuda-graph. Manual capture so we can wrap in autocast for bf16.
# ---------------------------------------------------------------------------

def bench_rollout_graph(model, obs, hx, dtype):
    # Static buffers on device, copied from `obs` once. Replays read from
    # these same addresses.
    static_obs = Observation(
        combat_hb=obs.combat_hb.clone(),
        combat_mask=obs.combat_mask.clone(),
        combat_kind_ids=obs.combat_kind_ids.clone(),
        combat_parent_ids=obs.combat_parent_ids.clone(),
        terrain_hb=obs.terrain_hb.clone(),
        terrain_mask=obs.terrain_mask.clone(),
        global_state=obs.global_state.clone(),
    )
    static_hx = hx.clone()

    # Warmup on a side stream so cuDNN/cuBLAS algo selection settles before
    # capture (CUDA graph capture is strict about this).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.no_grad():
            with autocast_ctx(dtype):
                for _ in range(20):
                    model.get_action_and_value(static_obs, hx=static_hx)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with autocast_ctx(dtype):
            with torch.cuda.graph(g):
                model.get_action_and_value(static_obs, hx=static_hx)

    def step():
        g.replay()
    return time_iters(step)


# ---------------------------------------------------------------------------
# Train step: forward_sequence + synthetic loss + backward + opt.step.
# Toy loss with gradient flow through every head; we measure compute, not
# convergence.
# ---------------------------------------------------------------------------

def _compute_loss(out, target_atk, target_def):
    log_probs, entropies, v_atk, v_def, _gru_info, _lp_a, _ent_a = out
    return (
        -log_probs.mean()
        + 0.5 * (v_atk - target_atk).pow(2).mean()
        + 0.5 * (v_def - target_def).pow(2).mean()
        - 0.01 * entropies.mean()
    )


def bench_train_eager(model, obs, hx, actions, target_atk, target_def, dtype):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    def step():
        opt.zero_grad(set_to_none=False)  # set_to_none=False -> graph-friendly even though we don't graph here
        with autocast_ctx(dtype):
            out = model.forward_sequence(obs, hx, actions)
            loss = _compute_loss(out, target_atk, target_def)
        loss.backward()
        opt.step()

    return time_iters(step)


def bench_train_graph(model, obs, hx, actions, target_atk, target_def, dtype):
    # Static buffers (capture records addresses, not values).
    static_obs = Observation(
        combat_hb=obs.combat_hb.clone(),
        combat_mask=obs.combat_mask.clone(),
        combat_kind_ids=obs.combat_kind_ids.clone(),
        combat_parent_ids=obs.combat_parent_ids.clone(),
        terrain_hb=obs.terrain_hb.clone(),
        terrain_mask=obs.terrain_mask.clone(),
        global_state=obs.global_state.clone(),
    )
    static_hx = hx.clone()
    static_actions = {k: v.clone() for k, v in actions.items()}
    static_tgt_atk = target_atk.clone()
    static_tgt_def = target_def.clone()

    # `capturable=True` is required so Adam keeps its step counter as a CUDA
    # tensor (incremented on-device) — otherwise it host-syncs to bump a
    # Python int, which is illegal inside graph capture.
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, capturable=True)

    # Warmup on a side stream. ALSO runs a real step so opt state tensors
    # exist before capture (otherwise opt.step allocates inside capture,
    # which is illegal).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(20):
            opt.zero_grad(set_to_none=False)
            with autocast_ctx(dtype):
                out = model.forward_sequence(static_obs, static_hx, static_actions)
                loss = _compute_loss(out, static_tgt_atk, static_tgt_def)
            loss.backward()
            opt.step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        opt.zero_grad(set_to_none=False)
        with autocast_ctx(dtype):
            out = model.forward_sequence(static_obs, static_hx, static_actions)
            loss = _compute_loss(out, static_tgt_atk, static_tgt_def)
        loss.backward()
        opt.step()

    def step():
        g.replay()
    return time_iters(step)


# ---------------------------------------------------------------------------
# Runner: per-model loop over the 8 cases. Resets peak mem before each.
# ---------------------------------------------------------------------------

def run_model(label, model_cls, cfg):
    print(f"\n=== {label} ===", flush=True)

    model = model_cls(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}", flush=True)

    results = []
    cases = [
        # (path, graphs, bf16)
        ("rollout", False, False),
        ("rollout", False, True),
        ("rollout", True, False),
        ("rollout", True, True),
        ("train", False, False),
        ("train", False, True),
        ("train", True, False),
        ("train", True, True),
    ]

    for path, use_graph, use_bf16 in cases:
        dtype = torch.bfloat16 if use_bf16 else None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        if path == "rollout":
            model.eval()
            obs, hx = make_rollout_inputs(cfg)
            if use_graph:
                med, p95 = bench_rollout_graph(model, obs, hx, dtype)
            else:
                med, p95 = bench_rollout_eager(model, obs, hx, dtype)
        else:
            model.train()
            obs, hx, actions, tatk, tdef = make_train_inputs(cfg)
            if use_graph:
                med, p95 = bench_train_graph(model, obs, hx, actions, tatk, tdef, dtype)
            else:
                med, p95 = bench_train_eager(model, obs, hx, actions, tatk, tdef, dtype)

        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        tag = f"{path:<7} g={int(use_graph)} bf16={int(use_bf16)}"
        print(f"  {tag}  med={med:7.3f} ms  p95={p95:7.3f} ms  peak={peak_mb:7.1f} MiB", flush=True)
        results.append({
            "label": label,
            "path": path,
            "graphs": use_graph,
            "bf16": use_bf16,
            "med_ms": med,
            "p95_ms": p95,
            "peak_mib": peak_mb,
            "params": n_params,
        })

    del model
    torch.cuda.empty_cache()
    return results


def print_table(all_results):
    print("\n\n=========================== SUMMARY ===========================")
    print(f"{'model':<8} {'path':<7} {'g':<3} {'bf16':<5} {'med ms':>8} {'p95 ms':>8} {'peak MiB':>10} {'params':>10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['label']:<8} {r['path']:<7} {int(r['graphs']):<3} {int(r['bf16']):<5} "
              f"{r['med_ms']:>8.3f} {r['p95_ms']:>8.3f} {r['peak_mib']:>10.1f} {r['params']:>10,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["old", "new"], default=None,
                        help="Restrict to one model (smoke-test)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch:  {torch.__version__}")
    print(f"shapes: B={B} L={L} N_COMBAT={N_COMBAT} N_TERRAIN={N_TERRAIN}")
    print(f"iters:  warmup={WARMUP} timed={ITERS}")

    all_results = []
    if args.only in (None, "old"):
        OldCls = load_old_model_class()
        old_cfg = make_old_config()
        all_results.extend(run_model("old", OldCls, old_cfg))
    if args.only in (None, "new"):
        NewCls = load_new_model_class()
        new_cfg = Config()
        all_results.extend(run_model("new", NewCls, new_cfg))

    print_table(all_results)


if __name__ == "__main__":
    main()
