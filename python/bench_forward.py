"""Standalone forward pass benchmark — run outside of training to isolate GPU perf."""
import time
import torch
from config import Config
from model import FullKnightActorCritic
from observation import Observation, GS

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = FullKnightActorCritic(config).to(device)
model.eval()

B = 6  # n_envs
N_COMBAT = 20
N_TERRAIN = 30
WARMUP = 50
ITERS = 500


def make_obs(device):
    obs = Observation(
        combat_hb=torch.randn(B, N_COMBAT, config.combat_feature_dim, device=device),
        combat_mask=torch.ones(B, N_COMBAT, device=device),
        combat_kind_ids=torch.zeros(B, N_COMBAT, dtype=torch.long, device=device),
        combat_parent_ids=torch.zeros(B, N_COMBAT, dtype=torch.long, device=device),
        terrain_hb=torch.randn(B, N_TERRAIN, config.terrain_feature_dim, device=device),
        terrain_mask=torch.ones(B, N_TERRAIN, device=device),
        global_state=torch.randn(B, config.global_state_dim, device=device),
    )
    obs.global_state[:, GS.HAS_DASH:] = (obs.global_state[:, GS.HAS_DASH:] > 0).float()
    return obs


obs = make_obs(device)

print(f"\nBenchmarking {ITERS} forward passes (batch={B}, combat={N_COMBAT}, terrain={N_TERRAIN})")

# Warmup
with torch.no_grad():
    for _ in range(WARMUP):
        model.get_action_and_value(obs)
torch.cuda.synchronize()

# Benchmark 1: GPU time only (cuda events)
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

with torch.no_grad():
    start_event.record()
    for _ in range(ITERS):
        model.get_action_and_value(obs)
    end_event.record()
torch.cuda.synchronize()
gpu_ms = start_event.elapsed_time(end_event)
print(f"\nGPU-timed (no sync per iter): {gpu_ms:.1f}ms total, {gpu_ms/ITERS:.3f}ms per call")

# Benchmark 2: Wall clock with sync every call (like our profiling code)
torch.cuda.synchronize()
with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(ITERS):
        model.get_action_and_value(obs)
        torch.cuda.synchronize()
    wall_sync = time.perf_counter() - t0
print(f"Wall-clock (sync per iter):   {wall_sync*1000:.1f}ms total, {wall_sync/ITERS*1000:.3f}ms per call")

# Benchmark 3: Wall clock without sync (pipelined)
torch.cuda.synchronize()
with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(ITERS):
        model.get_action_and_value(obs)
    torch.cuda.synchronize()
    wall_nosync = time.perf_counter() - t0
print(f"Wall-clock (no sync per iter):{wall_nosync*1000:.1f}ms total, {wall_nosync/ITERS*1000:.3f}ms per call")

# Benchmark 4: Full round-trip (numpy -> gpu -> forward -> cpu -> numpy) like collect_action
import numpy as np
np_obs = Observation(
    combat_hb=obs.combat_hb.cpu().numpy(),
    combat_mask=obs.combat_mask.cpu().numpy(),
    combat_kind_ids=obs.combat_kind_ids.cpu().numpy(),
    combat_parent_ids=obs.combat_parent_ids.cpu().numpy(),
    terrain_hb=obs.terrain_hb.cpu().numpy(),
    terrain_mask=obs.terrain_mask.cpu().numpy(),
    global_state=obs.global_state.cpu().numpy(),
)

torch.cuda.synchronize()
with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(ITERS):
        gpu_obs = Observation(
            combat_hb=torch.from_numpy(np_obs.combat_hb).float().to(device),
            combat_mask=torch.from_numpy(np_obs.combat_mask).float().to(device),
            combat_kind_ids=torch.from_numpy(np_obs.combat_kind_ids).long().to(device),
            combat_parent_ids=torch.from_numpy(np_obs.combat_parent_ids).long().to(device),
            terrain_hb=torch.from_numpy(np_obs.terrain_hb).float().to(device),
            terrain_mask=torch.from_numpy(np_obs.terrain_mask).float().to(device),
            global_state=torch.from_numpy(np_obs.global_state).float().to(device),
        )
        actions, lp, _, va, vd, _ = model.get_action_and_value(gpu_obs)
        _ = {k: v.cpu().numpy() for k, v in actions.items()}
        _ = lp.cpu().numpy()
        _ = va.cpu().numpy()
        _ = vd.cpu().numpy()
    torch.cuda.synchronize()
    wall_roundtrip = time.perf_counter() - t0
print(f"Full round-trip (np->gpu->np): {wall_roundtrip*1000:.1f}ms total, {wall_roundtrip/ITERS*1000:.3f}ms per call")

# Benchmark 5: torch.compile if available
try:
    compiled = torch.compile(model)
    with torch.no_grad():
        for _ in range(WARMUP):
            compiled.get_action_and_value(obs)
    torch.cuda.synchronize()

    start_event.record()
    with torch.no_grad():
        for _ in range(ITERS):
            compiled.get_action_and_value(obs)
    end_event.record()
    torch.cuda.synchronize()
    compiled_ms = start_event.elapsed_time(end_event)
    print(f"torch.compile GPU-timed:      {compiled_ms:.1f}ms total, {compiled_ms/ITERS:.3f}ms per call")
except Exception as e:
    print(f"torch.compile failed: {e}")
