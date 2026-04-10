"""Standalone forward pass benchmark — run outside of training to isolate GPU perf."""
import time
import torch
from config import Config
from model import FullKnightActorCritic

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

# Fixed-size dummy inputs
combat_hb = torch.randn(B, N_COMBAT, config.combat_feature_dim, device=device)
combat_mask = torch.ones(B, N_COMBAT, device=device)
combat_kind_ids = torch.zeros(B, N_COMBAT, dtype=torch.long, device=device)
terrain_hb = torch.randn(B, N_TERRAIN, config.terrain_feature_dim, device=device)
terrain_mask = torch.ones(B, N_TERRAIN, device=device)
global_state = torch.randn(B, config.global_state_dim, device=device)
# Make validity flags binary
global_state[:, 7:] = (global_state[:, 7:] > 0).float()

print(f"\nBenchmarking {ITERS} forward passes (batch={B}, combat={N_COMBAT}, terrain={N_TERRAIN})")

# Warmup
with torch.no_grad():
    for _ in range(WARMUP):
        model.get_action_and_value(combat_hb, combat_mask, combat_kind_ids, terrain_hb, terrain_mask, global_state)
torch.cuda.synchronize()

# Benchmark 1: GPU time only (cuda events)
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

with torch.no_grad():
    start_event.record()
    for _ in range(ITERS):
        model.get_action_and_value(combat_hb, combat_mask, combat_kind_ids, terrain_hb, terrain_mask, global_state)
    end_event.record()
torch.cuda.synchronize()
gpu_ms = start_event.elapsed_time(end_event)
print(f"\nGPU-timed (no sync per iter): {gpu_ms:.1f}ms total, {gpu_ms/ITERS:.3f}ms per call")

# Benchmark 2: Wall clock with sync every call (like our profiling code)
torch.cuda.synchronize()
with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(ITERS):
        model.get_action_and_value(combat_hb, combat_mask, combat_kind_ids, terrain_hb, terrain_mask, global_state)
        torch.cuda.synchronize()
    wall_sync = time.perf_counter() - t0
print(f"Wall-clock (sync per iter):   {wall_sync*1000:.1f}ms total, {wall_sync/ITERS*1000:.3f}ms per call")

# Benchmark 3: Wall clock without sync (pipelined)
torch.cuda.synchronize()
with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(ITERS):
        model.get_action_and_value(combat_hb, combat_mask, combat_kind_ids, terrain_hb, terrain_mask, global_state)
    torch.cuda.synchronize()
    wall_nosync = time.perf_counter() - t0
print(f"Wall-clock (no sync per iter):{wall_nosync*1000:.1f}ms total, {wall_nosync/ITERS*1000:.3f}ms per call")

# Benchmark 4: Full round-trip (numpy -> gpu -> forward -> cpu -> numpy) like collect_action
import numpy as np
chb_np = combat_hb.cpu().numpy()
cm_np = combat_mask.cpu().numpy()
ckid_np = combat_kind_ids.cpu().numpy()
thb_np = terrain_hb.cpu().numpy()
tm_np = terrain_mask.cpu().numpy()
gs_np = global_state.cpu().numpy()

torch.cuda.synchronize()
with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(ITERS):
        c = torch.from_numpy(chb_np).float().to(device)
        m = torch.from_numpy(cm_np).float().to(device)
        ckid = torch.from_numpy(ckid_np).long().to(device)
        t_ = torch.from_numpy(thb_np).float().to(device)
        tm_ = torch.from_numpy(tm_np).float().to(device)
        g = torch.from_numpy(gs_np).float().to(device)
        actions, lp, _, va, vd, _ = model.get_action_and_value(c, m, ckid, t_, tm_, g)
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
    # Warmup compile
    with torch.no_grad():
        for _ in range(WARMUP):
            compiled.get_action_and_value(combat_hb, combat_mask, combat_kind_ids, terrain_hb, terrain_mask, global_state)
    torch.cuda.synchronize()

    start_event.record()
    with torch.no_grad():
        for _ in range(ITERS):
            compiled.get_action_and_value(combat_hb, combat_mask, combat_kind_ids, terrain_hb, terrain_mask, global_state)
    end_event.record()
    torch.cuda.synchronize()
    compiled_ms = start_event.elapsed_time(end_event)
    print(f"torch.compile GPU-timed:      {compiled_ms:.1f}ms total, {compiled_ms/ITERS:.3f}ms per call")
except Exception as e:
    print(f"torch.compile failed: {e}")
