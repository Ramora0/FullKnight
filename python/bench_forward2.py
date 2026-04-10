"""Diagnose why forward pass is 87ms in training but 5ms in benchmark."""
import time
import random
import torch
import numpy as np
from config import Config
from model import FullKnightActorCritic
from observation import Observation, GS

config = Config()
device = torch.device("cuda")
model = FullKnightActorCritic(config).to(device)
model.eval()

B = 6
WARMUP = 50
ITERS = 341  # match one epoch


def bench(label, fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    print(f"  {label}: {ms:.0f}ms total, {ms/ITERS:.1f}ms/call")


def make_obs(nc=20, nt=30, dev=device):
    return Observation(
        combat_hb=torch.randn(B, nc, config.combat_feature_dim, device=dev),
        combat_mask=torch.ones(B, nc, device=dev),
        combat_kind_ids=torch.zeros(B, nc, dtype=torch.long, device=dev),
        combat_parent_ids=torch.zeros(B, nc, dtype=torch.long, device=dev),
        terrain_hb=torch.randn(B, nt, config.terrain_feature_dim, device=dev),
        terrain_mask=torch.ones(B, nt, device=dev),
        global_state=torch.randn(B, config.global_state_dim, device=dev),
    )


print("=== Test 1: Baseline (pre-allocated, fixed shape, no sync) ===")
obs = make_obs()
obs.global_state[:, GS.HAS_DASH:] = (obs.global_state[:, GS.HAS_DASH:] > 0).float()

with torch.no_grad():
    bench("pre-alloc fixed", lambda: model.get_action_and_value(obs))


print("\n=== Test 2: Pre-allocated, fixed shape, WITH .cpu().numpy() sync each call ===")
def test2():
    actions, lp, _, va, vd, _ = model.get_action_and_value(obs)
    {k: v.cpu().numpy() for k, v in actions.items()}
    lp.cpu().numpy()
    va.cpu().numpy()
    vd.cpu().numpy()

with torch.no_grad():
    bench("pre-alloc + cpu sync", test2)


print("\n=== Test 3: New tensors from numpy each call (fixed shape) ===")
np_obs = Observation(
    combat_hb=obs.combat_hb.cpu().numpy(),
    combat_mask=obs.combat_mask.cpu().numpy(),
    combat_kind_ids=obs.combat_kind_ids.cpu().numpy(),
    combat_parent_ids=obs.combat_parent_ids.cpu().numpy(),
    terrain_hb=obs.terrain_hb.cpu().numpy(),
    terrain_mask=obs.terrain_mask.cpu().numpy(),
    global_state=obs.global_state.cpu().numpy(),
)


def numpy_to_gpu_obs(o):
    return Observation(
        combat_hb=torch.from_numpy(o.combat_hb).float().to(device),
        combat_mask=torch.from_numpy(o.combat_mask).float().to(device),
        combat_kind_ids=torch.from_numpy(o.combat_kind_ids).long().to(device),
        combat_parent_ids=torch.from_numpy(o.combat_parent_ids).long().to(device),
        terrain_hb=torch.from_numpy(o.terrain_hb).float().to(device),
        terrain_mask=torch.from_numpy(o.terrain_mask).float().to(device),
        global_state=torch.from_numpy(o.global_state).float().to(device),
    )


def test3():
    gpu_obs = numpy_to_gpu_obs(np_obs)
    actions, lp, _, va, vd, _ = model.get_action_and_value(gpu_obs)
    {k: v.cpu().numpy() for k, v in actions.items()}
    lp.cpu().numpy()
    va.cpu().numpy()
    vd.cpu().numpy()

with torch.no_grad():
    bench("numpy->gpu + cpu sync", test3)


print("\n=== Test 4: Variable shapes each call (like real training) ===")
combat_sizes = [random.randint(1, 40) for _ in range(ITERS + WARMUP)]
terrain_sizes = [random.randint(1, 60) for _ in range(ITERS + WARMUP)]
call_idx = [0]


def make_numpy_obs(ci):
    nc = combat_sizes[ci]
    nt = terrain_sizes[ci]
    return Observation(
        combat_hb=np.random.randn(B, nc, config.combat_feature_dim).astype(np.float32),
        combat_mask=np.ones((B, nc), dtype=np.float32),
        combat_kind_ids=np.zeros((B, nc), dtype=np.int64),
        combat_parent_ids=np.zeros((B, nc), dtype=np.int64),
        terrain_hb=np.random.randn(B, nt, config.terrain_feature_dim).astype(np.float32),
        terrain_mask=np.ones((B, nt), dtype=np.float32),
        global_state=np_obs.global_state.copy(),
    )


def test4():
    o = make_numpy_obs(call_idx[0])
    call_idx[0] += 1
    gpu_obs = numpy_to_gpu_obs(o)
    actions, lp, _, va, vd, _ = model.get_action_and_value(gpu_obs)
    {k: v.cpu().numpy() for k, v in actions.items()}
    lp.cpu().numpy()
    va.cpu().numpy()
    vd.cpu().numpy()

with torch.no_grad():
    call_idx[0] = 0
    bench("variable shapes + numpy + sync", test4)


print("\n=== Test 5: Same as 4, but with async HK-like wait between calls ===")
call_idx[0] = 0
def test5_iter():
    o = make_numpy_obs(call_idx[0])
    call_idx[0] += 1
    gpu_obs = numpy_to_gpu_obs(o)
    actions, lp, _, va, vd, _ = model.get_action_and_value(gpu_obs)
    {k: v.cpu().numpy() for k, v in actions.items()}
    lp.cpu().numpy()
    va.cpu().numpy()
    vd.cpu().numpy()
    time.sleep(0.04)

print("(This one measures GPU time only via events, sleep shouldn't count)")
with torch.no_grad():
    for _ in range(WARMUP):
        test5_iter()
    torch.cuda.synchronize()
    call_idx[0] = 0
    events = []
    for i in range(ITERS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        o = make_numpy_obs(i)
        gpu_obs = numpy_to_gpu_obs(o)
        s.record()
        actions, lp, _, va, vd, _ = model.get_action_and_value(gpu_obs)
        e.record()
        {k: v.cpu().numpy() for k, v in actions.items()}
        lp.cpu().numpy()
        va.cpu().numpy()
        vd.cpu().numpy()
        events.append((s, e))
        time.sleep(0.04)
    torch.cuda.synchronize()
    total_gpu = sum(s.elapsed_time(e) for s, e in events)
    print(f"  with 40ms sleep: {total_gpu:.0f}ms GPU total, {total_gpu/ITERS:.1f}ms/call")


print("\n=== Test 6: After running a training-like pass (memory pressure) ===")
dummy_params = [torch.randn(256, 256, device=device, requires_grad=True) for _ in range(20)]
opt = torch.optim.Adam(dummy_params, lr=1e-3)
for _ in range(50):
    loss = sum(p.pow(2).sum() for p in dummy_params)
    opt.zero_grad()
    loss.backward()
    opt.step()
torch.cuda.synchronize()

call_idx[0] = 0
with torch.no_grad():
    bench("post-training + variable shapes", test4)
