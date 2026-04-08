"""Diagnose why forward pass is 87ms in training but 5ms in benchmark."""
import time
import random
import torch
import numpy as np
from config import Config
from model import FullKnightActorCritic

config = Config()
device = torch.device("cuda")
model = FullKnightActorCritic(config).to(device)
model.eval()

B = 6
WARMUP = 50
ITERS = 341  # match one epoch

def bench(label, fn):
    # warmup
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


print("=== Test 1: Baseline (pre-allocated, fixed shape, no sync) ===")
combat_hb = torch.randn(B, 20, config.combat_feature_dim, device=device)
combat_mask = torch.ones(B, 20, device=device)
terrain_hb = torch.randn(B, 30, config.terrain_feature_dim, device=device)
terrain_mask = torch.ones(B, 30, device=device)
gs = torch.randn(B, config.global_state_dim, device=device)
gs[:, 8:] = (gs[:, 8:] > 0).float()

with torch.no_grad():
    bench("pre-alloc fixed", lambda: model.get_action_and_value(
        combat_hb, combat_mask, terrain_hb, terrain_mask, gs))


print("\n=== Test 2: Pre-allocated, fixed shape, WITH .cpu().numpy() sync each call ===")
def test2():
    actions, lp, _, va, vd = model.get_action_and_value(
        combat_hb, combat_mask, terrain_hb, terrain_mask, gs)
    {k: v.cpu().numpy() for k, v in actions.items()}
    lp.cpu().numpy()
    va.cpu().numpy()
    vd.cpu().numpy()

with torch.no_grad():
    bench("pre-alloc + cpu sync", test2)


print("\n=== Test 3: New tensors from numpy each call (fixed shape) ===")
chb_np = combat_hb.cpu().numpy()
cm_np = combat_mask.cpu().numpy()
thb_np = terrain_hb.cpu().numpy()
tm_np = terrain_mask.cpu().numpy()
gs_np = gs.cpu().numpy()

def test3():
    c = torch.from_numpy(chb_np).float().to(device)
    m = torch.from_numpy(cm_np).float().to(device)
    t = torch.from_numpy(thb_np).float().to(device)
    tm_ = torch.from_numpy(tm_np).float().to(device)
    g = torch.from_numpy(gs_np).float().to(device)
    actions, lp, _, va, vd = model.get_action_and_value(c, m, t, tm_, g)
    {k: v.cpu().numpy() for k, v in actions.items()}
    lp.cpu().numpy()
    va.cpu().numpy()
    vd.cpu().numpy()

with torch.no_grad():
    bench("numpy->gpu + cpu sync", test3)


print("\n=== Test 4: Variable shapes each call (like real training) ===")
# Pre-generate random shapes
combat_sizes = [random.randint(1, 40) for _ in range(ITERS + WARMUP)]
terrain_sizes = [random.randint(1, 60) for _ in range(ITERS + WARMUP)]
call_idx = [0]

def make_numpy_obs(ci):
    nc = combat_sizes[ci]
    nt = terrain_sizes[ci]
    return (
        np.random.randn(B, nc, config.combat_feature_dim).astype(np.float32),
        np.ones((B, nc), dtype=np.float32),
        np.random.randn(B, nt, config.terrain_feature_dim).astype(np.float32),
        np.ones((B, nt), dtype=np.float32),
        gs_np.copy(),
    )

def test4():
    chb, cm, thb, tm_, g = make_numpy_obs(call_idx[0])
    call_idx[0] += 1
    c = torch.from_numpy(chb).float().to(device)
    m = torch.from_numpy(cm).float().to(device)
    t = torch.from_numpy(thb).float().to(device)
    tm2 = torch.from_numpy(tm_).float().to(device)
    g2 = torch.from_numpy(g).float().to(device)
    actions, lp, _, va, vd = model.get_action_and_value(c, m, t, tm2, g2)
    {k: v.cpu().numpy() for k, v in actions.items()}
    lp.cpu().numpy()
    va.cpu().numpy()
    vd.cpu().numpy()

with torch.no_grad():
    call_idx[0] = 0
    bench("variable shapes + numpy + sync", test4)


print("\n=== Test 5: Same as 4, but with async HK-like wait between calls ===")
# Simulate 40ms of HK step time between each forward pass
call_idx[0] = 0
def test5_iter():
    chb, cm, thb, tm_, g = make_numpy_obs(call_idx[0])
    call_idx[0] += 1
    c = torch.from_numpy(chb).float().to(device)
    m = torch.from_numpy(cm).float().to(device)
    t = torch.from_numpy(thb).float().to(device)
    tm2 = torch.from_numpy(tm_).float().to(device)
    g2 = torch.from_numpy(g).float().to(device)
    actions, lp, _, va, vd = model.get_action_and_value(c, m, t, tm2, g2)
    {k: v.cpu().numpy() for k, v in actions.items()}
    lp.cpu().numpy()
    va.cpu().numpy()
    vd.cpu().numpy()
    time.sleep(0.04)  # simulate HK step

print("(This one measures GPU time only via events, sleep shouldn't count)")
with torch.no_grad():
    # warmup
    for _ in range(WARMUP):
        test5_iter()
    torch.cuda.synchronize()
    call_idx[0] = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    events = []
    for i in range(ITERS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        chb, cm, thb, tm_, g = make_numpy_obs(i)
        c = torch.from_numpy(chb).float().to(device)
        m = torch.from_numpy(cm).float().to(device)
        t = torch.from_numpy(thb).float().to(device)
        tm2 = torch.from_numpy(tm_).float().to(device)
        g2 = torch.from_numpy(g).float().to(device)
        s.record()
        actions, lp, _, va, vd = model.get_action_and_value(c, m, t, tm2, g2)
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
# Simulate training memory pressure
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
