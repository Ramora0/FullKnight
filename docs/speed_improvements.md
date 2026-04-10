# Training Speed Improvements

Baseline timing (8 envs): ~72s per epoch
- HK sim: 22ms/step (C#), 31ms/step overhead (WS/JSON/Python), 52.8ms/step total wall
- Forward: 10.66s, Training: 6.24s, HK: 54.19s

## 1. Binary Protocol (~35% speedup, saves ~26s)

31ms of 52.8ms per-step wall time is JSON serialization/deserialization overhead. Over 1024 steps = 31.7s wasted per epoch.

Switch hitbox arrays to binary format. MessagePack has libraries on both sides (C#: `MessagePack-CSharp`, Python: `msgpack`). Could cut 31ms overhead to ~5ms.

Effort: 1-2 days. Mechanical change on both C# and Python sides.

## 2. Split-Group Pipelined Stepping (~17% speedup, saves ~11s)

GPU and HK never overlap: GPU computes 11ms, then idles 53ms during HK sim, then HK idles during GPU. Split 8 envs into 2 groups of 4, alternate:

```
Time 0ms:    send actions to group A -> A simulates
             forward pass on group B obs (6ms, hidden inside A's 53ms)
Time 53ms:   A results arrive
             send actions to group B -> B simulates
             forward pass on group A obs (6ms, hidden inside B's 53ms)
Time 106ms:  B results arrive
```

Two steps in 106ms instead of 128ms. Forward passes completely hidden. GPU goes from 15% to near-continuous utilization. Doesn't change RL algorithm.

Effort: 1 day. Restructure rollout loop in train.py and vec_env.py.

## 3. Overlap Training with Resets (~8% speedup, saves ~3s)

Training (6.24s) and reset scene loading (2s) are sequential. Run training in a background thread while staggered resets happen on the event loop:

```python
train_future = asyncio.get_event_loop().run_in_executor(
    None, lambda: agent.train_on_rollout(...)
)
_, reset_obs = await vec_env.reset_and_resume(reset_indices)
metrics = await train_future
```

Effort: 30 min.

## 4. Reduce HK Per-Frame Cost (variable, saves ~5-7s)

The 22ms C# time is 5 frames of Unity rendering + physics. Add to TrainingEnv.Setup():

- Minimize resolution: `-screen-width 320 -screen-height 240` launch args
- Disable VSync: `QualitySettings.vSyncCount = 0`
- Disable audio: `AudioListener.volume = 0` or destroy AudioListeners
- Lower quality: `QualitySettings.SetQualityLevel(0)`

Could cut 22ms to 12-15ms per step.

Effort: 30 min.

## 5. torch.compile (~7% speedup, saves ~4s)

With batch=8, kernel launch overhead dominates. Fuse operations:

```python
self.policy = torch.compile(FullKnightActorCritic(config)).to(self.device)
```

Matters more after binary protocol shrinks overhead (forward pass becomes larger fraction).

Effort: 1 line.

## Combined Estimate

| Change                  | Saves  | Effort   |
|-------------------------|--------|----------|
| Binary protocol         | ~26s   | 1-2 days |
| Split-group pipeline    | ~11s   | 1 day    |
| Overlap train+reset     | ~3s    | 30 min   |
| HK quality reduction    | ~5-7s  | 30 min   |
| torch.compile           | ~4s    | 1 line   |

Current: ~72s/epoch. With all: ~23s/epoch (~3x faster).
