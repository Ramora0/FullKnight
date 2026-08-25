# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

FullKnight is a reinforcement learning system for Hollow Knight boss fights. It has two halves:

- **C# mod** (`FullKnight.cs`, `Game/`, `Net/`, `Environment/`): A Hollow Knight mod that exposes a gym-like environment over WebSocket. It extracts observations (hitboxes, game state), receives actions, applies them via a virtual input device, and computes rewards.
- **Python trainer** (`python/`): A PPO training loop that connects to one or more running game instances, collects rollouts, and trains a set-based actor-critic network.

## Build Commands

### C# Mod

Requires Hollow Knight's managed DLLs. The `LocalRefs` MSBuild property must point to the game's `Managed` folder (set per-configuration in `FullKnight.csproj`).

```bash
# Debug: builds and copies DLL + dependencies to HK Mods folder
dotnet build -c Debug

# Release: builds and packages into Output/FullKnight.zip
dotnet build -c Release
```

### Python Trainer

```bash
cd python
uv pip install torch numpy websockets psutil tensorboard
python train.py
```

## Architecture

### Communication Flow

```
Python (VecEnv WebSocket server, port 8765)
  ↕ JSON messages (Protocol.cs defines Message/MessageData schema)
C# Mod (TrainingEnv coroutine loop inherits WebsocketEnv)
```

The Python side is the **server**. Each game instance connects as a client. `VecEnv` manages N parallel connections. Messages flow: `init` → `reset` → `action`/`step` loop, with `pause`/`resume` around training updates.

### Observation Space

- **Hitboxes** (variable-length sets): Split into combat (enemy + attack colliders) and terrain, each padded and masked for batching. Combat and terrain have different feature layouts (see below).
- **Global state** (33 floats): `[vel_x, vel_y, hp, soul, knight_w, knight_h, has_dash, has_wall_jump, has_double_jump, has_super_dash, has_dream_nail, has_acid_armour, has_nail_art, can_jump, can_double_jump, can_wall_jump, can_dash, can_attack, can_cast, can_nail_charge, can_dream_nail, can_super_dash, commit_locked, commit_releasing, commit_progress, commit_action_onehot × 8]`. Boss HP is not global; it lives per-hitbox in the `hp_raw` / `hp_max_raw` columns of combat features. Only indices 0..5 are continuous and z-scored; everything from `has_dash` on is a flag or a bounded `[0,1]` scalar that passes through raw (`n_binary_flags`), so new bounded columns append at the end.
- **Hard-commit proprioception** (global indices 22..32): `ActionDecoder`'s commit state machine overrides `action[2]` for the whole length of a hold — up to ~71 steps for dream_nail at `frames_per_wait=5` — while movement/direction/jump stay free. These columns tell the policy that a charge is in flight, which one, and how far through it is. Distinct from the `action_committed` wire bit, which is used only to mask the action head's gradient.
- **Combat hitbox features** (13 floats): `[rel_x, rel_y, w, h, vel_x, vel_y, is_trigger, gives_damage, takes_damage, is_target, is_invincible, hp_raw, hp_max_raw]` plus parallel kind/parent vocab IDs. `gives_damage`=hurts knight on contact, `takes_damage`=has reachable HealthManager (can be attacked), `is_target`=its HealthManager is in `BossSceneController.bosses` (the canonical objective), `is_invincible`=that HealthManager is currently untouchable (post-hit iframes / stagger), without which a staggered boss is indistinguishable from a hittable one and connecting swings look like misses. `vel_x` / `vel_y` are knight-relative per-step displacement computed C#-side against a per-collider position cache — combat rows are emitted from `HashSet` iteration, so row order is unstable between steps and no per-row instance identity crosses the wire, meaning motion is unrecoverable anywhere downstream. `hp_raw` / `hp_max_raw` bypass the running normalizer and get log1p-compressed instead — preserves high resolution near death while keeping the input range bounded (~[0, 8]).
- **Column ordering is load-bearing.** Combat features are grouped continuous → binary → hp tail: columns `[0, combat_normalized_dims)` are z-scored, the flags pass through raw, and the trailing hp pair is log1p-compressed. Adding a column means updating `config.py`, `binary_protocol.py`, `observation.py`'s index classes, the C# emitter, and — for anything with an x-component or a handedness — `mirror_observation`, where a miss is silent and corrupts half the augmented data.
- **Terrain hitbox features** (8 floats): `[mx, my, hdx, hdy, npx, npy, dist, is_trigger]` — midpoint, half-extent, nearest-point, and distance from knight. Composite-absorbed colliders are skipped to avoid double-counting.

### Action Space (Factored)

Four independent sub-actions decoded by `ActionDecoder.ApplyAction`:
- `action[0]` movement: 0=left, 1=right, 2=none
- `action[1]` direction: 0=up, 1=down, 2=none
- `action[2]` action: 0=attack(tap), 1=nail_charge(hold), 2=spell(tap), 3=focus(hold), 4=dash, 5=dream_nail(hold), 6=super_dash(hold), 7=none
- `action[3]` jump: 0=yes, 1=no

Tap vs hold: tap actions force a release-then-press transition for a fresh input event. Hold actions keep the key pressed across steps, enabling charge mechanics (nail arts, focus healing, dream nail, super dash). Hold actions only check `Can*` on the initial press; subsequent steps maintain the hold.

The model applies **validity masking** using the 9 `can_*` flags from global state to zero out impossible actions before sampling.

### Model (`model.py`)

`FullKnightActorCritic` uses set encoders to handle variable-length hitbox inputs:
- `CombatEncoder`: single-head attention pooling, queried by global state
- `TerrainEncoder`: sum pooling
- Outputs feed into a shared trunk → GRUCell (with residual + LayerNorm) → 4 actor heads (one per sub-action) + critic head

The GRU provides temporal memory across timesteps. Hidden state flows during rollout collection and is stored at chunk boundaries for truncated BPTT during training (`seq_len` steps). The residual connection ensures the GRU starts as a near-passthrough at initialization.

### Key C# Components

- `TrainingEnv`: Main environment loop. Handles reset/step/pause/resume. Reports per-step reward signals to Python — `damage_landed` (% of boss max HP dealt), `hits_taken` (raw HP lost, accumulated from the damage amount, *not* a hit count), `hp_healed` (raw HP restored). Reward is computed Python-side as `δ_attack/D − hits_taken + heal_coef·hp_healed` (`ppo.py:131`); no terminal win/loss bonus. Auto-resets on episode end (knight death or boss death).
- `ProxyController.cs` (`InputDeviceShim` + `ActionDecoder`): Virtual InControl device that injects actions. Checks `Can*` methods before applying actions.
- `HitboxObserver`: Tracks all active Collider2Ds via `HitboxReader` MonoBehaviour, classifies into Knight/Enemy/Attack/Terrain.
- `TimeScale`: IL-hooks `GameManager.FreezeMoment*` coroutines and shims `SetTimeScale` to maintain configurable game speed.
- `SaveFileProxy`: Loads an embedded completed save file (`Resource/save_file.json`) and disables saving.
- `SceneHooks`: Loads boss scenes via Hall of Gods transition sequence.

### Multi-Instance (`instance_manager.py`)

Windows-only. Creates junction-linked copies of the HK game directory to run N instances simultaneously. Configured via `Config.hk_path`.

### Environment

Episodes are real: both knight and boss have real HP and can actually die. Episode end (`done=True`) fires on either death, the env auto-resets, and Python's GAE bootstraps value to 0 at the boundary.

The reward shape, however, is intentionally non-terminal: there is no +1/-1 win/loss bonus. The signal is purely the per-step `δ_attack/D − hits_taken + heal_coef·hp_healed`. Rationale: the original episodic design with terminal rewards was reward-exploited (under discounting, idling beat dying, so the agent hid in a corner). The dense per-step damage signal encodes the actual objective without that pathology. `D` is a per-epoch adaptive scale measured from rollouts, not a hyperparameter to tune.

Both halves of the defense term are in raw HP: `hits_taken` accumulates the damage amount, so `heal_coef = 1.0` means healing exactly cancels the damage taken, and the net defense reward is the episode's net HP change regardless of whether a boss hits for one mask or two. Counting *hits* while measuring healing in *HP* — the previous shape — made getting hit and healing net **positive** against any 2-mask boss (−1 for the hit vs +2·0.65 for healing it back), an exploit that was inert only because every boss in the current pool deals 1 mask. Healing is not free at 1.0: focus costs soul earned by landing hits and roots the knight through the charge, so dodging still dominates.

## Config

All hyperparameters and environment settings are in `python/config.py` as a `@dataclass`. Key settings: `n_envs`, `level` (HK scene name), `frames_per_wait` (frame skip), `time_scale`.
