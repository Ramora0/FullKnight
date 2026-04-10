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

- **Hitboxes** (variable-length sets): Each hitbox is `[rel_x, rel_y, width, height, is_trigger]` relative to the knight. Split into combat (enemy + attack colliders) and terrain. Padded and masked for batching.
- **Global state** (22 floats): `[vel_x, vel_y, hp, soul, knight_w, knight_h, has_dash, has_wall_jump, has_double_jump, has_super_dash, has_dream_nail, has_acid_armour, has_nail_art, can_jump, can_double_jump, can_wall_jump, can_dash, can_attack, can_cast, can_nail_charge, can_dream_nail, can_super_dash]`. Boss HP is no longer global; it lives per-hitbox in the `hp_raw` column of combat features.
- **Combat hitbox features** (9 floats): `[rel_x, rel_y, w, h, is_trigger, gives_damage, takes_damage, is_target, hp_raw]` plus parallel kind/parent vocab IDs. `gives_damage`=hurts knight on contact, `takes_damage`=has reachable HealthManager (can be attacked), `is_target`=its HealthManager is in `BossSceneController.bosses` (the canonical objective). `hp_raw` is the HealthManager's current HP, raw — intentionally excluded from the running normalizer so the agent reads absolute "1-2 nail hits from death" vs "beefy" semantics.

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

- `TrainingEnv`: Main environment loop. Handles reset/step/pause/resume. Computes reward: step penalty + damage dealt fraction − damage taken fraction + win/loss bonuses. Auto-resets on episode end.
- `ProxyController.cs` (`InputDeviceShim` + `ActionDecoder`): Virtual InControl device that injects actions. Checks `Can*` methods before applying actions.
- `HitboxObserver`: Tracks all active Collider2Ds via `HitboxReader` MonoBehaviour, classifies into Knight/Enemy/Attack/Terrain.
- `TimeScale`: IL-hooks `GameManager.FreezeMoment*` coroutines and shims `SetTimeScale` to maintain configurable game speed.
- `SaveFileProxy`: Loads an embedded completed save file (`Resource/save_file.json`) and disables saving.
- `SceneHooks`: Loads boss scenes via Hall of Gods transition sequence.

### Multi-Instance (`instance_manager.py`)

Windows-only. Creates junction-linked copies of the HK game directory to run N instances simultaneously. Configured via `Config.hk_path`.

### Environment

Instead of training discretely on beating a boss, we train on an infinite boss game where the boss and player never die. Since the terminal state of a boss dying is just the sum of per step rewards, actually killing the boss isn't informative; its the progressive damage given vs taken thats the real, immediate reward signal.

## Config

All hyperparameters and environment settings are in `python/config.py` as a `@dataclass`. Key settings: `n_envs`, `level` (HK scene name), `frames_per_wait` (frame skip), `time_scale`.
