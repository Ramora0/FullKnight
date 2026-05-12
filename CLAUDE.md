# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

FullKnight is a reinforcement learning system for Hollow Knight boss fights. It has two halves:

- **C# mod** (`mod/` — `FullKnight.cs`, `Game/`, `Net/`, `Environment/`): A Hollow Knight mod that exposes a gym-like environment over WebSocket. It extracts observations (hitboxes, game state), receives actions, applies them via a virtual input device, and computes rewards.
- **Python trainer** (`python/`): A PPO training loop that connects to one or more running game instances, collects rollouts, and trains a set-based actor-critic network.

## Build Commands

### C# Mod

Requires Hollow Knight's managed DLLs. The `LocalRefs` MSBuild property must point to the game's `Managed` folder (set per-configuration in `mod/FullKnight.csproj`).

```bash
# Debug: builds and copies DLL + dependencies to HK Mods folder
dotnet build mod/FullKnight.csproj -c Debug

# Release: builds and packages into mod/Output/FullKnight.zip
dotnet build mod/FullKnight.csproj -c Release
```

**Always rebuild the mod after editing any `.cs` file under `mod/`.** The game loads the installed DLL, not the source — unbuilt changes are invisible at runtime (no new logs, no behavior change). Run `dotnet build mod/FullKnight.csproj -c Debug` before testing C# edits.

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
- **Global state** (22 floats): `[vel_x, vel_y, hp, soul, knight_w, knight_h, has_dash, has_wall_jump, has_double_jump, has_super_dash, has_dream_nail, has_acid_armour, has_nail_art, can_jump, can_double_jump, can_wall_jump, can_dash, can_attack, can_cast, can_nail_charge, can_dream_nail, can_super_dash]`. Boss HP is not global; it lives per-hitbox in the `hp_raw` / `hp_max_raw` columns of combat features.
- **Combat hitbox features** (10 floats): `[rel_x, rel_y, w, h, is_trigger, gives_damage, takes_damage, is_target, hp_raw, hp_max_raw]` plus parallel kind/parent vocab IDs. `gives_damage`=hurts knight on contact, `takes_damage`=has reachable HealthManager (can be attacked), `is_target`=its HealthManager is in `BossSceneController.bosses` (the canonical objective). `hp_raw` / `hp_max_raw` bypass the running normalizer and get log1p-compressed instead — preserves high resolution near death while keeping the input range bounded (~[0, 8]).
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

- `TrainingEnv`: Main environment loop. Handles reset/step/pause/resume. Reports per-step reward signals to Python — `damage_landed` (% of boss max HP dealt), `hits_taken` (integer hit count), `hp_healed` (raw HP restored). Reward is computed Python-side as `δ_attack/D − hits_taken + heal_coef·hp_healed` (`ppo.py:131`); no terminal win/loss bonus. Auto-resets on episode end (knight death or boss death).
- `ProxyController.cs` (`InputDeviceShim` + `ActionDecoder`): Virtual InControl device that injects actions. Checks `Can*` methods before applying actions.
- `HitboxObserver`: Tracks all active Collider2Ds via `HitboxReader` MonoBehaviour, classifies into Knight/Enemy/Attack/Terrain.
- Time control: `Time.captureDeltaTime` is set in `TrainingEnv.Reset()` to `0.0424 / frames_per_wait` and held constant. `Time.timeScale` toggles between 1 (running) and 0 (paused for inter-step Python obs handoff). The previous `TimeScale` IL-hook + multiplier infrastructure was removed; `time_scale` config field is now ignored.
- `SaveFileProxy`: Loads an embedded completed save file (`mod/Resource/save_file.json`) and disables saving.
- `SceneHooks`: Loads boss scenes via Hall of Gods transition sequence.

### Multi-Instance (`instance_manager.py`)

Windows-only. Creates junction-linked copies of the HK game directory to run N instances simultaneously. Configured via `Config.hk_path`.

### Environment

Episodes are real: both knight and boss have real HP and can actually die. Episode end (`done=True`) fires on either death, the env auto-resets, and Python's GAE bootstraps value to 0 at the boundary.

The reward shape, however, is intentionally non-terminal: there is no +1/-1 win/loss bonus. The signal is purely the per-step `δ_attack/D − hits_taken + heal_coef·hp_healed`. Rationale: the original episodic design with terminal rewards was reward-exploited (under discounting, idling beat dying, so the agent hid in a corner). The dense per-step damage signal encodes the actual objective without that pathology. `D` is a per-epoch adaptive scale measured from rollouts, not a hyperparameter to tune. `heal_coef = 0.65` makes a hit-then-heal sequence net to −0.35 instead of −1, creating a dodge > heal > tank ordering.

## Config

All hyperparameters and environment settings are in `python/config.py` as a `@dataclass`. Key settings: `n_envs`, `level` (HK scene name), `frames_per_wait` (frame skip), `time_scale`.
