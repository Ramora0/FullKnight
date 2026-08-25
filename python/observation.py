"""Observation dataclass and named index constants.

Centralizes "the seven things that flow together through the model and
training pipeline" so adding a per-hitbox or global-state field doesn't
require touching every function signature in the repo.
"""
from dataclasses import dataclass, fields, replace
from typing import Any, List
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Column index constants (use these instead of gs[N] / combat_hb[..., N]).
# When a column moves, fix it here and the rest of the codebase comes with it.
# ---------------------------------------------------------------------------

class GS:
    """Global state column indices (33 floats).

    Only indices 0..5 are continuous; everything from HAS_DASH onward is a
    binary flag or a bounded [0, 1] scalar and bypasses the running normalizer
    (see config.n_binary_flags). New bounded columns append at the end.
    """
    VEL_X = 0
    VEL_Y = 1
    HP = 2
    SOUL = 3
    KNIGHT_W = 4
    KNIGHT_H = 5
    # Ability unlock flags (7): indices 6..12
    HAS_DASH = 6
    HAS_WALL_JUMP = 7
    HAS_DOUBLE_JUMP = 8
    HAS_SUPER_DASH = 9
    HAS_DREAM_NAIL = 10
    HAS_ACID_ARMOUR = 11
    HAS_NAIL_ART = 12
    # Action validity flags (9): indices 13..21
    CAN_JUMP = 13
    CAN_DOUBLE_JUMP = 14
    CAN_WALL_JUMP = 15
    CAN_DASH = 16
    CAN_ATTACK = 17
    CAN_CAST = 18
    CAN_NAIL_CHARGE = 19
    CAN_DREAM_NAIL = 20
    CAN_SUPER_DASH = 21
    # Hard-commit proprioception (11): indices 22..32. The C# commit state
    # machine overrides action[2] for the length of a hold (up to ~71 steps for
    # dream_nail at frames_per_wait=5) while movement/direction/jump stay free;
    # these tell the policy a charge is in flight, which one, and how far along.
    COMMIT_LOCKED = 22     # in the locked phase of a hold
    COMMIT_RELEASING = 23  # this step forces action[2]=none, firing the hold
    COMMIT_PROGRESS = 24   # [0, 1] fraction of the locked phase elapsed
    COMMIT_ACTION_0 = 25   # one-hot over the 8 action slots, all zero when idle
    COMMIT_ACTION_7 = 32


class CB:
    """Combat hitbox feature column indices (13 floats).

    Grouped continuous-first: columns [0, config.combat_normalized_dims) are
    z-scored by the running normalizer, the binary flags pass through raw, and
    the hp tail is log1p-compressed. Preserve that grouping when adding columns.
    """
    REL_X = 0
    REL_Y = 1
    W = 2
    H = 3
    VEL_X = 4        # knight-relative displacement since the previous step
    VEL_Y = 5        # (0 on first sight or after the collider was inactive)
    IS_TRIGGER = 6
    GIVES_DAMAGE = 7
    TAKES_DAMAGE = 8
    IS_TARGET = 9
    IS_INVINCIBLE = 10  # HealthManager currently untouchable (iframes / stagger)
    HP_RAW = 11      # current HP, raw on the wire; log1p-compressed before the model
    HP_MAX_RAW = 12  # observed max HP (cached on first sight, refill-aware), same treatment


class TR:
    """Terrain segment feature column indices (8 floats).

    Every terrain collider is decomposed into line segments C#-side
    (boxes → 4 edges, edge colliders → polyline, polygons → closed paths,
    circles → 12-gon). Each segment is knight-relative.
    """
    MX = 0           # segment midpoint x
    MY = 1           # segment midpoint y
    HDX = 2          # half-vector x (midpoint → one endpoint), canonicalized HDX ≥ 0
    HDY = 3          # half-vector y (canonical tie-break: HDX == 0 ⇒ HDY ≥ 0)
    NPX = 4          # nearest point on the segment (clamped, not infinite line) x
    NPY = 5          # nearest point on the segment y
    DIST = 6         # L2 norm of (NPX, NPY) — pre-computed so attention can gate on it linearly
    IS_TRIGGER = 7   # 0/1, pass-through (not normalized)


# ---------------------------------------------------------------------------
# Observation: bundle of seven padded arrays/tensors that flow together.
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """Padded batch observation flowing through model + training pipeline.

    All fields are numpy arrays (during rollout collection) or torch tensors
    (during training). Shapes share a leading "batch axes" structure with
    optional time/minibatch dims:

      Per-frame collection: (B, ...)
      Per-rollout step (after stack): (T, B, ...)
      Per-training chunk: (B, L, ...)
    """
    combat_hb: Any         # (..., max_combat, 10)
    combat_mask: Any       # (..., max_combat)
    combat_kind_ids: Any   # (..., max_combat) int
    combat_parent_ids: Any # (..., max_combat) int
    terrain_hb: Any        # (..., max_terrain, 8)
    terrain_mask: Any      # (..., max_terrain)
    global_state: Any      # (..., 22)

    def replace(self, **kwargs) -> "Observation":
        """Functional update — returns a new Observation with the given fields replaced."""
        return replace(self, **kwargs)

    @staticmethod
    def stack(obs_list: List["Observation"]) -> "Observation":
        """Stack a list of T per-frame Observations into a single (T, B, ...) Observation.

        Pads combat/terrain dims to the global max across the list (per-frame
        observations have variable hitbox counts) before stacking.
        """
        T = len(obs_list)
        if T == 0:
            raise ValueError("stack() requires at least one Observation")
        max_combat = max(o.combat_hb.shape[1] for o in obs_list)
        max_combat = max(max_combat, 1)
        max_terrain = max(o.terrain_hb.shape[1] for o in obs_list)
        max_terrain = max(max_terrain, 1)

        N = obs_list[0].combat_hb.shape[0]
        c_feat = obs_list[0].combat_hb.shape[-1]
        t_feat = obs_list[0].terrain_hb.shape[-1]
        g_dim = obs_list[0].global_state.shape[-1]

        chb = np.zeros((T, N, max_combat, c_feat), dtype=np.float32)
        cm = np.zeros((T, N, max_combat), dtype=np.float32)
        ckid = np.zeros((T, N, max_combat), dtype=np.int64)
        cpid = np.zeros((T, N, max_combat), dtype=np.int64)
        thb = np.zeros((T, N, max_terrain, t_feat), dtype=np.float32)
        tm = np.zeros((T, N, max_terrain), dtype=np.float32)
        gs = np.zeros((T, N, g_dim), dtype=np.float32)

        for t, o in enumerate(obs_list):
            nc = o.combat_hb.shape[1]
            chb[t, :, :nc] = o.combat_hb
            cm[t, :, :nc] = o.combat_mask
            ckid[t, :, :nc] = o.combat_kind_ids
            cpid[t, :, :nc] = o.combat_parent_ids
            nt = o.terrain_hb.shape[1]
            thb[t, :, :nt] = o.terrain_hb
            tm[t, :, :nt] = o.terrain_mask
            gs[t] = o.global_state

        return Observation(
            combat_hb=chb,
            combat_mask=cm,
            combat_kind_ids=ckid,
            combat_parent_ids=cpid,
            terrain_hb=thb,
            terrain_mask=tm,
            global_state=gs,
        )

    def field_names(self) -> list:
        return [f.name for f in fields(self)]


# ---------------------------------------------------------------------------
# Horizontal-mirror augmentation. Hollow Knight is left/right symmetric, so
# every (obs, action) pair has a valid mirror twin under x-axis negation —
# free 2x data exposure in expectation when applied stochastically during
# training.
# ---------------------------------------------------------------------------

def mirror_observation(obs: "Observation") -> "Observation":
    """World x-axis flip on a torch-tensor Observation.

    - global_state: vel_x flips; hp/soul/sizes/ability+validity flags and the
      commit block (direction-agnostic) pass through.
    - combat: rel_x and vel_x flip; size/flags/hp pass through. Masks/kind/parent
      ids unchanged.
    - terrain: mx, npx flip (knight-relative). hdy flips iff hdx > 0 to keep
      the canonical hdx ≥ 0 invariant after mirroring (geometrically the same
      segment, just represented from the opposite endpoint). Padded rows
      (hdx == 0, hdy == 0) stay zero.

    Any new column carrying an x-component or a handedness MUST be flipped here
    — a miss is silent and corrupts half the augmented data.
    """
    gs = obs.global_state.clone()
    gs[..., GS.VEL_X] = -gs[..., GS.VEL_X]

    chb = obs.combat_hb.clone()
    chb[..., CB.REL_X] = -chb[..., CB.REL_X]
    chb[..., CB.VEL_X] = -chb[..., CB.VEL_X]

    thb = obs.terrain_hb.clone()
    thb[..., TR.MX] = -thb[..., TR.MX]
    thb[..., TR.NPX] = -thb[..., TR.NPX]
    hdx = thb[..., TR.HDX]
    thb[..., TR.HDY] = torch.where(hdx > 0, -thb[..., TR.HDY], thb[..., TR.HDY])

    return obs.replace(
        global_state=gs,
        combat_hb=chb,
        terrain_hb=thb,
    )


def mirror_movement(movement):
    """Swap movement labels: 0 (left) ↔ 1 (right); 2 (none) unchanged."""
    return torch.where(movement == 2, movement, 1 - movement)


# ---------------------------------------------------------------------------
# Terrain view-box gate. Drops segments whose nearest point lies outside the
# knight-relative axis-aligned box (view_w × view_h). Used uniformly by
# rollout collection (vec_env) and eval (batch_obs) so train/eval distributions
# match. Operates on raw (pre-normalization) terrain rows where NPX/NPY are
# in world units.
# ---------------------------------------------------------------------------

def filter_terrain_in_view(terrain_hb, view_w, view_h, kinds=None, parents=None):
    """Filter terrain rows whose nearest-point lies inside the knight-relative
    box (|NPX| <= view_w/2, |NPY| <= view_h/2). Returns the filtered terrain
    array; if `kinds`/`parents` are provided, also returns aligned filtered
    lists. No-op when view_w or view_h is falsy or when there are no rows.
    """
    arr = np.asarray(terrain_hb, dtype=np.float32) if not isinstance(terrain_hb, np.ndarray) else terrain_hb
    if not view_w or not view_h or arr.shape[0] == 0:
        if kinds is None and parents is None:
            return arr
        return arr, kinds, parents
    npx = arr[:, TR.NPX]
    npy = arr[:, TR.NPY]
    keep = (np.abs(npx) <= view_w / 2.0) & (np.abs(npy) <= view_h / 2.0)
    out = arr[keep]
    if kinds is None and parents is None:
        return out
    keep_idx = np.where(keep)[0].tolist()
    fk = [kinds[i] for i in keep_idx] if kinds is not None else None
    fp = [parents[i] for i in keep_idx] if parents is not None else None
    return out, fk, fp
