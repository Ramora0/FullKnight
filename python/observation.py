"""Observation dataclass and named index constants.

Centralizes "the seven things that flow together through the model and
training pipeline" so adding a per-hitbox or global-state field doesn't
require touching every function signature in the repo.
"""
from dataclasses import dataclass, fields, replace
from typing import Any, List
import numpy as np


# ---------------------------------------------------------------------------
# Column index constants (use these instead of gs[N] / combat_hb[..., N]).
# When a column moves, fix it here and the rest of the codebase comes with it.
# ---------------------------------------------------------------------------

class GS:
    """Global state column indices (22 floats)."""
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


class CB:
    """Combat hitbox feature column indices (10 floats)."""
    REL_X = 0
    REL_Y = 1
    W = 2
    H = 3
    IS_TRIGGER = 4
    GIVES_DAMAGE = 5
    TAKES_DAMAGE = 6
    IS_TARGET = 7
    HP_RAW = 8       # current HP, raw on the wire; log1p-compressed before the model
    HP_MAX_RAW = 9   # observed max HP (cached on first sight, refill-aware), same treatment


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
