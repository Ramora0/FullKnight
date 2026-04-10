"""Binary wire protocol for HK environment communication.

Replaces JSON serialization with flat binary packing for ~10x faster
encode/decode of hitbox arrays and observation data.
"""
import struct
import numpy as np

# Message type IDs (must match C# BinaryProtocol)
MSG_INIT   = 0
MSG_RESET  = 1
MSG_STEP   = 2
MSG_ACTION = 3
MSG_PAUSE  = 4
MSG_RESUME = 5
MSG_CLOSE  = 6

COMBAT_FEAT  = 7   # must match config.combat_feature_dim
TERRAIN_FEAT = 5   # must match config.terrain_feature_dim
GLOBAL_DIM   = 23  # must match config.global_state_dim

# --- Pack (Python -> C#) ---

def pack_init():
    return struct.pack('B', MSG_INIT)

def pack_reset(level, frames_per_wait, time_scale, eval_mode=False):
    level_bytes = level.encode('utf-8')
    return struct.pack(f'<BiiB H{len(level_bytes)}s',
                       MSG_RESET, frames_per_wait, time_scale,
                       1 if eval_mode else 0,
                       len(level_bytes), level_bytes)

def pack_action(action_vec):
    return struct.pack('<Biiii', MSG_ACTION, *action_vec)

def pack_pause():
    return struct.pack('B', MSG_PAUSE)

def pack_resume():
    return struct.pack('B', MSG_RESUME)

# --- Unpack (C# -> Python) ---

def unpack_obs(data, offset=1):
    """Unpack observation float fields (combat_hb, terrain_hb, global_state).
    Returns (combat_hb, terrain_hb, global_state, n_combat, new_offset).
    Kind strings are appended after step metadata; use unpack_kinds to read them."""
    n_combat, n_terrain = struct.unpack_from('<HH', data, offset)
    offset += 4

    combat_n_floats = n_combat * COMBAT_FEAT
    combat_hb = np.frombuffer(data, dtype='<f4', count=combat_n_floats, offset=offset).reshape(-1, COMBAT_FEAT).copy()
    offset += combat_n_floats * 4

    terrain_n_floats = n_terrain * TERRAIN_FEAT
    terrain_hb = np.frombuffer(data, dtype='<f4', count=terrain_n_floats, offset=offset).reshape(-1, TERRAIN_FEAT).copy()
    offset += terrain_n_floats * 4

    global_state = np.frombuffer(data, dtype='<f4', count=GLOBAL_DIM, offset=offset).copy()
    offset += GLOBAL_DIM * 4

    return combat_hb, terrain_hb, global_state, n_combat, offset

def unpack_kinds(data, offset, n):
    """Read n kind strings (u8 length + UTF-8 bytes each)."""
    kinds = []
    for _ in range(n):
        ln = data[offset]
        offset += 1
        kinds.append(bytes(data[offset:offset + ln]).decode('utf-8'))
        offset += ln
    return kinds, offset

def unpack_step(data):
    """Unpack a step response.
    Returns (combat_hb, terrain_hb, gs, combat_kinds, damage_landed, hits_taken,
             game_time, real_time, done)."""
    combat_hb, terrain_hb, gs, n_combat, offset = unpack_obs(data)
    damage_landed, hits_taken, game_time, real_time = struct.unpack_from('<ffff', data, offset)
    offset += 16
    done = data[offset] != 0
    offset += 1
    combat_kinds, offset = unpack_kinds(data, offset, n_combat)
    return combat_hb, terrain_hb, gs, combat_kinds, damage_landed, hits_taken, game_time, real_time, done

def unpack_reset(data):
    """Unpack a reset response. Returns (combat_hb, terrain_hb, gs, combat_kinds)."""
    combat_hb, terrain_hb, gs, n_combat, offset = unpack_obs(data)
    combat_kinds, _ = unpack_kinds(data, offset, n_combat)
    return combat_hb, terrain_hb, gs, combat_kinds
