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

COMBAT_FEAT  = 10  # must match config.combat_feature_dim
TERRAIN_FEAT = 8   # must match config.terrain_feature_dim
GLOBAL_DIM   = 22  # must match config.global_state_dim

# --- Pack (Python -> C#) ---

def pack_init(slot=0, use_shm=False):
    """Init handshake. Wire: [u8 MSG_INIT][u32 slot][u8 use_shm].
    `slot` is the per-instance index used to address the shared-memory
    channel for this env (Local\\fk_inst_{slot}_*). `use_shm` tells the C#
    side whether to actually open its end and route step() through shm
    (vs keep the WebSocket-only legacy path). Both trailers are read
    length-defensively, so an older mod DLL that ignores them still parses
    the leading byte cleanly."""
    return struct.pack('<BIB', MSG_INIT, int(slot), 1 if use_shm else 0)

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

_KIND_PROTOCOL_WARNED = False

def unpack_kinds(data, offset, n):
    """Read n kind strings (u8 length + UTF-8 bytes each).

    Defensive: if the buffer is exhausted (e.g. C# mod DLL is older than the
    Python side and never appends kinds), fall back to "unknown" for every
    hitbox and warn loudly once."""
    global _KIND_PROTOCOL_WARNED
    kinds = []
    for _ in range(n):
        if offset >= len(data):
            if not _KIND_PROTOCOL_WARNED:
                _KIND_PROTOCOL_WARNED = True
                bar = "!" * 78
                print("\n" + bar, flush=True)
                print("!!  KIND-PROTOCOL MISMATCH: C# response has no kind strings", flush=True)
                print(f"!!  Expected {n} kind entries, got 0 bytes left in buffer.", flush=True)
                print("!!  The Hollow Knight mod DLL is older than the Python protocol.", flush=True)
                print("!!  Rebuild the C# mod (`dotnet build -c Debug`) and reinstall.", flush=True)
                print("!!  Falling back to 'unknown' for all kinds until then.", flush=True)
                print(bar + "\n", flush=True)
            kinds.extend(["unknown"] * (n - len(kinds)))
            return kinds, offset
        ln = data[offset]
        offset += 1
        kinds.append(bytes(data[offset:offset + ln]).decode('utf-8'))
        offset += ln
    return kinds, offset

def unpack_terrain_debug(data, offset, n):
    """Read n terrain debug strings (u16 length + UTF-8 bytes each).

    Backwards-compatible: if the buffer is exhausted (old C# mod), returns
    empty strings and does not raise. The terrain debug channel is a
    temporary diagnostic for the observation viewer."""
    out = []
    for _ in range(n):
        if offset + 2 > len(data):
            out.extend([""] * (n - len(out)))
            return out, offset
        ln = struct.unpack_from('<H', data, offset)[0]
        offset += 2
        if offset + ln > len(data):
            out.append("")
            continue
        out.append(bytes(data[offset:offset + ln]).decode('utf-8', errors='replace'))
        offset += ln
    return out, offset


# Side channel: the last `terrain_debug` list pulled off the wire. Only
# populated by unpack_reset / unpack_step and only read by the debug viewer.
# Kept out of the main return tuple so training / model code is unchanged.
_last_terrain_debug: list = []

def pop_last_terrain_debug():
    """Return (and clear) the most-recently-received terrain debug strings."""
    global _last_terrain_debug
    out = _last_terrain_debug
    _last_terrain_debug = []
    return out


# Side channel: last FSM snapshot list pulled off the wire. Each entry is
# "<src>|<owner>|<fsm>|<state>" produced by FsmObserver C#-side. Only read
# by the debug viewer; training/model code never touches it.
_last_fsm_snapshots: list = []

def pop_last_fsm_snapshots():
    """Return (and clear) the most-recently-received FSM snapshot list."""
    global _last_fsm_snapshots
    out = _last_fsm_snapshots
    _last_fsm_snapshots = []
    return out


def _unpack_fsm_snapshots(data, offset):
    """Decode the trailing FSM-snapshot block.

    Layout: u16 count, then per-entry u16 length + UTF-8 bytes. Returns
    (list_of_strings, new_offset). Defensive: missing block (older mod DLL)
    returns []; truncated entry returns what we got so far without raising.
    """
    if offset + 2 > len(data):
        return [], offset
    n = struct.unpack_from('<H', data, offset)[0]
    offset += 2
    out = []
    for _ in range(n):
        if offset + 2 > len(data):
            return out, offset
        ln = struct.unpack_from('<H', data, offset)[0]
        offset += 2
        if offset + ln > len(data):
            return out, offset
        out.append(bytes(data[offset:offset + ln]).decode('utf-8', errors='replace'))
        offset += ln
    return out, offset


_DIAG_FMT = '<HHHif'  # enemy_cnt, attack_cnt, terrain_cnt, kind_cache_size, gc_heap_mb
_DIAG_SIZE = struct.calcsize(_DIAG_FMT)  # 14 bytes

# Reset phase trailer — 7 u16 phase deltas (ms) + 7 u16 phase frame counts
# + 1 u8 branch flag. Order matches C# TrainingEnv.Reset() LogPhase call
# sequence. Older C# DLLs only emit the 15-byte (ms+branch) variant; we
# detect by length and parse the frames block when present.
_RESET_PHASES_FMT_FULL = '<HHHHHHHHHHHHHHB'  # 7 ms + 7 frames + 1 branch = 29 bytes
_RESET_PHASES_FMT_LEGACY = '<HHHHHHHB'        # 7 ms + 1 branch = 15 bytes
_RESET_PHASES_SIZE_FULL = struct.calcsize(_RESET_PHASES_FMT_FULL)
_RESET_PHASES_SIZE_LEGACY = struct.calcsize(_RESET_PHASES_FMT_LEGACY)
RESET_PHASE_NAMES = (
    "pre_unload", "transition_out", "settle", "load_boss_scene",
    "recreate_reader", "init_boss_refs", "obs_final",
)
RESET_BRANCH_NAMES = {0: "workshop", 1: "natural_end"}

# Side channel: last diag tuple pulled off the wire. Populated by unpack_step,
# read by vec_env so we don't balloon the main return tuple on every caller.
_last_diag = {
    "enemy_count": 0,
    "attack_count": 0,
    "terrain_count": 0,
    "kind_cache_size": 0,
    "gc_heap_mb": 0.0,
}

def pop_last_diag():
    """Return the most-recently-received diag block (dict)."""
    return dict(_last_diag)


# Side channel: per-reset phase breakdown pulled off the wire by unpack_reset.
# Read by HKEnv.reset() and ferried up to vec_env's reap_completed_resets.
_last_reset_phases: dict = {}

def pop_last_reset_phases():
    """Return (and clear) the most-recently-parsed reset phase dict."""
    global _last_reset_phases
    out = _last_reset_phases
    _last_reset_phases = {}
    return out


def unpack_step(data):
    """Unpack a step response.
    Returns (combat_hb, terrain_hb, gs, combat_kinds, combat_parents,
             damage_landed, hits_taken, hp_healed, game_time, real_time, done,
             committed).
    `committed`: True iff action[2] was overridden by the C#-side hard-commit
    state machine this step. Python masks the action-head policy gradient on
    these steps. Defaults to False if absent (older C# DLL).
    Diag fields are stashed in _last_diag — read via pop_last_diag()."""
    global _last_terrain_debug, _last_diag, _last_fsm_snapshots
    combat_hb, terrain_hb, gs, n_combat, offset = unpack_obs(data)
    n_terrain = terrain_hb.shape[0]
    damage_landed, hits_taken, game_time, real_time, hp_healed = struct.unpack_from('<fffff', data, offset)
    offset += 20
    done = data[offset] != 0
    offset += 1
    committed = data[offset] != 0
    offset += 1
    combat_kinds, offset = unpack_kinds(data, offset, n_combat)
    combat_parents, offset = unpack_kinds(data, offset, n_combat)
    _last_terrain_debug, offset = unpack_terrain_debug(data, offset, n_terrain)
    # Diag trailer (14 bytes). Defensive: older C# mods don't send it.
    if offset + _DIAG_SIZE <= len(data):
        e, a, t, kc, heap_mb = struct.unpack_from(_DIAG_FMT, data, offset)
        offset += _DIAG_SIZE
        _last_diag = {
            "enemy_count": int(e),
            "attack_count": int(a),
            "terrain_count": int(t),
            "kind_cache_size": int(kc),
            "gc_heap_mb": float(heap_mb),
        }
    else:
        _last_diag = {
            "enemy_count": 0, "attack_count": 0, "terrain_count": 0,
            "kind_cache_size": 0, "gc_heap_mb": 0.0,
        }
    # FSM-snapshot block — appended after diag. Older C# mods don't send
    # it; _unpack_fsm_snapshots returns [] when the buffer is exhausted.
    _last_fsm_snapshots, offset = _unpack_fsm_snapshots(data, offset)
    return (combat_hb, terrain_hb, gs, combat_kinds, combat_parents,
            damage_landed, hits_taken, hp_healed, game_time, real_time, done,
            committed)

def unpack_reset(data):
    """Unpack a reset response.
    Returns (combat_hb, terrain_hb, gs, combat_kinds, combat_parents).
    Phase trailer (if present) is stashed in _last_reset_phases — read it
    via pop_last_reset_phases()."""
    global _last_terrain_debug, _last_reset_phases, _last_fsm_snapshots
    combat_hb, terrain_hb, gs, n_combat, offset = unpack_obs(data)
    n_terrain = terrain_hb.shape[0]
    combat_kinds, offset = unpack_kinds(data, offset, n_combat)
    combat_parents, offset = unpack_kinds(data, offset, n_combat)
    _last_terrain_debug, offset = unpack_terrain_debug(data, offset, n_terrain)
    # Reset phase trailer — defensive parse for older C# DLLs that don't
    # append it. Layout matches BinaryProtocol.Pack reset trailer. Prefer
    # the full (ms+frames+branch) layout; fall back to legacy (ms+branch).
    # The FSM-snapshot block (added later) sits AFTER this trailer, so we
    # advance `offset` past whichever variant we recognized before parsing it.
    remaining = len(data) - offset
    if remaining >= _RESET_PHASES_SIZE_FULL:
        vals = struct.unpack_from(_RESET_PHASES_FMT_FULL, data, offset)
        offset += _RESET_PHASES_SIZE_FULL
        phases = {n: float(v) for n, v in zip(RESET_PHASE_NAMES, vals[:7])}
        frames = {n: int(v) for n, v in zip(RESET_PHASE_NAMES, vals[7:14])}
        phases["frames"] = frames
        phases["branch"] = RESET_BRANCH_NAMES.get(int(vals[14]), "unknown")
        _last_reset_phases = phases
    elif remaining >= _RESET_PHASES_SIZE_LEGACY:
        vals = struct.unpack_from(_RESET_PHASES_FMT_LEGACY, data, offset)
        offset += _RESET_PHASES_SIZE_LEGACY
        phases = {n: float(v) for n, v in zip(RESET_PHASE_NAMES, vals[:7])}
        phases["frames"] = {n: 0 for n in RESET_PHASE_NAMES}
        phases["branch"] = RESET_BRANCH_NAMES.get(int(vals[7]), "unknown")
        _last_reset_phases = phases
    else:
        _last_reset_phases = {}
    # FSM-snapshot block — same defensive parse as in unpack_step.
    _last_fsm_snapshots, offset = _unpack_fsm_snapshots(data, offset)
    return combat_hb, terrain_hb, gs, combat_kinds, combat_parents
