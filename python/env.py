import os
import time
import numpy as np
from binary_protocol import (
    pack_init, pack_reset, pack_action, pack_pause, pack_resume,
    unpack_reset, unpack_step, pop_last_terrain_debug, pop_last_diag,
    pop_last_reset_phases, pop_last_fsm_snapshots, pop_last_fsm_dump,
    MSG_CLOSE,
)
import struct

# Threshold (seconds) above which a step's send/recv breakdown is printed.
# Send is normally <1ms; recv is dominated by the C# coroutine. If a slow step
# shows large send time, IPC is the issue; if recv dominates, look at the
# C# [Step-Timing] / [Phase-Timing] logs from the same epoch.
_SLOW_OP_THRESHOLD_S = 2.0

# Where one-shot per-scene PlayMaker graph dumps land.
FSM_DUMP_DIR = "fsm_dumps"


class HKEnv:
    """Wraps a single WebSocket connection to a Hollow Knight game instance.

    All wire-protocol details (binary packing) are encapsulated here.
    Callers only see numpy arrays and Python scalars.
    """

    def __init__(self, websocket, config):
        self.ws = websocket
        self.config = config
        # Debug-only: last terrain_debug strings pulled off the wire.
        # Populated after each reset/step by reading the protocol side channel.
        self.last_terrain_debug: list = []
        # Diag block (leak probes) from the most recent step — populated by
        # step(). vec_env reads this to aggregate per-epoch perf metrics.
        self.last_diag: dict = {
            "enemy_count": 0, "attack_count": 0, "terrain_count": 0,
            "kind_cache_size": 0, "gc_heap_mb": 0.0,
        }
        # Last-reset phase breakdown from the C# mod. Populated by reset()
        # and drained by VecEnv at reap time into the TimingTracker.
        self.last_reset_phases: dict = {}
        # FSM-snapshot list from the most recent step/reset. List of
        # "<src>|<owner>|<fsm>|<state>" strings produced by C# FsmObserver.
        # Consumed by the visualizer / fsm_tracker, never by training.
        self.last_fsm: list = []

    async def init(self):
        """Send init handshake and wait for ack."""
        await self.ws.send(pack_init())
        await self.ws.recv()

    async def reset(self, eval_mode=False, level=None):
        """Reset environment.
        Returns (combat_hb, terrain_hb, global_state, combat_kinds,
        combat_parents, combat_anims)."""
        t0 = time.perf_counter()
        await self.ws.send(pack_reset(
            level if level is not None else self.config.level,
            self.config.frames_per_wait,
            self.config.time_scale, eval_mode=eval_mode,
            fsm_debug=bool(getattr(self.config, "save_fsm_graph", False)
                           or getattr(self.config, "visualize", False)),
        ))
        t1 = time.perf_counter()
        data = await self.ws.recv()
        t2 = time.perf_counter()
        if (t2 - t0) > _SLOW_OP_THRESHOLD_S:
            print(f"    [reset-wire] level={level} send={(t1-t0)*1000:.0f}ms"
                  f" recv={(t2-t1)*1000:.0f}ms total={(t2-t0)*1000:.0f}ms",
                  flush=True)
        result = unpack_reset(data)
        self.last_terrain_debug = pop_last_terrain_debug()
        self.last_reset_phases = pop_last_reset_phases()
        self.last_fsm = pop_last_fsm_snapshots()
        self._save_fsm_dump(pop_last_fsm_dump(),
                            level if level is not None else self.config.level)
        return result

    @staticmethod
    def _save_fsm_dump(dump_json, level):
        """Persist the one-shot PlayMaker graph dump for a scene.

        The C# side sends this on the first load of each scene per process, so
        with N instances all N send an identical payload for the same boss.
        First writer wins; the rest are no-ops. This is ground truth about the
        boss (states, transitions, action parameters, animation clip tables) —
        the data FsmTracker currently reconstructs by observation.
        """
        if not dump_json:
            return
        try:
            os.makedirs(FSM_DUMP_DIR, exist_ok=True)
            path = os.path.join(FSM_DUMP_DIR, f"{level}.json")
            if os.path.exists(path):
                return
            # Write-then-rename so concurrent instances can't observe a
            # half-written file.
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(dump_json)
            os.replace(tmp, path)
            print(f"  [fsm_dump] wrote {path} ({len(dump_json):,} bytes)", flush=True)
        except Exception as e:
            print(f"  [fsm_dump] failed for {level}: {e}", flush=True)

    async def step(self, action_vec):
        """Take a step. action_vec = [movement, direction, action, jump].
        Returns (combat_hb, terrain_hb, global_state, combat_kinds, combat_parents,
                 combat_anims, damage_landed, hits_taken, hp_healed,
                 step_game_time, step_real_time, done, committed).
        """
        t0 = time.perf_counter()
        await self.ws.send(pack_action(action_vec))
        t1 = time.perf_counter()
        data = await self.ws.recv()
        t2 = time.perf_counter()
        if (t2 - t0) > _SLOW_OP_THRESHOLD_S:
            print(f"    [step-wire] action={action_vec} send={(t1-t0)*1000:.0f}ms"
                  f" recv={(t2-t1)*1000:.0f}ms total={(t2-t0)*1000:.0f}ms",
                  flush=True)
        (combat_hb, terrain_hb, gs, combat_kinds, combat_parents, combat_anims,
         damage_landed, hits_taken, hp_healed, game_time, real_time, done,
         committed) = unpack_step(data)
        self.last_terrain_debug = pop_last_terrain_debug()
        self.last_diag = pop_last_diag()
        self.last_fsm = pop_last_fsm_snapshots()
        return (combat_hb, terrain_hb, gs, combat_kinds, combat_parents,
                combat_anims, damage_landed, hits_taken, hp_healed, game_time,
                real_time, done, committed)

    async def step_eval(self, action_vec):
        """Like step() but also returns done flag. For eval mode."""
        await self.ws.send(pack_action(action_vec))
        data = await self.ws.recv()
        result = unpack_step(data)
        self.last_terrain_debug = pop_last_terrain_debug()
        return result

    async def pause(self):
        await self.ws.send(pack_pause())
        await self.ws.recv()

    async def resume(self):
        await self.ws.send(pack_resume())
        await self.ws.recv()

    async def close(self):
        await self.ws.send(struct.pack('B', MSG_CLOSE))
