import time
import numpy as np
from binary_protocol import (
    pack_init, pack_reset, pack_action, pack_pause, pack_resume,
    unpack_reset, unpack_step, pop_last_terrain_debug, pop_last_diag,
    pop_last_reset_phases, MSG_CLOSE,
)
import struct

# Threshold (seconds) above which a step's send/recv breakdown is printed.
# Send is normally <1ms; recv is dominated by the C# coroutine. If a slow step
# shows large send time, IPC is the issue; if recv dominates, look at the
# C# [Step-Timing] / [Phase-Timing] logs from the same epoch.
_SLOW_OP_THRESHOLD_S = 2.0


class HKEnv:
    """Wraps a single WebSocket connection to a Hollow Knight game instance.

    All wire-protocol details (binary packing) are encapsulated here.
    Callers only see numpy arrays and Python scalars.
    """

    def __init__(self, websocket, config, idx=0):
        self.ws = websocket
        self.config = config
        # Numeric env slot, used in debug-transitions wire prints so the
        # user can disambiguate which game instance issued each command.
        self.idx = idx
        # Debug-only: last terrain_debug strings pulled off the wire.
        # Populated after each reset/step by reading the protocol side channel.
        self.last_terrain_debug: list = []
        # Diag block (leak probes) from the most recent step — populated by
        # step(). vec_env reads this to aggregate per-epoch perf metrics.
        self.last_diag: dict = {
            "enemy_count": 0, "attack_count": 0, "terrain_count": 0,
            "kind_cache_size": 0, "gc_heap_mb": 0.0,
        }
        # Per-phase ms breakdown of the most recent reset, populated by
        # reset(). vec_env reads this in reap_completed_resets() so the
        # train-time diagnostic can attribute the 8s reset average.
        self.last_reset_phases: dict = {}

    def _dbg(self, msg):
        """Print a wire-level message in debug-transitions mode. Format:
        `[HH:MM:SS.mmm env <idx>] <msg>`. Always flush so a stuck await
        doesn't hide the most recent line in stdout buffering."""
        if getattr(self.config, "debug_transitions", False):
            ts = time.strftime("%H:%M:%S", time.localtime())
            ms = int((time.time() % 1) * 1000)
            print(f"  [{ts}.{ms:03d} env {self.idx}] {msg}", flush=True)

    async def init(self):
        """Send init handshake and wait for ack."""
        self._dbg("-> INIT")
        await self.ws.send(pack_init())
        await self.ws.recv()
        self._dbg("<- INIT ack")

    async def reset(self, eval_mode=False, level=None):
        """Reset environment.
        Returns (combat_hb, terrain_hb, global_state, combat_kinds, combat_parents)."""
        self._dbg(f"-> RESET (level={level}, eval={eval_mode})")
        t0 = time.perf_counter()
        await self.ws.send(pack_reset(
            level if level is not None else self.config.level,
            self.config.frames_per_wait,
            self.config.time_scale, eval_mode=eval_mode,
        ))
        t1 = time.perf_counter()
        data = await self.ws.recv()
        t2 = time.perf_counter()
        if (t2 - t0) > _SLOW_OP_THRESHOLD_S:
            print(f"    [reset-wire] env {self.idx} level={level} "
                  f"send={(t1-t0)*1000:.0f}ms"
                  f" recv={(t2-t1)*1000:.0f}ms total={(t2-t0)*1000:.0f}ms",
                  flush=True)
        result = unpack_reset(data)
        self.last_terrain_debug = pop_last_terrain_debug()
        self.last_reset_phases = pop_last_reset_phases()
        # Summarize what came back so the user can tell whether the boss
        # scene is really loaded (knight bounds non-zero, hp populated,
        # combat hitboxes present). Stuck-in-godhome typically shows up as
        # knight=0x0 with combat=0 right after a "successful" reset.
        if getattr(self.config, "debug_transitions", False):
            combat_hb, terrain_hb, gs, _, _ = result
            kn_w = float(gs[4]) if len(gs) > 4 else 0.0
            kn_h = float(gs[5]) if len(gs) > 5 else 0.0
            hp = int(gs[2]) if len(gs) > 2 else 0
            self._dbg(
                f"<- RESET {(t2-t1)*1000:.0f}ms "
                f"combat={len(combat_hb)} terrain={len(terrain_hb)} "
                f"knight={kn_w:.1f}x{kn_h:.1f} hp={hp}"
            )
        return result

    async def step(self, action_vec):
        """Take a step. action_vec = [movement, direction, action, jump].
        Returns (combat_hb, terrain_hb, global_state, combat_kinds, combat_parents,
                 damage_landed, hits_taken, hp_healed, step_game_time, step_real_time, done,
                 committed).
        """
        self._dbg(f"-> STEP action={action_vec}")
        t0 = time.perf_counter()
        await self.ws.send(pack_action(action_vec))
        t1 = time.perf_counter()
        data = await self.ws.recv()
        t2 = time.perf_counter()
        if (t2 - t0) > _SLOW_OP_THRESHOLD_S:
            print(f"    [step-wire] env {self.idx} action={action_vec} "
                  f"send={(t1-t0)*1000:.0f}ms"
                  f" recv={(t2-t1)*1000:.0f}ms total={(t2-t0)*1000:.0f}ms",
                  flush=True)
        (combat_hb, terrain_hb, gs, combat_kinds, combat_parents,
         damage_landed, hits_taken, hp_healed, game_time, real_time, done,
         committed) = unpack_step(data)
        self.last_terrain_debug = pop_last_terrain_debug()
        self.last_diag = pop_last_diag()
        if getattr(self.config, "debug_transitions", False):
            kn_w = float(gs[4]) if len(gs) > 4 else 0.0
            kn_h = float(gs[5]) if len(gs) > 5 else 0.0
            hp = int(gs[2]) if len(gs) > 2 else 0
            self._dbg(
                f"<- STEP {(t2-t1)*1000:.0f}ms done={done} "
                f"combat={len(combat_hb)} terrain={len(terrain_hb)} "
                f"knight={kn_w:.1f}x{kn_h:.1f} hp={hp} "
                f"dmg={damage_landed:.2f} hits={int(hits_taken)} "
                f"healed={int(hp_healed)}"
            )
        return (combat_hb, terrain_hb, gs, combat_kinds, combat_parents,
                damage_landed, hits_taken, hp_healed, game_time, real_time, done,
                committed)

    async def step_eval(self, action_vec):
        """Like step() but also returns done flag. For eval mode."""
        await self.ws.send(pack_action(action_vec))
        data = await self.ws.recv()
        result = unpack_step(data)
        self.last_terrain_debug = pop_last_terrain_debug()
        return result

    async def pause(self):
        self._dbg("-> PAUSE")
        await self.ws.send(pack_pause())
        await self.ws.recv()
        self._dbg("<- PAUSE ack")

    async def resume(self):
        self._dbg("-> RESUME")
        await self.ws.send(pack_resume())
        await self.ws.recv()
        self._dbg("<- RESUME ack")

    async def close(self):
        self._dbg("-> CLOSE")
        await self.ws.send(struct.pack('B', MSG_CLOSE))
