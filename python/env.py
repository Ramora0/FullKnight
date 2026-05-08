import numpy as np
from binary_protocol import (
    pack_init, pack_reset, pack_action, pack_pause, pack_resume,
    unpack_reset, unpack_step, pop_last_terrain_debug, pop_last_diag, MSG_CLOSE,
)
import struct


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

    async def init(self):
        """Send init handshake and wait for ack."""
        await self.ws.send(pack_init())
        await self.ws.recv()

    async def reset(self, eval_mode=False, level=None):
        """Reset environment.
        Returns (combat_hb, terrain_hb, global_state, combat_kinds, combat_parents)."""
        await self.ws.send(pack_reset(
            level if level is not None else self.config.level,
            self.config.frames_per_wait,
            self.config.time_scale, eval_mode=eval_mode,
        ))
        data = await self.ws.recv()
        result = unpack_reset(data)
        self.last_terrain_debug = pop_last_terrain_debug()
        return result

    async def step(self, action_vec):
        """Take a step. action_vec = [movement, direction, action, jump].
        Returns (combat_hb, terrain_hb, global_state, combat_kinds, combat_parents,
                 damage_landed, hits_taken, hp_healed, step_game_time, step_real_time, done,
                 committed).
        """
        await self.ws.send(pack_action(action_vec))
        data = await self.ws.recv()
        (combat_hb, terrain_hb, gs, combat_kinds, combat_parents,
         damage_landed, hits_taken, hp_healed, game_time, real_time, done,
         committed) = unpack_step(data)
        self.last_terrain_debug = pop_last_terrain_debug()
        self.last_diag = pop_last_diag()
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
        await self.ws.send(pack_pause())
        await self.ws.recv()

    async def resume(self):
        await self.ws.send(pack_resume())
        await self.ws.recv()

    async def close(self):
        await self.ws.send(struct.pack('B', MSG_CLOSE))
