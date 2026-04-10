import numpy as np
from binary_protocol import (
    pack_init, pack_reset, pack_action, pack_pause, pack_resume,
    unpack_reset, unpack_step, MSG_CLOSE,
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
        return unpack_reset(data)

    async def step(self, action_vec):
        """Take a step. action_vec = [movement, direction, action, jump].
        Returns (combat_hb, terrain_hb, global_state, combat_kinds, combat_parents,
                 damage_landed, hits_taken, step_game_time, step_real_time).
        """
        await self.ws.send(pack_action(action_vec))
        data = await self.ws.recv()
        (combat_hb, terrain_hb, gs, combat_kinds, combat_parents,
         damage_landed, hits_taken, game_time, real_time, done) = unpack_step(data)
        return (combat_hb, terrain_hb, gs, combat_kinds, combat_parents,
                damage_landed, hits_taken, game_time, real_time)

    async def step_eval(self, action_vec):
        """Like step() but also returns done flag. For eval mode."""
        await self.ws.send(pack_action(action_vec))
        data = await self.ws.recv()
        return unpack_step(data)

    async def pause(self):
        await self.ws.send(pack_pause())
        await self.ws.recv()

    async def resume(self):
        await self.ws.send(pack_resume())
        await self.ws.recv()

    async def close(self):
        await self.ws.send(struct.pack('B', MSG_CLOSE))
