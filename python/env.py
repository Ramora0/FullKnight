import json
import numpy as np


class HKEnv:
    """Wraps a single WebSocket connection to a Hollow Knight game instance."""

    def __init__(self, websocket, config):
        self.ws = websocket
        self.config = config

    async def send_and_recv(self, msg_type, data):
        message = json.dumps({"type": msg_type, "data": data, "sender": "server"})
        await self.ws.send(message)
        response = json.loads(await self.ws.recv())
        return response

    async def reset(self):
        """Reset environment. Returns (combat_hb, terrain_hb, global_state)."""
        resp = await self.send_and_recv("reset", {
            "level": self.config.level,
            "frames_per_wait": self.config.frames_per_wait,
            "time_scale": self.config.time_scale,
        })
        return self._parse_obs(resp["data"])

    async def step(self, action_vec):
        """Take a step. action_vec = [movement, direction, action, jump].
        Returns (combat_hb, terrain_hb, global_state, reward, done, info).
        """
        resp = await self.send_and_recv("action", {"action_vec": action_vec})
        d = resp["data"]
        combat_hb, terrain_hb, gs = self._parse_obs(d)
        return combat_hb, terrain_hb, gs, d.get("reward", 0.0), d.get("done", False), d.get("info", "")

    async def pause(self):
        await self.send_and_recv("pause", {})

    async def resume(self):
        await self.send_and_recv("resume", {})

    def _parse_obs(self, data):
        combat_hb = data.get("combat_hitboxes", [])
        terrain_hb = data.get("terrain_hitboxes", [])
        gs = np.array(data.get("global_state", [0.0] * self.config.global_state_dim), dtype=np.float32)
        return combat_hb, terrain_hb, gs
