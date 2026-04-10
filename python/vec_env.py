import asyncio
import time
import numpy as np
import websockets

from env import HKEnv
from vocab import KindVocab


class VecEnv:
    """Manages N parallel HK game instances via WebSocket connections."""

    def __init__(self, config):
        self.config = config
        self.n_envs = config.n_envs
        self.envs = [None] * self.n_envs
        self.connected = [asyncio.Event() for _ in range(self.n_envs)]
        self._ws_connections = [None] * self.n_envs
        self._server = None
        self.vocab = KindVocab(max_size=config.kind_vocab_size)

    async def start_server(self):
        """Start WebSocket server and wait for all N connections."""
        self._server = await websockets.serve(
            self._on_connect, self.config.server_host, self.config.server_port
        )
        print(f"WebSocket server started on ws://{self.config.server_host}:{self.config.server_port}")
        print(f"Waiting for {self.n_envs} game instances to connect...")
        await asyncio.gather(*[e.wait() for e in self.connected])
        print(f"All {self.n_envs} instances connected.")

    async def _on_connect(self, websocket):
        try:
            idx = self._ws_connections.index(None)
        except ValueError:
            print("Extra connection rejected — all slots full.")
            await websocket.close()
            return

        self._ws_connections[idx] = websocket
        self.envs[idx] = HKEnv(websocket, self.config)
        print(f"Instance {idx} connected.")

        await self.envs[idx].init()
        self.connected[idx].set()

        # Keep connection alive
        try:
            await asyncio.Future()
        except websockets.exceptions.ConnectionClosed:
            print(f"Instance {idx} disconnected.")
            self._ws_connections[idx] = None
            self.envs[idx] = None
            self.connected[idx].clear()

    async def _timed_op(self, label, idx, coro, loud=False):
        """Wrap a coroutine with per-env timing. Prints immediately on completion."""
        t0 = time.perf_counter()
        result = await coro
        dt = time.perf_counter() - t0
        if loud or dt > 2.0:
            print(f"  {label} env {idx}: done in {dt:.1f}s", flush=True)
        return idx, dt, result

    async def reset_all(self):
        """Reset all envs. Returns batched
        (combat_hb, combat_mask, combat_kinds, terrain_hb, terrain_mask, global_states)."""
        t0 = time.perf_counter()
        timed = await asyncio.gather(*[
            self._timed_op("reset", i, env.reset()) for i, env in enumerate(self.envs)
        ])
        total = time.perf_counter() - t0
        print(f"reset_all: {self.n_envs} envs in {total:.1f}s", flush=True)
        results = [r for _, _, r in sorted(timed, key=lambda x: x[0])]
        return self._batch_observations(results)

    async def step_all(self, actions):
        """Step all envs in parallel.
        actions: list of N action_vecs, each [movement, direction, action, jump].
        Returns (combat_hb, combat_mask, combat_kinds, terrain_hb, terrain_mask,
                 global_states, damage_landed, hits_taken, step_game_times, step_real_times).
        """
        t0 = time.perf_counter()
        timed = await asyncio.gather(*[
            self._timed_op("step", i, self.envs[i].step(actions[i])) for i in range(self.n_envs)
        ])
        total = time.perf_counter() - t0

        sorted_timed = sorted(timed, key=lambda x: x[0])
        results = [r for _, _, r in sorted_timed]
        (combat_lists, terrain_lists, gs_list, combat_kind_lists,
         damage_landed, hits_taken, step_game_times, step_real_times) = zip(*results)

        obs_batch = self._batch_observations(list(zip(
            combat_lists, terrain_lists, gs_list, combat_kind_lists)))
        damage_landed = np.array(damage_landed, dtype=np.float32)
        hits_taken = np.array(hits_taken, dtype=np.float32)
        step_game_times = np.array(step_game_times, dtype=np.float32)
        step_real_times = np.array(step_real_times, dtype=np.float32)
        return *obs_batch, damage_landed, hits_taken, step_game_times, step_real_times

    async def reset_and_resume(self, reset_indices):
        """Reset specified envs, resume the rest. Returns observations for reset envs only."""
        reset_set = set(reset_indices)
        tasks = []
        for i in range(self.n_envs):
            if i in reset_set:
                tasks.append(self._timed_op("reset", i, self.envs[i].reset()))
            else:
                tasks.append(self._timed_op("resume", i, self.envs[i].resume()))
        results = await asyncio.gather(*tasks)
        reset_obs = [results[i][2] for i in reset_indices]
        return reset_indices, self._batch_observations(reset_obs)

    async def pause_all(self):
        await asyncio.gather(*[env.pause() for env in self.envs])

    async def resume_all(self):
        await asyncio.gather(*[env.resume() for env in self.envs])

    def _batch_observations(self, obs_list):
        """Pad hitbox lists and stack into tensors.

        obs_list: list of (combat_hb, terrain_hb, global_state, combat_kinds) tuples.
        Returns (combat_hb_batch, combat_mask, combat_kind_ids,
                 terrain_hb_batch, terrain_mask, gs_batch).
        combat_kind_ids: int32 (B, max_combat); padding rows are 0 ("unknown" id).
        """
        combat_lists = [obs[0] for obs in obs_list]
        terrain_lists = [obs[1] for obs in obs_list]
        gs_list = [obs[2] for obs in obs_list]
        kind_lists = [obs[3] if len(obs) > 3 else [] for obs in obs_list]

        B = len(obs_list)

        # Pad combat hitboxes
        max_combat = max((len(c) for c in combat_lists), default=0)
        max_combat = max(max_combat, 1)  # at least 1 to avoid empty tensors

        combat_batch = np.zeros((B, max_combat, self.config.combat_feature_dim), dtype=np.float32)
        combat_mask = np.zeros((B, max_combat), dtype=np.float32)
        combat_kind_ids = np.zeros((B, max_combat), dtype=np.int32)
        for i, hb in enumerate(combat_lists):
            n = len(hb)
            if n > 0:
                combat_batch[i, :n, :] = hb
                combat_mask[i, :n] = 1.0
                ks = kind_lists[i]
                if ks:
                    combat_kind_ids[i, :n] = self.vocab.encode_list(ks[:n])

        # Pad terrain hitboxes
        max_terrain = max((len(t) for t in terrain_lists), default=0)
        max_terrain = max(max_terrain, 1)

        terrain_batch = np.zeros((B, max_terrain, self.config.terrain_feature_dim), dtype=np.float32)
        terrain_mask = np.zeros((B, max_terrain), dtype=np.float32)
        for i, hb in enumerate(terrain_lists):
            n = len(hb)
            if n > 0:
                terrain_batch[i, :n, :] = hb
                terrain_mask[i, :n] = 1.0

        gs_batch = np.stack(gs_list, axis=0)

        return combat_batch, combat_mask, combat_kind_ids, terrain_batch, terrain_mask, gs_batch
