import asyncio
import time
import numpy as np
import websockets

from env import HKEnv
from vocab import KindVocab
from observation import Observation


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
        # Latest level assigned per env. Annotates slow-op prints and lets
        # callers grab the boss name from dt-tagged results without having
        # to thread env_boss through step_all.
        self.env_levels = ["?"] * self.n_envs
        # In-flight background reset tasks: env_i -> asyncio.Task wrapping the
        # _timed_op reset coroutine. Caller polls with reap_completed_resets().
        self._reset_tasks: dict = {}

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
            boss = self.env_levels[idx] if idx < len(self.env_levels) else "?"
            print(f"  {label} env {idx} ({boss}): done in {dt:.1f}s", flush=True)
        return idx, dt, result

    async def reset_all(self, levels=None) -> Observation:
        """Reset all envs. Returns the batched Observation.
        levels: optional list of N scene names, one per env."""
        if levels is not None:
            for i, lv in enumerate(levels):
                if lv is not None:
                    self.env_levels[i] = lv
        t0 = time.perf_counter()
        timed = await asyncio.gather(*[
            self._timed_op("reset", i,
                           env.reset(level=(levels[i] if levels is not None else None)))
            for i, env in enumerate(self.envs)
        ])
        total = time.perf_counter() - t0
        print(f"reset_all: {self.n_envs} envs in {total:.1f}s", flush=True)
        results = [r for _, _, r in sorted(timed, key=lambda x: x[0])]
        return self._batch_observations(results)

    async def step_all(self, actions, active_indices=None):
        """Step envs in parallel.
        actions: list of action_vecs, one per active env, each
                 [movement, direction, action, jump]. Length must match
                 len(active_indices) (or n_envs if not provided).
        active_indices: optional list of env slots to step. When provided,
                        only those envs are stepped and returned arrays are
                        sized len(active_indices). Defaults to all envs.
        Returns (Observation, damage_landed, hits_taken, step_game_times,
                 step_real_times, step_wall_times). All arrays aligned to
                 active_indices ordering.
        """
        if active_indices is None:
            active_indices = list(range(self.n_envs))
        assert len(actions) == len(active_indices), \
            f"actions length {len(actions)} != active_indices length {len(active_indices)}"

        timed = await asyncio.gather(*[
            self._timed_op("step", env_i, self.envs[env_i].step(actions[local_i]))
            for local_i, env_i in enumerate(active_indices)
        ])

        # Sort results by position in active_indices (not env_i) to preserve
        # caller ordering. _timed_op returns (env_i, dt, result); build a map.
        by_env = {env_i: (dt, result) for env_i, dt, result in timed}
        ordered = [by_env[env_i] for env_i in active_indices]
        step_wall_times = np.array([dt for dt, _ in ordered], dtype=np.float32)
        results = [r for _, r in ordered]
        (combat_lists, terrain_lists, gs_list, combat_kind_lists, combat_parent_lists,
         damage_landed, hits_taken, step_game_times, step_real_times) = zip(*results)

        obs = self._batch_observations(list(zip(
            combat_lists, terrain_lists, gs_list, combat_kind_lists, combat_parent_lists)))
        damage_landed = np.array(damage_landed, dtype=np.float32)
        hits_taken = np.array(hits_taken, dtype=np.float32)
        step_game_times = np.array(step_game_times, dtype=np.float32)
        step_real_times = np.array(step_real_times, dtype=np.float32)
        return obs, damage_landed, hits_taken, step_game_times, step_real_times, step_wall_times

    async def start_resets(self, reset_indices, levels=None, resume_indices=None):
        """Kick off background resets for reset_indices and synchronously
        resume envs in resume_indices (defaults to all non-reset envs).

        Reset tasks are stored in self._reset_tasks and run in the background
        so the caller can immediately begin the next rollout on the remaining
        active envs. Call reap_completed_resets() at epoch boundaries to
        collect finished resets.

        levels: optional list aligned with reset_indices giving the new scene
                per reset env.
        """
        reset_set = set(reset_indices)
        level_for = {env_i: levels[k] for k, env_i in enumerate(reset_indices)} \
            if levels is not None else {}
        for env_i, lv in level_for.items():
            if lv is not None:
                self.env_levels[env_i] = lv

        if resume_indices is None:
            resume_indices = [i for i in range(self.n_envs) if i not in reset_set]

        # Fire-and-forget: wrap each reset in an asyncio.Task so it runs in
        # the background while we return control to the caller.
        for env_i in reset_indices:
            if env_i in self._reset_tasks:
                raise RuntimeError(
                    f"start_resets: env {env_i} already has a pending reset task"
                )
            coro = self._timed_op(
                "reset", env_i,
                self.envs[env_i].reset(level=level_for.get(env_i)),
                loud=True,
            )
            self._reset_tasks[env_i] = asyncio.create_task(coro)

        # Resume the remaining envs synchronously — these are fast (<10ms).
        if resume_indices:
            await asyncio.gather(*[
                self._timed_op("resume", i, self.envs[i].resume())
                for i in resume_indices
            ])

    def reap_completed_resets(self):
        """Collect any reset tasks that have finished. Returns a list of
        (env_i, raw_obs_tuple) where raw_obs_tuple is the per-env reset
        return value (combat_hb, terrain_hb, gs, combat_kinds, combat_parents).

        Completed tasks are removed from self._reset_tasks. Exceptions are
        re-raised loudly — a silently-failed reset should crash the run
        rather than leave an env stranded forever.
        """
        completed = []
        for env_i in list(self._reset_tasks.keys()):
            task = self._reset_tasks[env_i]
            if not task.done():
                continue
            _env_i, _dt, result = task.result()  # raises if the task errored
            completed.append((env_i, result))
            del self._reset_tasks[env_i]
        return completed

    async def await_all_resets(self):
        """Block until every in-flight reset task is done. Returns the same
        list shape as reap_completed_resets(). Useful as a safety net at
        epoch start if the reset cadence somehow exceeds the rollout budget."""
        if not self._reset_tasks:
            return []
        await asyncio.gather(*self._reset_tasks.values())
        return self.reap_completed_resets()

    async def pause_all(self):
        await asyncio.gather(*[env.pause() for env in self.envs])

    async def resume_all(self):
        await asyncio.gather(*[env.resume() for env in self.envs])

    def _batch_observations(self, obs_list):
        """Pad hitbox lists and stack into tensors.

        obs_list: list of (combat_hb, terrain_hb, global_state, combat_kinds, combat_parents) tuples.
        Returns (combat_hb_batch, combat_mask, combat_kind_ids, combat_parent_ids,
                 terrain_hb_batch, terrain_mask, gs_batch).
        combat_kind_ids/parent_ids: int32 (B, max_combat); padding rows are 0 ("unknown").
        """
        combat_lists = [obs[0] for obs in obs_list]
        terrain_lists = [obs[1] for obs in obs_list]
        gs_list = [obs[2] for obs in obs_list]
        kind_lists = [obs[3] if len(obs) > 3 else [] for obs in obs_list]
        parent_lists = [obs[4] if len(obs) > 4 else [] for obs in obs_list]

        B = len(obs_list)

        # Pad combat hitboxes
        max_combat = max((len(c) for c in combat_lists), default=0)
        max_combat = max(max_combat, 1)  # at least 1 to avoid empty tensors

        combat_batch = np.zeros((B, max_combat, self.config.combat_feature_dim), dtype=np.float32)
        combat_mask = np.zeros((B, max_combat), dtype=np.float32)
        combat_kind_ids = np.zeros((B, max_combat), dtype=np.int32)
        combat_parent_ids = np.zeros((B, max_combat), dtype=np.int32)
        for i, hb in enumerate(combat_lists):
            n = len(hb)
            if n > 0:
                combat_batch[i, :n, :] = hb
                combat_mask[i, :n] = 1.0
                ks = kind_lists[i]
                if ks:
                    combat_kind_ids[i, :n] = self.vocab.encode_list(ks[:n])
                ps = parent_lists[i]
                if ps:
                    combat_parent_ids[i, :n] = self.vocab.encode_list(ps[:n])

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

        return Observation(
            combat_hb=combat_batch,
            combat_mask=combat_mask,
            combat_kind_ids=combat_kind_ids,
            combat_parent_ids=combat_parent_ids,
            terrain_hb=terrain_batch,
            terrain_mask=terrain_mask,
            global_state=gs_batch,
        )
