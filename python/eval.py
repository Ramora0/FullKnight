"""Play the best trained agent in a real boss fight and optionally record the screen."""

import argparse
import asyncio
import os
import platform
import subprocess
import sys
import time as _time
import numpy as np
import psutil
import torch
import websockets
from tqdm import tqdm

from config import Config
from ppo import PPO
from instance_manager import InstanceManager
from env.env import HKEnv
from env.vec_env import VecEnv
from env.vocab import KindVocab
from env.observation import Observation, filter_terrain_in_view
from train import merge_obs_padded, slice_obs


async def eval_play(checkpoint_path, deterministic=False, time_scale=None,
                    level="GG_Mega_Moss_Charger", record=None, hk_path=None,
                    duration=0, no_agent=False, visualize=False,
                    terrain_max_dist=None, view_w=None, view_h=None):
    config = Config(n_envs=1, level=level)
    # Default to training time_scale so physics/animation/collision behavior
    # matches exactly what the agent was trained on. Override with --time-scale
    # if you want real-time viewing (accepts the distribution shift).
    if time_scale is not None:
        train_game_time = config.time_scale * config.frames_per_wait
        config.time_scale = time_scale
        config.frames_per_wait = train_game_time // time_scale
    if view_w is not None:
        config.view_w = view_w
    if view_h is not None:
        config.view_h = view_h

    agent = None
    vocab = KindVocab(max_size=config.kind_vocab_size)
    if no_agent:
        print(f"No-agent mode: sending noop actions, skipping PPO entirely.")
    else:
        agent = PPO(config)
        agent.load_checkpoint(checkpoint_path, vocab=vocab)
        agent.policy.eval()
        print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Level: {level} | Time scale: {config.time_scale}x | frames_per_wait: {config.frames_per_wait} | Deterministic: {deterministic}")

    vis = None
    if visualize:
        from visualizer import Visualizer
        # Pass config view-box to the visualizer so the outline matches the
        # active data-path gate. With the gate on, terrain handed to the
        # visualizer is already filtered, so the dim-ghost branch is a no-op
        # — kept only so disabling the gate (view_w=0) restores the preview.
        vis_w = config.view_w if config.view_w > 0 else None
        vis_h = config.view_h if config.view_h > 0 else None
        vis = Visualizer(
            vocab=vocab,
            terrain_max_dist=terrain_max_dist,
            view_w=vis_w, view_h=vis_h,
        )

    # Launch game instance with full graphics
    mgr = None
    launch_path = hk_path or config.hk_path
    if launch_path and os.path.exists(launch_path):
        print(f"Launching Hollow Knight from {launch_path} ...")
        mgr = InstanceManager(launch_path, config.hk_data_dir)
        mgr.spawn_n(1)
        mgr._disable_steam_api()
        mgr.start_instance("i0", graphical=True)
    else:
        print(f"hk_path not found ({launch_path}) — launch Hollow Knight manually.")

    # Wait for game to connect
    env = None
    connected = asyncio.Event()

    async def on_connect(websocket):
        nonlocal env
        env = HKEnv(websocket, config)
        await env.init()
        connected.set()
        try:
            await asyncio.Future()
        except websockets.exceptions.ConnectionClosed:
            pass

    server = await websockets.serve(on_connect, config.server_host, config.server_port)
    print(f"Waiting for game on ws://{config.server_host}:{config.server_port} ...")
    await connected.wait()
    print("Game connected!")

    # Reset. With an agent we use eval_mode=True (real HP, boss can die); in
    # no-agent viewer mode we stay in the infinite-fight training distribution
    # so nothing dies while you poke around.
    raw_obs = await env.reset(eval_mode=not no_agent)
    hx = np.zeros((1, config.gru_dim), dtype=np.float32)

    # Start screen recording
    recorder = None
    if record:
        recorder = start_recording(record)
        if recorder:
            print(f"Recording to: {record}")
        else:
            raise RuntimeError("Recording requested but ffmpeg failed to start. Install ffmpeg and try again.")

    step_count = 0
    total_damage = 0.0
    total_hits = 0

    import time as _time
    print(f"Fight started!{f' (time limit: {duration}s)' if duration else ''}")
    t_start = _time.perf_counter()

    try:
        while True:
            if no_agent:
                action_vec = [2, 2, 7, 1]  # idle: no move/dir/action/jump
            else:
                action_vec, hx = get_action(agent, raw_obs, config, deterministic, hx, vocab)

            if vis is not None:
                combat_hb_list, terrain_hb_list, gs_raw, c_kinds, c_parents = raw_obs
                kind_ids = vocab.encode_list(c_kinds)
                parent_ids = vocab.encode_list(c_parents)
                vis.update(
                    batch_obs(
                        combat_hb_list, terrain_hb_list, gs_raw,
                        kind_ids, parent_ids, config,
                    ),
                    terrain_debug=env.last_terrain_debug,
                    fsm_snapshots=env.last_fsm,
                )

            (combat_hb, terrain_hb, gs, combat_kinds, combat_parents,
             damage, hits, _, _, _, done, _) = await env.step_eval(action_vec)

            total_damage += damage
            total_hits += int(hits)
            step_count += 1

            if damage > 0 or hits > 0:
                print(f"  Step {step_count:5d} | {'DMG DEALT: ' + str(damage) if damage else ''}"
                      f"{'  HIT TAKEN' if hits else ''}"
                      f"  (totals: dealt={total_damage:.2f}  taken={total_hits})")

            elapsed = _time.perf_counter() - t_start
            if step_count % 100 == 0:
                print(f"  Step {step_count:5d} | Damage dealt: {total_damage:5.1f} | Hits taken: {total_hits} | {elapsed:.0f}s/{duration}s")

            if done or (duration and elapsed >= duration):
                reason = "TIME" if (duration and elapsed >= duration) else "DONE"
                print(f"\n{'=' * 50}")
                print(f"  Result:        {reason}")
                print(f"  Steps:         {step_count}")
                print(f"  Elapsed:       {elapsed:.1f}s")
                print(f"  Damage dealt:  {total_damage:.1f} nail hits")
                print(f"  Hits taken:    {total_hits}")
                print(f"{'=' * 50}")
                # Release all inputs so the knight stands still
                await env.step([2, 2, 7, 1])
                if recorder:
                    print("Recording 15 more seconds...")
                    await asyncio.sleep(15)
                break

            raw_obs = (combat_hb, terrain_hb, gs, combat_kinds, combat_parents)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        if recorder:
            stop_recording(recorder)
            size = os.path.getsize(record) if os.path.exists(record) else 0
            log_file = record + ".log"
            if size > 10000:
                print(f"Recording saved to: {record} ({size / 1024 / 1024:.1f} MB)")
            else:
                print(f"ERROR: Recording failed — {record} is only {size} bytes. See {log_file}")

        try:
            await env.close()
        except Exception:
            pass

        if mgr:
            mgr.stop_all()

        server.close()
        await server.wait_closed()


def _bootstrap_ci_mean(data, B=4000, alpha=0.05, seed=0):
    """Percentile-bootstrap 95% CI on the mean. Returns (mean, lo, hi).
    Returns NaN bounds if <2 samples."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.size < 2:
        m = float(arr.mean()) if arr.size else float("nan")
        return m, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(B, arr.size))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)


def _bootstrap_diff_ci(data_a, data_b, B=4000, alpha=0.05, seed=0):
    """Percentile-bootstrap CI on (mean(b) - mean(a)). Returns (diff, lo, hi).
    Convention: positive diff means B is larger than A."""
    a = np.asarray(data_a, dtype=np.float64)
    b = np.asarray(data_b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ai = rng.integers(0, a.size, size=(B, a.size))
    bi = rng.integers(0, b.size, size=(B, b.size))
    diffs = b[bi].mean(axis=1) - a[ai].mean(axis=1)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(b.mean() - a.mean()), float(lo), float(hi)


def _print_run_summary(label, results):
    """Print a single run's metrics with bootstrap CIs over completed episodes."""
    n_envs = results["n_envs"]
    steps_done = results["steps_done"]
    total_env_steps = steps_done * n_envs
    ep_dmg = results["ep_dmg"]
    ep_hits = results["ep_hits"]
    ep_steps = results["ep_steps"]

    # Per-step rates within episodes — one sample per completed episode. This
    # is the unit we bootstrap over (more independent than per-step samples).
    ep_dmg_per_step = [d / s for d, s in zip(ep_dmg, ep_steps) if s > 0]
    ep_hits_per_step = [h / s for h, s in zip(ep_hits, ep_steps) if s > 0]

    dmg_mean, dmg_lo, dmg_hi = _bootstrap_ci_mean(ep_dmg)
    hit_mean, hit_lo, hit_hi = _bootstrap_ci_mean(ep_hits)
    dps_mean, dps_lo, dps_hi = _bootstrap_ci_mean(ep_dmg_per_step)
    hps_mean, hps_lo, hps_hi = _bootstrap_ci_mean(ep_hits_per_step)

    print()
    print(f"{'=' * 70}")
    print(f"  {label}")
    print(f"{'-' * 70}")
    print(f"  n_envs={n_envs}  steps/env={steps_done}  total={total_env_steps}  "
          f"wall={results['wall']:.1f}s  episodes={len(ep_dmg)}")
    print(f"  per-env damage:   {[round(float(x), 1) for x in results['total_damage']]}")
    print(f"  per-env hits:     {results['total_hits'].tolist()}")
    print(f"  mean dmg/step (over all env-steps): "
          f"{results['total_damage'].sum() / max(total_env_steps, 1):.5f}")
    print(f"  mean hits/step (over all env-steps): "
          f"{results['total_hits'].sum() / max(total_env_steps, 1):.5f}")
    total_d = float(results['total_damage'].sum())
    total_h = int(results['total_hits'].sum())
    D_ratio = total_d / max(total_h, 1)
    print(f"  D = damage_landed / hits_taken (matches training curriculum): "
          f"{D_ratio:.2f}  (landed={total_d:.1f}  taken={total_h})")
    # Step-timing stats. Mismatch between training and eval here = different
    # effective decision rate for the same model — see distribution-shift
    # diagnostic. Training reference at the saved checkpoint can be inferred
    # from saved boss_state windows; print numbers here for direct compare.
    g = np.asarray(results.get('gtime_samples', []), dtype=np.float64)
    r = np.asarray(results.get('rtime_samples', []), dtype=np.float64)
    if g.size:
        print(f"  step game_time (s):  mean={g.mean():.4f}  "
              f"p50={np.percentile(g, 50):.4f}  "
              f"p95={np.percentile(g, 95):.4f}  "
              f"p99={np.percentile(g, 99):.4f}")
    if r.size:
        print(f"  step real_time (s):  mean={r.mean():.4f}  "
              f"p50={np.percentile(r, 50):.4f}  "
              f"p95={np.percentile(r, 95):.4f}  "
              f"p99={np.percentile(r, 99):.4f}")
    if g.size and r.size:
        ratio = g.sum() / max(r.sum(), 1e-9)
        print(f"  effective time_scale (game_time/real_time): {ratio:.3f}  "
              f"(configured: {3})")
    if len(ep_dmg) >= 2:
        print(f"  per-episode damage:  mean={dmg_mean:.2f}  95% CI [{dmg_lo:.2f}, {dmg_hi:.2f}]  (n={len(ep_dmg)})")
        print(f"  per-episode hits:    mean={hit_mean:.2f}  95% CI [{hit_lo:.2f}, {hit_hi:.2f}]")
        print(f"  per-episode dmg/step: mean={dps_mean:.5f}  95% CI [{dps_lo:.5f}, {dps_hi:.5f}]")
        print(f"  per-episode hits/step: mean={hps_mean:.5f}  95% CI [{hps_lo:.5f}, {hps_hi:.5f}]")
    else:
        print(f"  (only {len(ep_dmg)} episode(s) completed — no CI available; rerun with more steps)")
    print(f"{'=' * 70}")


def _print_comparison(label_a, results_a, label_b, results_b):
    """Bootstrap diff CI: positive diff = B - A. If 0 not in CI -> significant."""
    def _per_ep_rates(r):
        return ([d / s for d, s in zip(r["ep_dmg"], r["ep_steps"]) if s > 0],
                [h / s for h, s in zip(r["ep_hits"], r["ep_steps"]) if s > 0])

    dmg_a = results_a["ep_dmg"]
    dmg_b = results_b["ep_dmg"]
    hits_a = results_a["ep_hits"]
    hits_b = results_b["ep_hits"]
    dps_a, hps_a = _per_ep_rates(results_a)
    dps_b, hps_b = _per_ep_rates(results_b)

    print()
    print(f"{'=' * 70}")
    print(f"  COMPARISON: ({label_b}) - ({label_a})")
    print(f"  Positive = B is larger. 95% CI excluding 0 -> significant.")
    print(f"{'-' * 70}")
    if len(dmg_a) < 2 or len(dmg_b) < 2:
        print(f"  Insufficient episodes (A={len(dmg_a)}, B={len(dmg_b)}). Rerun with more steps.")
        print(f"{'=' * 70}")
        return

    for name, a, b in [
        ("per-episode damage     ", dmg_a, dmg_b),
        ("per-episode hits       ", hits_a, hits_b),
        ("per-episode dmg/step   ", dps_a, dps_b),
        ("per-episode hits/step  ", hps_a, hps_b),
    ]:
        diff, lo, hi = _bootstrap_diff_ci(a, b)
        sig = "* SIGNIFICANT" if (lo > 0 or hi < 0) else "  ns"
        print(f"  {name}  d={diff:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]   {sig}")
    print(f"{'=' * 70}")


async def extended_eval(checkpoint_path, n_envs, n_steps, time_scale=None,
                        level="GG_Mega_Moss_Charger", hk_path=None, no_agent=False,
                        graphical=None, label="extended",
                        view_w=None, view_h=None, use_cuda_graphs=False,
                        frames_per_wait=None):
    """Run K parallel envs through the training collection path (`collect_action`,
    sampled actions, batched policy forward) for `n_steps` steps per env, with
    synchronous reset on done. Tracks per-episode damage/hits/length.

    Returns a dict with totals + per-episode lists. If `graphical` is None,
    defaults to True only when n_envs==1; else respects the bool."""
    if graphical is None:
        graphical = (n_envs == 1)
    config = Config(n_envs=n_envs, level=level)
    if time_scale is not None:
        train_game_time = config.time_scale * config.frames_per_wait
        config.time_scale = time_scale
        config.frames_per_wait = max(1, train_game_time // time_scale)
    if view_w is not None:
        config.view_w = view_w
    if view_h is not None:
        config.view_h = view_h
    config.use_cuda_graphs = use_cuda_graphs
    if frames_per_wait is not None:
        config.frames_per_wait = frames_per_wait

    agent = None
    vocab = KindVocab(max_size=config.kind_vocab_size)
    if not no_agent:
        agent = PPO(config)
        agent.load_checkpoint(checkpoint_path, vocab=vocab)
        agent.policy.eval()
        print(f"[{label}] Loaded checkpoint: {checkpoint_path}")
    print(f"[{label}] n_envs={n_envs} | n_steps={n_steps} | level={level} | "
          f"time_scale={config.time_scale}x | frames_per_wait={config.frames_per_wait} | "
          f"graphical={graphical}")

    mgr = None
    launch_path = hk_path or config.hk_path
    if launch_path and os.path.exists(launch_path):
        print(f"[{label}] Spawning {n_envs} HK instance(s) (graphical={graphical})...")
        mgr = InstanceManager(launch_path, config.hk_data_dir)
        mgr.spawn_n(n_envs)
        mgr._disable_steam_api()
        mgr.start_all(graphical=graphical)
    else:
        print(f"[{label}] hk_path not found ({launch_path}) — launch HK manually.")

    vec_env = VecEnv(config)
    vec_env.vocab = vocab  # share vocab loaded from checkpoint
    await vec_env.start_server()

    obs = await vec_env.reset_all(levels=[level] * n_envs)
    if not no_agent:
        agent.reset_hidden(n_envs)

    total_damage = np.zeros(n_envs, dtype=np.float64)
    total_hits = np.zeros(n_envs, dtype=np.int64)

    # Per-episode accumulators. running_* track the current (incomplete) episode
    # in each env; on `done` we push to ep_* and reset. Trailing partial
    # episodes at end-of-run are discarded (would bias the mean).
    running_dmg = np.zeros(n_envs, dtype=np.float64)
    running_hits = np.zeros(n_envs, dtype=np.int64)
    running_steps = np.zeros(n_envs, dtype=np.int64)
    ep_dmg, ep_hits, ep_steps = [], [], []
    # Step-timing samples for distribution-shift diagnostics. game_time is
    # Time.deltaTime accumulated across frames_per_wait Unity frames (depends
    # on fps); real_time is wallclock for the same frames. Their ratio is
    # roughly time_scale; both depend on fps, which depends on CPU contention
    # (n_envs) and graphics. Mismatch between training and eval = different
    # effective decision rate for the same model.
    gtime_samples = []
    rtime_samples = []
    # Per-env Python-side wallclock for `await env.step(action)`. Mirrors
    # train.py's avg_wall_ms denominator so sim_pct lines up across modes.
    wtime_samples = []
    # Phase-attribution timers — every step of the main loop is partitioned
    # into these phases so we can see where the gap between wtime_mean and
    # per_step_wall_ms goes. All in seconds, summed across the whole run.
    # NOTE: t_step is the wallclock for `await vec_env.step_all(...)` itself
    # (one number per loop iter, ~max(wtime) under asyncio.gather), distinct
    # from wtime which is per-env step duration inside step_all.
    t_action_total = 0.0       # agent.collect_action + action_vec build
    t_step_total = 0.0         # await vec_env.step_all(...)
    t_postproc_total = 0.0     # totals/buffers/perf-sample bookkeeping
    t_done_total = 0.0         # done-handling: kick off async resets + bookkeeping
    t_pbar_total = 0.0         # tqdm.update + set_postfix (not free at 32 step/s × 8)
    n_resets = 0               # number of (env, done) events processed
    # Async-reset state. Resets fire as background tasks and resetting envs
    # drop out of `active_envs` until the task completes. obs_full is the
    # canonical (n_envs,...) Observation; we slice to active for stepping
    # and merge step results / reaped reset obs back in. total_env_steps
    # is the actual count of env-steps (sum of |active| per loop iter) —
    # may be less than n_steps × n_envs when envs are sitting out for resets.
    active_envs = list(range(n_envs))
    obs_full = obs
    total_env_steps = 0
    # HK process RSS + CPU% sampled every PERF_SAMPLE_PERIOD steps. RAM is a
    # single Win32 syscall (~10us); cpu_percent(interval=None) returns the
    # delta since the last call, so we prime once before stepping and then
    # sample at regular intervals — each sample reports CPU% averaged over
    # the elapsed window. Off the hot path.
    ram_samples = []
    cpu_samples = []  # (step, [pct_per_proc, ...], system_pct)
    gc_heap_samples = []  # (step, total_mono_gc_heap_mb_summed_across_envs)
    PERF_SAMPLE_PERIOD = 32
    cpu_total = psutil.cpu_count(logical=True) * 100  # 100% per logical core
    hk_procs = []  # psutil.Process handles, primed for cpu_percent
    if mgr is not None:
        for p in mgr._procs:
            try:
                pp = psutil.Process(p.pid)
                pp.cpu_percent(interval=None)  # prime
                hk_procs.append((p, pp))
            except Exception:
                pass
    psutil.cpu_percent(interval=None)  # prime system

    def _sample_perf(step_idx):
        total_rss = 0
        per_proc_cpu = []
        for popen, pp in hk_procs:
            try:
                if popen.poll() is None:
                    total_rss += pp.memory_info().rss
                    per_proc_cpu.append(pp.cpu_percent(interval=None))
            except Exception:
                pass
        sys_cpu = psutil.cpu_percent(interval=None)
        ram_samples.append((step_idx, total_rss / (1024 ** 2)))
        cpu_samples.append((step_idx, per_proc_cpu, sys_cpu))

    _sample_perf(-1)  # baseline (cpu values from this sample are 0 — primer)

    print(f"[{label}] Running {n_steps} steps per env ({n_steps * n_envs} total)...", flush=True)
    # Snapshot existing tasks so we can identify (and cancel) only the ones
    # spawned inside this extended_eval run during cleanup.
    tasks_at_start = set(asyncio.all_tasks())
    t_start = _time.perf_counter()
    step = -1

    pbar = tqdm(total=n_steps, desc=f"[{label}]", unit="step",
                dynamic_ncols=True, mininterval=0.5)

    try:
        for step in range(n_steps):
            ta0 = _time.perf_counter()

            # Reap any background resets that finished since the prior iter.
            # Splice their fresh obs into obs_full, zero hidden state, and add
            # them back to active_envs.
            reaped = vec_env.reap_completed_resets()
            if reaped:
                reaped_indices = [env_i for env_i, _ in reaped]
                reaped_obs_batch = vec_env._batch_observations(
                    [raw for _, raw in reaped]
                )
                obs_full = merge_obs_padded(obs_full, reaped_obs_batch, reaped_indices)
                if not no_agent:
                    agent.reset_hidden_for(reaped_indices)
                active_envs = sorted(set(active_envs) | set(reaped_indices))

            # If literally nothing is active (rare — all 8 dying simultaneously),
            # block until at least one reset finishes so we don't crash on empty
            # batch.
            if not active_envs:
                if vec_env._reset_tasks:
                    reaped = await vec_env.await_all_resets()
                    reaped_indices = [env_i for env_i, _ in reaped]
                    reaped_obs_batch = vec_env._batch_observations(
                        [raw for _, raw in reaped]
                    )
                    obs_full = merge_obs_padded(obs_full, reaped_obs_batch, reaped_indices)
                    if not no_agent:
                        agent.reset_hidden_for(reaped_indices)
                    active_envs = reaped_indices
                else:
                    break  # shouldn't happen

            N_active = len(active_envs)
            obs = slice_obs(obs_full, active_envs)

            if no_agent:
                action_vecs = [[2, 2, 7, 1] for _ in range(N_active)]
            else:
                actions_np, _, _, _, _ = agent.collect_action(obs, env_indices=active_envs)
                action_vecs = [
                    [int(actions_np["movement"][i]),
                     int(actions_np["direction"][i]),
                     int(actions_np["action"][i]),
                     int(actions_np["jump"][i])]
                    for i in range(N_active)
                ]
            ta1 = _time.perf_counter()
            t_action_total += ta1 - ta0

            (next_obs, damage, hits, hp_healed, done_flags,
             committed, gtimes, rtimes, wtimes, diag) = await vec_env.step_all(
                action_vecs, active_indices=active_envs)
            ts1 = _time.perf_counter()
            t_step_total += ts1 - ta1

            total_env_steps += N_active

            # Update per-env stats. damage[local_i] applies to active_envs[local_i].
            for local_i, env_i in enumerate(active_envs):
                total_damage[env_i] += damage[local_i]
                total_hits[env_i] += int(hits[local_i])
                running_dmg[env_i] += damage[local_i]
                running_hits[env_i] += int(hits[local_i])
                running_steps[env_i] += 1
            gtime_samples.extend(gtimes.tolist())
            rtime_samples.extend(rtimes.tolist())
            wtime_samples.extend(wtimes.tolist())
            if step % PERF_SAMPLE_PERIOD == 0 or step == n_steps - 1:
                _sample_perf(step)
                gc_heap_samples.append(
                    (step, float(diag["gc_heap_mb"].sum())))
            tp1 = _time.perf_counter()
            t_postproc_total += tp1 - ts1

            # Splice the just-stepped obs back into obs_full so the canonical
            # state is up-to-date. Done envs get terminal obs spliced too;
            # those slots are about to be overwritten on reset reap, so it's
            # harmless.
            obs_full = merge_obs_padded(obs_full, next_obs, active_envs)

            # Process done events: push episode stats, kick off background
            # resets, drop done envs from active_envs.
            if done_flags.any():
                local_done = np.where(done_flags)[0].tolist()
                done_envs = [active_envs[li] for li in local_done]
                n_resets += len(done_envs)
                for env_i in done_envs:
                    ep_dmg.append(float(running_dmg[env_i]))
                    ep_hits.append(int(running_hits[env_i]))
                    ep_steps.append(int(running_steps[env_i]))
                    running_dmg[env_i] = 0.0
                    running_hits[env_i] = 0
                    running_steps[env_i] = 0
                resume_indices = [i for i in active_envs if i not in set(done_envs)]
                await vec_env.start_resets(
                    done_envs, levels=[level] * len(done_envs),
                    resume_indices=resume_indices,
                )
                active_envs = resume_indices
            td1 = _time.perf_counter()
            t_done_total += td1 - tp1

            pbar.update(1)
            pbar.set_postfix({
                "dmg": f"{total_damage.sum():.1f}",
                "hits": int(total_hits.sum()),
                "eps": len(ep_dmg),
                "act": N_active,
            })
            t_pbar_total += _time.perf_counter() - td1

    except KeyboardInterrupt:
        print(f"\n[{label}] Interrupted.")
    finally:
        pbar.close()
        wall = _time.perf_counter() - t_start
        steps_done = max(step + 1, 0)

        # 0) Cancel any in-flight background reset tasks before closing the
        #    websockets they depend on — awaiting them after socket teardown
        #    would deadlock.
        for task in list(vec_env._reset_tasks.values()):
            task.cancel()
        vec_env._reset_tasks.clear()

        # 1) Tell each game we're closing (best-effort, short timeout).
        for env in vec_env.envs:
            if env is None:
                continue
            try:
                await asyncio.wait_for(env.close(), timeout=1.0)
            except Exception:
                pass

        # 2) Kill HK processes. Releases their TCP sockets, which lets the
        #    server-side _on_connect handlers exit (they're parked on
        #    asyncio.Future() and only unblock on ConnectionClosed).
        if mgr:
            mgr.stop_all()

        # 3) Force-close any server-side websockets that didn't drop yet.
        for ws in vec_env._ws_connections:
            if ws is None:
                continue
            try:
                await asyncio.wait_for(ws.close(), timeout=1.0)
            except Exception:
                pass

        # 4) Close server with a hard timeout. wait_closed() blocks until all
        #    handlers finish; if one is stuck we don't want to hang forever.
        if vec_env._server is not None:
            vec_env._server.close()
            try:
                await asyncio.wait_for(vec_env._server.wait_closed(), timeout=3.0)
            except asyncio.TimeoutError:
                print(f"[{label}] server close timed out — continuing anyway.")
            except Exception:
                pass

        # 5) Cancel any pending tasks that were spawned inside this run (the
        #    keep-alive Future inside _on_connect, websocket protocol tasks).
        #    Skip tasks that existed before this run started — those belong
        #    to the caller (e.g. the outer compare_graphics coroutine).
        cur = asyncio.current_task()
        for task in asyncio.all_tasks() - tasks_at_start:
            if task is cur or task.done():
                continue
            task.cancel()
        # Yield once so cancellations propagate before we return.
        await asyncio.sleep(0)

    # Pull PPO collect_action breakdown (CUDA event timers + CPU prep wall).
    # report_timing() returns sums in seconds; clears the internal log so a
    # second call would only see new data. We call it once here at end-of-run.
    if not no_agent and agent is not None:
        # Sync to make sure CUDA events are realized before we read elapsed_time.
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        ppo_timing = agent.report_timing() or {}
    else:
        ppo_timing = {}

    return {
        "label": label,
        "n_envs": n_envs,
        "n_steps": n_steps,
        "steps_done": steps_done,
        # total_env_steps is the actual count of (env × step) tuples executed.
        # Under the async-reset path it can be < n_envs × steps_done because
        # resetting envs sit out. Throughput should divide by this, not
        # n_envs × steps_done.
        "total_env_steps": total_env_steps,
        "wall": wall,
        "total_damage": total_damage,
        "total_hits": total_hits,
        "ep_dmg": ep_dmg,
        "ep_hits": ep_hits,
        "ep_steps": ep_steps,
        "gtime_samples": gtime_samples,
        "rtime_samples": rtime_samples,
        "wtime_samples": wtime_samples,
        "ram_samples": ram_samples,
        "cpu_samples": cpu_samples,
        "cpu_total": cpu_total,
        "gc_heap_samples": gc_heap_samples,
        "phase_timers": {
            "t_action_s": t_action_total,
            "t_step_s": t_step_total,
            "t_postproc_s": t_postproc_total,
            "t_done_s": t_done_total,
            "t_pbar_s": t_pbar_total,
            "n_resets": n_resets,
        },
        "ppo_timing": ppo_timing,
    }


async def compare_graphics(checkpoint_path, n_envs, n_steps, time_scale=None,
                           level="GG_Mega_Moss_Charger", hk_path=None, no_agent=False,
                           order="headless_first", view_w=None, view_h=None):
    """Run extended_eval twice — once headless, once graphical — and print a
    bootstrap-CI comparison. `order` controls which runs first (mostly to let
    the user check whether a warmup/order effect explains a difference).
    """
    runs = ("headless", "graphical") if order == "headless_first" else ("graphical", "headless")
    results = {}
    for mode in runs:
        graphical = (mode == "graphical")
        results[mode] = await extended_eval(
            checkpoint_path, n_envs=n_envs, n_steps=n_steps,
            time_scale=time_scale, level=level, hk_path=hk_path,
            no_agent=no_agent, graphical=graphical, label=mode,
            view_w=view_w, view_h=view_h,
        )

    _print_run_summary("HEADLESS", results["headless"])
    _print_run_summary("GRAPHICAL", results["graphical"])
    _print_comparison("HEADLESS", results["headless"],
                      "GRAPHICAL", results["graphical"])


def batch_obs(combat_hb_arr, terrain_hb_arr, gs, combat_kind_ids_arr, combat_parent_ids_arr, config) -> Observation:
    """Pack single-env raw obs into a batched (B=1) Observation."""
    # View-box gate matches what training applies in vec_env._batch_observations.
    terrain_hb_arr = filter_terrain_in_view(
        terrain_hb_arr, config.view_w, config.view_h,
    )
    n_combat = max(len(combat_hb_arr), 1)
    n_terrain = max(len(terrain_hb_arr), 1)

    chb = np.zeros((1, n_combat, config.combat_feature_dim), dtype=np.float32)
    cm = np.zeros((1, n_combat), dtype=np.float32)
    ckid = np.zeros((1, n_combat), dtype=np.int64)
    cpid = np.zeros((1, n_combat), dtype=np.int64)
    if len(combat_hb_arr) > 0:
        chb[0, :len(combat_hb_arr)] = combat_hb_arr
        cm[0, :len(combat_hb_arr)] = 1.0
        ckid[0, :len(combat_hb_arr)] = combat_kind_ids_arr
        cpid[0, :len(combat_hb_arr)] = combat_parent_ids_arr

    thb = np.zeros((1, n_terrain, config.terrain_feature_dim), dtype=np.float32)
    tm = np.zeros((1, n_terrain), dtype=np.float32)
    if len(terrain_hb_arr) > 0:
        thb[0, :len(terrain_hb_arr)] = terrain_hb_arr
        tm[0, :len(terrain_hb_arr)] = 1.0

    return Observation(
        combat_hb=chb,
        combat_mask=cm,
        combat_kind_ids=ckid,
        combat_parent_ids=cpid,
        terrain_hb=thb,
        terrain_mask=tm,
        global_state=gs.reshape(1, -1),
    )


@torch.no_grad()
def get_action(agent, raw_obs, config, deterministic, hx, vocab):
    """Get action from the policy with online normalizer updates (matching training).
    Returns (action_vec, hx_new). raw_obs is the per-env wire tuple."""
    combat_hb_list, terrain_hb_list, gs, combat_kinds, combat_parents = raw_obs
    kind_ids = vocab.encode_list(combat_kinds)
    parent_ids = vocab.encode_list(combat_parents)
    np_obs = batch_obs(
        combat_hb_list, terrain_hb_list, gs, kind_ids, parent_ids, config)

    # Normalize global state (continuous features only, flags pass through).
    # Update running stats first, matching training's collect_action.
    n_cont = config.global_state_dim - config.n_binary_flags
    agent.obs_normalizer.update(np_obs.global_state[..., :n_cont])
    gs_norm = np.empty_like(np_obs.global_state)
    gs_norm[..., :n_cont] = agent.obs_normalizer.normalize(np_obs.global_state[..., :n_cont])
    gs_norm[..., n_cont:] = np_obs.global_state[..., n_cont:]
    np_obs = np_obs.replace(global_state=gs_norm)

    # Normalize hitboxes via _normalize_hitboxes (updates running stats then
    # normalizes), matching training's collect_action exactly.
    chb_norm = agent._normalize_hitboxes(np_obs.combat_hb, np_obs.combat_mask, agent.combat_normalizer)
    PPO._log_compress_combat_hp(chb_norm)
    thb_norm = agent._normalize_hitboxes(np_obs.terrain_hb, np_obs.terrain_mask, agent.terrain_normalizer)
    np_obs = np_obs.replace(combat_hb=chb_norm, terrain_hb=thb_norm)

    device = agent.device
    obs_t = Observation(
        combat_hb=torch.from_numpy(np_obs.combat_hb).float().to(device),
        combat_mask=torch.from_numpy(np_obs.combat_mask).float().to(device),
        combat_kind_ids=torch.from_numpy(np_obs.combat_kind_ids).long().to(device),
        combat_parent_ids=torch.from_numpy(np_obs.combat_parent_ids).long().to(device),
        terrain_hb=torch.from_numpy(np_obs.terrain_hb).float().to(device),
        terrain_mask=torch.from_numpy(np_obs.terrain_mask).float().to(device),
        global_state=torch.from_numpy(np_obs.global_state).float().to(device),
    )
    hx_t = torch.from_numpy(hx).float().to(device)

    if deterministic:
        h, hx_new = agent.policy._encode(obs_t, hx=hx_t)
        logits_m = agent.policy.head_movement(h)
        logits_d = agent.policy.head_direction(h)
        logits_a = agent.policy.head_action(h).clone()
        logits_j = agent.policy.head_jump(h).clone()
        logits_a, logits_j = agent.policy._mask_logits(logits_a, logits_j, obs_t.global_state)

        a_m = logits_m.argmax(-1).item()
        a_d = logits_d.argmax(-1).item()
        a_a = logits_a.argmax(-1).item()
        a_j = logits_j.argmax(-1).item()
    else:
        actions, _, _, _, _, hx_new, _, _ = agent.policy.get_action_and_value(
            obs_t, hx=hx_t
        )
        a_m = actions["movement"].item()
        a_d = actions["direction"].item()
        a_a = actions["action"].item()
        a_j = actions["jump"].item()

    return [a_m, a_d, a_a, a_j], hx_new.cpu().numpy()


# ---------------------------------------------------------------------------
# Screen recording helpers (uses ffmpeg)
# ---------------------------------------------------------------------------

def _find_ffmpeg():
    """Locate ffmpeg binary, checking PATH and common Windows install locations."""
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        return path
    # WinGet installs here but bash shells may not inherit the Windows user PATH
    winget_path = os.path.expandvars(
        r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"
    )
    if os.path.isfile(winget_path):
        return winget_path
    return None


def start_recording(output_path):
    """Start an ffmpeg screen recording process. Returns the Popen handle."""
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        print("ERROR: ffmpeg not found on PATH or in WinGet Links.")
        print("  Install: winget install ffmpeg")
        return None

    system = platform.system()
    if system == "Darwin":
        cmd = [
            ffmpeg, "-y",
            "-f", "avfoundation",
            "-capture_cursor", "1",
            "-i", "1:none",
            "-r", "30",
            "-vcodec", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
    elif system == "Windows":
        cmd = [
            ffmpeg, "-y",
            "-f", "gdigrab",
            "-framerate", "30",
            "-i", "desktop",
            "-f", "dshow",
            "-i", "audio=CABLE Output (VB-Audio Virtual Cable)",
            "-map", "0:v", "-map", "1:a",
            "-vcodec", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-acodec", "aac",
            "-movflags", "frag_keyframe+empty_moov",
            output_path,
        ]
    else:
        cmd = [
            ffmpeg, "-y",
            "-f", "x11grab",
            "-i", ":0.0",
            "-r", "30",
            "-vcodec", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            output_path,
        ]

    try:
        ffmpeg_log = open(output_path + ".log", "w")
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=ffmpeg_log,
        )
        # Give ffmpeg time to initialize both capture devices
        import time as _time
        _time.sleep(3)
        ret = proc.poll()
        if ret is not None:
            ffmpeg_log.close()
            log_contents = open(output_path + ".log").read()
            print(f"ERROR: ffmpeg exited immediately (code {ret}):\n{log_contents[:1000]}")
            return None
        print(f"ffmpeg started: {ffmpeg} (log: {output_path}.log)")
        return proc
    except OSError as e:
        print(f"ERROR: failed to start ffmpeg: {e}")
        return None


def stop_recording(proc):
    """Gracefully stop ffmpeg by sending 'q' to stdin."""
    ret = proc.poll()
    if ret is not None:
        print(f"WARNING: ffmpeg already exited (code {ret}) — recording likely empty. Check .log file.")
        return
    try:
        proc.stdin.write(b"q")
        proc.stdin.flush()
        proc.stdin.close()
        proc.wait(timeout=15)
    except Exception:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def main():
    parser = argparse.ArgumentParser(description="Play trained agent in a real boss fight")
    parser.add_argument("checkpoint", nargs="?", default=None,
                        help="Path to model checkpoint (.pth). Omit with --no-agent.")
    parser.add_argument("--no-agent", action="store_true",
                        help="Skip PPO entirely; send noop actions. Intended for the observation viewer.")
    parser.add_argument("--visualize", action="store_true",
                        help="Open the live hitbox viewer (matplotlib).")
    parser.add_argument("--terrain-max-dist", type=float, default=None,
                        help="Visualizer-only preview: dim terrain segments with nearest-point distance > N"
                             " world units. Does NOT affect the agent's input.")
    parser.add_argument("--view-w", type=float, default=None,
                        help="Override Config.view_w (knight-relative box width, world units) for this run."
                             " Affects the agent's input — terrain outside the box is dropped before the model"
                             " sees it. Pair with --view-h. Pass 0 to disable the gate.")
    parser.add_argument("--view-h", type=float, default=None,
                        help="Override Config.view_h. Pair with --view-w. Pass 0 to disable.")
    parser.add_argument("--all-hitboxes", action="store_true",
                        help="Disable the view-box terrain gate entirely (sets view_w=view_h=0)."
                             " Use when evaluating a checkpoint trained without the gate.")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false",
                        help="Use stochastic sampling instead of greedy argmax")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--time-scale", type=int, default=None,
                        help="Game speed multiplier (default: training time_scale for matched physics; use 1 for real-time viewing)")
    parser.add_argument("--level", default="GG_Mega_Moss_Charger",
                        help="Boss scene name (default: GG_Mega_Moss_Charger)")
    parser.add_argument("--record", metavar="FILE", default="fight.mp4",
                        help="Record screen to video file (default: fight.mp4, --no-record to disable)")
    parser.add_argument("--no-record", dest="record", action="store_const", const=None,
                        help="Disable screen recording")
    parser.add_argument("--duration", type=int, default=0,
                        help="Max run time in seconds (default: 0 = no limit, ends on death)")
    parser.add_argument("--hk-path", default=None,
                        help="Path to Hollow Knight install (overrides config default)")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Extended-eval mode: number of parallel envs (default 1)."
                             " Triggers extended eval when used with --n-steps.")
    parser.add_argument("--n-steps", type=int, default=0,
                        help="Extended-eval mode: steps PER ENV (auto-resets on done)."
                             " When >0, switches to batched-policy eval via VecEnv;"
                             " ignores --duration/--record/--visualize/--deterministic.")
    parser.add_argument("--graphical", dest="graphical", action="store_const", const=True,
                        help="Force graphical for extended eval (default: only when n_envs=1).")
    parser.add_argument("--headless", dest="graphical", action="store_const", const=False,
                        help="Force headless for extended eval.")
    parser.set_defaults(graphical=None)
    parser.add_argument("--compare-graphics", action="store_true",
                        help="Run extended eval twice (headless then graphical) and"
                             " print a bootstrap-CI comparison of damage/hits.")
    parser.add_argument("--compare-order", choices=["headless_first", "graphical_first"],
                        default="headless_first",
                        help="Order for --compare-graphics (default: headless_first).")
    args = parser.parse_args()

    if not args.no_agent and not args.checkpoint:
        parser.error("checkpoint is required unless --no-agent is passed")
    if args.no_agent and args.record == "fight.mp4":
        args.record = None  # don't default-record in viewer mode
    if (args.view_w is None) != (args.view_h is None):
        parser.error("--view-w and --view-h must be set together")
    if args.all_hitboxes:
        if args.view_w is not None or args.view_h is not None:
            parser.error("--all-hitboxes is mutually exclusive with --view-w/--view-h")
        args.view_w = 0.0
        args.view_h = 0.0

    if args.compare_graphics:
        if args.n_steps <= 0:
            parser.error("--compare-graphics requires --n-steps > 0")
        asyncio.run(compare_graphics(
            args.checkpoint,
            n_envs=args.n_envs,
            n_steps=args.n_steps,
            time_scale=args.time_scale,
            level=args.level,
            hk_path=args.hk_path,
            no_agent=args.no_agent,
            order=args.compare_order,
            view_w=args.view_w, view_h=args.view_h,
        ))
        return

    if args.n_steps > 0 or args.n_envs > 1:
        # Extended eval: VecEnv-driven batched collection. Both single (n_envs=1)
        # and batched (n_envs>1) runs go through the same code path so the
        # comparison isolates batch effects.
        if args.n_steps <= 0:
            parser.error("--n-envs > 1 requires --n-steps > 0")
        results = asyncio.run(extended_eval(
            args.checkpoint,
            n_envs=args.n_envs,
            n_steps=args.n_steps,
            time_scale=args.time_scale,
            level=args.level,
            hk_path=args.hk_path,
            no_agent=args.no_agent,
            graphical=args.graphical,
            view_w=args.view_w, view_h=args.view_h,
        ))
        _print_run_summary("EXTENDED EVAL", results)
        return

    asyncio.run(eval_play(
        args.checkpoint,
        deterministic=args.deterministic,
        time_scale=args.time_scale,
        level=args.level,
        record=args.record,
        hk_path=args.hk_path,
        duration=args.duration,
        no_agent=args.no_agent,
        visualize=args.visualize,
        terrain_max_dist=args.terrain_max_dist,
        view_w=args.view_w,
        view_h=args.view_h,
    ))


if __name__ == "__main__":
    main()
