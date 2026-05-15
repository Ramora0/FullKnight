import asyncio
import os
import time
from collections import deque
import numpy as np
import torch
import wandb

from config import Config
from vec_env import VecEnv
from ppo import PPO
from instance_manager import InstanceManager


def merge_padded(old, new, indices, fill=0.0):
    """Overwrite rows in old with rows from new, expanding padding dims if needed.

    `indices` is a list of row positions in `old` to overwrite; `new[k]` goes
    into `old[indices[k]]`.
    """
    if old.shape[1:] != new.shape[1:]:
        pad = [(0, 0)] + [(0, max(0, n - o)) for o, n in zip(old.shape[1:], new.shape[1:])]
        old = np.pad(old, pad, constant_values=fill)
    for local_i, env_i in enumerate(indices):
        old[env_i] = fill
        idx = tuple([env_i] + [slice(0, s) for s in new.shape[1:]])
        old[idx] = new[local_i]
    return old


def merge_obs_padded(dst, src, indices):
    """Scatter rows of batched Observation `src` into `dst` at `indices`.
    Handles padding mismatch per-field via merge_padded. global_state has no
    padding so it's a plain row copy.
    """
    out = dst.replace(
        combat_hb=merge_padded(dst.combat_hb, src.combat_hb, indices),
        combat_mask=merge_padded(dst.combat_mask, src.combat_mask, indices),
        combat_kind_ids=merge_padded(dst.combat_kind_ids, src.combat_kind_ids, indices),
        combat_parent_ids=merge_padded(dst.combat_parent_ids, src.combat_parent_ids, indices),
        terrain_hb=merge_padded(dst.terrain_hb, src.terrain_hb, indices),
        terrain_mask=merge_padded(dst.terrain_mask, src.terrain_mask, indices),
    )
    for local_i, env_i in enumerate(indices):
        out.global_state[env_i] = src.global_state[local_i]
    return out


def slice_obs(obs, indices):
    """Return a new Observation containing only the given env rows (in order)."""
    idx = np.asarray(indices, dtype=np.int64)
    return obs.replace(
        combat_hb=obs.combat_hb[idx],
        combat_mask=obs.combat_mask[idx],
        combat_kind_ids=obs.combat_kind_ids[idx],
        combat_parent_ids=obs.combat_parent_ids[idx],
        terrain_hb=obs.terrain_hb[idx],
        terrain_mask=obs.terrain_mask[idx],
        global_state=obs.global_state[idx],
    )


def seed_everything(seed: int):
    """Seed all RNGs for deterministic model init and action sampling."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TimingTracker:
    """Single global wallclock breakdown for the train loop.

    Buckets partition the entire run wall time so percentages sum to 100.
    Reset cost is tracked separately as informational because background
    resets overlap rollout/train. A separate informational line attributes
    the throughput cost of NOT having mid-rollout env reactivation:
    after an env dies mid-rollout it sits idle for the rest of the rollout,
    producing zero-fill rows that get masked out at training time. The
    `dead_env_step_cells` count reflects exactly that — the data-yield
    that would be recovered by reactivating the slot mid-epoch.

    Hierarchy:
        rollout                          (= t_rollout)
          sim/combat                     (real C# sim, boss awake)
          sim/intro                      (real C# sim, boss asleep — wasted)
          sim/overhead                   (step_all wall - real sim:
                                          IPC, idle on slowest env)
          inference                      (norm + tensor_prep + h2d + fwd + d2h)
          misc                           (numpy bookkeeping in rollout body)
        train                            (= t_train)
          forward_gpu                    (cuda-event time of forward_sequence)
          backward_optim_gpu             (cuda-event time of bwd+clip+step)
          gae_cpu                        (per-env GAE compute)
          normalize_cpu                  (obs normalize + chunk reshape)
          h2d                            (host->device transfer + sync)
          misc                           (train_loop CPU overhead minus GPU
                                          phases, plus train wall outside
                                          named phases)
        reset_blocking                   (await_all_resets when no envs
                                          were active)
        between_epochs                   (logging, wandb, save, control flow)

    Sim wall is bounded by the slowest env per step (asyncio.gather), so
    sum_t(buf_step_all_wall) gives total step_all wall and sum_t(max_e
    real_time) approximates the productive-sim portion. The combat/intro
    split inside sim_real uses an env-second weighted ratio (real_time
    weighted by combat mask, summed) — unbiased over many epochs.

    Dead-env metric:
        dead_env_step_cells     (T*N cells where env had already died)
        dead_env_idle_ms        (rollout_wall * dead_fraction)
    Informational, not inside the 100%. The wall is real work (alive
    envs were stepping); what's wasted is the slot's data-yield. With
    mid-rollout reactivation those dead cells would carry valid samples.
    """

    BUCKETS = [
        # (key, group, indent_label)
        ("rollout/sim/combat",        "rollout", "sim/combat"),
        ("rollout/sim/intro",         "rollout", "sim/intro"),
        ("rollout/sim/overhead",      "rollout", "sim/overhead"),
        ("rollout/inference",         "rollout", "inference"),
        ("rollout/misc",              "rollout", "misc"),
        ("train/forward_gpu",         "train",   "forward_gpu"),
        ("train/backward_optim_gpu",  "train",   "backward_optim_gpu"),
        ("train/gae_cpu",             "train",   "gae_cpu"),
        ("train/normalize_cpu",       "train",   "normalize_cpu"),
        ("train/h2d",                 "train",   "h2d"),
        ("train/misc",                "train",   "misc"),
        ("reset_blocking",            "other",   "reset_blocking"),
        ("between_epochs",            "other",   "between_epochs"),
    ]

    RESET_PHASES = (
        "pre_unload", "transition_out", "settle",
        "load_boss_scene", "recreate_reader",
        "init_boss_refs", "obs_final",
    )

    def __init__(self):
        self.totals = {k: 0.0 for k, _, _ in self.BUCKETS}
        self.last = dict(self.totals)
        self.n_epochs = 0
        self.active_env_steps = 0    # cells with combat events
        self.total_env_steps = 0     # T*N cells total
        # Dead-env-cell tracking: cells where the env had already died.
        # With mid-rollout reactivation they would carry valid samples.
        self.dead_env_step_cells = 0
        self.cum_dead_wall_s = 0.0   # rollout_wall * dead_fraction, summed
        self.last_dead_cells = 0
        self.last_dead_fraction = 0.0
        self.last_dead_wall_s = 0.0
        self.reset_count = 0
        self.reset_wall_total_s = 0.0
        self.reset_phase_sums_ms = {p: 0.0 for p in self.RESET_PHASES}
        self.reset_phase_frames = {p: 0.0 for p in self.RESET_PHASES}
        self.reset_branch_counts = {"workshop": 0, "natural_end": 0, "unknown": 0}

    def record_epoch(self, *, t_rollout, t_train, t_reset_blocking, t_between,
                     sim_wall_per_step, real_time_arr,
                     combat_per_step, valid_arr, dones_arr,
                     inference_timing, train_phase_t, reset_dts,
                     active_env_steps, total_env_steps):
        """Pin one epoch's measurements into the cumulative totals."""
        # ---- Sim wall + real ----------------------------------------
        sim_wall_s = float(np.asarray(sim_wall_per_step).sum())
        if real_time_arr.size:
            sim_real_s = float(real_time_arr.max(axis=1).sum())
        else:
            sim_real_s = 0.0
        sim_real_s = min(sim_real_s, sim_wall_s)
        sim_overhead_s = max(sim_wall_s - sim_real_s, 0.0)

        # Combat vs intro split inside sim_real: env-second weighted ratio.
        valid_f = np.asarray(valid_arr, dtype=np.float32)
        combat_f = np.asarray(combat_per_step, dtype=np.float32) * valid_f
        intro_f = (1.0 - np.asarray(combat_per_step, dtype=np.float32)) * valid_f
        combat_es = float((real_time_arr * combat_f).sum())
        intro_es = float((real_time_arr * intro_f).sum())
        denom_es = combat_es + intro_es
        combat_frac = (combat_es / denom_es) if denom_es > 0 else 0.0
        sim_combat_s = sim_real_s * combat_frac
        sim_intro_s = sim_real_s * (1.0 - combat_frac)

        # ---- Inference (action selection per rollout step) ----------
        inf = inference_timing or {}
        inference_s = (
            inf.get("normalize_s", 0.0)
            + inf.get("tensor_prep_s", 0.0)
            + inf.get("h2d_s", 0.0)
            + inf.get("forward_s", 0.0)
            + inf.get("d2h_s", 0.0)
        )
        rollout_misc_s = max(t_rollout - sim_wall_s - inference_s, 0.0)

        # ---- Train breakdown ----------------------------------------
        tp = train_phase_t or {}
        gae_s = tp.get("gae", 0.0)
        norm_train_s = tp.get("normalize", 0.0)
        train_h2d_s = tp.get("h2d", 0.0)
        train_loop_s = tp.get("train_loop", 0.0)
        fwd_gpu_s = tp.get("forward_seq", 0.0)
        bwd_gpu_s = tp.get("backward_optim", 0.0)
        train_loop_cpu_s = max(train_loop_s - fwd_gpu_s - bwd_gpu_s, 0.0)
        train_named_s = gae_s + norm_train_s + train_h2d_s + train_loop_s
        train_outer_s = max(t_train - train_named_s, 0.0)
        train_misc_s = train_loop_cpu_s + train_outer_s

        # ---- Dead-env-cell metric (informational) -------------------
        # dones_arr[t, e] is True at the step where each env died (that
        # step itself produced the final valid sample, so we shift by
        # one and cumulative-OR to mark every step strictly AFTER the
        # first death). With mid-rollout reactivation those cells would
        # carry valid samples — so the count below is exactly the
        # data-yield reactivation would recover.
        dead_cells = 0
        dead_fraction = 0.0
        dead_wall_s = 0.0
        if dones_arr is not None:
            d = np.asarray(dones_arr, dtype=bool)
            if d.size:
                ever_done_before = np.zeros_like(d)
                if d.shape[0] > 1:
                    ever_done_before[1:] = np.cumsum(
                        d[:-1].astype(np.uint8), axis=0,
                    ) > 0
                dead_cells = int(ever_done_before.sum())
                dead_fraction = dead_cells / float(d.size)
                # Wall-equivalent throughput cost. Intuition: rollout
                # spent t_rollout producing data, but dead_fraction of
                # the (T*N) cells got zero-fill. With reactivation those
                # would carry samples, giving that fraction of rollout
                # back as productive throughput.
                dead_wall_s = float(t_rollout) * dead_fraction

        # ---- Commit deltas ------------------------------------------
        deltas = {
            "rollout/sim/combat":        sim_combat_s,
            "rollout/sim/intro":         sim_intro_s,
            "rollout/sim/overhead":      sim_overhead_s,
            "rollout/inference":         inference_s,
            "rollout/misc":              rollout_misc_s,
            "train/forward_gpu":         fwd_gpu_s,
            "train/backward_optim_gpu": bwd_gpu_s,
            "train/gae_cpu":             gae_s,
            "train/normalize_cpu":       norm_train_s,
            "train/h2d":                 train_h2d_s,
            "train/misc":                train_misc_s,
            "reset_blocking":            float(t_reset_blocking),
            "between_epochs":            max(float(t_between), 0.0),
        }
        for k, v in deltas.items():
            self.totals[k] += v
        self.last = deltas
        self.n_epochs += 1
        self.active_env_steps += int(active_env_steps)
        self.total_env_steps += int(total_env_steps)
        self.dead_env_step_cells += dead_cells
        self.cum_dead_wall_s += dead_wall_s
        self.last_dead_cells = dead_cells
        self.last_dead_fraction = dead_fraction
        self.last_dead_wall_s = dead_wall_s

        # ---- Resets (background, overlapped) ------------------------
        for entry in reset_dts:
            self.reset_count += 1
            self.reset_wall_total_s += float(entry.get("wall_dt", 0.0))
            for p in self.RESET_PHASES:
                self.reset_phase_sums_ms[p] += float(entry.get(p, 0.0))
            frames = entry.get("frames", {}) or {}
            for p in self.RESET_PHASES:
                self.reset_phase_frames[p] += float(frames.get(p, 0))
            br = entry.get("branch", "unknown")
            if br not in self.reset_branch_counts:
                br = "unknown"
            self.reset_branch_counts[br] += 1

    def epoch_line(self):
        """One-line summary of the most recent epoch's wallclock split."""
        d = self.last
        ep_total = sum(d.values())
        ro = (d["rollout/sim/combat"] + d["rollout/sim/intro"]
              + d["rollout/sim/overhead"] + d["rollout/inference"]
              + d["rollout/misc"])
        sim_real = d["rollout/sim/combat"] + d["rollout/sim/intro"]
        cmb_pct = 100 * d["rollout/sim/combat"] / sim_real if sim_real > 0 else 0
        tr = (d["train/forward_gpu"] + d["train/backward_optim_gpu"]
              + d["train/gae_cpu"] + d["train/normalize_cpu"]
              + d["train/h2d"] + d["train/misc"])
        gpu = d["train/forward_gpu"] + d["train/backward_optim_gpu"]
        gpu_pct = 100 * gpu / tr if tr > 0 else 0
        return (
            f"  time | wall {ep_total:.2f}s | rollout {ro:.2f}s "
            f"[sim {sim_real:.2f}s (cmb {cmb_pct:.0f}%) "
            f"oh {d['rollout/sim/overhead']*1000:.0f}ms "
            f"inf {d['rollout/inference']*1000:.0f}ms "
            f"py {d['rollout/misc']*1000:.0f}ms] "
            f"| train {tr:.2f}s [gpu {gpu_pct:.0f}%] "
            f"| reset_block {d['reset_blocking']*1000:.0f}ms "
            f"| dead_envs {self.last_dead_fraction*100:.0f}% "
            f"({self.last_dead_wall_s*1000:.0f}ms idle)"
        )

    def print_summary(self, label="cumulative"):
        n = max(self.n_epochs, 1)
        total = sum(self.totals.values())
        if total <= 0:
            return
        ms = lambda v: 1000.0 * v / n
        pct = lambda v: 100.0 * v / total
        bar = "=" * 74
        print()
        print(bar)
        print(f"  TIMING BREAKDOWN ({label}, {self.n_epochs} epochs, "
              f"{total:.1f}s wall, {total/n:.2f}s/epoch avg)")
        if self.total_env_steps:
            util = 100 * self.active_env_steps / self.total_env_steps
            print(f"  utilization: {self.active_env_steps}/{self.total_env_steps} "
                  f"env-step cells with combat events ({util:.1f}%)")
        print(bar)
        print(f"  {'bucket':<32s} {'ms/epoch':>11s} {'%':>7s}")
        print(f"  {'-'*32} {'-'*11} {'-'*7}")
        label_for = {b[0]: b[2] for b in self.BUCKETS}
        for group in ("rollout", "train", "other"):
            keys = [k for k, g, _ in self.BUCKETS if g == group]
            if group != "other":
                gtotal = sum(self.totals[k] for k in keys)
                print(f"  {group:<32s} {ms(gtotal):>9.1f} ms {pct(gtotal):>6.1f}%")
                for k in keys:
                    print(f"  {'  ' + label_for[k]:<32s} {ms(self.totals[k]):>9.1f} ms "
                          f"{pct(self.totals[k]):>6.1f}%")
            else:
                for k in keys:
                    print(f"  {label_for[k]:<32s} {ms(self.totals[k]):>9.1f} ms "
                          f"{pct(self.totals[k]):>6.1f}%")
        print(f"  {'-'*32} {'-'*11} {'-'*7}")
        print(f"  {'TOTAL':<32s} {ms(total):>9.1f} ms {100.0:>6.1f}%")
        # ---- Dead-env-cell informational ----------------------------
        # Quantifies the throughput cost of NOT reactivating env slots
        # mid-rollout. The wall is real work (alive envs were stepping);
        # what's wasted is the slot's data-yield. Express as both raw
        # cells and wallclock-equivalent so it's directly comparable to
        # the buckets above.
        if self.total_env_steps:
            dead_pct = 100.0 * self.dead_env_step_cells / self.total_env_steps
            avg_dead_wall_ms = 1000.0 * self.cum_dead_wall_s / n
            print()
            print(f"  dead-env idle (cost of no mid-rollout reactivation):")
            print(f"    {self.dead_env_step_cells}/{self.total_env_steps} cells "
                  f"({dead_pct:.1f}% of T*N) sat idle post-death")
            print(f"    {avg_dead_wall_ms:.0f} ms/epoch wallclock-equivalent "
                  f"({pct(self.cum_dead_wall_s):.1f}% of total wall)")
        # ---- Resets (overlapped, NOT in the 100%) -------------------
        if self.reset_count > 0:
            avg_ms = 1000.0 * self.reset_wall_total_s / self.reset_count
            sum_per_epoch_ms = 1000.0 * self.reset_wall_total_s / n
            sum_pct = pct(self.reset_wall_total_s)
            bc = self.reset_branch_counts
            print()
            print(f"  resets (background, overlapped — NOT in 100% above):")
            print(f"    {self.reset_count} resets ({self.reset_count/n:.2f}/epoch), "
                  f"avg {avg_ms:.0f}ms each, sum {sum_per_epoch_ms:.0f}ms/epoch "
                  f"({sum_pct:.1f}% of total wall if serial)")
            print(f"    branches    workshop={bc['workshop']} "
                  f"natural_end={bc['natural_end']} unknown={bc['unknown']}")
            phase_total = sum(self.reset_phase_sums_ms.values())
            if phase_total > 0:
                print(f"    phases (avg ms/reset, % of avg, frames, ms/f):")
                phase_total_avg = phase_total / self.reset_count
                for p in self.RESET_PHASES:
                    ms_v = self.reset_phase_sums_ms[p] / self.reset_count
                    f_v = self.reset_phase_frames[p] / self.reset_count
                    ms_per_f = ms_v / f_v if f_v > 0 else 0.0
                    p_pct = 100.0 * ms_v / phase_total_avg if phase_total_avg > 0 else 0.0
                    print(f"      {p:<18s} {ms_v:7.1f} ms  ({p_pct:5.1f}%)  "
                          f"{f_v:6.1f}f  {ms_per_f:5.1f}ms/f")
        print(bar, flush=True)


async def hard_restart_all(vec_env, mgr, env_boss, agent, graphical=False):
    """Synchronously kill and relaunch every HK instance, then reset to bosses.

    Patches multi-hour env-state drift that the policy starts exploiting.
    Pauses training for the duration (~30s typical). Returns the new full
    Observation. Caller is responsible for resetting active_envs to all slots
    and discarding any pending staggered-reset bookkeeping (those tasks are
    cancelled here because their websockets are dying anyway)."""
    n = vec_env.n_envs
    print(f"  hard-restart | killing all {n} HK instances...")

    # Cancel pending background level-reset tasks — their websockets are
    # about to be torn down, awaiting them would deadlock.
    for task in list(vec_env._reset_tasks.values()):
        task.cancel()
    vec_env._reset_tasks.clear()

    # Best-effort close server-side websockets so HK exits cleanly first.
    for ws in list(vec_env._ws_connections):
        if ws is not None:
            try:
                await asyncio.wait_for(ws.close(), timeout=1.0)
            except Exception:
                pass

    # Kill the OS processes (psutil terminate + 10s wait per instance).
    mgr.stop_all()

    # Reset vec_env's per-slot state — _on_connect fills these as instances
    # reconnect. Slot identity may permute (first-to-connect wins each slot),
    # which is fine because env_boss is reassigned via reset_all below.
    vec_env._ws_connections = [None] * n
    vec_env.envs = [None] * n
    for ev in vec_env.connected:
        ev.clear()

    # Relaunch processes; _disable_steam_api is idempotent (no-op second time).
    print(f"  hard-restart | relaunching {n} HK instances...")
    mgr.start_all(graphical=graphical)

    # Wait for all to reconnect via the existing _on_connect handler.
    await asyncio.gather(*[ev.wait() for ev in vec_env.connected])
    print(f"  hard-restart | all {n} reconnected; resetting to bosses...")

    # Drive every env to its assigned boss scene; returns batched Observation.
    obs = await vec_env.reset_all(levels=env_boss)

    # Zero hidden state across all envs — fresh-launch is an episode boundary.
    agent.reset_hidden(n)
    print(f"  hard-restart | done.")
    return obs


async def train(config: Config):
    if config.seed:
        seed_everything(config.seed)
        print(f"Seeded all RNGs with {config.seed}")

    vis = None
    if config.visualize:
        from visualizer import Visualizer
        vis = Visualizer()  # vocab attached after vec_env init
    # Launch game instances
    mgr = None
    if config.hk_path and os.path.exists(config.hk_path):
        print(f"Spawning {config.n_envs} HK instance(s)...")
        mgr = InstanceManager(config.hk_path, config.hk_data_dir)
        mgr.spawn_n(config.n_envs)
        mgr.start_all(graphical=(config.n_envs == 1))
    else:
        print(f"hk_path not found ({config.hk_path}) — launch Hollow Knight manually.")

    try:
        # Start vectorized environment server
        vec_env = VecEnv(config)
        await vec_env.start_server()
        if vis is not None:
            vis.vocab = vec_env.vocab

        bosses = config.boss_levels_list
        assert len(bosses) > 0, "config.boss_levels must list at least one scene"

        # D curriculum knobs (D_window, D_ema, D_max_delta) were tuned for
        # total_steps_per_epoch = 8192. Rescale them per-sample so behavior
        # stays consistent as we vary rollout length: a 1024-step run does
        # 8x more epochs per unit-data, so its per-epoch clamp should be 8x
        # smaller, its EMA decay 8x slower, and its window 8x wider.
        D_BASELINE_STEPS = 8192
        D_step_scale = config.total_steps_per_epoch / D_BASELINE_STEPS
        D_window_eff = max(1, int(round(config.D_window / D_step_scale)))
        D_ema_eff = config.D_ema ** D_step_scale
        D_max_delta_eff = config.D_max_delta * D_step_scale
        print(
            f"D curriculum: step_scale={D_step_scale:.4f} "
            f"window={D_window_eff} ema={D_ema_eff:.4f} "
            f"max_delta={D_max_delta_eff:.4f} (from config "
            f"{config.D_window}/{config.D_ema}/{config.D_max_delta})"
        )

        boss_state = {b: {
            "D": config.D_initial,
            "landed_window": deque(maxlen=D_window_eff),
            "taken_window":  deque(maxlen=D_window_eff),
        } for b in bosses}
        rng = np.random.default_rng(config.seed or None)
        env_boss = [bosses[int(rng.integers(len(bosses)))] for _ in range(config.n_envs)]
        print(f"Boss pool: {bosses}")
        print(f"Initial env_boss: {env_boss}")

        agent = PPO(config)
        start_env_steps = 0
        if config.resume:
            start_env_steps = agent.load_checkpoint(
                config.resume, vocab=vec_env.vocab, boss_state=boss_state
            )
            print(f"Resumed from: {config.resume}")
        print(f"Using device: {agent.device}")
        print(f"Model parameters: {sum(p.numel() for p in agent.policy.parameters()):,}")

        os.makedirs(os.path.dirname(config.save_path) or ".", exist_ok=True)
        # Time-budgeted runs default to wandb-off (quick local experiments),
        # but let the caller opt back in by setting WANDB_MODE=online explicitly.
        if config.time_budget and os.environ.get("WANDB_MODE") is None:
            os.environ["WANDB_MODE"] = "disabled"
        wandb.init(project=config.wandb_project, config=vars(config))

        time_budget = config.time_budget
        t_start = time.perf_counter()
        recent = deque(maxlen=20)

        # Single global wallclock breakdown. One source of truth for
        # "where does time go" — replaces the old timing|/collect|
        # log lines. Always on.
        tracker = TimingTracker()
        # Cadence for printing the full breakdown table during a run.
        # Per-epoch one-liner is always printed via tracker.epoch_line().
        TIMING_FULL_EVERY = 25

        # Slow-step bookkeeping: any per-env step whose wall time exceeds
        # `slow_step_threshold_s` is counted against the (env, boss) pair it
        # happened on. Lets us tell if slowness tracks a specific boss, a
        # specific env slot, or is scattered uniformly.
        slow_step_threshold_s = 2.0
        slow_count_by_boss = {b: 0 for b in bosses}
        slow_count_by_env = [0] * config.n_envs

        # First epoch: full reset to load boss scenes
        obs_full = await vec_env.reset_all(levels=env_boss)
        agent.reset_hidden(config.n_envs)
        active_envs = list(range(config.n_envs))

        # Staggered reset: every `steps_per_reset` accumulated env-steps,
        # schedule a reset for `envs_per_reset` envs round-robin. Resets run
        # as background tasks overlapping the next rollout; scheduled envs sit
        # out of the active set until their reset completes.
        envs_per_reset = max(1, config.n_envs // config.envs_per_reset_div)
        steps_since_last_reset = 0
        next_reset_env = 0  # round-robin cursor, advances only when scheduling

        # Step-driven training: total_env_steps bounds the run, save cadence
        # and LR annealing are both keyed on env_steps_collected (not epoch
        # count) so they stay meaningful as rollout size varies.
        env_steps_collected = start_env_steps
        last_save_step = start_env_steps
        epoch = -1  # local counter purely for logging
        # Wallclock-anchor for between_epochs bucket: the time spent on
        # logging/saving/resume/control-flow that lives outside t_rollout
        # and t_train. Updated at the end of each epoch (right after
        # tracker.record_epoch).
        t_prev_epoch_end = time.perf_counter()
        while env_steps_collected < config.total_env_steps:
            epoch += 1
            t_epoch_top = time.perf_counter()

            # Reap any background resets that have completed since we kicked
            # them off at the end of the prior epoch. Splice new obs into
            # obs_full, zero their hidden state, and readd to active_envs.
            reaped = vec_env.reap_completed_resets()
            # If nothing is active but resets are in flight (common with small
            # n_envs after a death-triggered reset), block until at least one
            # reset finishes — otherwise the rollout loop crashes on an empty
            # batch. Time the block separately: reset cost that COULDN'T
            # overlap with rollout/train.
            t_reset_block = 0.0
            if not reaped and not active_envs and vec_env._reset_tasks:
                _t_rb = time.perf_counter()
                reaped = await vec_env.await_all_resets()
                t_reset_block = time.perf_counter() - _t_rb
            if reaped:
                reaped_indices = [env_i for env_i, _ in reaped]
                reaped_obs_batch = vec_env._batch_observations(
                    [raw for _, raw in reaped]
                )
                obs_full = merge_obs_padded(obs_full, reaped_obs_batch, reaped_indices)
                agent.reset_hidden_for(reaped_indices)
                active_envs = sorted(set(active_envs) | set(reaped_indices))

            # Rollout runs over the currently-active subset. Buffers are
            # (T, N_active) shaped; N_active may be < n_envs if a reset from
            # the previous epoch hasn't finished yet.
            N_active = len(active_envs)
            active_set = set(active_envs)
            active_boss = [env_boss[i] for i in active_envs]

            # Rollout buffers
            buf_obs = []  # list of per-step Observations
            buf_actions = {k: [] for k in ["movement", "direction", "action", "jump"]}
            buf_log_probs = []
            buf_log_probs_action = []  # action-head log_prob alone, for hard-commit masking
            buf_values_atk = []
            buf_values_def = []
            buf_damage_landed = []
            buf_hits_taken = []
            buf_hp_healed = []
            buf_dones = []
            buf_committed = []  # bool, action[2] overridden by C# hard-commit state machine
            buf_hx = []
            buf_step_game_times = []
            buf_step_real_times = []
            buf_step_wall_times = []
            # step_all wallclock per step — bounds sim_wall in the
            # TimingTracker. step_wall_per_env (above) is per-env from C#'s
            # _timed_op; this is the perf_counter wrap around the whole
            # asyncio.gather(step_all). Sum_t(buf_step_all_wall) ≈ total
            # time the rollout spent inside step_all.
            buf_step_all_wall = []
            # Leak probes from C#. Each is (T, N_active).
            buf_diag_enemy = []
            buf_diag_attack = []
            buf_diag_terrain = []
            buf_diag_kind_cache = []
            buf_diag_gc_heap = []

            t_rollout_start = time.perf_counter()

            # Slice the active-env view out of obs_full for the first step.
            obs = slice_obs(obs_full, active_envs)

            for t in range(config.rollout_len):
                buf_hx.append(agent.get_hx_snapshot(env_indices=active_envs))
                (actions_np, log_probs, log_probs_action,
                 values_atk, values_def) = agent.collect_action(
                    obs, env_indices=active_envs
                )

                action_vecs = [
                    [
                        int(actions_np["movement"][i]),
                        int(actions_np["direction"][i]),
                        int(actions_np["action"][i]),
                        int(actions_np["jump"][i]),
                    ]
                    for i in range(N_active)
                ]

                t_step = time.perf_counter()
                (next_obs, damage_landed, hits_taken, hp_healed, done_flags,
                 committed_flags,
                 step_game_times, step_real_times,
                 step_wall_per_env, diag) = await vec_env.step_all(
                    action_vecs, active_indices=active_envs
                )
                wall_dt = time.perf_counter() - t_step
                buf_step_all_wall.append(wall_dt)

                buf_obs.append(obs)
                for k in buf_actions:
                    buf_actions[k].append(actions_np[k])
                buf_log_probs.append(log_probs)
                buf_log_probs_action.append(log_probs_action)
                buf_values_atk.append(values_atk)
                buf_values_def.append(values_def)
                buf_damage_landed.append(damage_landed)
                buf_hits_taken.append(hits_taken)
                buf_hp_healed.append(hp_healed)
                buf_dones.append(done_flags)
                buf_committed.append(committed_flags)
                buf_step_game_times.append(step_game_times)
                buf_step_real_times.append(step_real_times)
                buf_step_wall_times.append(step_wall_per_env)
                buf_diag_enemy.append(diag["enemy_count"])
                buf_diag_attack.append(diag["attack_count"])
                buf_diag_terrain.append(diag["terrain_count"])
                buf_diag_kind_cache.append(diag["kind_cache_size"])
                buf_diag_gc_heap.append(diag["gc_heap_mb"])

                obs = next_obs

                if vis is not None:
                    vis.update(obs)

            # Bootstrap final values
            _, _, _, final_vatk, final_vdef = agent.collect_action(
                obs, env_indices=active_envs
            )
            buf_values_atk.append(final_vatk)
            buf_values_def.append(final_vdef)

            # Scatter the final obs back into obs_full so the next epoch has a
            # consistent per-env canonical state to splice reaped resets into.
            obs_full = merge_obs_padded(obs_full, obs, active_envs)

            # Stack buffers: (T, N)
            damage_landed_arr = np.stack(buf_damage_landed)
            hits_taken_arr = np.stack(buf_hits_taken)
            hp_healed_arr = np.stack(buf_hp_healed)
            dones_arr = np.stack(buf_dones)
            committed_arr = np.stack(buf_committed)  # (T, N) bool
            log_probs_arr = np.stack(buf_log_probs)
            log_probs_action_arr = np.stack(buf_log_probs_action)
            values_atk_arr = np.stack(buf_values_atk)
            values_def_arr = np.stack(buf_values_def)
            actions_arr = {k: np.stack(v) for k, v in buf_actions.items()}
            buf_hx_arr = np.stack(buf_hx)  # (T, N, hidden_dim)

            # Diagnostic: first combat event per env, step timing
            any_event = (damage_landed_arr > 0) | (hits_taken_arr > 0)  # (T, N_active)
            active_steps = int(any_event.sum())
            total_steps_epoch = damage_landed_arr.shape[0] * damage_landed_arr.shape[1]
            first_event_steps = []
            for local_i in range(damage_landed_arr.shape[1]):
                col = any_event[:, local_i]
                idxs = np.where(col)[0]
                first_event_steps.append(int(idxs[0]) if len(idxs) > 0 else damage_landed_arr.shape[0])
            wall_time_arr = np.stack(buf_step_wall_times)  # (T, N_active)
            real_time_arr = np.stack(buf_step_real_times)  # (T, N_active) — C# unscaled sim time
            # First step may include intro skip — show it separately
            step0_ms = wall_time_arr[0].mean() * 1000 if wall_time_arr.shape[0] > 0 else 0
            avg_wall_ms = wall_time_arr[1:].mean() * 1000 if wall_time_arr.shape[0] > 1 else 0
            # Per-env slow-step inventory (skip step 0 which includes intro-skip).
            # For each env, report the max step time this epoch and every step
            # over `slow_step_threshold_s` gets counted against the boss.
            post_first = wall_time_arr[1:] if wall_time_arr.shape[0] > 1 else wall_time_arr
            real_post = real_time_arr[1:] if real_time_arr.shape[0] > 1 else real_time_arr
            per_env_max = post_first.max(axis=0) if post_first.shape[0] > 0 else np.zeros(N_active)
            per_env_slow_count = (post_first > slow_step_threshold_s).sum(axis=0)

            # === PERF DIAGNOSTICS ===
            # 1) Per-step spread across envs: how much wall time is wasted
            #    waiting on the slowest env? Big P99 vs P50 → straggler problem
            #    → queue/async stepping would help.
            if post_first.size:
                spread_s = post_first.max(axis=1) - post_first.min(axis=1)  # (T,)
                spread_p50_ms = float(np.percentile(spread_s, 50)) * 1000
                spread_p90_ms = float(np.percentile(spread_s, 90)) * 1000
                spread_p99_ms = float(np.percentile(spread_s, 99)) * 1000
                spread_max_ms = float(spread_s.max()) * 1000
            else:
                spread_p50_ms = spread_p90_ms = spread_p99_ms = spread_max_ms = 0.0

            # 2) C# real_dt vs Python wall_dt: how much of HK time is sim
            #    vs IPC/idle? If real_avg ≈ wall_avg the bottleneck is the
            #    game itself; if real_avg << wall_avg it's IPC/Python.
            #    real_dt is per-env; we average it over (T-1, N).
            real_avg_ms = float(real_post.mean()) * 1000 if real_post.size else 0.0
            overhead_ms = max(avg_wall_ms - real_avg_ms, 0.0)
            sim_pct = 100 * real_avg_ms / avg_wall_ms if avg_wall_ms > 0 else 0

            # 4) Leak probes. Trend-over-hours signal: if any of these rise
            #    monotonically while perf/sim_ms rises, the C# mod is the leak.
            #    hb_*    : HitboxReader HashSet sizes. Terrain should plateau
            #              per scene; enemy/attack growing indicates pooled
            #              prefabs accumulating via ModHooks.ColliderCreateHook.
            #    kind_*  : kindCache dict; entries for destroyed Unity objects
            #              linger until scene change (Dict.Equals uses C# refs).
            #    mono_*  : GC.GetTotalMemory — total managed heap. Rising =
            #              actual allocation leak (not just Unity object refs).
            #    rss_*   : OS-level resident memory from psutil. Rising while
            #              mono_heap is flat → native/Unity leak (textures,
            #              audio buffers, etc). Rising together → managed leak.
            if buf_diag_enemy:
                hb_enemy_arr = np.stack(buf_diag_enemy)       # (T, N_active)
                hb_attack_arr = np.stack(buf_diag_attack)
                hb_terrain_arr = np.stack(buf_diag_terrain)
                kind_cache_arr = np.stack(buf_diag_kind_cache)
                gc_heap_arr = np.stack(buf_diag_gc_heap)
                hb_enemy_avg = float(hb_enemy_arr.mean())
                hb_enemy_max = float(hb_enemy_arr.max())
                hb_attack_avg = float(hb_attack_arr.mean())
                hb_attack_max = float(hb_attack_arr.max())
                hb_terrain_avg = float(hb_terrain_arr.mean())
                hb_terrain_max = float(hb_terrain_arr.max())
                kind_cache_avg = float(kind_cache_arr.mean())
                kind_cache_max = float(kind_cache_arr.max())
                mono_heap_avg = float(gc_heap_arr.mean())
                mono_heap_max = float(gc_heap_arr.max())
            else:
                hb_enemy_avg = hb_enemy_max = 0.0
                hb_attack_avg = hb_attack_max = 0.0
                hb_terrain_avg = hb_terrain_max = 0.0
                kind_cache_avg = kind_cache_max = 0.0
                mono_heap_avg = mono_heap_max = 0.0

            # OS-level per-process memory. Walks psutil.Process(pid) for every
            # HK instance we launched; skips gracefully when instances are
            # attached manually (mgr is None) or a process has died.
            rss_mb_list = []
            if mgr is not None:
                try:
                    import psutil as _psutil
                    for p in getattr(mgr, "_procs", []):
                        try:
                            proc = _psutil.Process(p.pid)
                            rss_mb_list.append(proc.memory_info().rss / (1024 * 1024))
                        except (_psutil.NoSuchProcess, _psutil.AccessDenied):
                            continue
                except Exception:
                    pass
            hk_rss_avg = float(np.mean(rss_mb_list)) if rss_mb_list else 0.0
            hk_rss_max = float(np.max(rss_mb_list)) if rss_mb_list else 0.0

            # 3) Per-boss avg step time: which bosses are slow stragglers?
            #    Helps decide which bosses to drop or load-balance.
            per_boss_step_ms = {}
            if post_first.size:
                for boss in set(active_boss):
                    env_mask = np.array([b == boss for b in active_boss])
                    if env_mask.any():
                        per_boss_step_ms[boss] = float(post_first[:, env_mask].mean()) * 1000
            slow_events_epoch = []
            for local_i, env_i in enumerate(active_envs):
                cnt = int(per_env_slow_count[local_i])
                if cnt > 0:
                    boss = env_boss[env_i]
                    slow_count_by_boss[boss] = slow_count_by_boss.get(boss, 0) + cnt
                    slow_count_by_env[env_i] += cnt
                    slow_events_epoch.append(
                        f"env{env_i}({boss.replace('GG_', '')}):{cnt}×max{per_env_max[local_i]:.1f}s"
                    )
            slow_str = " ".join(slow_events_epoch) if slow_events_epoch else "none"
            # Step-driven reset scheduling: accumulate env-steps collected this
            # epoch, fire a reset batch each time we cross `steps_per_reset`.
            # Cadence is independent of rollout wall time, so the offline pool
            # stays bounded even when rollout << reset.
            steps_since_last_reset += total_steps_epoch
            reset_indices = []
            while steps_since_last_reset >= config.steps_per_reset:
                steps_since_last_reset -= config.steps_per_reset
                for _ in range(envs_per_reset):
                    reset_indices.append(next_reset_env)
                    next_reset_env = (next_reset_env + 1) % config.n_envs

            # Death-triggered resets: any env that emitted done=true during
            # this rollout needs to be reset now, not when the step-budget
            # counter gets around to it. HK has already started transitioning
            # back to GG_Workshop; Reset() waits for that before dreaming in.
            done_local = np.where(dones_arr.any(axis=0))[0]
            for local_i in done_local:
                env_i = active_envs[int(local_i)]
                if env_i not in reset_indices:
                    reset_indices.append(env_i)
            print(
                f"  diag | active_envs {N_active}/{config.n_envs} | "
                f"active_steps {active_steps}/{total_steps_epoch} "
                f"({100*active_steps/total_steps_epoch:.1f}%) | "
                f"first_event {first_event_steps} | "
                f"step0 {step0_ms:.0f}ms | avg_step {avg_wall_ms:.1f}ms | "
                f"reset_budget {steps_since_last_reset}/{config.steps_per_reset} | "
                f"reset_envs {reset_indices}"
            )
            print(
                f"  perf | spread P50/P90/P99/max "
                f"{spread_p50_ms:.0f}/{spread_p90_ms:.0f}/{spread_p99_ms:.0f}/{spread_max_ms:.0f}ms | "
                f"sim {real_avg_ms:.1f}ms ({sim_pct:.0f}%) overhead {overhead_ms:.1f}ms"
            )
            # Leak-probe line. Print raw counts (no formatting tricks) so an
            # upward trend over hours is visually obvious. Drop the line entirely
            # when all diag fields are 0 (old C# DLL / mod-side opt-out).
            if hb_enemy_avg or hb_terrain_avg or mono_heap_avg or hk_rss_avg:
                print(
                    f"  leak | hb_e/a/t avg {hb_enemy_avg:.0f}/{hb_attack_avg:.0f}/{hb_terrain_avg:.0f} "
                    f"(max {hb_enemy_max:.0f}/{hb_attack_max:.0f}/{hb_terrain_max:.0f}) | "
                    f"kcache {kind_cache_avg:.0f} | "
                    f"mono_heap {mono_heap_avg:.1f}MB | "
                    f"rss avg/max {hk_rss_avg:.0f}/{hk_rss_max:.0f}MB"
                )
            if per_boss_step_ms:
                boss_perf_str = " ".join(
                    f"{b.replace('GG_','')}:{per_boss_step_ms[b]:.0f}ms"
                    for b in sorted(per_boss_step_ms, key=lambda b: -per_boss_step_ms[b])
                )
                print(f"  perf | per_boss_step {boss_perf_str}")
            if slow_events_epoch:
                cum_boss = " ".join(
                    f"{b.replace('GG_', '')}:{slow_count_by_boss[b]}"
                    for b in bosses if slow_count_by_boss.get(b, 0) > 0
                )
                print(
                    f"  slow | this_epoch: {slow_str} | "
                    f"cum_by_boss: {cum_boss} | cum_by_env: {slow_count_by_env}"
                )

            # Per-boss adaptive D update. Only bosses represented in the
            # currently-active envs this epoch contribute; bosses with no
            # active envs (e.g. only assigned to a currently-resetting env)
            # are left untouched.
            d_ideal_epoch = {}
            d_ideal_window = {}
            for boss in set(active_boss):
                env_mask = np.array([b == boss for b in active_boss])
                landed_b = float(damage_landed_arr[:, env_mask].sum())
                taken_b = float(hits_taken_arr[:, env_mask].sum())
                bs = boss_state[boss]
                bs["landed_window"].append(landed_b)
                bs["taken_window"].append(taken_b)
                window_landed = sum(bs["landed_window"])
                window_taken = sum(bs["taken_window"])
                # D_ideal = the value D would take if it tracked perfectly
                # this epoch (or this window). Logged as a target for the
                # adaptive update; the gap D_ideal - D is the curriculum lag.
                d_ideal_epoch[boss] = (landed_b / taken_b) if taken_b > 0 else float("nan")
                d_ideal_window[boss] = (window_landed / window_taken) if window_taken > 0 else float("nan")

                if window_landed > 0 and window_taken > 0:
                    # Normal case: EMA toward the raw ratio, clamped.
                    D_raw = max(window_landed / window_taken, config.D_min)
                    if len(bs["landed_window"]) == 1:
                        bs["D"] = D_raw
                    else:
                        D_new = D_ema_eff * bs["D"] + (1 - D_ema_eff) * D_raw
                        bs["D"] = float(np.clip(
                            D_new,
                            bs["D"] * (1 - D_max_delta_eff),
                            bs["D"] * (1 + D_max_delta_eff),
                        ))
                elif window_landed == 0 and window_taken > 0:
                    # Policy is taking hits but landing nothing.
                    # Curriculum is too hard — drop D at the normal clamp rate.
                    bs["D"] = float(max(
                        bs["D"] * (1 - D_max_delta_eff),
                        config.D_min,
                    ))
                elif window_landed > 0 and window_taken == 0:
                    # Weaker signal: policy is landing damage without getting hit.
                    # Push D up at the normal clamp rate. No upper ceiling — D
                    # grows unbounded as the agent improves.
                    bs["D"] = float(bs["D"] * (1 + D_max_delta_eff))
                # else: both zero — no knight/boss interaction at all. Leave D
                # alone; this usually means the arena is broken, not a signal.

            D_per_env = np.array([boss_state[b]["D"] for b in active_boss], dtype=np.float32)

            # Pause game during training. Only pause active envs — resetting
            # envs are mid-scene-load and must not be paused.
            await asyncio.gather(*[
                vec_env.envs[i].pause() for i in active_envs
            ])

            t_rollout = time.perf_counter() - t_rollout_start

            torch.cuda.synchronize()
            inf_timing = agent.report_timing()

            # Mask post-death filler from training. Once an env reports done
            # mid-rollout, every subsequent step is a frozen all-zeros obs
            # (TrainingEnv.cs:209-224) until the end-of-epoch reset. The death
            # step itself is valid — it carries the real terminal transition.
            prev_dones = np.concatenate(
                [np.zeros((1, dones_arr.shape[1]), dtype=bool), dones_arr[:-1]],
                axis=0,
            )
            valid_arr = ~prev_dones  # (T, N_active)

            t0 = time.perf_counter()
            metrics = agent.train_on_rollout(
                buf_obs, actions_arr, log_probs_arr, log_probs_action_arr,
                damage_landed_arr, hits_taken_arr, hp_healed_arr,
                values_atk_arr, values_def_arr, D_per_env, buf_hx_arr,
                dones_arr, valid_arr, committed_arr,
            )
            torch.cuda.synchronize()
            t_train = time.perf_counter() - t0

            # Inference-timing locals — still consumed by wandb below.
            # The console "timing|" / "collect|" lines are gone; the
            # TimingTracker owns wallclock display now.
            inf = inf_timing or {}
            t_fwd = inf.get('forward_s', 0)
            t_norm = inf.get('normalize_s', 0)
            t_prep = inf.get('tensor_prep_s', 0)
            t_h2d = inf.get('h2d_s', 0)
            t_d2h = inf.get('d2h_s', 0)

            # Combat / intro per-step mask. The boss FSM keeps its colliders
            # disabled through the GG transition + intro animation, so
            # HitboxObserver emits nothing combat until the FSM flips them
            # on. We detect "boss awake" per env as the first step with an
            # enemy-flagged combat hitbox (combat feature col GIVES_DAMAGE
            # — set only for HitboxType.Enemy at HitboxObserver.cs:691, so
            # the knight's own attack swing doesn't falsely declare the
            # boss awake). The mask drives sim/combat vs sim/intro inside
            # TimingTracker.
            IDX_GIVES_DAMAGE = 5
            if real_time_arr.shape[0] > 0:
                has_hb_arr = np.stack([
                    ((o.combat_mask > 0) &
                     (o.combat_hb[..., IDX_GIVES_DAMAGE] > 0.5)).any(axis=-1)
                    for o in buf_obs
                ])
                T_hb = has_hb_arr.shape[0]
                first_awake = np.where(
                    has_hb_arr.any(axis=0),
                    has_hb_arr.argmax(axis=0),
                    T_hb,
                )
                step_idx = np.arange(T_hb)[:, None]
                combat_per_step = (step_idx >= first_awake[None, :])
            else:
                combat_per_step = np.zeros_like(real_time_arr, dtype=bool)

            # Single global timing record. Replaces the old
            # timing|/collect| bookkeeping. Buckets sum to 100% of epoch
            # wallclock; the dead-env-cell informational line measures
            # the throughput cost of NOT reactivating env slots
            # mid-rollout. Resets are tracked separately because they
            # overlap with rollout/train.
            t_now = time.perf_counter()
            t_between = max(t_epoch_top - t_prev_epoch_end, 0.0)
            tracker.record_epoch(
                t_rollout=t_rollout, t_train=t_train,
                t_reset_blocking=t_reset_block, t_between=t_between,
                sim_wall_per_step=np.array(buf_step_all_wall, dtype=np.float32),
                real_time_arr=real_time_arr,
                combat_per_step=combat_per_step, valid_arr=valid_arr,
                dones_arr=dones_arr,
                inference_timing=inf_timing,
                train_phase_t=metrics.get("train_phase_t", {}),
                reset_dts=vec_env.pop_reset_dts(),
                active_env_steps=active_steps,
                total_env_steps=int(config.rollout_len * config.n_envs),
            )
            t_prev_epoch_end = t_now
            print(tracker.epoch_line(), flush=True)
            if (epoch + 1) % TIMING_FULL_EVERY == 0:
                tracker.print_summary("cumulative")

            # Step-based linear LR annealing. Progress is measured in
            # env-steps collected (not epochs), so variable rollout sizes
            # and dropped-env epochs decay LR at the same rate per unit work.
            env_steps_collected += total_steps_epoch
            if config.anneal_lr:
                progress = min(1.0, env_steps_collected / config.total_env_steps)
                agent.set_lr(config.lr * (1.0 - progress))

            # Staggered reset: kick off resets for a subset of envs as
            # background tasks, resume everyone else synchronously, and drop
            # the reset envs from active_envs so the next rollout skips them.
            # Only schedule envs that are actually currently active — an env
            # still mid-reset from a prior epoch cannot be reset again.
            reset_indices = [i for i in reset_indices if i in active_set]
            new_bosses = [bosses[int(rng.integers(len(bosses)))] for _ in reset_indices]
            for env_i, b in zip(reset_indices, new_bosses):
                env_boss[env_i] = b
            resume_indices = [i for i in active_envs if i not in set(reset_indices)]
            await vec_env.start_resets(
                reset_indices, levels=new_bosses, resume_indices=resume_indices
            )
            active_envs = [i for i in active_envs if i not in set(reset_indices)]

            # Logging — per-env curriculum reward uses per-env D.
            heal_coef = config.heal_coef
            curriculum_reward = float(
                (damage_landed_arr / D_per_env[None, :] - hits_taken_arr + heal_coef * hp_healed_arr).mean()
            )
            total_steps = env_steps_collected
            Ds = np.array([boss_state[b]["D"] for b in bosses], dtype=np.float64)
            D_geomean = float(np.exp(np.log(np.maximum(Ds, 1e-6)).mean()))
            # Harmonic mean of D expressed in hits units. Dominated by the worst
            # boss — distinct from D_geomean (AM ≥ GM ≥ HM).
            avg_hits_per_boss = float((100.0 / np.maximum(Ds, 1e-6)).mean())

            # Balanced sample means: per-boss mean first, then average across
            # represented bosses. Weights each boss equally regardless of how
            # many envs happened to be assigned to it this epoch. Uses
            # active_boss (captured pre-reset) so the mask aligns with the
            # (T, N_active) rollout arrays.
            per_boss_landed_mean = []
            per_boss_taken_mean = []
            per_boss_healed_mean = []
            for boss in set(active_boss):
                env_mask = np.array([b == boss for b in active_boss])
                per_boss_landed_mean.append(float(damage_landed_arr[:, env_mask].mean()))
                per_boss_taken_mean.append(float(hits_taken_arr[:, env_mask].mean()))
                per_boss_healed_mean.append(float(hp_healed_arr[:, env_mask].mean()))
            balanced_landed = float(np.mean(per_boss_landed_mean))
            balanced_taken = float(np.mean(per_boss_taken_mean))
            balanced_healed = float(np.mean(per_boss_healed_mean))
            # Episode stats
            n_deaths = int(dones_arr.any(axis=0).sum())
            log = {
                "loss/surrogate": metrics["surrogate"],
                "loss/value_atk": metrics["value_atk"],
                "loss/value_def": metrics["value_def"],
                "metrics/ev_atk": metrics["ev_atk"],
                "metrics/ev_def": metrics["ev_def"],
                "metrics/pass_frac": metrics["pass_frac"],
                "metrics/adv_std_raw": metrics["adv_std_raw"],
                "metrics/atk_return_var": metrics["atk_return_var"],
                "metrics/def_return_var": metrics["def_return_var"],
                "loss/entropy": metrics["entropy"],
                "metrics/kl": metrics["kl"],
                "metrics/lr": agent.optimizer.param_groups[0]["lr"],
                "curriculum/D_geomean": D_geomean,
                "curriculum/avg_hits_per_boss": avg_hits_per_boss,
                "rollout/curriculum_reward": curriculum_reward,
                "rollout/damage_landed": balanced_landed,
                "rollout/hits_taken": balanced_taken,
                "rollout/hp_healed": balanced_healed,
                "rollout/deaths": n_deaths,
                "diag/committed_frac": float(committed_arr.mean()),
                "diag/active_step_pct": 100 * active_steps / total_steps_epoch,
                "diag/first_event_avg": np.mean(first_event_steps),
                "diag/step0_ms": step0_ms,
                "diag/avg_step_ms": avg_wall_ms,
                "diag/max_step_ms": float(post_first.max()) * 1000 if post_first.size else 0,
                "diag/slow_steps_epoch": int(per_env_slow_count.sum()),
                "diag/gru_norm": metrics["gru_norm"],
                # Perf diagnostics — see "PERF DIAGNOSTICS" block above for meaning.
                "perf/spread_p50_ms": spread_p50_ms,
                "perf/spread_p90_ms": spread_p90_ms,
                "perf/spread_p99_ms": spread_p99_ms,
                "perf/spread_max_ms": spread_max_ms,
                "perf/sim_ms": real_avg_ms,
                "perf/overhead_ms": overhead_ms,
                "perf/sim_pct": sim_pct,
                "perf/hb_enemy_avg": hb_enemy_avg,
                "perf/hb_enemy_max": hb_enemy_max,
                "perf/hb_attack_avg": hb_attack_avg,
                "perf/hb_attack_max": hb_attack_max,
                "perf/hb_terrain_avg": hb_terrain_avg,
                "perf/hb_terrain_max": hb_terrain_max,
                "perf/kind_cache_avg": kind_cache_avg,
                "perf/kind_cache_max": kind_cache_max,
                "perf/mono_heap_mb_avg": mono_heap_avg,
                "perf/mono_heap_mb_max": mono_heap_max,
                "perf/hk_rss_mb_avg": hk_rss_avg,
                "perf/hk_rss_mb_max": hk_rss_max,
                "perf/collect_norm_ms": t_norm * 1000,
                "perf/collect_prep_ms": t_prep * 1000,
                "perf/collect_h2d_ms": t_h2d * 1000,
                "perf/collect_fwd_ms": t_fwd * 1000,
                "perf/collect_d2h_ms": t_d2h * 1000,
                "epoch": epoch,
            }
            for boss, ms in per_boss_step_ms.items():
                log[f"perf/per_boss_step_ms/{boss}"] = ms
            for boss in bosses:
                log[f"curriculum/D/{boss}"] = boss_state[boss]["D"]
                if boss in d_ideal_epoch and not np.isnan(d_ideal_epoch[boss]):
                    log[f"curriculum/D_ideal_epoch/{boss}"] = d_ideal_epoch[boss]
                if boss in d_ideal_window and not np.isnan(d_ideal_window[boss]):
                    log[f"curriculum/D_ideal_window/{boss}"] = d_ideal_window[boss]
                log[f"diag/slow_cum/{boss}"] = slow_count_by_boss.get(boss, 0)
            for env_i in range(config.n_envs):
                log[f"diag/slow_cum_env/{env_i}"] = slow_count_by_env[env_i]
            wandb.log(log, step=total_steps)

            boss_ds = " ".join(f"{b.split('_')[-1]}:{boss_state[b]['D']:.2f}" for b in bosses)
            print(
                f"epoch {epoch:4d} | "
                f"steps {total_steps:8d} | "
                f"D[{boss_ds}] | "
                f"D_geo {D_geomean:6.2f} | "
                f"hits/boss {avg_hits_per_boss:6.1f} | "
                f"curr_rew {curriculum_reward:7.4f} | "
                f"dmg {balanced_landed:6.3f} | "
                f"taken {balanced_taken:6.3f} | "
                f"surr {metrics['surrogate']:7.4f} | "
                f"kl {metrics['kl']:6.4f}"
            )

            recent.append({
                'curriculum_reward': curriculum_reward,
                'damage_landed': balanced_landed,
                'hits_taken': balanced_taken,
                'entropy': metrics['entropy'],
                'kl': metrics['kl'],
                'D_geomean': D_geomean,
                'avg_hits_per_boss': avg_hits_per_boss,
                'surrogate': metrics['surrogate'],
                'pass_frac': metrics['pass_frac'],
            })

            if time_budget and (time.perf_counter() - t_start) >= time_budget:
                print(f"Time budget ({time_budget}s) reached after {epoch + 1} epochs")
                break

            if env_steps_collected - last_save_step >= config.save_every_steps:
                path = f"{config.save_path}_{env_steps_collected}.pth"
                agent.save_checkpoint(
                    path, vocab=vec_env.vocab, boss_state=boss_state,
                    env_steps=env_steps_collected,
                )
                print(f"  Saved checkpoint: {path}")
                last_save_step = env_steps_collected

            # Synchronous hard restart of every HK instance every N epochs.
            # Discards the staggered-reset bookkeeping above (those processes
            # are about to die) and reactivates all slots after relaunch.
            if (config.hard_restart_every_epochs > 0
                    and (epoch + 1) % config.hard_restart_every_epochs == 0):
                obs_full = await hard_restart_all(
                    vec_env, mgr, env_boss, agent,
                    graphical=(config.n_envs == 1),
                )
                active_envs = list(range(config.n_envs))

        # Print summary (used by autoresearch pipeline)
        if recent:
            n = len(recent)
            avg = {k: sum(m[k] for m in recent) / n for k in recent[0]}
            elapsed = time.perf_counter() - t_start
            print("\n---")
            print(f"curriculum_reward:      {avg['curriculum_reward']:.6f}")
            print(f"avg_damage_landed:      {avg['damage_landed']:.4f}")
            print(f"avg_hits_taken:         {avg['hits_taken']:.4f}")
            print(f"final_D_geomean:        {avg['D_geomean']:.2f}")
            print(f"final_avg_hits_per_boss: {avg['avg_hits_per_boss']:.1f}")
            for boss in bosses:
                print(f"final_D/{boss}: {boss_state[boss]['D']:.2f}")
            print(f"final_entropy:       {avg['entropy']:.6f}")
            print(f"final_kl:            {avg['kl']:.6f}")
            print(f"final_surrogate:     {avg['surrogate']:.6f}")
            print(f"pct_samples_trained: {100 * avg['pass_frac']:.1f}")
            print(f"epochs_completed:    {epoch + 1}")
            print(f"training_seconds:    {elapsed:.1f}")
            tracker.print_summary("final")

        agent.save_checkpoint(
            f"{config.save_path}_final.pth", vocab=vec_env.vocab,
            boss_state=boss_state, env_steps=env_steps_collected,
        )
        wandb.finish()

        if vis is not None:
            vis.close()
    finally:
        if mgr:
            print("Cleaning up instances...")
            mgr.stop_all()


def main():
    config = Config.from_cli()
    asyncio.run(train(config))


if __name__ == "__main__":
    main()
