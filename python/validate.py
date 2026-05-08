"""Single-config sim measurement: quality @ N=1, then throughput @ N=8.

Prints a `---` summary block at the end with the two headline numbers
(dmg/ep mean, env-steps/sec) plus bottleneck signals (rtime, CPU sat,
RAM, Mono GC heap). The autoresearch/sim loop scrapes this block.

This is intentionally a single-config tool. We can't run two C# mod
states at once — there's only one mod build active at a time — so A/B
comparison happens by reading historical runs from
`autoresearch/sim/results.tsv`, not by running both sides in one process.

Workflow:
  1. Edit C# mod files.
  2. `dotnet build -c Debug` (the autoresearch/sim/run_experiment.sh
     script does this for you).
  3. `python validate.py <chkpt>` — measures current mod state.
  4. Compare the printed summary against the previous baseline row in
     results.tsv. Improvement = higher throughput AND quality CIs that
     don't exclude zero-diff vs baseline.

Phases:
  - QUALITY @ N=1: per-episode dmg/hits with bootstrap 95% CIs. Single
    env eliminates contention so this is the cleanest behavior signal.
  - THROUGHPUT @ N=THROUGHPUT_N (default 8): env-steps/sec is the
    bottom line. CPU sat, RAM growth, and GC heap mean are the
    bottleneck signals you need to choose what to cut next.
"""
import argparse
import asyncio
import socket
import sys

import numpy as np

from config import Config
from eval import extended_eval, _bootstrap_ci_mean


def _abort_if_port_busy(host, port):
    """Bail before spawning anything if a training run is already on the port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        try:
            s.connect((host, port))
        except (ConnectionRefusedError, socket.timeout, OSError):
            return
    finally:
        s.close()
    print(
        f"ERROR: something is already listening on {host}:{port} "
        f"(probably a running train.py). validate.py would clobber it. "
        f"Stop the other process first.",
        file=sys.stderr,
    )
    sys.exit(2)


async def validate(checkpoint, quality_steps, throughput_steps,
                   throughput_n, hk_path, level, graphical, merged=False):
    """Run quality @ N=1 then throughput @ N=throughput_n. Print sections
    and a `---` summary block at the end.

    `graphical`: None (default — graphical at N=1, headless at N>1),
    True (force graphical both phases), False (force headless both phases).
    `merged`: when True, run a SINGLE phase at N=throughput_n for quality_steps
    steps and use it for both the quality CIs and the throughput readout.
    Cuts wall time roughly in half vs the split phases (no separate N=1 run)
    and tightens the quality CIs (more episodes per wallclock minute because
    N parallel envs accumulate them in parallel). Confounding risk: if Python-
    side contention at N>1 affects HK behavior, merged-quality CIs would
    differ from N=1 split-quality CIs. Verified to overlap on Moss Charger
    before adopting as default.
    """
    if merged:
        print("\n" + "#" * 78)
        print(f"#  MERGED QUALITY + THROUGHPUT @ N={throughput_n}")
        print("#" * 78)
        result = await extended_eval(checkpoint, n_envs=throughput_n,
                                     n_steps=quality_steps, level=level,
                                     hk_path=hk_path,
                                     graphical=graphical,
                                     label=f"merged-N{throughput_n}")
        _print_quality(result)
        _print_throughput(result, throughput_n)
        _print_summary_block(result, result, throughput_n)
        return {"merged": result}

    print("\n" + "#" * 78)
    print("#  PHASE 1: QUALITY @ N=1")
    print("#" * 78)
    quality = await extended_eval(checkpoint, n_envs=1, n_steps=quality_steps,
                                  level=level, hk_path=hk_path,
                                  graphical=graphical, label="quality")
    _print_quality(quality)

    print("\n" + "#" * 78)
    print(f"#  PHASE 2: THROUGHPUT @ N={throughput_n}")
    print("#" * 78)
    throughput = await extended_eval(checkpoint, n_envs=throughput_n,
                                     n_steps=throughput_steps, level=level,
                                     hk_path=hk_path,
                                     graphical=graphical,
                                     label=f"throughput-N{throughput_n}")
    _print_throughput(throughput, throughput_n)

    _print_summary_block(quality, throughput, throughput_n)
    return {"quality": quality, "throughput": throughput}


def _print_quality(r):
    """Quality at N=1: per-episode dmg/hits with bootstrap CIs."""
    print()
    print("=" * 78)
    print("  QUALITY  —  N=1, per-episode metrics with 95% bootstrap CI")
    print("-" * 78)
    ep_dmg, ep_hits = r["ep_dmg"], r["ep_hits"]
    n = len(ep_dmg)
    if n == 0:
        print(f"  0 completed episodes (rerun with more --n-steps)")
    else:
        dmg_m, dmg_lo, dmg_hi = _bootstrap_ci_mean(ep_dmg)
        hit_m, hit_lo, hit_hi = _bootstrap_ci_mean(ep_hits)
        print(f"  {n} episodes (wall={r['wall']:.1f}s):")
        print(f"     dmg/ep   mean={dmg_m:7.2f}   95% CI [{dmg_lo:7.2f}, {dmg_hi:7.2f}]")
        print(f"     hits/ep  mean={hit_m:7.2f}   95% CI [{hit_lo:7.2f}, {hit_hi:7.2f}]")
    print("=" * 78)


def _stats(r):
    """Pull the bottleneck-signal numbers out of a results dict."""
    rt = np.asarray(r.get("rtime_samples", []), dtype=np.float64)
    rt_mean = rt.mean() * 1000 if rt.size else float("nan")
    rt_p95 = np.percentile(rt, 95) * 1000 if rt.size else float("nan")
    # Game time per agent step — what the AGENT sees per decision. With
    # captureDeltaTime mode this is fixed; without, it's frames_per_wait
    # × Time.deltaTime per frame and varies with FPS.
    gt = np.asarray(r.get("gtime_samples", []), dtype=np.float64)
    gt_mean = gt.mean() if gt.size else float("nan")

    # Two sim_pct denominators, both useful and not the same:
    #   sim_pct_inner = mean(rt) / mean(wt) × 100
    #     wt = step_wall_times: Python's perf_counter for `await env.step()`
    #     This matches train.py's perf/sim_pct (line 404) and tells you "of
    #     the time inside step_all, how much was HK frame loop". Doesn't
    #     count Python forward / reset / loop-overhead between calls.
    #   sim_pct_total = mean(rt) / (total_wallclock / steps_done) × 100
    #     Includes everything (resets, agent forward, loop overhead). Bigger
    #     denominator → smaller fraction. The metric for "how dominant is
    #     sim in the WHOLE run".
    wt = np.asarray(r.get("wtime_samples", []), dtype=np.float64)
    wt_mean = wt.mean() * 1000 if wt.size else float("nan")  # ms

    steps_done = r.get("steps_done", 0)
    wall = r.get("wall", 0.0)
    per_step_wallclock_ms = (wall / steps_done * 1000) if steps_done else float("nan")
    sim_pct_inner = (rt_mean / wt_mean * 100) if (rt_mean and wt_mean) else float("nan")
    sim_pct_total = (rt_mean / per_step_wallclock_ms * 100) if (rt_mean and per_step_wallclock_ms) else float("nan")

    cpu_samples = r.get("cpu_samples", [])[1:]  # drop primer
    proc_sum = 0.0
    sat = 0.0
    sys_mean = 0.0
    sys_peak = 0.0
    if cpu_samples:
        sys_vals = [s[2] for s in cpu_samples]
        sys_mean = sum(sys_vals) / len(sys_vals)
        sys_peak = max(sys_vals)
        if cpu_samples[0][1]:
            n_procs = len(cpu_samples[0][1])
            per_proc_means = [
                sum(s[1][i] for s in cpu_samples if i < len(s[1]))
                / max(sum(1 for s in cpu_samples if i < len(s[1])), 1)
                for i in range(n_procs)
            ]
            proc_sum = sum(per_proc_means)
        sat = proc_sum / max(r.get("cpu_total", 100), 1) * 100

    ram = r.get("ram_samples", [])
    if ram:
        ram_vals = [v for _, v in ram]
        ram_init, ram_final = ram_vals[0], ram_vals[-1]
        ram_mean = sum(ram_vals) / len(ram_vals)
    else:
        ram_init = ram_final = ram_mean = 0.0

    gc = r.get("gc_heap_samples", [])
    if gc:
        gc_vals = [v for _, v in gc]
        gc_mean = sum(gc_vals) / len(gc_vals)
        gc_growth = gc_vals[-1] - gc_vals[0] if len(gc_vals) >= 2 else 0.0
    else:
        gc_mean = gc_growth = 0.0

    # Phase-attribution timers from extended_eval. Each is total wall over
    # the run divided by steps_done — gives per-iter ms for that phase.
    # Residual = per_step_wallclock - (action+step+postproc+done+pbar). Should
    # be small if instrumentation is exhaustive (asyncio sleep(0), gc, etc.).
    pt = r.get("phase_timers", {})
    if steps_done > 0 and pt:
        t_action_ms = pt.get("t_action_s", 0.0) / steps_done * 1000
        t_step_ms = pt.get("t_step_s", 0.0) / steps_done * 1000
        t_postproc_ms = pt.get("t_postproc_s", 0.0) / steps_done * 1000
        t_done_ms = pt.get("t_done_s", 0.0) / steps_done * 1000
        t_pbar_ms = pt.get("t_pbar_s", 0.0) / steps_done * 1000
        t_residual_ms = max(0.0,
            per_step_wallclock_ms
            - (t_action_ms + t_step_ms + t_postproc_ms + t_done_ms + t_pbar_ms))
        n_resets_total = int(pt.get("n_resets", 0))
        # Per-reset wallclock: total time spent in done-handling / n_resets.
        # Resets are wallclock-expensive (1-3s in HK); this surfaces it.
        reset_mean_ms = (pt.get("t_done_s", 0.0) / n_resets_total * 1000) \
                        if n_resets_total else 0.0
    else:
        t_action_ms = t_step_ms = t_postproc_ms = 0.0
        t_done_ms = t_pbar_ms = t_residual_ms = 0.0
        n_resets_total = 0
        reset_mean_ms = 0.0

    # PPO collect_action breakdown — CUDA event sums (seconds) + CPU prep wall.
    # Convert to per-call (per loop iter, not per env-step) ms. count is the
    # number of collect_action calls = steps_done.
    pp = r.get("ppo_timing", {})
    if pp and pp.get("count", 0):
        c = pp["count"]
        ppo_norm_ms = pp.get("normalize_s", 0.0) / c * 1000
        ppo_prep_ms = pp.get("tensor_prep_s", 0.0) / c * 1000
        ppo_h2d_ms = pp.get("h2d_s", 0.0) / c * 1000
        ppo_fwd_ms = pp.get("forward_s", 0.0) / c * 1000
        ppo_d2h_ms = pp.get("d2h_s", 0.0) / c * 1000
    else:
        ppo_norm_ms = ppo_prep_ms = ppo_h2d_ms = ppo_fwd_ms = ppo_d2h_ms = 0.0

    return dict(rt_mean=rt_mean, rt_p95=rt_p95,
                gt_mean=gt_mean,
                wt_mean=wt_mean,
                per_step_wallclock_ms=per_step_wallclock_ms,
                sim_pct_inner=sim_pct_inner,
                sim_pct_total=sim_pct_total,
                proc_sum=proc_sum, sat=sat,
                sys_mean=sys_mean, sys_peak=sys_peak,
                ram_init=ram_init, ram_final=ram_final, ram_mean=ram_mean,
                ram_growth=ram_final - ram_init,
                gc_mean=gc_mean, gc_growth=gc_growth,
                t_action_ms=t_action_ms, t_step_ms=t_step_ms,
                t_postproc_ms=t_postproc_ms, t_done_ms=t_done_ms,
                t_pbar_ms=t_pbar_ms, t_residual_ms=t_residual_ms,
                n_resets=n_resets_total, reset_mean_ms=reset_mean_ms,
                ppo_norm_ms=ppo_norm_ms, ppo_prep_ms=ppo_prep_ms,
                ppo_h2d_ms=ppo_h2d_ms, ppo_fwd_ms=ppo_fwd_ms,
                ppo_d2h_ms=ppo_d2h_ms)


def _print_throughput(r, n_envs):
    """Throughput @ N: env-steps/sec + CPU/RAM/GC bottleneck signals."""
    print()
    print("=" * 78)
    print(f"  THROUGHPUT  —  N={n_envs}, env-steps/sec + CPU/RAM/GC bottleneck signals")
    print("-" * 78)
    env_steps = r["n_envs"] * r["steps_done"]
    wall = max(r["wall"], 1e-9)
    eps = env_steps / wall
    s = _stats(r)
    print(f"  throughput = {eps:7.2f} env-steps/sec  "
          f"(wall={wall:.1f}s, env-steps={env_steps})")
    print(f"     rtime    mean={s['rt_mean']:6.2f} ms   p95={s['rt_p95']:6.2f} ms")
    print(f"     CPU      HK sum={s['proc_sum']:6.0f}%   "
          f"machine sat={s['sat']:5.1f}%   "
          f"system mean={s['sys_mean']:5.1f}% peak={s['sys_peak']:5.1f}%")
    print(f"     RAM      initial={s['ram_init']:6.0f} MB  "
          f"final={s['ram_final']:6.0f} MB  "
          f"mean={s['ram_mean']:6.0f} MB  "
          f"growth={s['ram_growth']:+6.0f} MB")
    print(f"     GC heap  mean={s['gc_mean']:6.1f} MB  growth={s['gc_growth']:+6.1f} MB")
    print("=" * 78)


def _print_summary_block(quality, throughput, throughput_n):
    """Machine-scrapeable `---` block. autoresearch/sim/run_experiment.sh
    extracts everything from `^---$` to EOF and prints it as the run summary.
    Keep keys stable — the autoresearch loop reads these by name.
    """
    ep_dmg = quality["ep_dmg"]
    ep_hits = quality["ep_hits"]
    n_ep = len(ep_dmg)
    if n_ep > 0:
        dmg_m, dmg_lo, dmg_hi = _bootstrap_ci_mean(ep_dmg)
        hit_m, hit_lo, hit_hi = _bootstrap_ci_mean(ep_hits)
    else:
        dmg_m = dmg_lo = dmg_hi = hit_m = hit_lo = hit_hi = float("nan")

    env_steps = throughput["n_envs"] * throughput["steps_done"]
    wall = max(throughput["wall"], 1e-9)
    eps = env_steps / wall
    s = _stats(throughput)

    print()
    print("---")
    print(f"dmg_per_ep_mean:    {dmg_m:.4f}")
    print(f"dmg_per_ep_lo:      {dmg_lo:.4f}")
    print(f"dmg_per_ep_hi:      {dmg_hi:.4f}")
    print(f"hits_per_ep_mean:   {hit_m:.4f}")
    print(f"hits_per_ep_lo:     {hit_lo:.4f}")
    print(f"hits_per_ep_hi:     {hit_hi:.4f}")
    print(f"quality_episodes:   {n_ep}")
    print(f"quality_wall_s:     {quality['wall']:.1f}")
    print(f"throughput_eps:     {eps:.4f}")
    print(f"throughput_n:       {throughput_n}")
    print(f"throughput_wall_s:  {wall:.1f}")
    print(f"rtime_mean_ms:      {s['rt_mean']:.4f}")
    print(f"rtime_p95_ms:       {s['rt_p95']:.4f}")
    print(f"gtime_mean_s:       {s['gt_mean']:.6f}")
    print(f"wtime_mean_ms:      {s['wt_mean']:.4f}")
    print(f"per_step_wall_ms:   {s['per_step_wallclock_ms']:.4f}")
    print(f"sim_pct_inner:      {s['sim_pct_inner']:.2f}")
    print(f"sim_pct_total:      {s['sim_pct_total']:.2f}")
    print(f"cpu_hk_sum_pct:     {s['proc_sum']:.2f}")
    print(f"cpu_machine_sat:    {s['sat']:.2f}")
    print(f"cpu_system_mean:    {s['sys_mean']:.2f}")
    print(f"cpu_system_peak:    {s['sys_peak']:.2f}")
    print(f"ram_initial_mb:     {s['ram_init']:.0f}")
    print(f"ram_final_mb:       {s['ram_final']:.0f}")
    print(f"ram_growth_mb:      {s['ram_growth']:.0f}")
    print(f"gc_heap_mean_mb:    {s['gc_mean']:.2f}")
    print(f"gc_heap_growth_mb:  {s['gc_growth']:.2f}")
    # Phase-attribution diagnostic — partitions the per-step wallclock into
    # the components of the rollout loop. Sum should ≈ per_step_wall_ms;
    # t_residual_ms is whatever's left after the named phases (asyncio
    # scheduling, gc.collect, sleep(0), import-time stuff). Tells us where
    # the gap between wtime_mean (inside `await env.step`) and per_step_wall
    # (full loop) actually goes.
    print(f"t_action_ms:        {s['t_action_ms']:.4f}")
    print(f"t_step_ms:          {s['t_step_ms']:.4f}")
    print(f"t_postproc_ms:      {s['t_postproc_ms']:.4f}")
    print(f"t_done_ms:          {s['t_done_ms']:.4f}")
    print(f"t_pbar_ms:          {s['t_pbar_ms']:.4f}")
    print(f"t_residual_ms:      {s['t_residual_ms']:.4f}")
    print(f"n_resets:           {s['n_resets']}")
    print(f"reset_mean_ms:      {s['reset_mean_ms']:.1f}")
    # PPO collect_action breakdown — per-call ms within t_action.
    # normalize/tensor_prep are CPU wall; h2d/forward/d2h are CUDA event sums.
    print(f"ppo_normalize_ms:   {s['ppo_norm_ms']:.4f}")
    print(f"ppo_tensor_prep_ms: {s['ppo_prep_ms']:.4f}")
    print(f"ppo_h2d_ms:         {s['ppo_h2d_ms']:.4f}")
    print(f"ppo_forward_ms:     {s['ppo_fwd_ms']:.4f}")
    print(f"ppo_d2h_ms:         {s['ppo_d2h_ms']:.4f}")


def main():
    p = argparse.ArgumentParser(
        description="Single-config sim measurement (quality @ N=1, throughput @ N=8).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for workflow.",
    )
    p.add_argument("checkpoint", help="Path to .pth checkpoint")
    p.add_argument("--quality-steps", type=int, default=8192,
                   help="Steps for the quality phase at N=1 (default 8192). "
                        "Higher = more episodes = tighter dmg/ep CI. "
                        "Episodes at Moss Charger are ~200-400 steps each.")
    p.add_argument("--throughput-steps", type=int, default=1024,
                   help="Steps per env for the throughput phase (default 1024). "
                        "Just needs to be long enough for the env-steps/sec "
                        "rate to stabilize past ramp-up.")
    p.add_argument("--throughput-n", type=int, default=8,
                   help="n_envs for the throughput phase (default 8). "
                        "Set this to your training-time n_envs.")
    p.add_argument("--level", default=None,
                   help="HK scene name (default: config.level).")
    p.add_argument("--hk-path", default=None,
                   help="Override config.hk_path (Hollow Knight install dir).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--graphical", action="store_const", const=True,
                   default=None, dest="graphical",
                   help="Force graphical mode for both phases.")
    g.add_argument("--headless", action="store_const", const=False,
                   default=None, dest="graphical",
                   help="Force headless mode for both phases.")
    p.add_argument("--merged", action="store_true",
                   help="Single phase at N=throughput_n for quality_steps; "
                        "extracts quality CIs and throughput from one run.")

    args = p.parse_args()

    cfg = Config()
    _abort_if_port_busy(cfg.server_host, cfg.server_port)

    asyncio.run(validate(
        args.checkpoint,
        quality_steps=args.quality_steps,
        throughput_steps=args.throughput_steps,
        throughput_n=args.throughput_n,
        hk_path=args.hk_path,
        level=args.level if args.level is not None else cfg.level,
        graphical=args.graphical,
        merged=args.merged,
    ))


if __name__ == "__main__":
    main()
