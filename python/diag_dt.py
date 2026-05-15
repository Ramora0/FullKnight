"""Per-step game-time dt distribution with 8 envs (training-equivalent setup).

Spawns 8 HK instances via the train.py flow (InstanceManager + VecEnv), resets
each to a random boss from config.boss_levels_list, and runs noop steps. For
each step it records `step_game_time` (sum of Time.deltaTime across the
frames_per_wait frames inside that agent step). Filters zero-dt filler steps
emitted after a death.

Prints percentile summary + 5ms-bucket histogram, plus real_time / wall_time
side-stats. Stops the instances on exit.
"""
import asyncio
import time
import numpy as np

from config import Config
from vec_env import VecEnv
from instance_manager import InstanceManager


async def main(n_steps: int = 400, warmup_steps: int = 5):
    config = Config(n_envs=8)
    print(f"Spawning {config.n_envs} HK instances...")
    mgr = InstanceManager(config.hk_path, config.hk_data_dir)
    mgr.spawn_n(config.n_envs)
    mgr.start_all(graphical=False)

    try:
        vec_env = VecEnv(config)
        await vec_env.start_server()

        bosses = config.boss_levels_list
        rng = np.random.default_rng(0)
        env_boss = [bosses[int(rng.integers(len(bosses)))] for _ in range(config.n_envs)]
        print(f"Env bosses: {env_boss}")
        print(f"frames_per_wait={config.frames_per_wait}  time_scale={config.time_scale}")
        print(f"Expected gtime/step ~ 5 frames * 0.00283s * 3x = {5*0.00283*3*1000:.2f}ms")

        await vec_env.reset_all(levels=env_boss)

        # Noop actions — the dt is dominated by Unity frame stepping under
        # captureDeltaTime + timeScale, not by what the agent did.
        noop = [2, 2, 7, 1]
        actions = [noop for _ in range(config.n_envs)]

        gts: list[float] = []
        rts: list[float] = []
        walls: list[float] = []
        gts_by_env: list[list[float]] = [[] for _ in range(config.n_envs)]
        dones_count = np.zeros(config.n_envs, dtype=int)
        alive = np.ones(config.n_envs, dtype=bool)

        print(f"Collecting over {n_steps} steps (first {warmup_steps} skipped: includes intro-skip)...")
        t_start = time.perf_counter()
        for t in range(n_steps):
            (_, _, _, _, dones, _,
             step_game, step_real, step_wall, _) = await vec_env.step_all(actions)

            if t >= warmup_steps:
                for e in range(config.n_envs):
                    if step_game[e] > 0:  # filter zero-dt post-death filler
                        gts.append(float(step_game[e]))
                        rts.append(float(step_real[e]))
                        walls.append(float(step_wall[e]))
                        gts_by_env[e].append(float(step_game[e]))

            for e in range(config.n_envs):
                if dones[e]:
                    dones_count[e] += 1
                    alive[e] = False

            if (t + 1) % 50 == 0:
                print(f"  step {t+1}/{n_steps}  samples={len(gts)}  "
                      f"alive={int(alive.sum())}/{config.n_envs}  deaths={dones_count.sum()}")

        wall_total = time.perf_counter() - t_start
        g = np.array(gts)
        r = np.array(rts)
        w = np.array(walls)

        if g.size == 0:
            print("No game-time samples collected (all envs died immediately?).")
            return

        print(f"\n=== step_game_time distribution (n={g.size}, post-warmup, alive-only) ===")
        print(f"  mean={g.mean()*1000:6.2f}ms  std={g.std()*1000:5.2f}ms  "
              f"min={g.min()*1000:6.2f}ms  max={g.max()*1000:7.2f}ms")
        pct_labels = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
        pct_vals = [np.percentile(g, p) * 1000 for p in pct_labels]
        print("  " + "  ".join(f"p{p}={v:.2f}ms" for p, v in zip(pct_labels, pct_vals)))

        # Histogram. Use a fixed 2ms bucket up to max, log-scale bar.
        bucket_ms = 2.0
        max_ms = float(g.max() * 1000)
        edges = np.arange(0, max_ms + bucket_ms, bucket_ms)
        hist, _ = np.histogram(g * 1000, bins=edges)
        hmax = int(hist.max()) if hist.size else 1
        print(f"\nHistogram ({bucket_ms:.0f}ms buckets):")
        for lo, count in zip(edges[:-1], hist):
            if count == 0:
                continue
            bar = "#" * max(1, int(60 * count / hmax))
            print(f"  [{lo:5.1f}-{lo+bucket_ms:5.1f}ms] {count:6d}  {bar}")

        print(f"\n  step_real_time (unscaled): mean={r.mean()*1000:.2f}ms  "
              f"p50={np.percentile(r,50)*1000:.2f}ms  p99={np.percentile(r,99)*1000:.2f}ms")
        print(f"  step_wall_time (Python):   mean={w.mean()*1000:.2f}ms  "
              f"p50={np.percentile(w,50)*1000:.2f}ms  p99={np.percentile(w,99)*1000:.2f}ms")
        print(f"  effective time_scale (sum_game/sum_real): {g.sum()/r.sum():.3f}  "
              f"(configured: {config.time_scale})")

        print(f"\nPer-env summary (mean gtime ms / std / samples):")
        for e in range(config.n_envs):
            arr = np.array(gts_by_env[e]) * 1000
            if arr.size == 0:
                print(f"  env{e} ({env_boss[e].replace('GG_',''):<20}): no samples")
            else:
                print(f"  env{e} ({env_boss[e].replace('GG_',''):<20}): "
                      f"mean={arr.mean():.2f}  std={arr.std():.2f}  n={arr.size}  "
                      f"deaths={dones_count[e]}")

        print(f"\nTotal wall: {wall_total:.1f}s  deaths={dones_count.sum()}")

    finally:
        print("Stopping instances...")
        mgr.stop_all()


if __name__ == "__main__":
    asyncio.run(main())
