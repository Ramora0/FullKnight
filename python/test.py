"""
Minimal reset-repro for the GG_Gruz_Mother "sleeping on ceiling" bug.

What this does:
  1. Spawn ONE graphical HK instance.
  2. Reset into GG_Gruz_Mother.
  3. Idle-step (no-op actions) for ~1 second of game time.
  4. Reset into GG_Gruz_Mother AGAIN.
  5. Idle-step for another ~1 second.
  6. Print hitbox / global-state diagnostics for each step.

All the heavy-lifting diagnostics live on the C# side (see
TrainingEnv.LogBossDiag) and go to the HK mod log:
    %AppData%\..\LocalLow\Team Cherry\Hollow Knight\ModLog.txt
This script prints the Python-visible observation so you can correlate the
timeline against that log.

Usage:
    cd python
    python test.py
    # or: python test.py --level GG_Gruz_Mother --steps 30
"""
import argparse
import asyncio
import os
import sys
import time

import numpy as np

from config import Config
from vec_env import VecEnv
from instance_manager import InstanceManager


NOOP_ACTION = [2, 2, 7, 1]  # movement=none, direction=none, action=none, jump=no


def summarize_obs(tag, obs, vocab):
    """One-line observation summary: enemy count + positions + target HP + knight vel."""
    from observation import GS, CB
    mask = obs.combat_mask[0]
    n_combat = int(mask.sum())
    g = obs.global_state[0]
    vel = (float(g[GS.VEL_X]), float(g[GS.VEL_Y]))
    hp = float(g[GS.HP])

    hb_strs = []
    target_hp_str = ""
    for i in range(min(n_combat, 6)):
        row = obs.combat_hb[0, i]
        rx, ry = row[CB.REL_X], row[CB.REL_Y]
        w, h = row[CB.W], row[CB.H]
        gives = int(row[CB.GIVES_DAMAGE])
        is_target = int(row[CB.IS_TARGET])
        hp_raw = float(row[CB.HP_RAW])
        kid = int(obs.combat_kind_ids[0, i])
        kind = vocab._i2s[kid] if 0 <= kid < len(vocab._i2s) else f"id{kid}"
        flag = ('*' if is_target else '') + ('!' if gives else '')
        hb_strs.append(f"{kind}@({rx:+.1f},{ry:+.1f}){w:.1f}x{h:.1f}{flag}")
        if is_target and not target_hp_str:
            target_hp_str = f" target_hp={hp_raw:.0f}"
    n_terrain = int(obs.terrain_mask[0].sum())
    print(
        f"  [{tag}] combat={n_combat} terrain={n_terrain} "
        f"vel=({vel[0]:+.2f},{vel[1]:+.2f}) hp={hp:.0f}{target_hp_str} | "
        + " ".join(hb_strs),
        flush=True,
    )


async def run(args):
    config = Config()
    config.n_envs = 1
    config.level = args.level
    config.boss_levels = args.level
    config.frames_per_wait = args.frames_per_wait
    config.time_scale = args.time_scale
    config.visualize = False

    mgr = None
    if config.hk_path and os.path.exists(config.hk_path):
        print(f"Spawning 1 HK instance at {config.hk_path} ...", flush=True)
        mgr = InstanceManager(config.hk_path, config.hk_data_dir)
        mgr.spawn_n(1)
        mgr.start_all(graphical=True)
    else:
        print(
            f"hk_path not found ({config.hk_path}) — start Hollow Knight manually "
            f"with the FullKnight mod loaded.",
            flush=True,
        )

    try:
        vec_env = VecEnv(config)
        await vec_env.start_server()

        action_vec = [NOOP_ACTION]

        for episode in range(args.episodes):
            print(f"\n=== EPISODE {episode} ===", flush=True)
            t0 = time.perf_counter()
            levels = [args.level]
            obs = await vec_env.reset_all(levels=levels)
            t_reset = time.perf_counter() - t0
            print(f"  reset done in {t_reset:.1f}s", flush=True)
            summarize_obs("post-reset", obs, vec_env.vocab)

            # Idle-step. At time_scale=3 and frames_per_wait=5, each step is
            # ~5 frames ≈ 0.08s game time. args.steps controls how many.
            for step_i in range(args.steps):
                (obs, damage_landed, hits_taken, step_game_times, step_real_times,
                 step_wall_times) = await vec_env.step_all(action_vec)
                if step_i < 5 or step_i % 5 == 0:
                    summarize_obs(f"step {step_i:02d}", obs, vec_env.vocab)

            # Pause between episodes so the scene-load doesn't race the previous step.
            await vec_env.pause_all()

        print("\nDone. Check ModLog.txt for [DIAG ...] blocks from the C# side —")
        print("compare 'reset#1 POST-INITBOSSREFS' vs 'reset#2 POST-INITBOSSREFS'.")
    finally:
        if mgr:
            print("Cleaning up instances...", flush=True)
            mgr.stop_all()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", default="GG_Gruz_Mother")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--time_scale", type=int, default=1,
                   help="Game speed. Keep at 1 for visual inspection.")
    p.add_argument("--frames_per_wait", type=int, default=5)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
