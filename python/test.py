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


def summarize_obs(tag, combat_hb, combat_mask, combat_kind_ids, terrain_hb, terrain_mask, gs, vocab):
    """One-line observation summary: enemy count + positions + boss HP + knight vel."""
    # Env 0 only — this test always runs with n_envs=1.
    mask = combat_mask[0]
    n_combat = int(mask.sum())
    # global_state layout: [vel_x, vel_y, hp, soul, boss_hp, knight_w, knight_h, ...]
    g = gs[0]
    vel = (float(g[0]), float(g[1]))
    hp = float(g[2])
    boss_hp = float(g[4])

    # First few combat hitboxes: [rel_x, rel_y, w, h, is_trigger, hurts_knight, is_target]
    hb_strs = []
    for i in range(min(n_combat, 6)):
        rx, ry, w, h = combat_hb[0, i, 0], combat_hb[0, i, 1], combat_hb[0, i, 2], combat_hb[0, i, 3]
        hurts = int(combat_hb[0, i, 5])
        kid = int(combat_kind_ids[0, i])
        # vocab has _i2s (list) but no decode() method — index directly.
        kind = vocab._i2s[kid] if 0 <= kid < len(vocab._i2s) else f"id{kid}"
        hb_strs.append(f"{kind}@({rx:+.1f},{ry:+.1f}){w:.1f}x{h:.1f}{'!' if hurts else ''}")
    n_terrain = int(terrain_mask[0].sum())
    print(
        f"  [{tag}] combat={n_combat} terrain={n_terrain} "
        f"vel=({vel[0]:+.2f},{vel[1]:+.2f}) hp={hp:.0f} boss_hp={boss_hp:.0f} | "
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
            combat_hb, combat_mask, combat_kind_ids, combat_parent_ids, terrain_hb, terrain_mask, gs = \
                await vec_env.reset_all(levels=levels)
            t_reset = time.perf_counter() - t0
            print(f"  reset done in {t_reset:.1f}s", flush=True)
            summarize_obs("post-reset", combat_hb, combat_mask, combat_kind_ids,
                          terrain_hb, terrain_mask, gs, vec_env.vocab)

            # Idle-step. At time_scale=3 and frames_per_wait=5, each step is
            # ~5 frames ≈ 0.08s game time. args.steps controls how many.
            for step_i in range(args.steps):
                (combat_hb, combat_mask, combat_kind_ids, combat_parent_ids,
                 terrain_hb, terrain_mask, gs,
                 damage_landed, hits_taken, step_game_times, step_real_times) = \
                    await vec_env.step_all(action_vec)
                if step_i < 5 or step_i % 5 == 0:
                    summarize_obs(f"step {step_i:02d}", combat_hb, combat_mask,
                                  combat_kind_ids, terrain_hb, terrain_mask, gs,
                                  vec_env.vocab)

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
