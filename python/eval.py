"""Play the best trained agent in a real boss fight and optionally record the screen."""

import argparse
import asyncio
import os
import platform
import subprocess
import sys
import numpy as np
import torch
import websockets

from config import Config
from ppo import PPO
from instance_manager import InstanceManager
from env import HKEnv
from vocab import KindVocab
from observation import Observation, GS, CB


async def eval_play(checkpoint_path, deterministic=False, time_scale=1,
                    level="GG_Mega_Moss_Charger", record=None, hk_path=None,
                    duration=0):
    config = Config(n_envs=1, level=level)
    # Scale frames_per_wait to preserve game-time-per-step from training.
    # Training: time_scale=3, frames_per_wait=5 → 15 frame-equivalents per step.
    # At eval time_scale=1: need frames_per_wait=15 for same observation spacing.
    train_game_time = config.time_scale * config.frames_per_wait
    config.time_scale = time_scale
    config.frames_per_wait = train_game_time // time_scale

    # Load agent (vocab loaded from checkpoint to keep embedding rows aligned)
    agent = PPO(config)
    vocab = KindVocab(max_size=config.kind_vocab_size)
    agent.load_checkpoint(checkpoint_path, vocab=vocab)
    agent.policy.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Level: {level} | Time scale: {time_scale}x | frames_per_wait: {config.frames_per_wait} | Deterministic: {deterministic}")

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

    # Reset with eval mode (real HP, boss can die). raw_obs is the per-env tuple
    # straight from the wire: (combat_hb_list, terrain_hb_list, gs, kinds, parents).
    raw_obs = await env.reset(eval_mode=True)
    # Capture max knight HP from first obs (right after reset = full health).
    # Boss HP is now per-hitbox (hp_raw column); we record the per-target max
    # on first sight so we can spoof targets back to full each step (matching the
    # infinite-fight training distribution where boss HP is restored every frame).
    max_knight_hp = raw_obs[2][GS.HP]
    target_max_hp = {}          # kind string -> max hp_raw seen
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
            # Spoof knight HP so agent sees max like during training.
            raw_obs[2][GS.HP] = max_knight_hp
            # Spoof every is_target hitbox's hp_raw back to its observed max.
            combat_hb_now = raw_obs[0]
            combat_kinds_now = raw_obs[3]
            for hb_i in range(len(combat_hb_now)):
                row = combat_hb_now[hb_i]
                if row[CB.IS_TARGET] > 0.5:
                    key = combat_kinds_now[hb_i] if hb_i < len(combat_kinds_now) else ""
                    cur = float(row[CB.HP_RAW])
                    if cur > target_max_hp.get(key, 0.0):
                        target_max_hp[key] = cur
                    row[CB.HP_RAW] = target_max_hp.get(key, cur)
            action_vec, hx = get_action(agent, raw_obs, config, deterministic, hx, vocab)

            (combat_hb, terrain_hb, gs, combat_kinds, combat_parents,
             damage, hits, _, _, done) = await env.step_eval(action_vec)

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


def batch_obs(combat_hb_arr, terrain_hb_arr, gs, combat_kind_ids_arr, combat_parent_ids_arr, config) -> Observation:
    """Pack single-env raw obs into a batched (B=1) Observation."""
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
    """Get action from the policy using frozen normalizers (no stats update).
    Returns (action_vec, hx_new). raw_obs is the per-env wire tuple."""
    combat_hb_list, terrain_hb_list, gs, combat_kinds, combat_parents = raw_obs
    kind_ids = vocab.encode_list(combat_kinds)
    parent_ids = vocab.encode_list(combat_parents)
    np_obs = batch_obs(
        combat_hb_list, terrain_hb_list, gs, kind_ids, parent_ids, config)

    # Normalize global state (continuous features only, flags pass through)
    n_cont = config.global_state_dim - config.n_binary_flags
    gs_norm = np.empty_like(np_obs.global_state)
    gs_norm[..., :n_cont] = agent.obs_normalizer.normalize(np_obs.global_state[..., :n_cont])
    gs_norm[..., n_cont:] = np_obs.global_state[..., n_cont:]
    np_obs = np_obs.replace(global_state=gs_norm)

    # Normalize hitboxes — combat normalizer covers only the leading
    # combat_normalized_dims columns; the trailing hp_raw column passes through raw.
    n_norm_c = config.combat_normalized_dims
    chb = np_obs.combat_hb
    cm = np_obs.combat_mask
    for i in range(chb.shape[0]):
        nc = int(cm[i].sum())
        if nc > 0:
            chb[i, :nc, :n_norm_c] = agent.combat_normalizer.normalize(chb[i, :nc, :n_norm_c])
    thb = np_obs.terrain_hb
    tm = np_obs.terrain_mask
    for i in range(thb.shape[0]):
        nt = int(tm[i].sum())
        if nt > 0:
            thb[i, :nt] = agent.terrain_normalizer.normalize(thb[i, :nt])

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
        actions, _, _, _, _, hx_new = agent.policy.get_action_and_value(
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
    parser.add_argument("checkpoint", help="Path to model checkpoint (.pth)")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false",
                        help="Use stochastic sampling instead of greedy argmax")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--time-scale", type=int, default=1,
                        help="Game speed multiplier (default: 1 for real-time, frames_per_wait auto-scaled)")
    parser.add_argument("--level", default="GG_Mega_Moss_Charger",
                        help="Boss scene name (default: GG_Mega_Moss_Charger)")
    parser.add_argument("--record", metavar="FILE", default="fight.mp4",
                        help="Record screen to video file (default: fight.mp4, --record none to disable)")
    parser.add_argument("--no-record", dest="record", action="store_const", const=None,
                        help="Disable screen recording")
    parser.add_argument("--duration", type=int, default=0,
                        help="Max run time in seconds (default: 0 = no limit, ends on death)")
    parser.add_argument("--hk-path", default=None,
                        help="Path to Hollow Knight install (overrides config default)")
    args = parser.parse_args()

    asyncio.run(eval_play(
        args.checkpoint,
        deterministic=args.deterministic,
        time_scale=args.time_scale,
        level=args.level,
        record=args.record,
        hk_path=args.hk_path,
        duration=args.duration,
    ))


if __name__ == "__main__":
    main()
