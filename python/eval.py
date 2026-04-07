"""Play the best trained agent in a real boss fight and optionally record the screen."""

import argparse
import asyncio
import json
import platform
import subprocess
import sys
import numpy as np
import torch
import websockets

from config import Config
from ppo import PPO


async def eval_play(checkpoint_path, deterministic=False, time_scale=1,
                    level="GG_Mega_Moss_Charger", record=None):
    config = Config(n_envs=1, level=level, time_scale=time_scale)

    # Load agent
    agent = PPO(config)
    agent.load_checkpoint(checkpoint_path)
    agent.policy.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Level: {level} | Time scale: {time_scale}x | Deterministic: {deterministic}")

    # Wait for game to connect
    game_ws = None
    connected = asyncio.Event()

    async def on_connect(websocket):
        nonlocal game_ws
        game_ws = websocket
        await websocket.send(json.dumps({"type": "init", "data": {}, "sender": "server"}))
        await websocket.recv()
        connected.set()
        try:
            await asyncio.Future()
        except websockets.exceptions.ConnectionClosed:
            pass

    server = await websockets.serve(on_connect, config.server_host, config.server_port)
    print(f"Waiting for game on ws://{config.server_host}:{config.server_port} ...")
    await connected.wait()
    print("Game connected!")

    async def send_recv(msg_type, data):
        await game_ws.send(json.dumps({"type": msg_type, "data": data, "sender": "server"}))
        return json.loads(await game_ws.recv())

    # Reset with eval mode
    resp = await send_recv("reset", {
        "level": config.level,
        "frames_per_wait": config.frames_per_wait,
        "time_scale": config.time_scale,
        "eval": True,
    })

    obs = parse_obs(resp["data"], config)

    # Start screen recording
    recorder = None
    if record:
        recorder = start_recording(record)
        if recorder:
            print(f"Recording to: {record}")

    step_count = 0
    total_damage = 0.0
    total_hits = 0

    print("Fight started!")

    try:
        while True:
            action_vec = get_action(agent, obs, config, deterministic)

            resp = await send_recv("action", {"action_vec": action_vec})
            d = resp["data"]

            damage = d.get("damage_landed", 0) or 0
            hits = d.get("hits_taken", 0) or 0
            total_damage += damage
            total_hits += hits
            step_count += 1

            done = d.get("done", False)
            info = d.get("info", "")

            if step_count % 100 == 0:
                print(f"  Step {step_count:5d} | Damage dealt: {total_damage:5.1f} | Hits taken: {total_hits}")

            if done:
                result = "WIN" if info == "win" else "LOSS"
                print(f"\n{'=' * 50}")
                print(f"  Result:        {result}")
                print(f"  Steps:         {step_count}")
                print(f"  Damage dealt:  {total_damage:.1f} nail hits")
                print(f"  Hits taken:    {total_hits}")
                print(f"{'=' * 50}")
                break

            obs = parse_obs(d, config)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    # Stop recording
    if recorder:
        stop_recording(recorder)
        print(f"Recording saved to: {record}")

    # Close connection
    try:
        await game_ws.send(json.dumps({"type": "close", "data": {}, "sender": "server"}))
    except Exception:
        pass
    server.close()
    await server.wait_closed()


def parse_obs(data, config):
    combat_hb = data.get("combat_hitboxes", [])
    terrain_hb = data.get("terrain_hitboxes", [])
    gs = np.array(data.get("global_state", [0.0] * config.global_state_dim), dtype=np.float32)
    return combat_hb, terrain_hb, gs


def batch_obs(combat_hb_list, terrain_hb_list, gs, config):
    """Convert single-env obs into batched numpy arrays (B=1)."""
    n_combat = max(len(combat_hb_list), 1)
    n_terrain = max(len(terrain_hb_list), 1)

    chb = np.zeros((1, n_combat, config.combat_feature_dim), dtype=np.float32)
    cm = np.zeros((1, n_combat), dtype=np.float32)
    for i, hb in enumerate(combat_hb_list):
        chb[0, i] = np.array(hb, dtype=np.float32)
        cm[0, i] = 1.0

    thb = np.zeros((1, n_terrain, config.terrain_feature_dim), dtype=np.float32)
    tm = np.zeros((1, n_terrain), dtype=np.float32)
    for i, hb in enumerate(terrain_hb_list):
        thb[0, i] = np.array(hb, dtype=np.float32)
        tm[0, i] = 1.0

    gs_batch = gs.reshape(1, -1)
    return chb, cm, thb, tm, gs_batch


@torch.no_grad()
def get_action(agent, obs, config, deterministic=False):
    """Get action from the policy using frozen normalizers (no stats update)."""
    combat_hb_list, terrain_hb_list, gs = obs
    chb, cm, thb, tm, gs_batch = batch_obs(combat_hb_list, terrain_hb_list, gs, config)

    # Normalize global state (continuous features only, flags pass through)
    n_cont = config.global_state_dim - config.n_validity_flags
    gs_norm = np.empty_like(gs_batch)
    gs_norm[..., :n_cont] = agent.obs_normalizer.normalize(gs_batch[..., :n_cont])
    gs_norm[..., n_cont:] = gs_batch[..., n_cont:]

    # Normalize hitboxes
    for i in range(chb.shape[0]):
        nc = int(cm[i].sum())
        if nc > 0:
            chb[i, :nc] = agent.combat_normalizer.normalize(chb[i, :nc])
    for i in range(thb.shape[0]):
        nt = int(tm[i].sum())
        if nt > 0:
            thb[i, :nt] = agent.terrain_normalizer.normalize(thb[i, :nt])

    device = agent.device
    chb_t = torch.from_numpy(chb).float().to(device)
    cm_t = torch.from_numpy(cm).float().to(device)
    thb_t = torch.from_numpy(thb).float().to(device)
    tm_t = torch.from_numpy(tm).float().to(device)
    gs_t = torch.from_numpy(gs_norm).float().to(device)

    if deterministic:
        h = agent.policy._encode(chb_t, cm_t, thb_t, tm_t, gs_t)
        logits_m = agent.policy.head_movement(h)
        logits_d = agent.policy.head_direction(h)
        logits_a = agent.policy.head_action(h).clone()
        logits_j = agent.policy.head_jump(h).clone()
        logits_a, logits_j = agent.policy._mask_logits(logits_a, logits_j, gs_t)

        a_m = logits_m.argmax(-1).item()
        a_d = logits_d.argmax(-1).item()
        a_a = logits_a.argmax(-1).item()
        a_j = logits_j.argmax(-1).item()
    else:
        actions, _, _, _, _ = agent.policy.get_action_and_value(
            chb_t, cm_t, thb_t, tm_t, gs_t
        )
        a_m = actions["movement"].item()
        a_d = actions["direction"].item()
        a_a = actions["action"].item()
        a_j = actions["jump"].item()

    return [a_m, a_d, a_a, a_j]


# ---------------------------------------------------------------------------
# Screen recording helpers (uses ffmpeg)
# ---------------------------------------------------------------------------

def start_recording(output_path):
    """Start an ffmpeg screen recording process. Returns the Popen handle."""
    system = platform.system()
    if system == "Darwin":
        cmd = [
            "ffmpeg", "-y",
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
            "ffmpeg", "-y",
            "-f", "gdigrab",
            "-i", "desktop",
            "-r", "30",
            "-vcodec", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-f", "x11grab",
            "-i", ":0.0",
            "-r", "30",
            "-vcodec", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            output_path,
        ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc
    except FileNotFoundError:
        print("WARNING: ffmpeg not found — recording disabled. Install with: brew install ffmpeg")
        return None


def stop_recording(proc):
    """Gracefully stop ffmpeg by sending 'q' to stdin."""
    try:
        proc.stdin.write(b"q")
        proc.stdin.flush()
        proc.wait(timeout=10)
    except Exception:
        proc.terminate()
        proc.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description="Play trained agent in a real boss fight")
    parser.add_argument("checkpoint", help="Path to model checkpoint (.pth)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use greedy (argmax) action selection instead of sampling")
    parser.add_argument("--time-scale", type=int, default=1,
                        help="Game speed multiplier (default: 1 for real-time)")
    parser.add_argument("--level", default="GG_Mega_Moss_Charger",
                        help="Boss scene name (default: GG_Mega_Moss_Charger)")
    parser.add_argument("--record", metavar="FILE",
                        help="Record screen to video file (requires ffmpeg)")
    args = parser.parse_args()

    asyncio.run(eval_play(
        args.checkpoint,
        deterministic=args.deterministic,
        time_scale=args.time_scale,
        level=args.level,
        record=args.record,
    ))


if __name__ == "__main__":
    main()
