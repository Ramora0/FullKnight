"""Benchmark IPC round-trip latency: shared memory vs TCP+NODELAY.

Simulates the hot path: send ~17 bytes (action), receive ~800 bytes (observation).
"""
import asyncio
import time
import os
import mmap
import multiprocessing
import multiprocessing.shared_memory
import socket as sock_mod

ITERS = 5000
OBS_SIZE = 694   # typical observation payload
ACTION_SIZE = 17  # action payload

obs_payload = os.urandom(OBS_SIZE)
action_payload = os.urandom(ACTION_SIZE)


# --- Shared memory (multiprocessing, no GIL contention) ---

def shm_worker(shm_name, size, n_iters):
    """Child process: waits for signal=1, writes obs, sets signal=2."""
    shm = multiprocessing.shared_memory.SharedMemory(name=shm_name)
    buf = shm.buf
    for _ in range(n_iters):
        while buf[0] != 1:
            pass
        buf[1:1+OBS_SIZE] = obs_payload
        buf[0] = 2
    shm.close()

def bench_shared_memory():
    size = 1 + OBS_SIZE + 64
    shm = multiprocessing.shared_memory.SharedMemory(create=True, size=size)
    buf = shm.buf
    buf[0] = 0

    total = ITERS + 100  # warmup included
    p = multiprocessing.Process(target=shm_worker, args=(shm.name, size, total))
    p.start()

    # warmup
    for _ in range(100):
        buf[0] = 1
        while buf[0] != 2:
            pass

    t0 = time.perf_counter()
    for _ in range(ITERS):
        buf[0] = 1
        while buf[0] != 2:
            pass
        # simulate reading obs
        _ = bytes(buf[1:1+OBS_SIZE])
    elapsed = time.perf_counter() - t0

    p.join()
    shm.close()
    shm.unlink()
    return elapsed


# --- TCP with NODELAY ---

async def tcp_server_handler(reader, writer):
    writer.transport.get_extra_info('socket').setsockopt(
        sock_mod.IPPROTO_TCP, sock_mod.TCP_NODELAY, 1)
    try:
        while True:
            data = await reader.readexactly(ACTION_SIZE)
            writer.write(obs_payload)
            await writer.drain()
    except (asyncio.IncompleteReadError, ConnectionResetError):
        pass

async def bench_tcp_nodelay():
    host, port = "127.0.0.1", 19877
    srv = await asyncio.start_server(tcp_server_handler, host, port)
    await asyncio.sleep(0.05)

    reader, writer = await asyncio.open_connection(host, port)
    writer.transport.get_extra_info('socket').setsockopt(
        sock_mod.IPPROTO_TCP, sock_mod.TCP_NODELAY, 1)

    # warmup
    for _ in range(100):
        writer.write(action_payload)
        await writer.drain()
        await reader.readexactly(OBS_SIZE)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        writer.write(action_payload)
        await writer.drain()
        await reader.readexactly(OBS_SIZE)
    elapsed = time.perf_counter() - t0

    writer.close()
    srv.close()
    await srv.wait_closed()
    return elapsed


# --- Raw Unix domain socket ---

async def uds_server_handler(reader, writer):
    try:
        while True:
            data = await reader.readexactly(ACTION_SIZE)
            writer.write(obs_payload)
            await writer.drain()
    except (asyncio.IncompleteReadError, ConnectionResetError):
        pass

async def bench_unix_socket():
    path = "/tmp/bench_ipc_uds.sock"
    if os.path.exists(path):
        os.unlink(path)
    srv = await asyncio.start_unix_server(uds_server_handler, path)
    await asyncio.sleep(0.05)

    reader, writer = await asyncio.open_unix_connection(path)

    # warmup
    for _ in range(100):
        writer.write(action_payload)
        await writer.drain()
        await reader.readexactly(OBS_SIZE)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        writer.write(action_payload)
        await writer.drain()
        await reader.readexactly(OBS_SIZE)
    elapsed = time.perf_counter() - t0

    writer.close()
    srv.close()
    await srv.wait_closed()
    os.unlink(path)
    return elapsed


async def main():
    print(f"Benchmarking {ITERS} round-trips, action={ACTION_SIZE}B obs={OBS_SIZE}B")
    print()

    shm_time = bench_shared_memory()
    us = shm_time / ITERS * 1e6
    print(f"Shared memory:  {us:7.1f} us/rt  ({shm_time/ITERS*1024*1000:.0f}ms over 1024 steps)")

    uds_time = await bench_unix_socket()
    us = uds_time / ITERS * 1e6
    print(f"Unix socket:    {us:7.1f} us/rt  ({uds_time/ITERS*1024*1000:.0f}ms over 1024 steps)")

    tcp_time = await bench_tcp_nodelay()
    us = tcp_time / ITERS * 1e6
    print(f"TCP+NODELAY:    {us:7.1f} us/rt  ({tcp_time/ITERS*1024*1000:.0f}ms over 1024 steps)")

    print()
    print("Note: WebSocket adds framing + masking + asyncio overhead on top of TCP.")
    print("Your current measured overhead is ~31ms/step = ~31000us/rt.")

if __name__ == "__main__":
    asyncio.run(main())
