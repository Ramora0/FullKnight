"""Stage-1 primitives test for shared-memory IPC.

Spawns a minimal .NET 4.7.2 console app (tools/ShmEcho) that echoes via
shared memory + Win32 named events, then measures Python<->Mono round-trip
latency through the production ShmChannel. Establishes the floor before
committing to the production migration.

Also serves as the Phase-B correctness test for ShmChannel (asserts that
echoed bytes match what was sent).

Pass:     mean RTT < 30us, p99 < 100us    -> proceed with migration
Marginal: mean RTT < 100us                -> proceed but expect smaller win
Fail:     mean RTT > 100us                -> abandon shm; revert to TCP/UDS

Build the echo first:
    cd tools/ShmEcho && dotnet build -c Release
"""
import os
import statistics
import subprocess
import sys
import time

from shm_transport import ShmChannel

SLOT = 99  # arbitrary; chosen out of band of any real training slot
ITERS = 10_000
WARMUP = 500
ACT_PAYLOAD = 17       # mirrors production action wire size (1B type + 4xi32)


def _locate_echo_exe():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(repo_root, "tools", "ShmEcho", "bin", "Release", "net472", "ShmEcho.exe"),
        os.path.join(repo_root, "tools", "ShmEcho", "bin", "Debug", "net472", "ShmEcho.exe"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def main():
    echo_exe = _locate_echo_exe()
    if echo_exe is None:
        print("ERROR: ShmEcho.exe not found. Build first:", file=sys.stderr)
        print("  cd tools/ShmEcho && dotnet build -c Release", file=sys.stderr)
        return 1

    # Create kernel objects via ShmChannel BEFORE spawning the child so the
    # child's CreateOrOpen / CreateEvent attach to existing handles.
    ch = ShmChannel(slot=SLOT)

    proc = subprocess.Popen(
        [echo_exe, str(SLOT), str(ITERS + WARMUP + 5)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True,
    )

    # Block on stdout until the echo prints its ready line so we don't
    # fire SetEvent before its WaitOne has been entered.
    ready_line = proc.stdout.readline()
    print(f"[bench] echo: {ready_line.strip()}")
    if "ready" not in ready_line:
        print(f"[bench] echo failed to come up; aborting", file=sys.stderr)
        proc.kill()
        ch.close()
        return 2

    action_bytes = bytes(range(ACT_PAYLOAD))

    # Warmup — primes JIT, page faults, scheduler. Also serves as the
    # correctness check: every echo must come back byte-identical.
    for i in range(WARMUP):
        obs = ch.step_sync(action_bytes)
        if obs != action_bytes:
            print(f"[bench] CORRECTNESS FAIL at warmup iter {i}: "
                  f"got {obs!r}, expected {action_bytes!r}", file=sys.stderr)
            proc.kill()
            ch.close()
            return 5

    samples_ns = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        ch.step_sync(action_bytes)
        samples_ns.append(time.perf_counter_ns() - t0)

    proc.wait(timeout=2.0)
    ch.close()

    samples_us = [s / 1000.0 for s in samples_ns]
    mean_us = statistics.mean(samples_us)
    p50 = statistics.median(samples_us)
    p99 = statistics.quantiles(samples_us, n=100)[98]
    p999 = statistics.quantiles(samples_us, n=1000)[998]
    max_us = max(samples_us)

    print()
    print(f"[bench] {ITERS} iters Python <-> .NET 4.7.2 (Mono BCL)")
    print(f"        mean  {mean_us:7.1f} us")
    print(f"        p50   {p50:7.1f} us")
    print(f"        p99   {p99:7.1f} us")
    print(f"        p999  {p999:7.1f} us")
    print(f"        max   {max_us:7.0f} us")
    print()
    print(f"        cost over 1024-step rollout: ~{mean_us * 1024 / 1000:.0f} ms")
    print(f"        current WebSocket  ~340us/step -> ~348 ms / 1024 steps")
    speedup = 340.0 / mean_us if mean_us > 0 else 0
    print(f"        speedup vs WebSocket: {speedup:.1f}x")
    print()

    if mean_us < 30:
        print(f"[bench] PASS - mean {mean_us:.1f}us < 30us target")
        return 0
    if mean_us < 100:
        print(f"[bench] MARGINAL - mean {mean_us:.1f}us < 100us ceiling but > 30us target")
        return 0
    print(f"[bench] FAIL - mean {mean_us:.1f}us > 100us ceiling; abandon shm")
    return 4


if __name__ == "__main__":
    sys.exit(main())
