"""Shared-memory action/obs channel for the step hot path.

One ShmChannel per Hollow Knight instance, addressed by slot ID. Replaces
the WebSocket round-trip for `step` only — init/reset/pause/resume/close
stay on WebSocket. The binary protocol (binary_protocol.pack_action /
unpack_step) is unchanged; this just swaps the transport.

Layout:
    Local\\fk_inst_{slot}_act     MMF, ACT_SIZE bytes; Python writes action
    Local\\fk_inst_{slot}_obs     MMF, OBS_SIZE bytes; C# writes [u32 len][payload]
    Local\\fk_inst_{slot}_act_evt EventWaitHandle, AutoReset; Python -> C# signal
    Local\\fk_inst_{slot}_obs_evt EventWaitHandle, AutoReset; C# -> Python signal

Python creates the kernel objects before HK is spawned so the C# side's
CreateOrOpen / CreateEvent attach to the existing handles (no race).

Stage-1 validated this delivers ~7us mean RTT vs ~340us for WebSocket
(see python/bench_ipc_mono.py).
"""
import mmap

import win32api
import win32event


# Kernel-object name template. The Local\ prefix scopes to the current logon
# session — multiple users / RDP sessions on the same machine don't collide.
_ACT_NAME_FMT = r"Local\fk_inst_{slot}_act"
_OBS_NAME_FMT = r"Local\fk_inst_{slot}_obs"
_ACT_EVT_NAME_FMT = r"Local\fk_inst_{slot}_act_evt"
_OBS_EVT_NAME_FMT = r"Local\fk_inst_{slot}_obs_evt"

# Buffer sizes. ACT is fixed-size; the protocol's MSG_ACTION is 17 bytes
# (1 type + 4xi32). 64 covers any future action expansion + cacheline align.
# OBS handles the worst-case observation envelope: max combat ~256 hitboxes
# x 32 bytes + max terrain ~256 x 24 + global state + diag trailer + utf8
# kind/parent strings. 8192 has comfortable headroom; the first 4 bytes
# carry an i32 little-endian payload length so the reader knows how much
# of the buffer is live.
ACT_SIZE = 64
OBS_SIZE = 8192
OBS_HEADER = 4  # u32 length prefix

# Win32 wait return codes.
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 258


class ChannelClosedError(IOError):
    """Raised when the peer (Hollow Knight process) is no longer alive
    during a step round-trip. Mirrors websockets.ConnectionClosedError so
    callers can treat both transports uniformly."""


class ShmChannel:
    """Synchronous shared-memory channel for a single env step round-trip.

    Use via `step_sync(action_bytes, alive_check=...) -> obs_bytes`. From
    asyncio code, wrap the call in `loop.run_in_executor(None, ...)` so the
    event loop isn't blocked on `WaitForSingleObject`.

    The channel is owned by the Python side: it creates the kernel objects
    in __init__ and tears them down in close(). The C# side opens its end
    via MemoryMappedFile.CreateOrOpen / new EventWaitHandle(named) — both
    are idempotent w.r.t. an already-created object.
    """

    def __init__(self, slot: int):
        self.slot = int(slot)
        self._closed = False

        # Order matters slightly: the events are tiny and cheap; the MMFs
        # actually back commit pages. Create both before touching them.
        self.act_mm = mmap.mmap(-1, ACT_SIZE, _ACT_NAME_FMT.format(slot=self.slot))
        self.obs_mm = mmap.mmap(-1, OBS_SIZE, _OBS_NAME_FMT.format(slot=self.slot))
        # AutoReset events (manualReset=False, initialState=False). The signal
        # is consumed atomically by the waiting WaitOne, so we never need to
        # ResetEvent manually.
        self.act_evt = win32event.CreateEvent(
            None, False, False, _ACT_EVT_NAME_FMT.format(slot=self.slot)
        )
        self.obs_evt = win32event.CreateEvent(
            None, False, False, _OBS_EVT_NAME_FMT.format(slot=self.slot)
        )

    def step_sync(self, action_bytes: bytes, *,
                  timeout_ms: int = 5000,
                  alive_check=None) -> bytes:
        """Send action, block until obs is ready, return obs bytes.

        action_bytes: raw binary_protocol.pack_action() output.
        timeout_ms: wait deadline before invoking alive_check / raising.
        alive_check: optional callable -> bool. On timeout, called to decide
            whether the peer has died (raise ChannelClosedError) vs the
            timeout itself was spurious (raise TimeoutError).

        Returns the obs payload (without the 4-byte length prefix), suitable
        for passing straight into binary_protocol.unpack_step.
        """
        if self._closed:
            raise ChannelClosedError(f"slot {self.slot}: channel is closed")
        n = len(action_bytes)
        if n > ACT_SIZE:
            raise ValueError(f"action {n}B exceeds ACT_SIZE={ACT_SIZE}")

        # Write action then signal. The mmap on Windows is backed by the
        # unified page cache, so the C# side sees the bytes the moment its
        # WaitOne unblocks.
        self.act_mm.seek(0)
        self.act_mm.write(action_bytes)
        win32event.SetEvent(self.act_evt)

        rc = win32event.WaitForSingleObject(self.obs_evt, timeout_ms)
        if rc != WAIT_OBJECT_0:
            if alive_check is not None and not alive_check():
                raise ChannelClosedError(
                    f"slot {self.slot}: peer dead, no obs after {timeout_ms}ms"
                )
            raise TimeoutError(
                f"slot {self.slot}: obs timeout after {timeout_ms}ms (rc={rc})"
            )

        self.obs_mm.seek(0)
        length = int.from_bytes(self.obs_mm.read(OBS_HEADER), "little", signed=False)
        if length < 0 or length > OBS_SIZE - OBS_HEADER:
            raise ValueError(
                f"slot {self.slot}: obs length {length} out of range [0, {OBS_SIZE - OBS_HEADER}]"
            )
        return self.obs_mm.read(length)

    def close(self):
        """Tear down kernel objects. Idempotent."""
        if self._closed:
            return
        self._closed = True
        # Close the mmap views first; the kernel section ref-counts down
        # when both sides drop.
        try:
            self.act_mm.close()
        except Exception:
            pass
        try:
            self.obs_mm.close()
        except Exception:
            pass
        # win32event handles are auto-closed on GC, but be explicit so the
        # name becomes free for re-use immediately if the slot is recycled.
        try:
            win32api.CloseHandle(self.act_evt)
        except Exception:
            pass
        try:
            win32api.CloseHandle(self.obs_evt)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        # Best-effort cleanup if the caller forgot close().
        try:
            self.close()
        except Exception:
            pass
