using System;
using System.IO.MemoryMappedFiles;
using System.Threading;

namespace FullKnight.Net
{
    // Shared-memory action/obs channel — mirror of python/shm_transport.py.
    // Used for the step hot path only. Init/reset/pause/resume/close stay
    // on WebSocket (less frequent control plane, variable-shaped payloads).
    //
    // Layout (per-instance, addressed by slot ID):
    //   Local\fk_inst_{slot}_act     MMF, ACT_SIZE bytes; Python writes action
    //   Local\fk_inst_{slot}_obs     MMF, OBS_SIZE bytes; we write [u32 len][payload]
    //   Local\fk_inst_{slot}_act_evt EventWaitHandle, AutoReset; Python -> us
    //   Local\fk_inst_{slot}_obs_evt EventWaitHandle, AutoReset; us -> Python
    //
    // Lifecycle: Python creates the kernel objects before launching HK.
    // CreateOrOpen / new EventWaitHandle(named) on this side just attaches
    // to the existing handles. Both sides Close() / Dispose() on shutdown;
    // OS reaps the named objects when the last ref drops.
    //
    // Stage-1 measurements (python/bench_ipc_mono.py): ~7us mean RTT, ~16us
    // p99. Validated against .NET 4.7.2 BCL on Windows; same surfaces are
    // exposed in Unity Mono so production should track within a small
    // constant.
    public sealed class ShmChannel : IDisposable
    {
        public const int ACT_SIZE = 64;
        public const int OBS_SIZE = 8192;
        public const int OBS_HEADER = 4;     // u32 little-endian payload length
        public const int ACT_PAYLOAD = 17;   // 1B type + 4 x i32 (matches BinaryProtocol)

        public int Slot { get; }

        private readonly MemoryMappedFile _actMmf;
        private readonly MemoryMappedFile _obsMmf;
        private readonly MemoryMappedViewAccessor _actView;
        private readonly MemoryMappedViewAccessor _obsView;
        private readonly EventWaitHandle _actEvt;
        private readonly EventWaitHandle _obsEvt;
        private readonly byte[] _actScratch = new byte[ACT_PAYLOAD];

        private bool _disposed;

        public ShmChannel(int slot)
        {
            Slot = slot;
            string actName = $"Local\\fk_inst_{slot}_act";
            string obsName = $"Local\\fk_inst_{slot}_obs";
            string actEvtName = $"Local\\fk_inst_{slot}_act_evt";
            string obsEvtName = $"Local\\fk_inst_{slot}_obs_evt";

            // CreateOrOpen: Python creates first; we attach to the existing
            // section. If we somehow win the race, we create with the same
            // size — Python's mmap.mmap will then attach successfully.
            _actMmf = MemoryMappedFile.CreateOrOpen(actName, ACT_SIZE);
            _obsMmf = MemoryMappedFile.CreateOrOpen(obsName, OBS_SIZE);
            _actView = _actMmf.CreateViewAccessor(0, ACT_SIZE);
            _obsView = _obsMmf.CreateViewAccessor(0, OBS_SIZE);

            // EventWaitHandle's named ctor opens an existing event with the
            // given name, or creates one with the supplied initialState +
            // EventResetMode. AutoReset clears the signal atomically when
            // a waiter unblocks, so we never need ResetEvent.
            _actEvt = new EventWaitHandle(false, EventResetMode.AutoReset, actEvtName);
            _obsEvt = new EventWaitHandle(false, EventResetMode.AutoReset, obsEvtName);
        }

        // Non-blocking poll for a pending action. Returns true and fills the
        // 4-element action vector if one is ready; returns false otherwise.
        // Designed for use from a Unity coroutine that yields each frame —
        // WaitOne(0) is a cheap user-mode peek when the event is unsignaled.
        public bool TryReadAction(out int[] actionVec)
        {
            if (!_actEvt.WaitOne(0))
            {
                actionVec = null;
                return false;
            }
            // Action wire format from binary_protocol.pack_action:
            //   [u8 type=MSG_ACTION][i32 movement][i32 direction][i32 action][i32 jump]
            // We accept any type byte here (caller doesn't dispatch), so we
            // just read the 4 trailing i32s.
            _actView.ReadArray(0, _actScratch, 0, ACT_PAYLOAD);
            actionVec = new int[4];
            actionVec[0] = BitConverter.ToInt32(_actScratch, 1);
            actionVec[1] = BitConverter.ToInt32(_actScratch, 5);
            actionVec[2] = BitConverter.ToInt32(_actScratch, 9);
            actionVec[3] = BitConverter.ToInt32(_actScratch, 13);
            return true;
        }

        // Write a packed observation payload into the obs region and signal
        // Python that it's ready. payload is the body that Python's
        // unpack_step expects (i.e., the same bytes BinaryProtocol.Pack
        // would have produced minus the WebSocket framing).
        public void SendObs(byte[] payload)
        {
            if (payload.Length > OBS_SIZE - OBS_HEADER)
                throw new ArgumentException(
                    $"obs payload {payload.Length}B exceeds buffer ({OBS_SIZE - OBS_HEADER}B)");
            _obsView.Write(0, payload.Length);
            _obsView.WriteArray(OBS_HEADER, payload, 0, payload.Length);
            _obsEvt.Set();
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            // Order matches Python: views first, then sections, then events.
            try { _actView?.Dispose(); } catch { }
            try { _obsView?.Dispose(); } catch { }
            try { _actMmf?.Dispose(); } catch { }
            try { _obsMmf?.Dispose(); } catch { }
            try { _actEvt?.Dispose(); } catch { }
            try { _obsEvt?.Dispose(); } catch { }
        }
    }
}
