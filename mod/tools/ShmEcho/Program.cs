using System;
using System.IO.MemoryMappedFiles;
using System.Threading;

namespace ShmEcho
{
    // Stage-1 primitives test counterpart for python/bench_ipc_mono.py.
    // Targets net472 (Mono-compatible) so we measure under the same BCL
    // surfaces that Hollow Knight's Unity runtime exposes. The production
    // migration won't ship; this exists only to validate floor latency
    // before committing to the shared-memory IPC migration.
    internal static class Program
    {
        // Match the production buffer/name conventions from the plan.
        // ACT_SIZE includes a 64B cacheline-aligned action region; OBS_SIZE
        // is the worst-case observation envelope. The echo path writes a
        // 4B length prefix + ECHO_PAYLOAD bytes copied from action into obs.
        private const int ACT_SIZE = 64;
        private const int OBS_SIZE = 8192;
        private const int ECHO_PAYLOAD = 17;

        private static int Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.Error.WriteLine("usage: ShmEcho <slot> [iters]");
                return 2;
            }

            int slot = int.Parse(args[0]);
            // Default cap is generous so the bench can drive arbitrary iter counts;
            // the Python side controls the actual number of round-trips.
            long iters = args.Length > 1 ? long.Parse(args[1]) : 1_000_000_000L;

            string actName = $"Local\\fk_inst_{slot}_act";
            string obsName = $"Local\\fk_inst_{slot}_obs";
            string actEvtName = $"Local\\fk_inst_{slot}_act_evt";
            string obsEvtName = $"Local\\fk_inst_{slot}_obs_evt";

            // CreateOrOpen lets either side initialize first; whichever process gets
            // there second just gets a handle to the existing kernel object.
            using var actMmf = MemoryMappedFile.CreateOrOpen(actName, ACT_SIZE);
            using var obsMmf = MemoryMappedFile.CreateOrOpen(obsName, OBS_SIZE);
            using var actView = actMmf.CreateViewAccessor(0, ACT_SIZE);
            using var obsView = obsMmf.CreateViewAccessor(0, OBS_SIZE);
            // AutoReset semantics: WaitOne returns + clears the signal in one atomic op.
            // Initial state is unsignaled so we don't echo a phantom request before Python
            // has written one.
            using var actEvt = new EventWaitHandle(false, EventResetMode.AutoReset, actEvtName);
            using var obsEvt = new EventWaitHandle(false, EventResetMode.AutoReset, obsEvtName);

            // Stdout sync: Python waits for this line before starting the bench, so we
            // can't race the very first SetEvent against an unready WaitOne.
            Console.Out.WriteLine($"[ShmEcho] slot={slot} iters={iters} ready");
            Console.Out.Flush();

            byte[] buf = new byte[ECHO_PAYLOAD];
            for (long i = 0; i < iters; i++)
            {
                // 1s timeout to keep this from hanging if the parent dies.
                if (!actEvt.WaitOne(1000))
                {
                    Console.Error.WriteLine($"[ShmEcho] timeout after {i} iters; exiting");
                    return 1;
                }
                actView.ReadArray(0, buf, 0, ECHO_PAYLOAD);
                obsView.Write(0, ECHO_PAYLOAD);            // 4B length prefix
                obsView.WriteArray(4, buf, 0, ECHO_PAYLOAD);
                obsEvt.Set();
            }
            return 0;
        }
    }
}
