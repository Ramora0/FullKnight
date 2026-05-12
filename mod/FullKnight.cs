using System;
using System.IO;
using Modding;

namespace FullKnight
{
	internal class FullKnight : Mod
	{
		internal static FullKnight Instance { get; private set; }

		// Slot ID stamped by TrainingEnv.Setup when MSG_INIT arrives. Until
		// then, log lines are tagged with the instance's pre-MSG_INIT tag
		// (derived from Application.dataPath — see InstanceLog).
		public static int SlotId = -1;

		// Per-instance log file. The Modding API's central ModLog.txt is
		// shared across all junction-linked HK processes (they all map to the
		// same %APPDATALOW%/Team Cherry/Hollow Knight directory), so when 6
		// instances run in parallel only one of them actually wins the file
		// lock and the rest silently drop their lines. We bypass that here
		// by maintaining our own writer per Unity process, keyed off the
		// per-instance dataPath suffix (i0_Data → "i0", etc).
		public static void LogS(string msg)
		{
			string tagged = $"[s{SlotId}] {msg}";
			InstanceLog.Write(tagged);
		}

		private string _serverUrl = "ws://localhost:8765";

		public override void Initialize()
		{
			Instance = this;
			// First line of the per-instance log records where it lives, so
			// the central ModLog (which still gets this line via Mod.Log) tells
			// the user which file to tail for which slot.
			Log($"FullKnight initializing — per-instance log: {InstanceLog.Path}");
			InstanceLog.Write($"FullKnight initializing — tag={InstanceLog.Tag}");
			var env = new Environment.TrainingEnv(_serverUrl);
			env.Start();
		}

		public override string GetVersion() => "1.0.0";
	}

	internal static class InstanceLog
	{
		private static readonly object _lock = new object();
		private static StreamWriter _writer;
		private static string _path;
		private static string _tag;
		private static bool _initFailed;

		public static string Path { get { EnsureOpen(); return _path ?? "(unopened)"; } }
		public static string Tag { get { EnsureOpen(); return _tag ?? "(unknown)"; } }

		private static string ComputeTag()
		{
			// Application.dataPath looks like "...\\iN_Data\\Hollow Knight_Data"
			// or just "...\\iN_Data" depending on the Unity version. Strip the
			// trailing "_Data" segment and use it as the instance tag. Falls
			// back to PID if Application.dataPath is unavailable / unexpected.
			try
			{
				string dp = UnityEngine.Application.dataPath ?? "";
				string dir = System.IO.Path.GetFileName(dp);
				if (!string.IsNullOrEmpty(dir) && dir.EndsWith("_Data") && dir.Length > 5)
					return dir.Substring(0, dir.Length - 5);
			}
			catch { }
			try { return "pid" + System.Diagnostics.Process.GetCurrentProcess().Id; }
			catch { return "unknown"; }
		}

		private static void EnsureOpen()
		{
			if (_writer != null || _initFailed) return;
			lock (_lock)
			{
				if (_writer != null || _initFailed) return;
				try
				{
					_tag = ComputeTag();
					string baseDir = UnityEngine.Application.persistentDataPath;
					_path = System.IO.Path.Combine(baseDir, $"FullKnight_{_tag}.log");
					// FileMode.Create truncates so each run starts fresh; the
					// Read share lets the user tail -F from another process.
					var fs = new FileStream(_path, FileMode.Create, FileAccess.Write, FileShare.Read);
					_writer = new StreamWriter(fs) { AutoFlush = true };
					_writer.WriteLine($"[{DateTime.Now:HH:mm:ss.fff}] InstanceLog opened path={_path} tag={_tag}");
				}
				catch (Exception ex)
				{
					_initFailed = true;
					// Last resort: surface the open failure on the central log
					// so the user knows logging is degraded for this instance.
					try { FullKnight.Instance?.Log($"[InstanceLog] open failed: {ex.Message}"); }
					catch { }
				}
			}
		}

		public static void Write(string msg)
		{
			EnsureOpen();
			if (_writer == null) return;
			try
			{
				// Lock around the write — Unity coroutines are single-threaded,
				// but background hooks (e.g. FreezeMoment intercepts) and the
				// shm dispatch path can land here from off-main contexts.
				lock (_lock)
				{
					_writer.WriteLine($"[{DateTime.Now:HH:mm:ss.fff}] {msg}");
				}
			}
			catch { /* never let a logging failure propagate */ }
		}
	}
}
