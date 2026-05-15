using System.Collections.Generic;

namespace FullKnight.Net
{
	public class Message
	{
		public string type;
		public string sender;
		public MessageData data;
	}

	public class MessageData
	{
		// Config (Python -> C#, sent during reset)
		public string level;
		public int? frames_per_wait;
		public int? time_scale;

		// Observation (C# -> Python)
		public List<float[]> combat_hitboxes;
		public List<string> combat_kinds;    // parallel: leaf-kind id string per combat hitbox
		public List<string> combat_parents;  // parallel: HealthManager-root name per combat hitbox ("" if none)
		public List<float[]> terrain_hitboxes;
		// Debug-only: parallel to terrain_hitboxes. Pipe-delimited fields describing
		// the underlying Collider2D so the Python viewer can explain ghost terrain
		// boxes (disabled colliders, tilemap/composite interactions, etc.).
		public List<string> terrain_debug;
		public float[] global_state;

		// Reward / done
		public float? reward;
		public bool? done;
		public string info;

		// Raw reward signals (for Python-side reward computation)
		public float? damage_landed;  // % of boss max HP dealt this step
		public int? hits_taken;
		public float? hp_healed;      // HP restored this step (e.g. via focus)

		// Diagnostic: time elapsed during frame skip
		public float? step_game_time;   // scaled (Time.deltaTime)
		public float? step_real_time;   // unscaled (Time.unscaledDeltaTime)

		// Diagnostic: long-run memory/leak probes. All optional, populated each
		// step so Python can avg them per epoch and watch for monotone growth.
		// Sizes are raw HashSet/Dictionary counts inside HitboxReader — expected
		// to plateau per scene; unbounded rise = pooled prefabs accumulating via
		// ModHooks.ColliderCreateHook and will track perf/sim_ms upward.
		public ushort? diag_enemy_count;      // HitboxType.Enemy set size
		public ushort? diag_attack_count;     // HitboxType.Attack set size
		public ushort? diag_terrain_count;    // HitboxType.Terrain set size
		public int? diag_kind_cache_size;     // kindCache dict size (stale Unity refs never GC here)
		public float? diag_gc_heap_mb;        // GC.GetTotalMemory(false) in MB — mono heap total

		// Mode (Python -> C#, sent during reset)
		public bool? eval;

		// Action (Python -> C#)
		public int[] action_vec;

		// True iff action[2] was overridden by the hard-commit state machine
		// this step. Python masks the action-head's policy gradient on these
		// steps (the agent didn't make a free choice; movement/direction/jump
		// stay free and get normal gradient).
		public bool? action_committed;

		// Reset-only phase telemetry. Populated by TrainingEnv.Reset and
		// serialized as a fixed-order trailer on reset messages (see
		// BinaryProtocol.Pack). reset_branch encodes which transition path
		// the env took: 0=workshop (no wait), 1=natural_end, 2=unknown (suicide
		// or anything else). reset_phase_ms / reset_phase_frames are sized 7 and
		// indexed in PHASE_KEYS order: [pre_unload, transition_out, settle,
		// load_boss_scene, recreate_reader, init_boss_refs, obs_final]. Python's
		// TimingTracker aggregates these into the per-reset breakdown.
		public byte? reset_branch;
		public float[] reset_phase_ms;
		public ushort[] reset_phase_frames;
	}

	public static class ResetPhase
	{
		// Canonical phase keys, in wire order. Must match Python's
		// TimingTracker.RESET_PHASES tuple.
		public static readonly string[] Keys = new[] {
			"pre_unload", "transition_out", "settle",
			"load_boss_scene", "recreate_reader",
			"init_boss_refs", "obs_final",
		};
		public const int Count = 7;

		public const byte BranchWorkshop    = 0;
		public const byte BranchNaturalEnd  = 1;
		public const byte BranchUnknown     = 2;

		public static int IndexOf(string key)
		{
			for (int i = 0; i < Keys.Length; i++)
				if (Keys[i] == key) return i;
			return -1;
		}
	}
}
