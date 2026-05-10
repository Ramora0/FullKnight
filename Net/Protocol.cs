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

		// Reset-only: per-phase wallclock deltas (ms) collected by LogPhase.
		// Order is fixed and matches Python's unpack_reset trailer:
		//   [pre_unload, transition_out, settle, load_boss_scene,
		//    recreate_reader, init_boss_refs, obs_final].
		// Empty/null on non-reset messages.
		public float[] reset_phase_deltas_ms;
		// Parallel to reset_phase_deltas_ms: how many Unity frames each phase
		// occupied. ms / frames disambiguates "many normal frames" from
		// "few stalled frames". Empty/null on non-reset messages.
		public int[] reset_phase_frames;
		// 0 = already-in-workshop (no transition needed),
		// 1 = natural-end (waited for HK's death/win transition),
		// 2 = suicide (synthetic kill, mid-fight reset).
		public byte? reset_branch;

		// Env slot index, sent by Python on MSG_INIT. Used to address the
		// per-instance shared-memory step channel (Local\fk_inst_{slot}_*).
		// Null when the wire didn't carry a slot (old Python client).
		public uint? slot;

		// True if Python wants the step hot path routed through ShmChannel.
		// When false (or null = old client), step actions/responses stay on
		// the WebSocket. Both transports always coexist in the C# code; this
		// flag just gates which one is wired up at Setup time.
		public bool? use_shm;
	}
}
