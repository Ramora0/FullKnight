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
		public List<float[]> terrain_hitboxes;
		public float[] global_state;

		// Reward / done
		public float? reward;
		public bool? done;
		public string info;

		// Raw reward signals (for Python-side reward computation)
		public float? damage_landed;  // normalized to nail-hit equivalents
		public int? hits_taken;

		// Diagnostic: time elapsed during frame skip
		public float? step_game_time;   // scaled (Time.deltaTime)
		public float? step_real_time;   // unscaled (Time.unscaledDeltaTime)

		// Mode (Python -> C#, sent during reset)
		public bool? eval;

		// Action (Python -> C#)
		public int[] action_vec;
	}
}
