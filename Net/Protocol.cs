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

		// Raw hit count signals (for Python-side reward computation)
		public int? hits_landed;
		public int? hits_taken;

		// Action (Python -> C#)
		public int[] action_vec;
	}
}
