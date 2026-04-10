using System;
using System.IO;
using System.Collections.Generic;

namespace FullKnight.Net
{
	public static class BinaryProtocol
	{
		public const byte MSG_INIT   = 0;
		public const byte MSG_RESET  = 1;
		public const byte MSG_STEP   = 2;
		public const byte MSG_ACTION = 3;
		public const byte MSG_PAUSE  = 4;
		public const byte MSG_RESUME = 5;
		public const byte MSG_CLOSE  = 6;

		private static readonly Dictionary<byte, string> IdToType = new()
		{
			{MSG_INIT, "init"}, {MSG_RESET, "reset"}, {MSG_STEP, "step"},
			{MSG_ACTION, "action"}, {MSG_PAUSE, "pause"}, {MSG_RESUME, "resume"},
			{MSG_CLOSE, "close"}
		};

		private static readonly Dictionary<string, byte> TypeToId = new()
		{
			{"init", MSG_INIT}, {"reset", MSG_RESET}, {"step", MSG_STEP},
			{"action", MSG_ACTION}, {"pause", MSG_PAUSE}, {"resume", MSG_RESUME},
			{"close", MSG_CLOSE}
		};

		/// <summary>Pack a C# -> Python response as binary.</summary>
		public static byte[] Pack(Message message)
		{
			using var ms = new MemoryStream(512);
			using var w = new BinaryWriter(ms);

			w.Write(TypeToId[message.type]);
			var d = message.data;

			if (message.type == "step" || message.type == "reset")
			{
				var combat = d.combat_hitboxes ?? new List<float[]>();
				var terrain = d.terrain_hitboxes ?? new List<float[]>();
				var gs = d.global_state ?? Array.Empty<float>();
				var kinds = d.combat_kinds ?? new List<string>();
				var parents = d.combat_parents ?? new List<string>();

				w.Write((ushort)combat.Count);
				w.Write((ushort)terrain.Count);

				foreach (var hb in combat)
					for (int i = 0; i < hb.Length; i++)
						w.Write(hb[i]);

				foreach (var hb in terrain)
					for (int i = 0; i < hb.Length; i++)
						w.Write(hb[i]);

				for (int i = 0; i < gs.Length; i++)
					w.Write(gs[i]);

				if (message.type == "step")
				{
					w.Write(d.damage_landed ?? 0f);
					w.Write((float)(d.hits_taken ?? 0));
					w.Write(d.step_game_time ?? 0f);
					w.Write(d.step_real_time ?? 0f);
					w.Write(d.done == true ? (byte)1 : (byte)0);
				}

				// Combat kind strings: one per combat hitbox, in the same order.
				// Format: u8 length + UTF-8 bytes. Length capped at 255 (truncated).
				for (int i = 0; i < combat.Count; i++)
				{
					string k = i < kinds.Count ? (kinds[i] ?? "unknown") : "unknown";
					var bytes = System.Text.Encoding.UTF8.GetBytes(k);
					int len = bytes.Length > 255 ? 255 : bytes.Length;
					w.Write((byte)len);
					w.Write(bytes, 0, len);
				}

				// Combat parent strings: same format, also one per combat hitbox.
				// Empty string means "no parent HealthManager reachable".
				for (int i = 0; i < combat.Count; i++)
				{
					string p = i < parents.Count ? (parents[i] ?? "") : "";
					var bytes = System.Text.Encoding.UTF8.GetBytes(p);
					int len = bytes.Length > 255 ? 255 : bytes.Length;
					w.Write((byte)len);
					w.Write(bytes, 0, len);
				}
			}

			return ms.ToArray();
		}

		/// <summary>Unpack a Python -> C# request from binary.</summary>
		public static Message Unpack(byte[] data)
		{
			using var ms = new MemoryStream(data);
			using var r = new BinaryReader(ms);

			byte typeId = r.ReadByte();
			var msg = new Message { type = IdToType[typeId], data = new MessageData() };

			switch (typeId)
			{
				case MSG_RESET:
					msg.data.frames_per_wait = r.ReadInt32();
					msg.data.time_scale = r.ReadInt32();
					msg.data.eval = r.ReadByte() != 0;
					ushort len = r.ReadUInt16();
					msg.data.level = System.Text.Encoding.UTF8.GetString(r.ReadBytes(len));
					break;
				case MSG_ACTION:
					msg.data.action_vec = new int[4];
					for (int i = 0; i < 4; i++)
						msg.data.action_vec[i] = r.ReadInt32();
					break;
			}

			return msg;
		}
	}
}
