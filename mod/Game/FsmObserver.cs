using System.Collections.Generic;
using HutongGames.PlayMaker;
using UnityEngine;

namespace FullKnight.Game
{
	// Snapshots every PlayMakerFSM relevant to "what is happening in the fight"
	// each step. Three sources are sampled, tagged in the output by single char:
	//   B  — every FSM in the subtree rooted at each boss HealthManager. This
	//        is where the attack-naming states live (e.g. "Slash Antic",
	//        "Throw"). Spawned projectiles are reparented to scene root
	//        (SpawnFromPool.cs) so they're NOT in this subtree.
	//   E  — every FSM on a currently-active Enemy-class collider (damages_hero,
	//        i.e. anything that can hit the knight). Catches in-flight boss
	//        projectile state machines that the boss subtree no longer contains.
	//   A  — every FSM on a currently-active Attack-class collider (damages_enemy,
	//        the knight's nail / spells). Useful for cross-referencing agent
	//        action against attack-window FSM state.
	//
	// Only FSMs whose name is in NameWhitelist are emitted. HK's convention is
	// that the boss's main attack-picking FSM is named "Control" — sibling FSMs
	// on the same GameObject (e.g. "Constrain Y" with a "Fluctuate" state,
	// "Audio", "Health", "FSM") are positional/auxiliary and pollute the
	// attack-segmentation graph. Extend NameWhitelist if a boss is encountered
	// whose controller uses a different name.
	//
	// Output is a flat list of "<src>|<owner>|<fsm>|<state>" strings. Attack
	// segmentation happens entirely Python-side from the observed state-
	// transition graph — no structural action introspection here.
	public class FsmObserver
	{
		public static readonly HashSet<string> NameWhitelist = new HashSet<string>
		{
			"Control",
		};


		public List<string> Snapshot(
			HashSet<HealthManager> bossHMs,
			HashSet<Collider2D> enemyColliders,
			HashSet<Collider2D> attackColliders)
		{
			var entries = new List<string>(32);

			if (bossHMs != null)
			{
				foreach (var hm in bossHMs)
				{
					if (hm == null) continue;
					var go = hm.gameObject;
					if (go == null) continue;
					string owner = go.name;
					var fsms = go.GetComponentsInChildren<PlayMakerFSM>(true);
					if (fsms == null) continue;
					for (int i = 0; i < fsms.Length; i++)
					{
						var fsm = fsms[i];
						if (fsm == null) continue;
						// Skip FSMs on inactive children — they're not driving
						// anything right now and only add noise to the panel.
						if (!fsm.isActiveAndEnabled) continue;
						if (!NameWhitelist.Contains(fsm.FsmName)) continue;
						AppendEntry(entries, "B", owner, fsm);
					}
				}
			}

			AppendCollidersFsms(entries, enemyColliders, "E");
			AppendCollidersFsms(entries, attackColliders, "A");

			return entries;
		}

		private static void AppendCollidersFsms(List<string> entries, HashSet<Collider2D> colliders, string src)
		{
			if (colliders == null) return;
			foreach (var col in colliders)
			{
				if (col == null) continue;
				if (!col.isActiveAndEnabled) continue;
				var go = col.gameObject;
				if (go == null) continue;
				string owner = go.name;
				var fsms = go.GetComponents<PlayMakerFSM>();
				if (fsms == null) continue;
				for (int i = 0; i < fsms.Length; i++)
				{
					var fsm = fsms[i];
					if (fsm == null) continue;
					if (!fsm.isActiveAndEnabled) continue;
					if (!NameWhitelist.Contains(fsm.FsmName)) continue;
					AppendEntry(entries, src, owner, fsm);
				}
			}
		}

		private static void AppendEntry(List<string> entries, string src, string owner, PlayMakerFSM fsm)
		{
			string fsmName = fsm.FsmName ?? "?";
			string state = fsm.ActiveStateName ?? "(none)";
			entries.Add($"{src}|{owner}|{fsmName}|{state}");
		}
	}
}
