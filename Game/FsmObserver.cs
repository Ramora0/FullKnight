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
	// Selection is blacklist-based, not whitelist: every active FSM in the boss
	// subtree (or on a relevant collider) is emitted except those whose name is
	// in NameBlacklist. HK's main attack-picking FSM is conventionally named
	// "Control" on most bosses, but several use specialized names (e.g.
	// "FalseyControl", "Mossy Charger", "Gruz Mother Top"). A whitelist of just
	// "Control" silently dropped those, so multi-boss runs only ever produced
	// data for Hornet. The blacklist instead enumerates the known auxiliary
	// FSMs (audio/health/stun/positional/etc.) that pollute the segmentation
	// graph and lets the boss controller through regardless of its name.
	//
	// Output is a flat list of "<src>|<owner>|<fsm>|<state>" strings. Attack
	// segmentation happens entirely Python-side from the observed state-
	// transition graph — no structural action introspection here.
	public class FsmObserver
	{
		// FSM names known to be auxiliary noise rather than attack pickers.
		// Drawn from decomp grep + HK-modder convention. Any FSM whose name
		// matches is dropped before reaching the segmenter. Boss controller
		// FSMs (whatever they're called) pass through.
		public static readonly HashSet<string> NameBlacklist = new HashSet<string>
		{
			// Damage / health plumbing
			"damages_enemy", "damages_hero",
			"health_manager_enemy", "health_manager", "Health",
			"Set HP",
			// Stun / hit reaction (a boss's stun FSM is its own state machine
			// with hit-react states; useful for debugging but inflates the
			// segmenter with interrupt edges — same reason the viewer hides
			// Stun Air / Stun Land)
			"Stun Control", "Stun", "Stun Damage",
			// Audio / FX
			"Audio", "Sounds", "Music Region",
			"Death", "Death Effects", "Crash Effect", "Shake", "shudder",
			"CameraShake",
			// Positional constraints
			"Constrain X", "Constrain Y", "Bobble",
			// Camera / hero locks (block hero input during boss intros etc.)
			"Camera Lock", "Roar Lock", "Hero Lock",
			// Generic catch-all name HK uses for unnamed FSMs on auxiliary GOs
			"FSM",
			// Range / detection
			"Detect Range",
			// Recoil
			"Recoil",
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
						if (NameBlacklist.Contains(fsm.FsmName)) continue;
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
					if (NameBlacklist.Contains(fsm.FsmName)) continue;
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
