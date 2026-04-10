using System.Collections.Generic;
using UnityEngine;
using GlobalEnums;
using Modding;
using UnityEngine.SceneManagement;

namespace FullKnight.Game
{
	public enum HitboxType : int
	{
		Knight,
		Enemy,
		Attack,
		Terrain
	}

	public class HitboxReader : MonoBehaviour
	{
		public readonly SortedDictionary<HitboxType, HashSet<Collider2D>> colliders = new()
		{
			{HitboxType.Knight, new HashSet<Collider2D>()},
			{HitboxType.Enemy, new HashSet<Collider2D>()},
			{HitboxType.Attack, new HashSet<Collider2D>()},
			{HitboxType.Terrain, new HashSet<Collider2D>()}
		};

		// Cached per-collider kind/parent strings + HealthManager ref. Computed
		// once on first classification (cheap parent walk) and reused thereafter,
		// since HK reparents rarely.
		public readonly Dictionary<Collider2D, string> kindCache = new();
		public readonly Dictionary<Collider2D, string> parentCache = new();
		public readonly Dictionary<Collider2D, HealthManager> hmCache = new();

		private void Start()
		{
			foreach (Collider2D collider2D in Resources.FindObjectsOfTypeAll<Collider2D>())
			{
				AddHitbox(collider2D);
			}
		}

		public void UpdateHitbox(GameObject go)
		{
			foreach (Collider2D col in go.GetComponentsInChildren<Collider2D>(true))
			{
				AddHitbox(col);
			}
		}

		public void PruneDestroyed()
		{
			foreach (var kvp in colliders)
				kvp.Value.RemoveWhere(c => c == null);
			var dead = new List<Collider2D>();
			foreach (var k in kindCache.Keys) if (k == null) dead.Add(k);
			foreach (var k in dead) { kindCache.Remove(k); parentCache.Remove(k); hmCache.Remove(k); }
		}

		private void AddHitbox(Collider2D collider2D)
		{
			if (collider2D == null) return;

			if (collider2D is BoxCollider2D or PolygonCollider2D or EdgeCollider2D or CircleCollider2D)
			{
				GameObject go = collider2D.gameObject;
				if (collider2D.GetComponent<DamageHero>() || collider2D.gameObject.LocateMyFSM("damages_hero"))
				{
					colliders[HitboxType.Enemy].Add(collider2D);
				}
				else if (go.layer == (int)PhysLayers.TERRAIN && !collider2D.isTrigger)
				{
					colliders[HitboxType.Terrain].Add(collider2D);
				}
				else if (go == HeroController.instance?.gameObject && !collider2D.isTrigger)
				{
					colliders[HitboxType.Knight].Add(collider2D);
				}
				else if (go.GetComponent<DamageEnemies>() || go.LocateMyFSM("damages_enemy") || go.name == "Damager" && go.LocateMyFSM("Damage"))
				{
					colliders[HitboxType.Attack].Add(collider2D);
				}
			}
		}

		/// <summary>
		/// Return a stable class-level kind string for a combat collider, computed
		/// from what HK already stores. Rule (first match wins):
		///   1. DamageEnemies on the collider → stripped GameObject name (knight attacks).
		///   2. DamageHero on the collider → stripped GameObject name (enemy projectiles,
		///      attack volumes, hazards).
		///   3. Walk parents to the nearest HealthManager → that root's stripped name.
		///      Catches enemy body colliders, AND uses the same string ClassifyParent
		///      returns so the body's leaf id and an attack's parent id collapse to one
		///      vocab row (and one shared embedding).
		///   4. Fallback: stripped own name, or "unknown".
		/// No role prefix (knight/enemy/other): hurts_knight already separates knight
		/// from enemy hitboxes, and prefixes would prevent the body↔parent merge.
		/// </summary>
		public string GetKind(Collider2D col)
		{
			if (col == null) return "unknown";
			if (kindCache.TryGetValue(col, out string cached)) return cached;

			string result = ClassifyKind(col);
			kindCache[col] = result;
			return result;
		}

		/// <summary>
		/// Walk parents to the nearest HealthManager and return its stripped
		/// root name (e.g. "Mega Moss Charger"). Empty string if none reachable
		/// (detached projectiles, knight-owned colliders, terrain). Used as the
		/// "parent identity" channel in the factored kind embedding: leaf kind
		/// pools across bosses, parent name specializes per boss.
		/// </summary>
		public string GetParentKind(Collider2D col)
		{
			if (col == null) return "";
			if (parentCache.TryGetValue(col, out string cached)) return cached;

			ClassifyParent(col, out string name, out HealthManager hm);
			parentCache[col] = name;
			hmCache[col] = hm;  // may be null
			return name;
		}

		/// <summary>Returns the cached HealthManager (or null) for a collider.
		/// Populated by GetParentKind on first lookup.</summary>
		public HealthManager GetParentHm(Collider2D col)
		{
			if (col == null) return null;
			if (hmCache.TryGetValue(col, out HealthManager cached)) return cached;
			// Force population.
			GetParentKind(col);
			return hmCache.TryGetValue(col, out cached) ? cached : null;
		}

		private void ClassifyParent(Collider2D col, out string name, out HealthManager hm)
		{
			Transform t = col.transform;
			int depth = 0;
			while (t != null && depth < 8)
			{
				var found = t.GetComponent<HealthManager>();
				if (found != null)
				{
					name = Strip(t.gameObject.name);
					hm = found;
					return;
				}
				t = t.parent;
				depth++;
			}
			name = "";
			hm = null;
		}

		private string ClassifyKind(Collider2D col)
		{
			GameObject go = col.gameObject;

			if (go.GetComponent<DamageEnemies>() != null)
				return Strip(go.name);

			if (go.GetComponent<DamageHero>() != null)
				return Strip(go.name);

			Transform t = go.transform;
			int depth = 0;
			while (t != null && depth < 8)
			{
				var hm = t.GetComponent<HealthManager>();
				if (hm != null) return Strip(t.gameObject.name);
				t = t.parent;
				depth++;
			}

			string n = Strip(go.name);
			return string.IsNullOrEmpty(n) ? "unknown" : n;
		}

		private static string Strip(string name)
		{
			if (string.IsNullOrEmpty(name)) return "";
			int i = name.IndexOf("(Clone)");
			if (i >= 0) name = name.Substring(0, i);
			return name.Trim();
		}
	}

	public class HitboxReaderHook
	{
		private HitboxReader _hitboxReader;
		public bool loaded = false;

		public void Load()
		{
			Unload();
			UnityEngine.SceneManagement.SceneManager.activeSceneChanged += CreateHitboxReader;
			ModHooks.ColliderCreateHook += UpdateHitboxReader;
			CreateHitboxReader();
			loaded = true;
		}

		/// <summary>Force-recreate the reader for the current scene
		/// (workaround for missed activeSceneChanged events under multi-instance load).</summary>
		public void RecreateReader() => CreateHitboxReader();

		public void Unload()
		{
			UnityEngine.SceneManagement.SceneManager.activeSceneChanged -= CreateHitboxReader;
			ModHooks.ColliderCreateHook -= UpdateHitboxReader;
			DestroyHitboxReader();
		}

		private void CreateHitboxReader(Scene current, Scene next) => CreateHitboxReader();

		private void CreateHitboxReader()
		{
			DestroyHitboxReader();
			if (GameManager.instance.IsGameplayScene())
			{
				_hitboxReader = new GameObject().AddComponent<HitboxReader>();
			}
		}

		private void DestroyHitboxReader()
		{
			if (_hitboxReader != null)
			{
				Object.Destroy(_hitboxReader);
				_hitboxReader = null;
			}
		}

		private void UpdateHitboxReader(GameObject go)
		{
			if (_hitboxReader != null)
			{
				_hitboxReader.UpdateHitbox(go);
			}
		}

		public SortedDictionary<HitboxType, HashSet<Collider2D>> GetHitboxes()
		{
			return _hitboxReader?.colliders
				?? new SortedDictionary<HitboxType, HashSet<Collider2D>>()
				{
					{ HitboxType.Knight, new HashSet<Collider2D>() },
					{ HitboxType.Enemy, new HashSet<Collider2D>() },
					{ HitboxType.Attack, new HashSet<Collider2D>() },
					{ HitboxType.Terrain, new HashSet<Collider2D>() }
				};
		}

		public HitboxReader GetReader() => _hitboxReader;
	}

	public class HitboxObserver
	{
		private HitboxReaderHook _hook = new();

		public void Load() => _hook.Load();
		public void Unload() => _hook.Unload();
		public void RecreateReader() => _hook.RecreateReader();
		public SortedDictionary<HitboxType, HashSet<Collider2D>> GetHitboxes() => _hook.GetHitboxes();

		public struct SplitObservation
		{
			public List<float[]> CombatHitboxes;
			public List<string> CombatKinds;
			public List<string> CombatParents;
			public List<float[]> TerrainHitboxes;
			public float KnightWidth;
			public float KnightHeight;
		}

		/// <summary>
		/// Extract hitbox features split by type.
		/// Combat (Enemy + Attack): 9 floats per hitbox —
		///   [rel_x, rel_y, width, height, is_trigger,
		///    gives_damage, takes_damage, is_target, hp_raw]
		///   gives_damage = collider hurts the knight on contact (Enemy bucket).
		///   takes_damage = a HealthManager is reachable from this collider.
		///   is_target    = that HealthManager is in the supplied bossHms set.
		///   hp_raw       = current HP of the reached HealthManager (0 if none).
		///                  Raw value, NOT normalized — the agent should be able to
		///                  read "1-2 nail hits from death" vs "beefy".
		/// CombatKinds / CombatParents: parallel string lists.
		/// Terrain: [rel_x, rel_y, width, height, is_trigger]
		/// Knight: bounds only (width, height), folded into global state.
		/// </summary>
		public SplitObservation GetSplitFeatures(System.Collections.Generic.HashSet<HealthManager> bossHms = null)
		{
			var hitboxes = _hook.GetHitboxes();
			var reader = _hook.GetReader();
			// Prune destroyed colliders to prevent set growth over long episodes
			foreach (var kvp in hitboxes)
				kvp.Value.RemoveWhere(c => c == null);
			var knightPos = HeroController.instance.transform.position;

			var combat = new List<float[]>();
			var combatKinds = new List<string>();
			var combatParents = new List<string>();
			var terrain = new List<float[]>();
			float knightW = 0f, knightH = 0f;

			foreach (var kvp in hitboxes)
			{
				foreach (var col in kvp.Value)
				{
					if (col == null || !col.isActiveAndEnabled) continue;

					var bounds = col.bounds;

					if (kvp.Key == HitboxType.Knight)
					{
						knightW = bounds.size.x;
						knightH = bounds.size.y;
						continue;
					}

					float relX = bounds.center.x - knightPos.x;
					float relY = bounds.center.y - knightPos.y;
					float w = bounds.size.x;
					float h = bounds.size.y;
					float isTrigger = col.isTrigger ? 1f : 0f;

					if (kvp.Key == HitboxType.Enemy || kvp.Key == HitboxType.Attack)
					{
						string parent = reader != null ? reader.GetParentKind(col) : "";
						HealthManager hm = reader != null ? reader.GetParentHm(col) : null;

						float givesDamage = kvp.Key == HitboxType.Enemy ? 1f : 0f;
						float takesDamage = hm != null ? 1f : 0f;
						float isTarget = (hm != null && bossHms != null && bossHms.Contains(hm)) ? 1f : 0f;
						float hpRaw = hm != null ? (float)hm.hp : 0f;

						combat.Add(new float[] {
							relX, relY, w, h, isTrigger,
							givesDamage, takesDamage, isTarget, hpRaw
						});
						combatKinds.Add(reader != null ? reader.GetKind(col) : "unknown");
						combatParents.Add(parent);
					}
					else if (kvp.Key == HitboxType.Terrain)
					{
						terrain.Add(new float[] { relX, relY, w, h, isTrigger });
					}
				}
			}

			return new SplitObservation
			{
				CombatHitboxes = combat,
				CombatKinds = combatKinds,
				CombatParents = combatParents,
				TerrainHitboxes = terrain,
				KnightWidth = knightW,
				KnightHeight = knightH
			};
		}
	}
}
