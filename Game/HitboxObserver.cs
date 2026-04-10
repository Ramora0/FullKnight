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

		// Cached per-collider kind/parent strings. Computed once on first
		// classification (cheap parent walk) and reused thereafter, since HK
		// reparents rarely.
		public readonly Dictionary<Collider2D, string> kindCache = new();
		public readonly Dictionary<Collider2D, string> parentCache = new();

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
			foreach (var k in dead) { kindCache.Remove(k); parentCache.Remove(k); }
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
		///   1. DamageHero component on the collider → stripped GameObject name.
		///      This is the prefab that defines "what hurts the knight" — projectiles,
		///      per-attack damage volumes, area hazards. Shared across bosses when
		///      the prefab is shared (e.g. "Needle", "Fireball").
		///   2. DamageEnemies component → stripped GameObject name. Knight attacks:
		///      "Slash", "AltSlash", "Fireball(Clone)" → "Fireball", etc.
		///   3. Walk parents to the nearest HealthManager → that root's stripped name.
		///      Catches enemy body colliders (the boss collider itself).
		///   4. Fallback: stripped own name, or "unknown".
		/// Namespaced with a prefix so knight/enemy/terrain keys never collide.
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

			string result = ClassifyParent(col);
			parentCache[col] = result;
			return result;
		}

		private string ClassifyParent(Collider2D col)
		{
			Transform t = col.transform;
			int depth = 0;
			while (t != null && depth < 8)
			{
				if (t.GetComponent<HealthManager>() != null)
					return Strip(t.gameObject.name);
				t = t.parent;
				depth++;
			}
			return "";
		}

		private string ClassifyKind(Collider2D col)
		{
			GameObject go = col.gameObject;

			// Knight-owned attack: DamageEnemies sits on the prefab that dealt damage.
			if (go.GetComponent<DamageEnemies>() != null)
			{
				return "knight/" + Strip(go.name);
			}

			// Enemy-owned damage: DamageHero identifies the hurting prefab (projectile,
			// attack volume, hazard). Named after the prefab — "Needle", "Fireball".
			if (go.GetComponent<DamageHero>() != null)
			{
				return "enemy/" + Strip(go.name);
			}

			// Enemy body: walk up to the nearest HealthManager, use its root name.
			// This is where "which boss is this" lives — on the boss's own collider,
			// distinct from any projectile it fires.
			Transform t = go.transform;
			int depth = 0;
			while (t != null && depth < 8)
			{
				var hm = t.GetComponent<HealthManager>();
				if (hm != null) return "enemy/" + Strip(t.gameObject.name);
				t = t.parent;
				depth++;
			}

			// Nothing identifying. Use own name so at least identical unknowns group.
			string n = Strip(go.name);
			return string.IsNullOrEmpty(n) ? "unknown" : "other/" + n;
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
		/// Combat (Enemy + Attack): [rel_x, rel_y, width, height, is_trigger, hurts_knight, is_target]
		/// CombatKinds: parallel string list, one kind-id per combat hitbox.
		/// Terrain: [rel_x, rel_y, width, height, is_trigger]
		/// Knight: bounds only (width, height), folded into global state.
		/// </summary>
		public SplitObservation GetSplitFeatures()
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
						float hurtsKnight = kvp.Key == HitboxType.Enemy ? 1f : 0f;
						float isTarget = kvp.Key == HitboxType.Enemy ? 1f : 0f;
						combat.Add(new float[] { relX, relY, w, h, isTrigger, hurtsKnight, isTarget });
						combatKinds.Add(reader != null ? reader.GetKind(col) : "unknown");
						combatParents.Add(reader != null ? reader.GetParentKind(col) : "");
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
