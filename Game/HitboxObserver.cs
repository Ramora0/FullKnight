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
		// HK's HealthManager has no public maxHp field — max is whatever the
		// prefab serialized into `hp` at OnEnable. We cache the highest hp
		// we've ever seen per HM, populated on first sight (when hp == max in
		// every normal flow) and bumped if a phase refill ever pushes it higher.
		// Lives on the reader so it dies with the scene, same as the other caches.
		public readonly Dictionary<HealthManager, int> hmMaxHpCache = new();

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
			// Drop max-hp entries for HMs that have been destroyed.
			var deadHms = new List<HealthManager>();
			foreach (var k in hmMaxHpCache.Keys) if (k == null) deadHms.Add(k);
			foreach (var k in deadHms) hmMaxHpCache.Remove(k);
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

		/// <summary>Return the observed max HP for an HM: max of (cached, current).
		/// First sight populates the cache; phase refills (current > cached) bump it.</summary>
		public int ObserveMaxHp(HealthManager hm)
		{
			if (hm == null) return 0;
			int cur = hm.hp;
			if (hmMaxHpCache.TryGetValue(hm, out int seen))
			{
				if (cur > seen) { hmMaxHpCache[hm] = cur; return cur; }
				return seen;
			}
			hmMaxHpCache[hm] = cur;
			return cur;
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
			// Parallel to TerrainHitboxes; one pipe-delimited debug string per box.
			// Format: "name|path|colType|layer|enabled|active|trigger|usedByComposite|bx,by,bw,bh"
			public List<string> TerrainDebug;
			public float KnightWidth;
			public float KnightHeight;
		}

		private static string BuildTerrainDebug(Collider2D col, Vector3 knightPos, Collider2D knightCol)
		{
			if (col == null) return "null";
			var go = col.gameObject;
			string name = go.name ?? "";
			// Build path from up to 4 parents for traceability in-scene.
			var sb = new System.Text.StringBuilder();
			Transform t = go.transform.parent;
			int depth = 0;
			var parts = new List<string>();
			while (t != null && depth < 4)
			{
				parts.Add(t.gameObject.name);
				t = t.parent;
				depth++;
			}
			parts.Reverse();
			string path = string.Join("/", parts);
			string colType = col.GetType().Name;
			string layer = LayerMask.LayerToName(go.layer);
			bool enabled = col.enabled;
			bool active = go.activeInHierarchy;
			bool trigger = col.isTrigger;
			bool usedByComposite = false;
			try { usedByComposite = col.usedByComposite; } catch { }
			var b = col.bounds;
			sb.Append(name).Append('|')
			  .Append(path).Append('|')
			  .Append(colType).Append('|')
			  .Append(layer).Append('|')
			  .Append(enabled ? "1" : "0").Append('|')
			  .Append(active ? "1" : "0").Append('|')
			  .Append(trigger ? "1" : "0").Append('|')
			  .Append(usedByComposite ? "1" : "0").Append('|')
			  .Append(b.center.x.ToString("0.00")).Append(',')
			  .Append(b.center.y.ToString("0.00")).Append(',')
			  .Append(b.size.x.ToString("0.00")).Append(',')
			  .Append(b.size.y.ToString("0.00"));

			// Decompose into actual collision segments (knight-relative world coords)
			// so the viewer can draw the real geometry on top of the lying AABB.
			// Emitted as a trailing pipe-delimited field: "|segments=x1,y1,x2,y2;..."
			// Only emitted for shapes where the AABB is a lie (EdgeCollider2D,
			// PolygonCollider2D). BoxCollider2D's AABB already matches its collision
			// surface modulo rotation, so we skip it to keep the payload small.
			var segs = new System.Text.StringBuilder();
			Transform tr = col.transform;
			if (col is EdgeCollider2D ec)
			{
				var pts = ec.points;
				for (int i = 0; i + 1 < pts.Length; i++)
				{
					Vector3 a = tr.TransformPoint(new Vector3(pts[i].x + ec.offset.x, pts[i].y + ec.offset.y, 0));
					Vector3 bw = tr.TransformPoint(new Vector3(pts[i + 1].x + ec.offset.x, pts[i + 1].y + ec.offset.y, 0));
					if (segs.Length > 0) segs.Append(';');
					segs.Append((a.x - knightPos.x).ToString("0.00")).Append(',')
					    .Append((a.y - knightPos.y).ToString("0.00")).Append(',')
					    .Append((bw.x - knightPos.x).ToString("0.00")).Append(',')
					    .Append((bw.y - knightPos.y).ToString("0.00"));
				}
			}
			else if (col is PolygonCollider2D pc)
			{
				for (int pi = 0; pi < pc.pathCount; pi++)
				{
					var pts = pc.GetPath(pi);
					if (pts == null || pts.Length == 0) continue;
					for (int i = 0; i < pts.Length; i++)
					{
						var p0 = pts[i];
						var p1 = pts[(i + 1) % pts.Length]; // closed path
						Vector3 a = tr.TransformPoint(new Vector3(p0.x + pc.offset.x, p0.y + pc.offset.y, 0));
						Vector3 bw = tr.TransformPoint(new Vector3(p1.x + pc.offset.x, p1.y + pc.offset.y, 0));
						if (segs.Length > 0) segs.Append(';');
						segs.Append((a.x - knightPos.x).ToString("0.00")).Append(',')
						    .Append((a.y - knightPos.y).ToString("0.00")).Append(',')
						    .Append((bw.x - knightPos.x).ToString("0.00")).Append(',')
						    .Append((bw.y - knightPos.y).ToString("0.00"));
					}
				}
			}
			if (segs.Length > 0)
				sb.Append("|segments=").Append(segs);

			// ---- Reachability / "will this actually collide with the knight?" probes ----
			// Appended as trailing key=value fields. Each is independently defensive:
			// any exception downgrades the field to "?" rather than dropping the whole
			// debug string, because these queries touch live Unity physics state.
			//
			// Heroic (hero) layer + the knight collider are the ground truth for "can
			// the knight touch this". The parent pair GetIgnoreLayerCollision +
			// GetIgnoreCollision tells us if the layer-matrix or an explicit per-pair
			// override rules the collider out before physics even looks at the shape.
			// PhysLayers.HERO doesn't exist in HK's enum. Read the hero layer directly
			// from the live HeroController GameObject — ground truth, no enum lookup.
			int heroLayer = HeroController.instance != null
				? HeroController.instance.gameObject.layer
				: -1;
			{
				// Layer collision matrix: true if physics ignores this layer pair.
				string v = "?";
				try { v = Physics2D.GetIgnoreLayerCollision(heroLayer, go.layer) ? "1" : "0"; } catch { }
				sb.Append("|layer_ignore=").Append(v);
			}
			{
				// Per-pair override (takes precedence over the matrix, set at runtime).
				string v = "?";
				try { v = knightCol != null && Physics2D.GetIgnoreCollision(knightCol, col) ? "1" : "0"; } catch { }
				sb.Append("|pair_ignore=").Append(v);
			}
			// Rigidbody2D attached to the terrain collider. If absent the collider
			// acts as a static collider attached to the implicit static body, which
			// is the normal case for terrain.
			var rb = col.attachedRigidbody;
			sb.Append("|rb=").Append(rb != null ? "1" : "0");
			if (rb != null)
			{
				sb.Append("|rb_sim=").Append(rb.simulated ? "1" : "0");
				sb.Append("|rb_type=").Append(rb.bodyType.ToString().Substring(0, 3).ToLower());
			}
			// Signed minimum distance from knight collider to this collider. Negative
			// means currently penetrating (overlap). `isValid=false` means physics
			// couldn't compute — usually because one side isn't simulated.
			if (knightCol != null)
			{
				try
				{
					var cd = Physics2D.Distance(knightCol, col);
					sb.Append("|dist=")
					  .Append(cd.isValid ? cd.distance.ToString("0.00") : "inv");
				}
				catch { sb.Append("|dist=err"); }

				try
				{
					sb.Append("|touching=").Append(col.IsTouching(knightCol) ? "1" : "0");
				}
				catch { sb.Append("|touching=?"); }
			}
			// Linecast from knight center to the collider's closest point on its
			// surface. Two questions answered:
			//   ray_self  : does the first hit along that line belong to `col`?
			//               1 = yes (line-of-sight reachable), 0 = something else
			//               blocks first, ? = no hit at all.
			//   ray_first : name of whatever DID get hit first (useful when
			//               ray_self=0, so we can see what's eclipsing this one).
			//   ray_dist  : distance along the ray to the first hit.
			try
			{
				Vector2 start = knightPos;
				Vector2 end;
				try { end = col.ClosestPoint(start); }
				catch { end = (Vector2)col.bounds.center; }
				var hit = Physics2D.Linecast(start, end);
				if (hit.collider == null)
				{
					sb.Append("|ray_self=?|ray_first=none");
				}
				else
				{
					sb.Append("|ray_self=").Append(hit.collider == col ? "1" : "0");
					sb.Append("|ray_first=").Append(hit.collider.gameObject.name);
					sb.Append("|ray_dist=").Append(hit.distance.ToString("0.00"));
				}
			}
			catch { sb.Append("|ray_self=err"); }

			// OverlapPoint at the collider's AABB center, all layers. Tells us what
			// (if anything) actually sits at that spot — helps spot render-only
			// GameObjects whose bounds don't correspond to any solid collider.
			try
			{
				var hits = Physics2D.OverlapPointAll(b.center);
				bool sawSelf = false;
				int solidCount = 0;
				foreach (var h in hits)
				{
					if (h == null) continue;
					if (h == col) sawSelf = true;
					if (!h.isTrigger) solidCount++;
				}
				sb.Append("|overlap_self=").Append(sawSelf ? "1" : "0");
				sb.Append("|overlap_n=").Append(solidCount);
			}
			catch { sb.Append("|overlap_self=err"); }

			return sb.ToString();
		}

		/// <summary>
		/// Extract hitbox features split by type.
		/// Combat (Enemy + Attack): 10 floats per hitbox —
		///   [rel_x, rel_y, width, height, is_trigger,
		///    gives_damage, takes_damage, is_target, hp_raw, hp_max_raw]
		///   gives_damage = collider hurts the knight on contact (Enemy bucket).
		///   takes_damage = a HealthManager is reachable from this collider.
		///   is_target    = that HealthManager is in the supplied bossHms set.
		///   hp_raw       = current HP of the reached HealthManager (0 if none).
		///   hp_max_raw   = observed max HP (cached on first sight, bumped on refills).
		/// Both hp_raw and hp_max_raw are emitted RAW; the Python side log1p-compresses
		/// them so the network sees a sane magnitude while preserving high resolution
		/// in the "1-2 nail hits from death" regime.
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
			// Pick a representative knight collider to use as the "query" side of
			// every reachability probe in BuildTerrainDebug. Prefer a non-trigger
			// collider from the Knight bucket; fall back to any Collider2D on the
			// hero GameObject.
			Collider2D knightCol = null;
			if (hitboxes.ContainsKey(HitboxType.Knight))
			{
				foreach (var kc in hitboxes[HitboxType.Knight])
				{
					if (kc != null && kc.isActiveAndEnabled && !kc.isTrigger)
					{
						knightCol = kc;
						break;
					}
				}
			}
			if (knightCol == null && HeroController.instance != null)
				knightCol = HeroController.instance.GetComponent<Collider2D>();

			var combat = new List<float[]>();
			var combatKinds = new List<string>();
			var combatParents = new List<string>();
			var terrain = new List<float[]>();
			var terrainDebug = new List<string>();
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
						float hpMaxRaw = hm != null && reader != null ? (float)reader.ObserveMaxHp(hm) : 0f;

						combat.Add(new float[] {
							relX, relY, w, h, isTrigger,
							givesDamage, takesDamage, isTarget, hpRaw, hpMaxRaw
						});
						combatKinds.Add(reader != null ? reader.GetKind(col) : "unknown");
						combatParents.Add(parent);
					}
					else if (kvp.Key == HitboxType.Terrain)
					{
						terrain.Add(new float[] { relX, relY, w, h, isTrigger });
						terrainDebug.Add(BuildTerrainDebug(col, knightPos, knightCol));
					}
				}
			}

			return new SplitObservation
			{
				CombatHitboxes = combat,
				CombatKinds = combatKinds,
				CombatParents = combatParents,
				TerrainHitboxes = terrain,
				TerrainDebug = terrainDebug,
				KnightWidth = knightW,
				KnightHeight = knightH
			};
		}
	}
}
