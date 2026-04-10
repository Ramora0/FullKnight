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
	}

	public class HitboxObserver
	{
		private HitboxReaderHook _hook = new();

		public void Load() => _hook.Load();
		public void Unload() => _hook.Unload();
		public SortedDictionary<HitboxType, HashSet<Collider2D>> GetHitboxes() => _hook.GetHitboxes();

		public struct SplitObservation
		{
			public List<float[]> CombatHitboxes;
			public List<float[]> TerrainHitboxes;
			public float KnightWidth;
			public float KnightHeight;
		}

		/// <summary>
		/// Extract hitbox features split by type.
		/// Combat (Enemy + Attack): [rel_x, rel_y, width, height, is_trigger, hurts_knight, is_target]
		/// Terrain: [rel_x, rel_y, width, height, is_trigger]
		/// Knight: bounds only (width, height), folded into global state.
		/// </summary>
		public SplitObservation GetSplitFeatures()
		{
			var hitboxes = _hook.GetHitboxes();
			// Prune destroyed colliders to prevent set growth over long episodes
			foreach (var kvp in hitboxes)
				kvp.Value.RemoveWhere(c => c == null);
			var knightPos = HeroController.instance.transform.position;

			var combat = new List<float[]>();
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
				TerrainHitboxes = terrain,
				KnightWidth = knightW,
				KnightHeight = knightH
			};
		}
	}
}
