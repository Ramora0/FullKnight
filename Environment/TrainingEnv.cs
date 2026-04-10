using System.Collections;
using System.Collections.Generic;
using System.Text;
using FullKnight.Net;
using FullKnight.Game;
using HutongGames.PlayMaker;
using InControl;
using Modding;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace FullKnight.Environment
{
	public class TrainingEnv : WebsocketEnv
	{
		private string _level;
		private int _frameSkipCount;
		private int _timeScaleValue;
		private int _hitsTakenInStep;
		private float _damageLandedInStep;
		private int _bossMaxHP;
		private int _knightMaxHP;

		// Eval mode: real damage, real death, episode ends on kill
		private bool _evalMode;
		private bool _bossDied;
		private bool _episodeDone;
		private string _episodeResult;
		private HealthManager _bossHM;

		// Boss intro skip: keep simulating internally until combat starts
		private bool _combatStarted;

		// Diagnostics: count resets so logs are correlatable across episodes
		private int _resetCount;

		private HitboxObserver _hitboxObserver = new();
		private InputDeviceShim _inputShim = new();
		private Game.TimeScale _timeManager;

		public TrainingEnv(string url, params string[] protocols) : base(url, protocols) { }

		protected override IEnumerator OnMessage(Message message)
		{
			switch (message.type)
			{
				case "close":
					_terminate = true;
					break;
				case "action":
					yield return Step(message.data);
					break;
				case "reset":
					yield return Reset(message.data);
					break;
				case "pause":
					yield return Pause(message.data);
					break;
				case "resume":
					yield return Resume(message.data);
					break;
			}
		}

		private IEnumerator Pause(MessageData data)
		{
			Time.timeScale = 0;
			SendMessage(new Message { type = "pause", data = data });
			yield break;
		}

		private IEnumerator Resume(MessageData data)
		{
			// Step() owns freeze/unfreeze (lines 136/184). Unfreezing here opens a gap
			// where the previous rollout's held inputs tick frames and leak damage into
			// step 0 before the new action is applied.
			SendMessage(new Message { type = "resume", data = data });
			yield break;
		}

		private IEnumerator Reset(MessageData data)
		{
			_level = data.level ?? _level;
			_frameSkipCount = data.frames_per_wait ?? _frameSkipCount;
			_timeScaleValue = data.time_scale ?? _timeScaleValue;
			_evalMode = data.eval ?? false;
			_hitsTakenInStep = 0;
			_damageLandedInStep = 0;
			_bossDied = false;
			_episodeDone = false;
			_episodeResult = null;
			_combatStarted = false;
			_resetCount++;

			LogBossDiag($"reset#{_resetCount} PRE-UNLOAD (still in old scene)");

			// Release any inputs held over from the previous episode before the
			// scene transition unfreezes time — otherwise a stuck "left" or "jump"
			// runs the knight for the entire transition + intro-skip window.
			ActionDecoder.ApplyAction(_inputShim, new int[] { 2, 2, 7, 1 });

			// Unpause so scene transition and WaitForSeconds can proceed
			Time.timeScale = _timeScaleValue;

			yield return SceneHooks.LoadBossScene(_level);

			LogBossDiag($"reset#{_resetCount} POST-SCENELOAD (before reader recreate)");

			// Force-recreate the hitbox reader for the boss scene. The activeSceneChanged
			// event is unreliable under multi-instance load (some instances miss it), so
			// we explicitly rebuild the reader here and yield a frame for Start() to scan.
			_hitboxObserver.RecreateReader();
			yield return null;

			InitBossRefs();
			LogBossDiag($"reset#{_resetCount} POST-INITBOSSREFS");
			_knightMaxHP = PlayerData.instance.maxHealth;

			UnhookDamage();
			HookDamage();

			if (_timeManager != null) _timeManager.Dispose();
			_timeManager = new Game.TimeScale(_timeScaleValue);

			var obs = _hitboxObserver.GetSplitFeatures();
			var gs = StateExtractor.GetGlobalState(obs.KnightWidth, obs.KnightHeight);

			data.combat_hitboxes = obs.CombatHitboxes;
			data.combat_kinds = obs.CombatKinds;
			data.combat_parents = obs.CombatParents;
			data.terrain_hitboxes = obs.TerrainHitboxes;
			data.global_state = gs;

			Time.timeScale = 0;
			SendMessage(new Message { type = "reset", data = data });
			yield break;
		}

		private IEnumerator Step(MessageData data)
		{
			// If episode already ended, keep returning done
			if (_episodeDone)
			{
				data.done = true;
				data.info = _episodeResult;
				data.combat_hitboxes = new List<float[]>();
				data.terrain_hitboxes = new List<float[]>();
				data.global_state = new float[19];
				data.damage_landed = 0;
				data.hits_taken = 0;
				data.step_game_time = 0;
				data.step_real_time = 0;
				SendMessage(new Message { type = "step", data = data });
				yield break;
			}

			Time.timeScale = _timeScaleValue;

			ActionDecoder.ApplyAction(_inputShim, data.action_vec);

			// In training mode, restore knight HP each step for infinite fighting
			if (!_evalMode)
				PlayerData.instance.health = _knightMaxHP;

			float gameTimeElapsed = 0f;
			float realTimeElapsed = 0f;
			for (int i = 0; i < _frameSkipCount; i++)
			{
				yield return null;
				gameTimeElapsed += Time.deltaTime;
				realTimeElapsed += Time.unscaledDeltaTime;
				// In eval mode, break early on death
				if (_evalMode && (_bossDied || PlayerData.instance.health <= 0))
					break;
			}

			// If boss intro is still playing, fast-forward until combat starts
			if (!_combatStarted)
			{
				LogBossDiag($"reset#{_resetCount} INTRO-SKIP START");
				Time.timeScale = 20f;
				int introFrames = 0;
				while (!HasActiveCombatHitboxes())
				{
					introFrames++;
					// Dense early logging (every 10 frames for first 200, then every 100)
					// lets us pinpoint the exact frame the boss glitches to the ceiling.
					bool shouldDiag = introFrames <= 200
						? (introFrames % 10 == 0)
						: (introFrames % 100 == 0);
					if (shouldDiag)
						LogBossDiag($"reset#{_resetCount} INTRO-SKIP f{introFrames}");
					if (introFrames > 5000)
					{
						var hb = _hitboxObserver.GetHitboxes();
						Log($"IntroSkip: TIMEOUT after {introFrames} frames — "
							+ $"enemy={hb[HitboxType.Enemy].Count} terrain={hb[HitboxType.Terrain].Count} "
							+ $"scene={UnityEngine.SceneManagement.SceneManager.GetActiveScene().name}");
						LogBossDiag($"reset#{_resetCount} INTRO-SKIP TIMEOUT");
						break;
					}
					yield return null;
				}
				LogBossDiag($"reset#{_resetCount} INTRO-SKIP DONE (after {introFrames} frames)");
				_combatStarted = true;
				// Clear any accidental reward signals from intro
				_hitsTakenInStep = 0;
				_damageLandedInStep = 0;
				// Run one normal frame skip at real speed so first obs is clean
				Time.timeScale = _timeScaleValue;
				for (int i = 0; i < _frameSkipCount; i++)
					yield return null;
			}

			Time.timeScale = 0;
			data.step_game_time = gameTimeElapsed;
			data.step_real_time = realTimeElapsed;

			// Check for episode end in eval mode
			if (_evalMode && !_episodeDone)
			{
				if (_bossDied)
				{
					_episodeDone = true;
					_episodeResult = "win";
				}
				else if (PlayerData.instance.health <= 0)
				{
					_episodeDone = true;
					_episodeResult = "loss";
				}
			}

			// Record reward signals
			data.damage_landed = _damageLandedInStep;
			data.hits_taken = _hitsTakenInStep;
			_hitsTakenInStep = 0;
			_damageLandedInStep = 0;

			if (_episodeDone)
			{
				data.done = true;
				data.info = _episodeResult;
				data.combat_hitboxes = new List<float[]>();
				data.terrain_hitboxes = new List<float[]>();
				data.global_state = new float[19];
				SendMessage(new Message { type = "step", data = data });
				yield break;
			}

			// Build observation
			var obs = _hitboxObserver.GetSplitFeatures();
			var gs = StateExtractor.GetGlobalState(obs.KnightWidth, obs.KnightHeight);

			data.combat_hitboxes = obs.CombatHitboxes;
			data.combat_kinds = obs.CombatKinds;
			data.combat_parents = obs.CombatParents;
			data.terrain_hitboxes = obs.TerrainHitboxes;
			data.global_state = gs;
			data.done = false;

			SendMessage(new Message { type = "step", data = data });
			yield break;
		}

		private void Log(string msg) => FullKnight.Instance.Log($"[TrainingEnv] {msg}");

		/// <summary>
		/// Dump every piece of state useful for diagnosing boss-reset bugs:
		/// active scene, knight position, boss GameObject/HealthManager status,
		/// FSM active state name, enemy hitbox inventory, BossSceneController
		/// state, key PlayerData flags. Called at multiple points during Reset
		/// and the intro-skip loop so log timelines tell the whole story.
		/// </summary>
		private void LogBossDiag(string tag)
		{
			var sb = new StringBuilder();
			sb.Append("[DIAG ").Append(tag).Append("]\n");
			try
			{
				var scene = SceneManager.GetActiveScene();
				sb.Append("  scene=").Append(scene.name)
				  .Append(" loaded=").Append(scene.isLoaded)
				  .Append(" time=").Append(Time.timeScale.ToString("0.00"))
				  .Append(" combatStarted=").Append(_combatStarted)
				  .Append('\n');

				var hc = HeroController.instance;
				if (hc != null)
				{
					var p = hc.transform.position;
					sb.Append("  knight pos=(").Append(p.x.ToString("0.00")).Append(',')
					  .Append(p.y.ToString("0.00")).Append(")\n");
				}
				else sb.Append("  knight=NULL\n");

				var bsc = BossSceneController.Instance;
				if (bsc != null)
				{
					sb.Append("  BossSceneController: bosses.len=")
					  .Append(bsc.bosses != null ? bsc.bosses.Length : -1)
					  .Append(" BossLevel=").Append(bsc.BossLevel)
					  .Append('\n');
					if (bsc.bosses != null)
					{
						for (int i = 0; i < bsc.bosses.Length; i++)
						{
							var b = bsc.bosses[i];
							if (b == null) { sb.Append("    [").Append(i).Append("] NULL\n"); continue; }
							var go = b.gameObject;
							sb.Append("    [").Append(i).Append("] name=").Append(go.name)
							  .Append(" active=").Append(go.activeInHierarchy)
							  .Append(" pos=(").Append(go.transform.position.x.ToString("0.00"))
							  .Append(',').Append(go.transform.position.y.ToString("0.00")).Append(")");
							var hm = go.GetComponent<HealthManager>();
							if (hm != null)
								sb.Append(" hp=").Append(hm.hp).Append(" dead=").Append(hm.isDead)
								  .Append(" invincible=").Append(hm.IsInvincible);
							var rb = go.GetComponent<Rigidbody2D>();
							if (rb != null)
								sb.Append(" vel=(").Append(rb.velocity.x.ToString("0.00"))
								  .Append(',').Append(rb.velocity.y.ToString("0.00")).Append(")");
							sb.Append('\n');
							// Dump every PlayMakerFSM on boss + its children so we can
							// see which state the sleep/wake machine is sitting in.
							var fsms = go.GetComponentsInChildren<PlayMakerFSM>(true);
							foreach (var fsm in fsms)
							{
								sb.Append("      fsm='").Append(fsm.FsmName)
								  .Append("' state='")
								  .Append(fsm.ActiveStateName ?? "<none>")
								  .Append("' on ").Append(fsm.gameObject.name).Append('\n');
							}
						}
					}
				}
				else sb.Append("  BossSceneController.Instance=NULL\n");

				// Hitbox inventory: what does the observer currently see?
				var hitboxes = _hitboxObserver.GetHitboxes();
				int enemyCount = 0, attackCount = 0, terrainCount = 0;
				if (hitboxes != null)
				{
					if (hitboxes.ContainsKey(HitboxType.Enemy)) enemyCount = hitboxes[HitboxType.Enemy].Count;
					if (hitboxes.ContainsKey(HitboxType.Attack)) attackCount = hitboxes[HitboxType.Attack].Count;
					if (hitboxes.ContainsKey(HitboxType.Terrain)) terrainCount = hitboxes[HitboxType.Terrain].Count;
				}
				sb.Append("  hitboxes: enemy=").Append(enemyCount)
				  .Append(" attack=").Append(attackCount)
				  .Append(" terrain=").Append(terrainCount).Append('\n');
				// Enumerate live enemy hitboxes with positions — this is the clearest
				// signal of "where did the boss actually go" independent of FSM guesses.
				if (hitboxes != null && hitboxes.ContainsKey(HitboxType.Enemy))
				{
					int i = 0;
					foreach (var col in hitboxes[HitboxType.Enemy])
					{
						if (col == null) continue;
						var c = col.bounds.center;
						sb.Append("    enemy[").Append(i++).Append("] ")
						  .Append(col.gameObject.name)
						  .Append(" active=").Append(col.isActiveAndEnabled)
						  .Append(" pos=(").Append(c.x.ToString("0.00"))
						  .Append(',').Append(c.y.ToString("0.00"))
						  .Append(") size=(").Append(col.bounds.size.x.ToString("0.00"))
						  .Append(',').Append(col.bounds.size.y.ToString("0.00"))
						  .Append(")\n");
						if (i >= 12) { sb.Append("    ...\n"); break; }
					}
				}

				// PlayerData flags that can influence whether a boss intro plays
				// (if HK reads them for sleep/wake FSMs in the HoG variant).
				var pd = PlayerData.instance;
				if (pd != null)
				{
					sb.Append("  pd:");
					foreach (var key in new[] { "killedBigFly", "killedGruzMother", "newGruzMother" })
					{
						try { sb.Append(' ').Append(key).Append('=').Append(pd.GetBool(key)); }
						catch { sb.Append(' ').Append(key).Append("=?"); }
					}
					sb.Append('\n');
				}
			}
			catch (System.Exception e)
			{
				sb.Append("  EXCEPTION: ").Append(e.GetType().Name).Append(": ").Append(e.Message).Append('\n');
			}
			FullKnight.Instance.Log(sb.ToString());
		}

		protected override IEnumerator Setup()
		{
			Connect();
			yield return new Socket.WaitForMessage(socket);
			Message message = socket.UnreadMessages.Dequeue();
			if (message.type != "init")
			{
				Log($"Setup: expected init, got '{message.type}' — retrying");
				yield return Setup();
				yield break;
			}

			On.GameManager.SaveGame += SaveFileProxy.DisableSaveGame;
			SaveFileProxy.LoadCompletedSave();
			GameManager.instance.ContinueGame();
			yield return new SceneHooks.WaitForSceneLoad("GG_Workshop");
			yield return new WaitForFinishedEnteringScene();
			yield return new WaitForSeconds(2f);

			_hitboxObserver.Load();
			InputManager.AttachDevice(_inputShim);
			SendMessage(message);
		}

		protected override IEnumerator Dispose()
		{
			UnhookDamage();
			_timeManager?.Dispose();
			InputManager.DetachDevice(_inputShim);
			_hitboxObserver.Unload();
			CloseSocket();
			yield break;
		}

		private void HookDamage()
		{
			ModHooks.AfterTakeDamageHook += OnKnightDamaged;
			On.HealthManager.TakeDamage += OnBossDamaged;
		}

		private void UnhookDamage()
		{
			ModHooks.AfterTakeDamageHook -= OnKnightDamaged;
			On.HealthManager.TakeDamage -= OnBossDamaged;
		}

		private int OnKnightDamaged(int damageType, int damage)
		{
			_hitsTakenInStep++;
			// Eval: let real damage through. Training: minimal damage (HP restored next step)
			return _evalMode ? damage : 1;
		}

		private void OnBossDamaged(On.HealthManager.orig_TakeDamage orig, HealthManager self, HitInstance hitInstance)
		{
			// Only track the one designated boss. Minions, summons, and ambient
			// HealthManagers in the arena pass through unmodified.
			if (self != _bossHM) { orig(self, hitInstance); return; }

			_damageLandedInStep += hitInstance.DamageDealt / (float)_bossMaxHP * 100f;

			if (!_evalMode)
			{
				// Training: prevent boss death, restore HP for infinite fighting
				if (self.hp - hitInstance.DamageDealt <= 0)
					self.hp = _bossMaxHP;
				orig(self, hitInstance);
				self.hp = _bossMaxHP;
			}
			else
			{
				// Eval: let damage through, detect boss death
				bool wouldDie = _bossHM != null && self == _bossHM
					&& self.hp - hitInstance.DamageDealt <= 0;
				orig(self, hitInstance);
				if (wouldDie)
					_bossDied = true;
			}
		}

		private bool HasActiveCombatHitboxes()
		{
			var hitboxes = _hitboxObserver.GetHitboxes();
			foreach (var col in hitboxes[HitboxType.Enemy])
			{
				if (col != null && col.isActiveAndEnabled)
					return true;
			}
			return false;
		}

		private void InitBossRefs()
		{
			_bossHM = null;
			_bossMaxHP = 1;
			try
			{
				if (BossSceneController.Instance?.bosses != null
					&& BossSceneController.Instance.bosses.Length > 0)
				{
					_bossHM = BossSceneController.Instance.bosses[0]
						.gameObject.GetComponent<HealthManager>();
					if (_bossHM != null) _bossMaxHP = _bossHM.hp;
				}
			}
			catch { }
		}
	}
}
