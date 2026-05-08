using System.Collections;
using System.Collections.Generic;
using System.Text;
using FullKnight.Net;
using FullKnight.Game;
using HutongGames.PlayMaker;
using HutongGames.PlayMaker.Actions;
using InControl;
using Modding;
using UnityEngine;

namespace FullKnight.Environment
{
	public class TrainingEnv : WebsocketEnv
	{
		private string _level;
		private int _frameSkipCount;
		private int _timeScaleValue;
		private int _hitsTakenInStep;
		private float _damageLandedInStep;
		private float _hpHealedInStep;
		private int _knightHpAtStepStart;
		private int _knightMaxHP;

		// Eval mode: real damage, real death, episode ends on kill
		private bool _evalMode;
		private bool _bossDied;
		private bool _episodeDone;
		private string _episodeResult;
		// Set during synthetic suicide (Reset path) so OnKnightDamaged skips
		// the hit-taken counter — otherwise the forced kill would inject a
		// phantom -1 hit penalty into the next step's reward.
		private bool _syntheticKill;
		// Set of HealthManagers considered the "true target(s)" for is_target and
		// reward. Populated from BossSceneController.bosses on each reset; used by
		// HitboxObserver.GetSplitFeatures to flag which combat hitboxes the agent
		// should actually try to kill (vs. minions / projectiles), and by the
		// damage hook to credit reward and apply training-mode immortality.
		private readonly HashSet<HealthManager> _bossHMs = new();
		// Max HP at reset for each boss in _bossHMs. Stored per-hm because multi-
		// boss fights (Oro/Mato, God Tamer) have asymmetric HP pools and damage
		// must be normalized against each boss's own maxHP.
		private readonly Dictionary<HealthManager, int> _bossMaxHPs = new();

		// Boss intro skip: keep simulating internally until combat starts
		private bool _combatStarted;

		// Diagnostics: count resets so logs are correlatable across episodes
		private int _resetCount;
		// Diagnostics: count steps within an episode for slow-step correlation
		private int _stepCount;
		// Phase-timing scratch state. Stopwatch-style (Time.realtimeSinceStartup is
		// wall time, unaffected by Time.timeScale) so we can attribute wall-time
		// stalls to a specific sub-phase of Reset/Step. _phaseStart marks the
		// beginning of the current operation; _phaseLast marks the most recent
		// LogPhase call so each line shows both delta and total.
		private float _phaseStart;
		private float _phaseLast;

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
			PhaseBegin();
			_level = data.level ?? _level;
			_frameSkipCount = data.frames_per_wait ?? _frameSkipCount;
			_timeScaleValue = data.time_scale ?? _timeScaleValue;
			_evalMode = data.eval ?? false;
			_hitsTakenInStep = 0;
			_damageLandedInStep = 0;
			_hpHealedInStep = 0;
			_stepCount = 0;
			// Capture episode-end flags before clearing — the suicide-vs-wait
			// branch below depends on whether this reset follows a natural end
			// (boss death = win, knight death = loss) or a mid-fight cut. Note:
			// reading PlayerData.instance.health here is unreliable, since HK's
			// death sequence may have already restored HP between the death step
			// and the time Reset() arrives.
			bool naturalEnd = _bossDied || _episodeResult == "loss";
			_bossDied = false;
			_episodeDone = false;
			_episodeResult = null;
			_combatStarted = false;
			_resetCount++;

			LogBossDiag($"reset#{_resetCount} PRE-UNLOAD (still in old scene)");
			LogPhase("Reset", "PRE-UNLOAD");

			// Release any inputs held over from the previous episode before the
			// scene transition unfreezes time — otherwise a stuck "left" or "jump"
			// runs the knight for the entire transition + intro-skip window.
			// Also clear any hard-commit lock left over from a mid-charge death.
			_inputShim.ResetCommit();
			ActionDecoder.ApplyAction(_inputShim, new int[] { 2, 2, 7, 1 },
				_frameSkipCount, _timeScaleValue);

			// Unpause so scene transition and WaitForSeconds can proceed
			Time.timeScale = _timeScaleValue;

			// Three paths out of the boss arena:
			//  (a) Already out — death/win cleanup landed before reset arrived.
			//      Nothing to wait for.
			//  (b) Natural end (win or loss) — HK's own DoDreamReturn / death
			//      transition is queued and time just resumed, so it's about to
			//      run. We don't care WHERE it sends the knight (boss-complete
			//      can land in a non-Workshop scene like a pantheon completion
			//      room); we just need them out of the arena before LoadBossScene
			//      fires. Wait for the scene to change away from preScene; the
			//      LoadBossScene bounce handles getting back to GG_Workshop from
			//      wherever HK dropped us. Suiciding here would race the queued
			//      natural transition and corrupt state — that was the original
			//      "boss killed but never re-entered arena" bug.
			//  (c) Mid-fight (step-budget reset on a live knight + live boss) —
			//      yanking that state into a new scene leaves FSMs / HUD
			//      inconsistent. Force a synthetic suicide so HK's natural death
			//      cleanup brings us home cleanly.
			var preScene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
			if (preScene == "GG_Workshop")
			{
				// Cleanup already landed; nothing to wait for.
				LogBossDiag($"reset#{_resetCount} ALREADY-IN-WORKSHOP");
				LogPhase("Reset", "ALREADY-IN-WORKSHOP");
			}
			else if (naturalEnd)
			{
				LogBossDiag($"reset#{_resetCount} NATURAL-END WAIT (in {preScene})");
				yield return WaitForSceneChange(preScene);
				var afterScene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
				LogBossDiag($"reset#{_resetCount} NATURAL-END DONE (now in {afterScene})");
				LogPhase("Reset", $"NATURAL-END (pre={preScene} post={afterScene})");
			}
			else
			{
				LogBossDiag($"reset#{_resetCount} PRE-SUICIDE (alive in {preScene})");
				yield return KillKnight();
				LogBossDiag($"reset#{_resetCount} POST-SUICIDE");
				LogPhase("Reset", $"SUICIDE (pre={preScene})");
			}

			// Let any in-flight transition (the one that just left the arena, or
			// any post-load scene wiring) settle before LoadBossScene kicks off
			// our own. LoadBossScene then bounces through GG_Workshop if needed.
			yield return new WaitForFinishedEnteringScene();
			LogPhase("Reset", "WaitForFinishedEnteringScene (settle)");

			yield return SceneHooks.LoadBossScene(_level);
			LogPhase("Reset", "LoadBossScene");

			LogBossDiag($"reset#{_resetCount} POST-SCENELOAD (before reader recreate)");

			// Force-recreate the hitbox reader for the boss scene. The activeSceneChanged
			// event is unreliable under multi-instance load (some instances miss it), so
			// we explicitly rebuild the reader here and yield a frame for Start() to scan.
			_hitboxObserver.RecreateReader();
			yield return null;
			LogPhase("Reset", "RecreateReader+frame");

			InitBossRefs();
			LogBossDiag($"reset#{_resetCount} POST-INITBOSSREFS");
			LogPhase("Reset", "InitBossRefs");
			// One-line pass/fail signal for the same-scene-reload bug. Grep for
			// "[BounceCheck]" to audit every reset at a glance.
			bool bossAwake = HasActiveCombatHitboxes();
			Log($"[BounceCheck] reset#{_resetCount} level={_level} bossAwake={bossAwake}");
			_knightMaxHP = PlayerData.instance.maxHealth;

			UnhookDamage();
			HookDamage();

			if (_timeManager != null) _timeManager.Dispose();
			_timeManager = new Game.TimeScale(_timeScaleValue);

			var obs = _hitboxObserver.GetSplitFeatures(_bossHMs, emitTerrainDebug: _evalMode);
			var gs = StateExtractor.GetGlobalState(obs.KnightWidth, obs.KnightHeight);

			data.combat_hitboxes = obs.CombatHitboxes;
			data.combat_kinds = obs.CombatKinds;
			data.combat_parents = obs.CombatParents;
			data.terrain_hitboxes = obs.TerrainHitboxes;
			data.terrain_debug = obs.TerrainDebug;
			data.global_state = gs;

			Time.timeScale = 0;
			LogPhase("Reset", "obs+freeze (final)");
			float resetTotalMs = (Time.realtimeSinceStartup - _phaseStart) * 1000f;
			Log($"[Reset-Timing] reset#{_resetCount} TOTAL {resetTotalMs:F0}ms level={_level}");
			SendMessage(new Message { type = "reset", data = data });
			yield break;
		}

		private IEnumerator Step(MessageData data)
		{
			PhaseBegin();
			_stepCount++;
			// If episode already ended, keep returning done
			if (_episodeDone)
			{
				data.done = true;
				data.info = _episodeResult;
				data.combat_hitboxes = new List<float[]>();
				data.terrain_hitboxes = new List<float[]>();
				data.global_state = new float[22];
				data.damage_landed = 0;
				data.hits_taken = 0;
				data.hp_healed = 0;
				data.step_game_time = 0;
				data.step_real_time = 0;
				SendMessage(new Message { type = "step", data = data });
				yield break;
			}

			Time.timeScale = _timeScaleValue;

			bool committedThisStep = ActionDecoder.ApplyAction(
				_inputShim, data.action_vec, _frameSkipCount, _timeScaleValue);
			data.action_committed = committedThisStep;

			// Track HP at step start for heal detection
			_knightHpAtStepStart = PlayerData.instance.health;

			// Force per-frame dt to a fixed value during the agent's frame-skip
			// loop. captureDeltaTime overrides Time.deltaTime regardless of
			// real wallclock between frames or Time.timeScale, so any future
			// per-frame Update() speedup translates to wallclock throughput
			// rather than changing dt — preserving agent dynamics by
			// construction. Disabled outside this loop because (a) Unity
			// ignores timeScale=0 under capture (would break the inter-step
			// pause) and (b) the intro-skip fast-forward (timeScale=20) needs
			// real-time. Value is calibrated against the pre-uncap baseline:
			// baseline rtime_mean=12.9ms − ~0.5ms I/O = ~12.4ms for 5 frames =
			// ~2.5ms/frame at scaled fps × timeScale=3 → dt ≈ 0.0075s/frame.
			Time.captureDeltaTime = 0.0075f;

			float frameSkipT0 = Time.realtimeSinceStartup;
			float gameTimeElapsed = 0f;
			float realTimeElapsed = 0f;
			int frameSkipFrames = 0;
			for (int i = 0; i < _frameSkipCount; i++)
			{
				yield return null;
				frameSkipFrames++;
				gameTimeElapsed += Time.deltaTime;
				realTimeElapsed += Time.unscaledDeltaTime;
				// Break early on death (both modes now have real HP)
				if (_bossDied || PlayerData.instance.health <= 0)
					break;
			}
			Time.captureDeltaTime = 0;  // restore real-time so timeScale=0 pauses
			float frameSkipMs = (Time.realtimeSinceStartup - frameSkipT0) * 1000f;

			// If boss intro is still playing, fast-forward until combat starts
			float introT0 = Time.realtimeSinceStartup;
			int introFrames = 0;
			bool introSkipRan = !_combatStarted;
			bool introSkipTimedOut = false;
			float introSettleMs = 0f;
			if (!_combatStarted)
			{
				LogBossDiag($"reset#{_resetCount} INTRO-SKIP START");
				Time.timeScale = 20f;
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
						introSkipTimedOut = true;
						break;
					}
					yield return null;
				}
				LogBossDiag($"reset#{_resetCount} INTRO-SKIP DONE (after {introFrames} frames)");
				_combatStarted = true;
				// Clear any accidental reward signals from intro
				_hitsTakenInStep = 0;
				_damageLandedInStep = 0;
				// Run one normal frame skip at real speed so first obs is clean.
				// Capture mode here too so the settle window matches a normal
				// agent step's per-frame game-time.
				float settleT0 = Time.realtimeSinceStartup;
				Time.timeScale = _timeScaleValue;
				Time.captureDeltaTime = 0.0075f;
				for (int i = 0; i < _frameSkipCount; i++)
					yield return null;
				Time.captureDeltaTime = 0;
				introSettleMs = (Time.realtimeSinceStartup - settleT0) * 1000f;
			}
			float introTotalMs = (Time.realtimeSinceStartup - introT0) * 1000f;

			Time.timeScale = 0;
			data.step_game_time = gameTimeElapsed;
			data.step_real_time = realTimeElapsed;

			// Check for episode end (both modes now have real HP and death)
			if (!_episodeDone)
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

			// Compute HP healed this step (positive delta = healing occurred)
			int hpNow = PlayerData.instance.health;
			int hpDelta = hpNow - _knightHpAtStepStart;
			_hpHealedInStep = hpDelta > 0 ? (float)hpDelta : 0f;

			// Record reward signals
			data.damage_landed = _damageLandedInStep;
			data.hits_taken = _hitsTakenInStep;
			data.hp_healed = _hpHealedInStep;
			_hitsTakenInStep = 0;
			_damageLandedInStep = 0;
			_hpHealedInStep = 0;

			// Long-run leak probes. Cheap: cache sizes are integer field reads,
			// GC.GetTotalMemory(false) is non-blocking (no collection). Populated
			// even on episode-done steps so epoch averaging stays unbiased.
			var sizes = _hitboxObserver.GetCacheSizes();
			data.diag_enemy_count = (ushort)System.Math.Min(sizes.EnemyCount, ushort.MaxValue);
			data.diag_attack_count = (ushort)System.Math.Min(sizes.AttackCount, ushort.MaxValue);
			data.diag_terrain_count = (ushort)System.Math.Min(sizes.TerrainCount, ushort.MaxValue);
			data.diag_kind_cache_size = sizes.KindCacheCount;
			data.diag_gc_heap_mb = System.GC.GetTotalMemory(false) / (1024f * 1024f);

			// Slow-step diagnostic. Wall time is dominated by the frame-skip and
			// intro-skip loops; obs build / hooks / GC probes are <1ms. Always
			// log the first step after a reset (intro-skip lives there) and any
			// step over 1s wall. Emitted before the done-branch so dying-during-
			// intro-skip steps get attributed too.
			float stepWallMs = (Time.realtimeSinceStartup - _phaseStart) * 1000f;
			if (stepWallMs > 1000f || introSkipRan)
			{
				Log($"[Step-Timing] reset#{_resetCount} step#{_stepCount} "
					+ $"total={stepWallMs:F0}ms frameSkip={frameSkipMs:F0}ms"
					+ $"({frameSkipFrames}f) "
					+ $"introSkip={(introSkipRan ? introTotalMs : 0):F0}ms"
					+ $"({introFrames}f, settle={introSettleMs:F0}ms"
					+ (introSkipTimedOut ? ", TIMEOUT" : "")
					+ $") gameTime={gameTimeElapsed * 1000:F0}ms "
					+ $"realTime={realTimeElapsed * 1000:F0}ms "
					+ $"timeScale={_timeScaleValue} done={_episodeDone}");
			}

			if (_episodeDone)
			{
				data.done = true;
				data.info = _episodeResult;
				data.combat_hitboxes = new List<float[]>();
				data.terrain_hitboxes = new List<float[]>();
				data.global_state = new float[22];
				SendMessage(new Message { type = "step", data = data });
				yield break;
			}

			// Build observation
			var obs = _hitboxObserver.GetSplitFeatures(_bossHMs, emitTerrainDebug: _evalMode);
			var gs = StateExtractor.GetGlobalState(obs.KnightWidth, obs.KnightHeight);

			data.combat_hitboxes = obs.CombatHitboxes;
			data.combat_kinds = obs.CombatKinds;
			data.combat_parents = obs.CombatParents;
			data.terrain_hitboxes = obs.TerrainHitboxes;
			data.terrain_debug = obs.TerrainDebug;
			data.global_state = gs;
			data.done = false;

			SendMessage(new Message { type = "step", data = data });
			yield break;
		}

		private void Log(string msg) => FullKnight.Instance.Log($"[TrainingEnv] {msg}");

		// Phase timing helpers. PhaseBegin() resets the stopwatch at the start of
		// Reset() / Step(); LogPhase() emits a "[Phase-Timing]" line with the
		// delta since the last call and the cumulative total. Wall-clock based
		// (Time.realtimeSinceStartup) so Time.timeScale gymnastics during reset
		// don't confuse the readings.
		private void PhaseBegin()
		{
			_phaseStart = Time.realtimeSinceStartup;
			_phaseLast = _phaseStart;
		}

		private void LogPhase(string scope, string label)
		{
			float now = Time.realtimeSinceStartup;
			float deltaMs = (now - _phaseLast) * 1000f;
			float totalMs = (now - _phaseStart) * 1000f;
			_phaseLast = now;
			Log($"[Phase-Timing] {scope}#{_resetCount} {label}: +{deltaMs:F0}ms (total {totalMs:F0}ms)");
		}

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
				var scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene();
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

			// Uncap Unity's frame loop. With -nographics there's no display to
			// vsync against; the only thing throttling Update() is targetFrameRate
			// (default 60 on Windows). cpu_machine_sat=28% on the merged-baseline
			// run says we're not CPU-bound — we're frame-rate-bound. Setting
			// vSyncCount=0 + targetFrameRate=-1 lets Unity tick the main loop as
			// fast as the CPU can carry it, which directly raises env-steps/sec.
			QualitySettings.vSyncCount = 0;
			Application.targetFrameRate = -1;

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

		// AfterTakeDamageHook fires past HeroController.TakeDamage's iframe
		// short-circuit, so it only ticks on hits that actually land — using
		// TakeDamageHook here counted iframe-blocked contacts and inflated
		// hits_taken ~16×, wrecking the reward signal. Return value replaces
		// the applied damage; pass `damage` through unchanged.
		private int OnKnightDamaged(int damageType, int damage)
		{
			if (!_syntheticKill) _hitsTakenInStep++;
			return damage;
		}

		private void OnBossDamaged(On.HealthManager.orig_TakeDamage orig, HealthManager self, HitInstance hitInstance)
		{
			// Only track HealthManagers in the designated boss set. Minions,
			// summons, and ambient HealthManagers pass through unmodified.
			if (!_bossMaxHPs.TryGetValue(self, out int maxHP))
			{
				orig(self, hitInstance);
				return;
			}

			// Equal-weight-per-boss normalization: each boss contributes 100/N
			// percent when fully killed, independent of its HP pool. Collapses
			// to the single-boss formula when N=1. Asymmetric fights (Oro/Mato,
			// God Tamer) get correct 50/50 weighting instead of HP-proportional.
			int n = _bossHMs.Count;
			_damageLandedInStep += hitInstance.DamageDealt / (float)(n * maxHP) * 100f;

			// Real damage in both modes — episode ends when all bosses are dead
			bool wouldDie = self.hp - hitInstance.DamageDealt <= 0;
			orig(self, hitInstance);
			if (wouldDie)
			{
				bool allDead = true;
				foreach (var hm in _bossHMs)
				{
					if (hm == null) continue;
					if (hm == self) continue;
					if (hm.hp > 0) { allDead = false; break; }
				}
				if (allDead) _bossDied = true;
			}
		}

		// Force-kill the knight via HK's natural damage path. Step-budget resets
		// fire mid-fight on a live knight + live boss; yanking that into a new
		// scene leaves FSM coroutines / HUD bindings in inconsistent states
		// (missing healthbar, stuck-on-ceiling boss). Riding the death sequence
		// gives HK its blessed cleanup: death anim → DreamReturn → GG_Workshop.
		// The _syntheticKill guard suppresses hit-counter inflation in
		// OnKnightDamaged for both this fatal hit and any incidental damage
		// during the death animation.
		private IEnumerator KillKnight()
		{
			_syntheticKill = true;
			HeroController.instance.TakeDamage(
				HeroController.instance.gameObject,
				GlobalEnums.CollisionSide.other,
				9999, 0);
			int timeoutFrames = 2000;
			while (UnityEngine.SceneManagement.SceneManager.GetActiveScene().name != "GG_Workshop"
				&& --timeoutFrames > 0)
			{
				yield return null;
			}
			if (timeoutFrames <= 0)
			{
				Log($"KillKnight: TIMEOUT waiting for GG_Workshop "
					+ $"(still in {UnityEngine.SceneManagement.SceneManager.GetActiveScene().name})");
			}
			yield return new WaitForFinishedEnteringScene();
			_syntheticKill = false;
		}

		// Wait for HK's natural end-of-episode transition (boss-complete
		// DoDreamReturn on a win, hero-death dream return on a loss) to leave
		// the current arena. Destination-agnostic: boss-complete can land us in
		// a non-Workshop scene (pantheon completion rooms etc.), and we don't
		// need to know which — LoadBossScene will bounce through Workshop from
		// wherever HK puts us. We just need to be out of `fromScene` so our
		// scene load doesn't race the queued natural transition.
		private IEnumerator WaitForSceneChange(string fromScene)
		{
			int timeoutFrames = 5000;
			while (UnityEngine.SceneManagement.SceneManager.GetActiveScene().name == fromScene
				&& --timeoutFrames > 0)
			{
				yield return null;
			}
			if (timeoutFrames <= 0)
			{
				Log($"WaitForSceneChange: TIMEOUT (still in {fromScene})");
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
			_bossHMs.Clear();
			_bossMaxHPs.Clear();
			try
			{
				if (BossSceneController.Instance?.bosses != null
					&& BossSceneController.Instance.bosses.Length > 0)
				{
					foreach (var b in BossSceneController.Instance.bosses)
					{
						if (b == null) continue;
						var hm = b.gameObject.GetComponent<HealthManager>();
						if (hm == null) continue;
						_bossHMs.Add(hm);
						_bossMaxHPs[hm] = hm.hp;
					}
				}
			}
			catch { }
		}
	}
}
