using System.Collections;
using System.Collections.Generic;
using FullKnight.Net;
using FullKnight.Game;
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
		private int _phaseLastFrame;
		// Per-reset phase deltas, populated by LogPhase. Cleared at PhaseBegin.
		// Shipped to Python via a fixed-shape trailer on the reset response so
		// the train-time diagnostic can break down the 8s reset average.
		private readonly List<float> _resetPhaseDeltasMs = new();
		// Parallel frame-count deltas: how many Unity frames each phase took.
		// Combined with the ms deltas this disambiguates "many normal frames"
		// from "few stalled frames" — e.g. 4000 ms / 4 frames is a render
		// stall; 4000 ms / 800 frames is just a slow loop at 200fps.
		private readonly List<int> _resetPhaseFrames = new();
		// Which exit-from-arena branch ran in the current Reset(). Mirrors the
		// reset_branch wire field. 0 = already in GG_Workshop, 1 = waited for
		// natural end transition.
		private byte _resetBranch;

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
			_bossDied = false;
			_episodeDone = false;
			_episodeResult = null;
			_resetCount++;

			LogPhase("Reset", "PRE-UNLOAD");

			// Release any inputs held over from the previous episode before the
			// scene transition unfreezes time — otherwise a stuck "left" or "jump"
			// runs the knight for the entire transition + intro-skip window.
			// Also clear any hard-commit lock left over from a mid-charge death.
			_inputShim.ResetCommit();
			ActionDecoder.ApplyAction(_inputShim, new int[] { 2, 2, 7, 1 },
				_frameSkipCount, _timeScaleValue);

			// FIRST-PRINCIPLES SIMPLIFICATION: removed the timeScale=20 crank
			// during reset (was lines 142-164 + 230-237 — dual SetMultiplier
			// to fast-forward HK transition coroutines and back). Testing
			// whether the crank is actually load-bearing or whether resets
			// work fine at the normal _timeScaleValue. Also removed the
			// suicide branch (forced mid-fight resets are disabled upstream)
			// and consolidated the two-branch arena-exit logic into a single
			// "wait for scene change away from boss" check.
			//
			// Always SetMultiplier — Step() ends with Time.timeScale = 0 to
			// freeze for the Python obs handoff, and we need to unfreeze
			// here so HK's queued DoDreamReturn / hero-death transition
			// can actually tick during WaitForSceneChange.
			if (_timeManager == null)
				_timeManager = new Game.TimeScale(_timeScaleValue);
			else
				_timeManager.SetMultiplier(_timeScaleValue);

			var preScene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
			if (preScene != "GG_Workshop")
			{
				// HK's natural DoDreamReturn (win) / hero-death (loss) transition
				// is queued from the previous step and just needs time to leave
				// the boss arena. Loading on top of it would race and corrupt
				// state — wait for the scene to actually change first.
				_resetBranch = 1;
				yield return WaitForSceneChange(preScene);
				var afterScene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
				LogPhase("Reset", $"NATURAL-END (pre={preScene} post={afterScene})");
			}
			else
			{
				_resetBranch = 0;
				LogPhase("Reset", "ALREADY-IN-WORKSHOP");
			}

			// FIRST-PRINCIPLES: removed the WaitForFinishedEnteringScene "settle"
			// that used to live here — LoadBossScene's BounceThroughWorkshop
			// already does its own enter-scene wait, so this was redundant.
			// Empty LogPhase preserves the 7-slot wire alignment that Python
			// expects (binary_protocol.py:RESET_PHASE_NAMES); without it the
			// debug breakdown labels every phase one slot to the left.
			LogPhase("Reset", "settle (skipped)");

			yield return SceneHooks.LoadBossScene(_level);
			LogPhase("Reset", "LoadBossScene");

			// Force-recreate the hitbox reader for the boss scene. The activeSceneChanged
			// event is unreliable under multi-instance load (some instances miss it), so
			// we explicitly rebuild the reader here and yield a frame for Start() to scan.
			_hitboxObserver.RecreateReader();
			yield return null;
			LogPhase("Reset", "RecreateReader+frame");

			InitBossRefs();
			LogPhase("Reset", "InitBossRefs");
			// One-line pass/fail signal for the same-scene-reload bug. Grep for
			// "[BounceCheck]" to audit every reset at a glance.
			bool bossAwake = HasActiveCombatHitboxes();
			Log($"[BounceCheck] reset#{_resetCount} level={_level} bossAwake={bossAwake}");
			_knightMaxHP = PlayerData.instance.maxHealth;

			UnhookDamage();
			HookDamage();

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
			data.reset_phase_deltas_ms = _resetPhaseDeltasMs.ToArray();
			data.reset_phase_frames = _resetPhaseFrames.ToArray();
			data.reset_branch = _resetBranch;
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

			// Pin Time.deltaTime to a constant during the agent's frame-skip so
			// per-step game-time is decoupled from Unity's wallclock framerate.
			// Without this, faster wallclock fps = smaller deltaTime = the agent
			// sees a different gtime regime than it was trained on, even when
			// nothing about the simulation changed.
			//
			// EMPIRICAL: under capture, Time.deltaTime = captureDeltaTime ×
			// Time.timeScale (NOT just captureDeltaTime as Unity docs imply).
			// First attempt at 0.00848 produced gtime=0.126/step (3× baseline)
			// because timeScale=3 multiplied through; quality cratered.
			// Calibration: gtime_baseline / (frames × timeScale) = 0.0424 /
			// (5 × 3) ≈ 0.00283, giving Time.deltaTime ≈ 0.00848/frame and
			// gtime ≈ 0.0424/step — matches the trained regime.
			//
			// Restore captureDeltaTime=0 between steps because Unity ignores
			// Time.timeScale=0 under capture (would break the inter-step pause
			// and the intro-skip timeScale=20 fast-forward).
			//
			// Computed (not const) so that lowering _frameSkipCount automatically
			// raises captureDeltaTime to preserve gtime. Baseline regime is
			// gtime=0.0424s/step; we keep that invariant and trade Unity frames
			// for per-frame game-time. Going from frames=5 to frames=2 means
			// 60% fewer Unity frames per agent step → ~60% less Unity sim cost,
			// while the agent's gtime/step stays exactly 0.0424.
			const float kBaselineGtime = 0.0424f;
			float kStepDeltaTime = kBaselineGtime / (_frameSkipCount * _timeScaleValue);
			Time.captureDeltaTime = kStepDeltaTime;

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
			Time.captureDeltaTime = 0;  // restore real-time so timeScale gymnastics work
			float frameSkipMs = (Time.realtimeSinceStartup - frameSkipT0) * 1000f;

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

			// Slow-step diagnostic. Wall time is dominated by the frame-skip
			// loop; obs build / hooks / GC probes are <1ms.
			float stepWallMs = (Time.realtimeSinceStartup - _phaseStart) * 1000f;
			if (stepWallMs > 1000f)
			{
				Log($"[Step-Timing] reset#{_resetCount} step#{_stepCount} "
					+ $"total={stepWallMs:F0}ms frameSkip={frameSkipMs:F0}ms"
					+ $"({frameSkipFrames}f) "
					+ $"gameTime={gameTimeElapsed * 1000:F0}ms "
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
			_phaseLastFrame = Time.frameCount;
			_resetPhaseDeltasMs.Clear();
			_resetPhaseFrames.Clear();
		}

		private void LogPhase(string scope, string label)
		{
			float now = Time.realtimeSinceStartup;
			int frameNow = Time.frameCount;
			float deltaMs = (now - _phaseLast) * 1000f;
			float totalMs = (now - _phaseStart) * 1000f;
			int deltaFrames = frameNow - _phaseLastFrame;
			_phaseLast = now;
			_phaseLastFrame = frameNow;
			_resetPhaseDeltasMs.Add(deltaMs);
			_resetPhaseFrames.Add(deltaFrames);
			float msPerFrame = deltaFrames > 0 ? deltaMs / deltaFrames : 0f;
			Log($"[Phase-Timing] {scope}#{_resetCount} {label}: +{deltaMs:F0}ms / {deltaFrames}f ({msPerFrame:F1}ms/f) (total {totalMs:F0}ms)");
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
			// (default cap on Windows). Pairing this with captureDeltaTime in
			// Step() decouples wallclock framerate from per-step game-time, so
			// faster frames don't shrink dt out from under the agent. Uncap-alone
			// (commit e094f23, reverted) gave +63% throughput but cratered
			// quality via the regime shift; capture-alone (a2f7136) preserves
			// regime but Unity stays at ~360fps cap and there's no speedup.
			// Combined, the uncap delivers the wallclock win and capture holds
			// the regime steady.
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
			_hitsTakenInStep++;
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

		// Wait for HK's natural end-of-episode transition (boss-complete
		// DoDreamReturn on a win, hero-death dream return on a loss) to leave
		// the current arena AND fully finish its Finish callback chain.
		//
		// Both gates matter. Scene-name flip alone fires during sceneLoad's
		// ActivationComplete, but `gm.sceneLoad` only nulls in the later Finish
		// callback. Issuing BeginSceneTransition in that window is rejected
		// with "Cannot scene transition while a scene transition is in
		// progress" (logged by GameManager) and HK ends up stuck in
		// ENTERING_LEVEL forever — the boss scene never loads. Polling
		// IsInSceneTransition (cleared inside Finish, after sceneLoad = null)
		// closes that race.
		private IEnumerator WaitForSceneChange(string fromScene)
		{
			int timeoutFrames = 5000;
			int waited = 0;
			float t0 = Time.realtimeSinceStartup;
			var gm = GameManager.instance;
			while (--timeoutFrames > 0)
			{
				bool sceneFlipped = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name != fromScene;
				bool transitionDone = gm == null || !gm.IsInSceneTransition;
				if (sceneFlipped && transitionDone)
					break;
				waited++;
				// Periodic heartbeat with HK-side state. Lets us tell, when
				// a reset is hanging, whether HK's death/win transition is
				// actually progressing or whether we're frozen on something.
				// Sampled every 60 frames so the log isn't drowned but the
				// signal arrives quickly enough to catch a stuck timescale.
				if (waited % 60 == 0)
				{
					var pd = PlayerData.instance;
					string gs = gm != null ? gm.GetSceneNameString() : "?";
					string state = "?";
					try { state = gm != null ? gm.gameState.ToString() : "?"; }
					catch { }
					int hp = pd != null ? pd.health : -1;
					Log($"[WaitForSceneChange] reset#{_resetCount} f{waited} "
						+ $"still={fromScene} gm.scene={gs} state={state} "
						+ $"inTransition={(gm != null && gm.IsInSceneTransition)} "
						+ $"timeScale={Time.timeScale:F2} hp={hp} "
						+ $"elapsed={(Time.realtimeSinceStartup - t0) * 1000f:F0}ms");
				}
				yield return null;
			}
			if (timeoutFrames <= 0)
			{
				Log($"WaitForSceneChange: TIMEOUT after {waited} frames "
					+ $"(still in {fromScene}, "
					+ $"elapsed={(Time.realtimeSinceStartup - t0) * 1000f:F0}ms)");
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
				var bsc = BossSceneController.Instance;
				if (bsc?.bosses != null && bsc.bosses.Length > 0)
				{
					foreach (var b in bsc.bosses)
					{
						if (b == null) continue;
						var hm = b.gameObject.GetComponent<HealthManager>();
						if (hm == null) continue;
						_bossHMs.Add(hm);
						_bossMaxHPs[hm] = hm.hp;
					}
				}
				// Strip the 5-second `bossesDeadWaitTime` HK adds after a boss
				// dies. EndSceneDelayed yields `WaitForSeconds(bossesDeadWaitTime)`
				// before any transition logic runs (BossSceneController.cs:234 in
				// the decomp); zeroing it is the single largest win-side wallclock
				// reduction available without breaking the GG transition FSM
				// chain. We DO leave doTransitionOut=true (default): turning it
				// off skips the `GG TRANSITION OUT` event broadcast that HK's
				// transition FSM listens for, and the fallback path is strictly
				// slower (+1s wallclock measured).
				if (bsc != null)
					bsc.bossesDeadWaitTime = 0f;
			}
			catch { }
		}
	}
}
