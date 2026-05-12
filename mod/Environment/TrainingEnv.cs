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
		private int _hitsTakenInStep;
		private float _damageLandedInStep;
		private float _hpHealedInStep;
		private int _knightHpAtStepStart;

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
		// OnDeath subscribers per HM, stored so we can unsubscribe on the next
		// reset (or Dispose). False Knight and similar multi-phase bosses
		// restore hp on stagger via some path that bypasses HM.OnDeath, so
		// the event only fires on the truly-final death — same signal HK's
		// BossSceneController uses to end the scene. Subscribing here means
		// our _bossDied can't trip before HK itself decides the boss is gone.
		private readonly Dictionary<HealthManager, HealthManager.DeathEvent> _bossDeathHandlers = new();

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

		// Silent-HP-damage probe state. _inHeroTakeDamage is set true while
		// HeroController.TakeDamage is executing (its hooks fire AfterTakeDamage
		// and OnKnightDamaged for us). Any PlayerData.TakeHealth call that
		// decrements health while this flag is false bypassed the standard
		// damage path — i.e. an FSM action / charm / direct write performed
		// the hit. Logged with a stack trace so we can attribute the source.
		private bool _inHeroTakeDamage;
		private int _silentHpDeltaInStep;
		private int _silentHpEventsInStep;

		// Recoil-duration probe. When _debugRecoil is set (via FK_DEBUG_RECOIL),
		// Step() detects rising/falling edges of (cState.recoiling || recoilFrozen)
		// and emits a [Recoil] log line per knockback event with duration measured
		// in agent-steps, scaled game-time, and wallclock time. Used to verify
		// fps_cap / frames_per_wait don't change how long the agent perceives
		// knockback. The horizontal recoil is FixedUpdate-driven (recoilSteps
		// counter); freeze + invul windows are WaitForSeconds-driven on scaled
		// time. Both should translate to a fixed agent-step count given the
		// pinned captureDeltaTime, but the probe verifies that empirically.
		private bool _debugRecoil;
		private bool _recoilActive;
		private int _recoilStartStep;
		private float _recoilStartGameTime;
		private float _recoilStartRealTime;

		// Fake-reset: with probability _fakeResetProb, intercept lethal damage
		// (boss or knight) by clamping it to leave 1 HP, then immediately
		// restoring both knight and all boss HPs to max in-place. This avoids
		// the ~85%-of-wallclock scene-transition Reset() while still emitting
		// a done=true boundary so the curriculum / GAE bootstrap fire normally.
		// Set via FK_FAKE_RESET_PROB env var. _fakeResetPending is flipped on
		// inside the damage hooks; Step() converts it into _episodeDone with
		// info="fake_reset". _lastEpisodeWasFake then takes Reset() through
		// a fast-path that skips the scene-load entirely.
		private float _fakeResetProb;
		private bool _fakeResetPending;
		private bool _lastEpisodeWasFake;
		private int _fakeResetCount;

		// Stored hook delegates so Dispose can detach them. The FreezeMoment kill
		// is a global override — installed once in Setup() and held for the
		// process lifetime — so we keep refs to the actual delegates we hooked.
		private On.GameManager.hook_FreezeMoment_float_float_float_float _killFreezeFloat;
		private On.GameManager.hook_FreezeMoment_float_float_float_bool _killFreezeBool;
		private On.GameManager.hook_FreezeMoment_int _killFreezeInt;
		private On.GameManager.hook_FreezeMomentGC _killFreezeMomentGC;

		private HitboxObserver _hitboxObserver = new();
		private FsmObserver _fsmObserver = new();
		private InputDeviceShim _inputShim = new();

		// Shared-memory channel for the step hot path. Opened in Setup() when
		// Python's MSG_INIT carries use_shm=true; null otherwise (then step
		// actions/responses stay on the WebSocket). The _shmActionLoop
		// background coroutine polls _shm.TryReadAction every frame and
		// dispatches Step() when an action arrives. Type lookup uses the
		// `using FullKnight.Net;` at the top of this file — naming the type
		// `FullKnight.Net.ShmChannel` here collides with the FullKnight class
		// (CS0426: namespace vs. type ambiguity).
		private ShmChannel _shm;

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
			Log($"[ResetEntry] reset#{_resetCount + 1} requestedLevel={data.level ?? "(null)"} "
				+ $"eval={data.eval} fpw={data.frames_per_wait}");
			// [ResetDiag] Capture entry state BEFORE any field mutations so we can
			// see exactly what HK looked like when Python issued the reset — in
			// particular whether scene was still PLAYING (premature reset from
			// a false done=true) vs. legitimately EXITING_LEVEL.
			{
				var _gm = GameManager.instance;
				string _entryScene = UnityEngine.SceneManagement.SceneManager
					.GetActiveScene().name;
				string _gmState = "?";
				try { _gmState = _gm != null ? _gm.gameState.ToString() : "?"; }
				catch { }
				bool _inTrans = _gm != null && _gm.IsInSceneTransition;
				int _hpNow = PlayerData.instance != null ? PlayerData.instance.health : -1;
				var _bossHpStr = new System.Text.StringBuilder();
				foreach (var hm in _bossHMs)
				{
					if (_bossHpStr.Length > 0) _bossHpStr.Append(",");
					_bossHpStr.Append(hm == null ? "null" : hm.hp.ToString());
				}
				Log($"[ResetDiag] reset#{_resetCount + 1} entryScene={_entryScene} "
					+ $"gmState={_gmState} inTransition={_inTrans} knightHp={_hpNow} "
					+ $"prevResult={_episodeResult ?? "(none)"} bossHps=[{_bossHpStr}] "
					+ $"stepCount={_stepCount} bossDied={_bossDied} "
					+ $"episodeDone={_episodeDone} level={data.level ?? _level}");
			}
			// Fake-reset is only valid if the next episode targets the SAME
			// boss arena we're already in. Python's boss_rotation_period
			// keeps an env on the same boss for N episodes; on rotation it
			// picks a new boss and we MUST do a real scene-load reset.
			string requestedLevel = data.level ?? _level;
			string activeScene = UnityEngine.SceneManagement.SceneManager
				.GetActiveScene().name;
			if (_lastEpisodeWasFake && requestedLevel != activeScene)
			{
				_lastEpisodeWasFake = false;  // boss rotated: fall through to full reset
			}
			_level = requestedLevel;
			_frameSkipCount = data.frames_per_wait ?? _frameSkipCount;
			_evalMode = data.eval ?? false;
			_hitsTakenInStep = 0;
			_damageLandedInStep = 0;
			_hpHealedInStep = 0;
			_silentHpDeltaInStep = 0;
			_silentHpEventsInStep = 0;
			_inHeroTakeDamage = false;
			_recoilActive = false;
			_stepCount = 0;
			_bossDied = false;
			_episodeDone = false;
			_episodeResult = null;
			_resetCount++;

			// Fake-reset fast-path. Damage hooks already restored both knight
			// and boss HPs to max in-place; no HK death FSM ran (we clamped
			// the lethal hit at 1 HP). Scene state is whatever it was mid-fight,
			// so the only work we have to do here is rebuild the obs and reply.
			// Skips ~85% of typical reset wallclock (PRE-UNLOAD + LoadBossScene).
			if (_lastEpisodeWasFake)
			{
				_lastEpisodeWasFake = false;
				// captureDeltaTime was pinned at last real reset and is held
				// constant through fake resets. timeScale was set to 0 by the
				// previous Step()'s freeze; leave it there since obs build
				// doesn't need ticks.
				const float kBaselineGtime_fake = 0.0424f;
				Time.captureDeltaTime = kBaselineGtime_fake;
				_resetBranch = 2;
				// Stub LogPhase calls to keep wire alignment with the 7-slot
				// reset_phase_deltas_ms array Python expects (binary_protocol).
				LogPhase("Reset", "FAKE-PRE-UNLOAD");
				LogPhase("Reset", "FAKE-NATURAL-END");
				LogPhase("Reset", "FAKE-settle");
				LogPhase("Reset", "FAKE-LoadBossScene");
				LogPhase("Reset", "FAKE-RecreateReader");
				LogPhase("Reset", "FAKE-InitBossRefs");
				var fakeObs = _hitboxObserver.GetSplitFeatures(_bossHMs, emitTerrainDebug: _evalMode);
				var fakeGs = StateExtractor.GetGlobalState(fakeObs.KnightWidth, fakeObs.KnightHeight);
				data.combat_hitboxes = fakeObs.CombatHitboxes;
				data.combat_kinds = fakeObs.CombatKinds;
				data.combat_parents = fakeObs.CombatParents;
				data.terrain_hitboxes = fakeObs.TerrainHitboxes;
				data.terrain_debug = fakeObs.TerrainDebug;
				data.global_state = fakeGs;
				data.fsm_snapshots = SnapshotFsms();
				LogPhase("Reset", "obs+freeze (fake)");
				float fakeResetMs = (Time.realtimeSinceStartup - _phaseStart) * 1000f;
				Log($"[Reset-Timing] reset#{_resetCount} FAKE TOTAL {fakeResetMs:F1}ms (count={_fakeResetCount})");
				data.reset_phase_deltas_ms = _resetPhaseDeltasMs.ToArray();
				data.reset_phase_frames = _resetPhaseFrames.ToArray();
				data.reset_branch = _resetBranch;
				SendMessage(new Message { type = "reset", data = data });
				yield break;
			}

			LogPhase("Reset", "PRE-UNLOAD");

			// Release any inputs held over from the previous episode before the
			// scene transition unfreezes time — otherwise a stuck "left" or "jump"
			// runs the knight for the entire transition + intro-skip window.
			// Also clear any hard-commit lock left over from a mid-charge death.
			_inputShim.ResetCommit();
			ActionDecoder.ApplyAction(_inputShim, new int[] { 2, 2, 7, 1 }, _frameSkipCount);

			// Step() ends with Time.timeScale=0 to freeze for the Python obs
			// handoff. Unfreeze here so HK's queued DoDreamReturn / hero-death
			// transition can tick during WaitForSceneChange and downstream
			// LoadBossScene coroutines.
			Time.timeScale = 1f;
			// Pin per-frame game time so it doesn't depend on Unity's wallclock
			// fps. Held constant from here on (never restored to 0); the inter-
			// step pause uses Time.timeScale=0 instead, which gives deltaTime=0
			// even with captureDeltaTime non-zero.
			const float kBaselineGtime = 0.0424f;
			Time.captureDeltaTime = kBaselineGtime;

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

			// Diagnostic: dump the state of the TARGET statue's FSM after returning
			// from a boss. The previous run found "any" statue (always Inert) which
			// was misleading. Target the specific bossScene matching _level so we
			// see the same FSM LoadBossScene will use.
			if (_resetBranch == 1)
			{
				BossStatue target = null;
				foreach (var s in UnityEngine.Object.FindObjectsOfType<BossStatue>())
				{
					var bs = s.bossScene;
					if (bs == null) continue;
					if (bs.Tier1Scene == _level || bs.Tier2Scene == _level || bs.Tier3Scene == _level)
					{
						target = s;
						break;
					}
				}
				if (target == null)
				{
					Log("[WakeDiag] target statue NOT FOUND for level=" + _level);
				}
				else
				{
					var pmFsm = target.bossUIControlFSM;
					var fsm = pmFsm?.Fsm;
					string goActive = target.gameObject.activeInHierarchy.ToString();
					string compEnabled = (pmFsm != null) ? pmFsm.enabled.ToString() : "null";
					string fsmActiveState = fsm?.ActiveStateName ?? "(null fsm)";
					bool fsmFinished = fsm != null && fsm.Finished;
					string startState = fsm?.StartState ?? "(none)";
					Log($"[WakeDiag] target={target.gameObject.name} goActive={goActive} "
						+ $"compEnabled={compEnabled} fsmActiveState='{fsmActiveState}' "
						+ $"fsmFinished={fsmFinished} startState={startState}");
				}
			}

			yield return SceneHooks.LoadBossScene(_level);
			LogPhase("Reset", "LoadBossScene");

			// Force-recreate the hitbox reader for the boss scene. The activeSceneChanged
			// event is unreliable under multi-instance load (some instances miss it), so
			// we explicitly rebuild the reader here and yield a frame for Start() to scan.
			_hitboxObserver.RecreateReader();
			yield return null;
			LogPhase("Reset", "RecreateReader+frame");

			InitBossRefs();
			// BossSceneController.bosses can populate lazily — retry until
			// colliders enable. Replaces a fixed-duration realtime settle that
			// gave 0.4–3.5 sim-s of FSM advance depending on fps.
			int wakeFrames = 0;
			const int kMaxWakeFrames = 600;
			while (wakeFrames < kMaxWakeFrames)
			{
				if (_bossHMs.Count == 0) InitBossRefs();
				if (_bossHMs.Count > 0 && HasActiveCombatHitboxes()) break;
				yield return null;
				wakeFrames++;
			}
			bool bossAwake = HasActiveCombatHitboxes();
			LogPhase("Reset", "InitBossRefs+BossWake");
			Log($"[BounceCheck] reset#{_resetCount} level={_level} "
				+ $"bossAwake={bossAwake} wakeFrames={wakeFrames}");

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
			data.fsm_snapshots = SnapshotFsms();

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
			// First few steps of each episode get an entry log so we can tell
			// whether Step is firing at all when training appears frozen.
			if (_stepCount <= 3)
			{
				int aLen = data.action_vec != null ? data.action_vec.Length : -1;
				string aStr = aLen == 4
					? $"[{data.action_vec[0]},{data.action_vec[1]},{data.action_vec[2]},{data.action_vec[3]}]"
					: $"(len={aLen})";
				Log($"[StepEntry] reset#{_resetCount} step#{_stepCount} "
					+ $"action={aStr} episodeDone={_episodeDone} "
					+ $"timeScale={Time.timeScale:F2}");
			}
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
				EmitStepResponse(data);
				yield break;
			}

			Time.timeScale = 1f;

			bool committedThisStep = ActionDecoder.ApplyAction(
				_inputShim, data.action_vec, _frameSkipCount);
			data.action_committed = committedThisStep;

			// Track HP at step start for heal detection
			_knightHpAtStepStart = PlayerData.instance.health;

			// captureDeltaTime is pinned in Reset() and held constant — it
			// makes per-frame game time deterministic regardless of machine
			// wallclock fps. We don't toggle it here; inter-step pause uses
			// Time.timeScale=0 (deltaTime=captureDeltaTime*0=0).

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
			float frameSkipMs = (Time.realtimeSinceStartup - frameSkipT0) * 1000f;

			Time.timeScale = 0;
			data.step_game_time = gameTimeElapsed;
			data.step_real_time = realTimeElapsed;

			if (_debugRecoil)
			{
				var hc = HeroController.instance;
				bool recoilNow = hc != null
					&& (hc.cState.recoiling
						|| hc.cState.recoilFrozen
						|| hc.cState.recoilingLeft
						|| hc.cState.recoilingRight);
				if (recoilNow && !_recoilActive)
				{
					_recoilActive = true;
					_recoilStartStep = _stepCount;
					_recoilStartGameTime = Time.time;
					_recoilStartRealTime = Time.realtimeSinceStartup;
				}
				else if (!recoilNow && _recoilActive)
				{
					_recoilActive = false;
					int dSteps = _stepCount - _recoilStartStep;
					float dGtMs = (Time.time - _recoilStartGameTime) * 1000f;
					float dRtMs = (Time.realtimeSinceStartup - _recoilStartRealTime) * 1000f;
					Log($"[Recoil] reset#{_resetCount} steps={dSteps} "
						+ $"gt={dGtMs:F0}ms rt={dRtMs:F0}ms "
						+ $"fpw={_frameSkipCount}");
				}
			}

			// Check for episode end (both modes now have real HP and death)
			if (!_episodeDone)
			{
				if (_fakeResetPending)
				{
					// Damage hook clamped a lethal hit; HPs already restored.
					// Mark the boundary so curriculum / GAE bootstrap fire,
					// but signal Reset() to take the fast-path (no scene load).
					_fakeResetPending = false;
					_lastEpisodeWasFake = true;
					_fakeResetCount++;
					_episodeDone = true;
					_episodeResult = "fake_reset";
					Log($"[EpisodeEnd] reset#{_resetCount} step#{_stepCount} result=fake_reset");
				}
				else if (_bossDied)
				{
					_episodeDone = true;
					_episodeResult = "win";
					// [EpisodeEnd] Log live boss HPs so we can tell whether the
					// boss actually died (hp<=0) or our OnBossDamaged hook flipped
					// _bossDied via a damageOverride / clamp false positive
					// during stagger.
					var _bossHpStr = new System.Text.StringBuilder();
					foreach (var hm in _bossHMs)
					{
						if (_bossHpStr.Length > 0) _bossHpStr.Append(",");
						_bossHpStr.Append(hm == null ? "null" : hm.hp.ToString());
					}
					Log($"[EpisodeEnd] reset#{_resetCount} step#{_stepCount} result=win "
						+ $"knightHp={PlayerData.instance.health} bossHps=[{_bossHpStr}]");
				}
				else if (PlayerData.instance.health <= 0)
				{
					// Knight dropped to 0 outside the TakeDamage hook (silent
					// FSM/charm/spike write). Roll fake here too so this path
					// also benefits — clamp HP back up and treat as fake.
					if (_fakeResetProb > 0f && UnityEngine.Random.value < _fakeResetProb)
					{
						RestoreFightHPs();
						_lastEpisodeWasFake = true;
						_fakeResetCount++;
						_episodeDone = true;
						_episodeResult = "fake_reset";
						Log($"[EpisodeEnd] reset#{_resetCount} step#{_stepCount} result=fake_reset (knight)");
					}
					else
					{
						_episodeDone = true;
						_episodeResult = "loss";
						Log($"[EpisodeEnd] reset#{_resetCount} step#{_stepCount} result=loss knightHp={PlayerData.instance.health}");
					}
				}
				// Glitch detector disabled: it false-fires during legitimate
				// knight-death sequences (scene/bosses flip before health hits 0).
#if false
				else
				{
					// Glitch detector: target the specific "knight + boss
					// disappeared, no done event" failure mode. Three
					// independent same-step signals; any of them flipping
					// is enough to declare the episode lost. None of these
					// trigger in normal mid-fight play (unlike a count of
					// "N steps without events"), so detection is immediate
					// and false-positive-resistant.
					//
					//   1. BossSceneController.endedScene: set the moment
					//      any tracked boss HM's OnDeath fires (regardless
					//      of whether it routed through TakeDamage), or any
					//      manual EndBossScene call. Catches FSM-driven
					//      boss kills that bypass our OnBossDamaged hook.
					//   2. Active scene name no longer matches _level: the
					//      scene flipped under us during the step. Catches
					//      transitions kicked off by anything other than
					//      our Reset() coroutine.
					//   3. All tracked _bossHMs entries are null /
					//      destroyed / disabled / hp<=0: every boss is gone
					//      but neither (1) nor _bossDied flagged it. Catches
					//      Object.Destroy on the boss GameObject without an
					//      OnDeath dispatch (rare but observed in dump).
					//
					// _bossHMs.Count == 0 means InitBossRefs hasn't run
					// yet (we're still inside the first Step after a
					// just-completed Reset where boss spawn lags) — skip
					// the "all gone" check there to avoid a false positive.
					// endedScene is a private bool on BossSceneController (HK
					// source line ~217). We read it via ReflectionHelper —
					// same pattern StateExtractor uses for HeroController.rb2d.
					var bsc = BossSceneController.Instance;
					bool sceneEnded = false;
					if (bsc != null)
					{
						try
						{
							sceneEnded = ReflectionHelper
								.GetField<BossSceneController, bool>(
									bsc, "endedScene");
						}
						catch { sceneEnded = false; }
					}

					string activeScene = UnityEngine.SceneManagement
						.SceneManager.GetActiveScene().name;
					bool sceneFlipped = !string.IsNullOrEmpty(_level)
						&& activeScene != _level;

					bool allBossesGone = false;
					if (_bossHMs.Count > 0)
					{
						allBossesGone = true;
						foreach (var hm in _bossHMs)
						{
							if (hm == null) continue;
							if (hm.gameObject == null) continue;
							if (!hm.gameObject.activeInHierarchy) continue;
							if (hm.hp <= 0) continue;
							allBossesGone = false;
							break;
						}
					}

					if (sceneEnded || sceneFlipped || allBossesGone)
					{
						_episodeDone = true;
						string reason = sceneEnded ? "glitch_scene_ended"
							: sceneFlipped ? "glitch_scene_flipped"
							: "glitch_bosses_gone";
						_episodeResult = reason;
						int liveHms = 0;
						foreach (var hm in _bossHMs)
						{
							if (hm != null && hm.gameObject != null
								&& hm.gameObject.activeInHierarchy
								&& hm.hp > 0) liveHms++;
						}
						Log($"[GlitchDetector] reset#{_resetCount} "
							+ $"step#{_stepCount} reason={reason} "
							+ $"scene={activeScene} expected={_level} "
							+ $"bsc.endedScene={sceneEnded} "
							+ $"hmCount={_bossHMs.Count} liveHms={liveHms} "
							+ $"hp={PlayerData.instance.health} "
							+ $"silentHpEvents={_silentHpEventsInStep} "
							+ $"silentHpDelta={_silentHpDeltaInStep}");
					}
				}
#endif
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
			// Silent-HP probe is per-step: clear after the step completes so
			// each [SilentHP] log line is attributable to its own step. The
			// glitch-detector log above already includes this step's accumulator
			// before we wipe it.
			_silentHpDeltaInStep = 0;
			_silentHpEventsInStep = 0;

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
					+ $"done={_episodeDone}");
			}

			if (_episodeDone)
			{
				data.done = true;
				data.info = _episodeResult;
				data.combat_hitboxes = new List<float[]>();
				data.terrain_hitboxes = new List<float[]>();
				data.global_state = new float[22];
				EmitStepResponse(data);
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
			data.fsm_snapshots = SnapshotFsms();
			data.done = false;

			EmitStepResponse(data);
			yield break;
		}

		// Snapshot every FSM relevant to the current fight: every PlayMakerFSM
		// in the subtree of each tracked boss, plus every FSM on currently-
		// active Enemy-class colliders (catches in-flight boss projectiles
		// whose pool reparented them out of the boss subtree), plus every
		// FSM on Attack-class colliders (the knight's nail / spell hitboxes,
		// for cross-referencing the agent's action against attack windows).
		// Routed to the Python visualizer only — never consumed by training.
		private int _fsmDiagTicks;
		private List<string> SnapshotFsms()
		{
			HashSet<Collider2D> enemies = null;
			HashSet<Collider2D> attacks = null;
			try
			{
				var hb = _hitboxObserver.GetHitboxes();
				if (hb != null)
				{
					if (hb.ContainsKey(HitboxType.Enemy)) enemies = hb[HitboxType.Enemy];
					if (hb.ContainsKey(HitboxType.Attack)) attacks = hb[HitboxType.Attack];
				}
			}
			catch (System.Exception e) { Log($"[FsmDiag] GetHitboxes threw: {e.Message}"); }

			List<string> result;
			try
			{
				result = _fsmObserver.Snapshot(_bossHMs, enemies, attacks);
			}
			catch (System.Exception e)
			{
				Log($"[FsmDiag] Snapshot threw: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
				result = new List<string>();
			}

			// Heartbeat log every 60 steps so we can see why the panel is empty
			// without spamming the log. Always logs the first step.
			_fsmDiagTicks++;
			if (_fsmDiagTicks == 1 || _fsmDiagTicks % 60 == 0)
			{
				int eCount = enemies != null ? enemies.Count : -1;
				int aCount = attacks != null ? attacks.Count : -1;
				Log($"[FsmDiag] tick={_fsmDiagTicks} bossHMs={_bossHMs.Count} "
					+ $"enemyColliders={eCount} attackColliders={aCount} "
					+ $"snapshot={result.Count}");
				if (result.Count > 0 && result.Count <= 3)
				{
					for (int i = 0; i < result.Count; i++)
						Log($"[FsmDiag]   [{i}] {result[i]}");
				}
			}
			return result;
		}

		private void Log(string msg) => FullKnight.LogS($"[TrainingEnv] {msg}");

		// Phase timing helpers. PhaseBegin() resets the stopwatch at the start of
		// Reset() / Step(); LogPhase() emits a "[Phase-Timing]" line with the
		// delta since the last call and the cumulative total. Wall-clock based
		// (Time.realtimeSinceStartup) so Time.timeScale=0 freezes during the
		// Python obs handoff don't confuse the readings.
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
			// Stamp the global SlotId so every Log call site (TrainingEnv,
			// SceneHooks, FsmObserver, …) can prefix [s{slot}] and we can
			// pick out a single instance's lines from the shared ModLog.
			if (message.data.slot.HasValue)
				FullKnight.SlotId = (int)message.data.slot.Value;
			Log($"[Setup] received init slot={FullKnight.SlotId} useShm={message.data.use_shm}");

			// Uncap Unity's frame loop. With -nographics there's no display to
			// vsync against; the only thing throttling Update() is targetFrameRate
			// (default cap on Windows). Pairing this with captureDeltaTime
			// decouples wallclock framerate from per-step game-time, so faster
			// frames don't shrink dt out from under the agent. Uncap-alone
			// (commit e094f23, reverted) gave +63% throughput but cratered
			// quality via the regime shift; capture-alone (a2f7136) preserves
			// regime but Unity stays at ~360fps cap and there's no speedup.
			// Combined, the uncap delivers the wallclock win and capture holds
			// the regime steady.
			//
			// FK_FPS_CAP env var (set by train.py from config.fps_cap) overrides
			// the uncap with a positive integer cap — useful when watching
			// graphical runs locally so HK doesn't burn CPU at hundreds of fps.
			int fpsCap = -1;
			var capStr = System.Environment.GetEnvironmentVariable("FK_FPS_CAP");
			if (!string.IsNullOrEmpty(capStr) && int.TryParse(capStr, out int parsed) && parsed > 0)
				fpsCap = parsed;
			QualitySettings.vSyncCount = 0;
			Application.targetFrameRate = fpsCap;
			Log($"[Setup] targetFrameRate={fpsCap} (FK_FPS_CAP={capStr ?? "unset"})");

			var dbgRecoil = System.Environment.GetEnvironmentVariable("FK_DEBUG_RECOIL");
			_debugRecoil = !string.IsNullOrEmpty(dbgRecoil) && dbgRecoil != "0";
			Log($"[Setup] debugRecoil={_debugRecoil} (FK_DEBUG_RECOIL={dbgRecoil ?? "unset"})");

			_fakeResetProb = 0f;
			var fakeProbStr = System.Environment.GetEnvironmentVariable("FK_FAKE_RESET_PROB");
			if (!string.IsNullOrEmpty(fakeProbStr)
				&& float.TryParse(fakeProbStr, System.Globalization.NumberStyles.Float,
					System.Globalization.CultureInfo.InvariantCulture, out float parsedProb))
			{
				_fakeResetProb = Mathf.Clamp01(parsedProb);
			}
			Log($"[Setup] fakeResetProb={_fakeResetProb:F3} (FK_FAKE_RESET_PROB={fakeProbStr ?? "unset"})");

			// Kill HK's hit-stop. On every hit, HeroController.StartRecoil yields
			// to gm.FreezeMoment(0.01, 0.35, 0.1, 0.0001), which ramps Time.timeScale
			// down to 0.0001 and back up to 1 via TimeController.GenericTimeScale.
			// That ramp coroutine runs across our inter-step boundaries — and our
			// explicit `Time.timeScale = 0/1` writes in Step() bypass TimeController,
			// so any subsequent `genericTimeScale` mutation by the ramp recomputes
			// and clobbers our value (TimeController.cs:69-80). Net effect: a hit
			// can leak a fractional timeScale into the following agent step, making
			// per-step game-time non-deterministic and dependent on Python wallclock
			// pause cadence. For RL training the visual hit-stop has no value, so
			// we no-op all FreezeMoment overloads. The void `FreezeMoment(int)`
			// overload internally StartCoroutine's the IEnumerator overloads, so
			// hooking those is sufficient — we hook the int form too for paranoia.
			_killFreezeFloat = (orig, self, rd, w, ru, ts) => EmptyCoroutine();
			_killFreezeBool = (orig, self, rd, w, ru, gc) => EmptyCoroutine();
			_killFreezeMomentGC = (orig, self, rd, w, ru, ts) => EmptyCoroutine();
			_killFreezeInt = (orig, self, type) => { /* no-op */ };
			On.GameManager.FreezeMoment_float_float_float_float += _killFreezeFloat;
			On.GameManager.FreezeMoment_float_float_float_bool += _killFreezeBool;
			On.GameManager.FreezeMomentGC += _killFreezeMomentGC;
			On.GameManager.FreezeMoment_int += _killFreezeInt;
			Log("[Setup] FreezeMoment killed (hit-stop disabled)");

			On.GameManager.SaveGame += SaveFileProxy.DisableSaveGame;
			SaveFileProxy.LoadCompletedSave();
			GameManager.instance.ContinueGame();
			yield return new SceneHooks.WaitForSceneLoad("GG_Workshop");
			yield return new SceneHooks.WaitForEntryFinished();
			yield return new WaitForSeconds(2f);

			_hitboxObserver.Load();
			InputManager.AttachDevice(_inputShim);

			// If Python negotiated the shared-memory step transport during
			// init, open our end of the channel and spin up a background
			// coroutine that polls the action event each frame and dispatches
			// Step(). init/reset/pause/resume/close stay on the WebSocket.
			if ((message.data.use_shm ?? false) && message.data.slot.HasValue)
			{
				int slot = (int)message.data.slot.Value;
				try
				{
					_shm = new ShmChannel(slot);
					Log($"[Setup] shm step channel opened (slot={slot})");
					GameManager.instance.StartCoroutine(_shmActionLoop());
				}
				catch (System.Exception e)
				{
					Log($"[Setup] shm open failed for slot={slot}: {e.Message}; falling back to WebSocket");
					_shm = null;
				}
			}
			else
			{
				Log($"[Setup] shm step channel disabled (use_shm={message.data.use_shm}, slot={message.data.slot}); using WebSocket");
			}

			SendMessage(message);
		}

		// Background dispatch for the shm step transport. WaitOne(0) inside
		// TryReadAction is a cheap user-mode check when the event isn't
		// signaled, so per-frame polling cost is essentially zero. When an
		// action arrives we run Step() inline (yields through frame_skip
		// frames just like the WebSocket dispatch did); EmitStepResponse
		// writes the obs back via shm and signals Python.
		//
		// Heartbeat: every 600 polled frames (~10s at 60fps), log liveness
		// + seconds-since-last-action so a stuck Python or stuck Step is
		// distinguishable from a healthy idle loop in the HK mod log.
		private IEnumerator _shmActionLoop()
		{
			long polls = 0;
			int actionsSeen = 0;
			float lastActionT = Time.realtimeSinceStartup;
			float lastHeartbeatT = Time.realtimeSinceStartup;
			Log("[ShmLoop] started");
			while (!_terminate && _shm != null)
			{
				if (_shm.TryReadAction(out int[] action))
				{
					actionsSeen++;
					lastActionT = Time.realtimeSinceStartup;
					var data = new MessageData { action_vec = action };
					yield return Step(data);
				}
				polls++;
				if (Time.realtimeSinceStartup - lastHeartbeatT > 10f)
				{
					float idle = Time.realtimeSinceStartup - lastActionT;
					Log($"[ShmLoop] alive polls={polls} actions={actionsSeen} "
						+ $"idleSinceLastAction={idle:F1}s "
						+ $"resetCount={_resetCount} stepCount={_stepCount} "
						+ $"episodeDone={_episodeDone}");
					lastHeartbeatT = Time.realtimeSinceStartup;
				}
				yield return null;
			}
			Log($"[ShmLoop] exiting (terminate={_terminate}, shmNull={_shm == null})");
		}

		// Emit a step response over whichever transport is active for this
		// instance. The packed bytes are identical between paths
		// (BinaryProtocol.Pack); only the wire changes.
		private void EmitStepResponse(MessageData data)
		{
			var msg = new Message { type = "step", data = data, sender = "client" };
			if (_shm != null)
			{
				byte[] bytes = BinaryProtocol.Pack(msg);
				_shm.SendObs(bytes);
			}
			else
			{
				SendMessage(msg);
			}
		}

		protected override IEnumerator Dispose()
		{
			UnhookDamage();
			foreach (var kvp in _bossDeathHandlers)
			{
				if (kvp.Key != null) kvp.Key.OnDeath -= kvp.Value;
			}
			_bossDeathHandlers.Clear();
			if (_killFreezeFloat != null) On.GameManager.FreezeMoment_float_float_float_float -= _killFreezeFloat;
			if (_killFreezeBool != null) On.GameManager.FreezeMoment_float_float_float_bool -= _killFreezeBool;
			if (_killFreezeMomentGC != null) On.GameManager.FreezeMomentGC -= _killFreezeMomentGC;
			if (_killFreezeInt != null) On.GameManager.FreezeMoment_int -= _killFreezeInt;
			InputManager.DetachDevice(_inputShim);
			_hitboxObserver.Unload();
			if (_shm != null)
			{
				_shm.Dispose();
				_shm = null;
			}
			CloseSocket();
			yield break;
		}

		private static IEnumerator EmptyCoroutine() { yield break; }

		private void HookDamage()
		{
			ModHooks.AfterTakeDamageHook += OnKnightDamaged;
			On.HealthManager.TakeDamage += OnBossDamaged;
			// Silent-HP-damage probe: wrap HeroController.TakeDamage so we can
			// distinguish HP decrements that came through the normal damage
			// pipeline (hooks fire) from FSM/charm/direct-write paths that
			// bypass it (only PlayerData.TakeHealth fires).
			On.HeroController.TakeDamage += OnHeroTakeDamage;
			On.PlayerData.TakeHealth += OnPlayerTakeHealth;
		}

		private void UnhookDamage()
		{
			ModHooks.AfterTakeDamageHook -= OnKnightDamaged;
			On.HealthManager.TakeDamage -= OnBossDamaged;
			On.HeroController.TakeDamage -= OnHeroTakeDamage;
			On.PlayerData.TakeHealth -= OnPlayerTakeHealth;
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
			// [BossDmgDiag] Snapshot pre-state so we can compare against post-orig
			// values and see whether HK actually applied the full damage or clamped
			// via damageOverride / multiplier / invulnerability. False-positive
			// _bossDied is the leading hypothesis for premature reset on stagger.
			int _hpBefore = self.hp;
			bool _damageOverride = self.damageOverride;
			float _multiplier = hitInstance.Multiplier;

			// Fake-reset: if this hit would kill the LAST live boss, with
			// probability _fakeResetProb clamp it so the boss survives at 1 HP,
			// then immediately restore all boss HPs and the knight HP to max.
			// No HK death FSM fires because hp never reaches 0.
			if (wouldDie && _fakeResetProb > 0f)
			{
				bool wouldFinish = true;
				foreach (var hm in _bossHMs)
				{
					if (hm == null) continue;
					if (hm == self) continue;
					if (hm.hp > 0) { wouldFinish = false; break; }
				}
				if (wouldFinish && UnityEngine.Random.value < _fakeResetProb)
				{
					var clamped = hitInstance;
					clamped.DamageDealt = Mathf.Max(0, self.hp - 1);
					orig(self, clamped);
					RestoreFightHPs();
					_fakeResetPending = true;
					return;
				}
			}

			orig(self, hitInstance);
			int _hpAfter = self.hp;
			// _bossDied is no longer set here. Death detection moved to
			// OnBossActualDeath, subscribed to each tracked HM.OnDeath in
			// InitBossRefs. False Knight stagger restores hp:26->260 via a path
			// that bypasses OnDeath, so OnDeath is the only signal that aligns
			// with HK's own "boss truly gone" judgment.
			if (wouldDie || _hpBefore <= maxHP / 2)
			{
				Log($"[BossDmgDiag] reset#{_resetCount} step#{_stepCount} "
					+ $"hm={self.gameObject.name} hp:{_hpBefore}->{_hpAfter} "
					+ $"dmg={hitInstance.DamageDealt} mult={_multiplier:F2} "
					+ $"override={_damageOverride} wouldDie={wouldDie} "
					+ $"maxHP={maxHP} hmCount={_bossHMs.Count}");
			}
		}

		// OnDeath handler — bound per HM via a closure in InitBossRefs. Fires
		// when HK's Die() reaches SendDeathEvent (i.e., the phase-restore path
		// did NOT intercept). Matches the signal HK's own BossSceneController
		// uses to end the scene, so we can never trip _bossDied earlier than
		// HK itself.
		private void OnBossActualDeath(HealthManager hm)
		{
			Log($"[BossOnDeath] reset#{_resetCount} step#{_stepCount} "
				+ $"hm={(hm != null ? hm.gameObject.name : "null")} "
				+ $"hp={(hm != null ? hm.hp : -1)}");
			bool allDead = true;
			foreach (var other in _bossHMs)
			{
				if (other == null) continue;
				if (other == hm) continue;
				if (other.hp > 0) { allDead = false; break; }
			}
			if (allDead)
			{
				_bossDied = true;
				Log($"[BossDied] reset#{_resetCount} step#{_stepCount} "
					+ $"hm={(hm != null ? hm.gameObject.name : "null")} via=OnDeath");
			}
		}

		// Hooks for silent-HP detection. HeroController.TakeDamage is the
		// canonical "knight got hit" path — its hooks (ModHooks.OnTakeDamage,
		// AfterTakeDamageHook) fire reliably from inside it. We wrap it just
		// to flip a flag so the PlayerData.TakeHealth hook below can tell
		// whether the decrement came from inside that path or from an
		// external write (FSM SetIntValue, charm self-damage, etc.).
		private void OnHeroTakeDamage(On.HeroController.orig_TakeDamage orig,
			HeroController self, GameObject go, GlobalEnums.CollisionSide damageSide,
			int damageAmount, int hazardType)
		{
			// Fake-reset: if this hit would kill the knight, with probability
			// _fakeResetProb clamp it so knight survives at 1 HP, then restore
			// HPs to max and flag fake reset. Avoids the death FSM entirely.
			if (_fakeResetProb > 0f && damageAmount > 0)
			{
				int currentHp = PlayerData.instance.health;
				if (currentHp - damageAmount <= 0
					&& UnityEngine.Random.value < _fakeResetProb)
				{
					int safeDmg = Mathf.Max(0, currentHp - 1);
					_inHeroTakeDamage = true;
					try { orig(self, go, damageSide, safeDmg, hazardType); }
					finally { _inHeroTakeDamage = false; }
					RestoreFightHPs();
					_fakeResetPending = true;
					return;
				}
			}

			_inHeroTakeDamage = true;
			try { orig(self, go, damageSide, damageAmount, hazardType); }
			finally { _inHeroTakeDamage = false; }
		}

		// Restore knight HP and all tracked boss HPs to their max values, in-place.
		// Called by the damage hooks after they've clamped a lethal hit. The FSM
		// state for both is whatever it was mid-hit; HP is now safely above zero
		// so the death FSM never triggered.
		//
		// Knight: must go through HeroController.MaxHealth() — a raw write to
		// PlayerData.instance.health skips proxyFSM "HeroCtrl-MaxHealth" (HUD
		// mask redraw), prevHealth bookkeeping (delta-listeners desync),
		// blockerHits reset, and UpdateBlueHealth(). MaxHealth() also reads
		// CurrentMaxHealth live, so charm-modified caps (Joni's / Fragile Heart)
		// stay correct without our own cache.
		// Boss: HealthManager.hp is a plain public field with no listeners on
		// write, so direct assignment is fine.
		private void RestoreFightHPs()
		{
			var hc = HeroController.instance;
			if (hc != null) hc.MaxHealth();
			foreach (var hm in _bossHMs)
			{
				if (hm == null) continue;
				if (hm.gameObject == null) continue;
				if (!_bossMaxHPs.TryGetValue(hm, out int maxHp)) continue;
				hm.hp = maxHp;
			}
		}

		// PlayerData.TakeHealth is the lowest-rung HP write inside HK.
		// Damage paths we observe via hooks all funnel through here, but so
		// do paths we DON'T observe (FSM-driven hits, direct calls). When
		// the "in HeroController.TakeDamage" flag is false and HP went down,
		// the hit bypassed AfterTakeDamageHook — log full attribution.
		private void OnPlayerTakeHealth(On.PlayerData.orig_TakeHealth orig,
			PlayerData self, int amount)
		{
			int hpBefore = self.health;
			orig(self, amount);
			int delta = hpBefore - self.health;
			if (delta <= 0) return;
			if (_inHeroTakeDamage) return;  // normal path; AfterTakeDamage will log it
			_silentHpDeltaInStep += delta;
			_silentHpEventsInStep++;
			// Skip 2 frames so the printed top is the unobserved caller, not
			// PlayerData.TakeHealth or our own hook. Limit depth so the line
			// doesn't blow up the mod log on a deep FSM chain.
			string callers = "?";
			try
			{
				var st = new System.Diagnostics.StackTrace(skipFrames: 2,
					fNeedFileInfo: false);
				int n = System.Math.Min(st.FrameCount, 8);
				var parts = new List<string>(n);
				for (int i = 0; i < n; i++)
				{
					var m = st.GetFrame(i)?.GetMethod();
					if (m == null) continue;
					string typeName = m.DeclaringType != null
						? m.DeclaringType.Name : "?";
					parts.Add(typeName + "." + m.Name);
				}
				callers = string.Join(" <- ", parts);
			}
			catch { }
			Log($"[SilentHP] reset#{_resetCount} step#{_stepCount} amount={amount} "
				+ $"hp:{hpBefore}->{self.health} delta={delta} stack: {callers}");
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
			// Unsubscribe any previous-episode OnDeath handlers before clearing.
			// HMs from the prior fight may have been destroyed; skip nulls. The
			// `-=` on a destroyed HM is a no-op anyway (the event holder is gone),
			// but iterating cleanly keeps the dict in sync.
			foreach (var kvp in _bossDeathHandlers)
			{
				if (kvp.Key != null) kvp.Key.OnDeath -= kvp.Value;
			}
			_bossDeathHandlers.Clear();
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
						// Capture hm in a local so the closure binds the current
						// HM, not the loop variable. Store the delegate so we can
						// unsubscribe by reference on the next reset.
						var capturedHm = hm;
						HealthManager.DeathEvent handler = () => OnBossActualDeath(capturedHm);
						_bossDeathHandlers[hm] = handler;
						hm.OnDeath += handler;
					}
				}
			}
			catch { }
		}
	}
}
