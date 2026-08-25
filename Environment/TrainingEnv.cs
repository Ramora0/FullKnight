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
		// OnDeath subscribers per HM, stored so we can unsubscribe on the next
		// reset (or Dispose). False Knight and similar multi-phase bosses
		// restore hp on stagger via some path that bypasses HM.OnDeath, so
		// the event only fires on the truly-final death — same signal HK's
		// BossSceneController uses to end the scene. Subscribing here means
		// our _bossDied can't trip before HK itself decides the boss is gone.
		private readonly Dictionary<HealthManager, HealthManager.DeathEvent> _bossDeathHandlers = new();

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
		// Per-phase ms / frame counts for the current Reset, indexed by
		// ResetPhase.Keys order. Filled by LogResetPhase and shipped on the
		// reset message so Python's TimingTracker can break down resets.
		private readonly float[] _resetPhaseMs = new float[Net.ResetPhase.Count];
		private readonly ushort[] _resetPhaseFrames = new ushort[Net.ResetPhase.Count];
		private byte _resetBranch;
		// Frame at the last LogResetPhase call — delta is folded into the
		// phase frame count. Wall-clock complement to _phaseLast.
		private int _phaseLastFrame;

		private HitboxObserver _hitboxObserver = new();
		private FsmObserver _fsmObserver = new();
		private InputDeviceShim _inputShim = new();
		private Game.TimeScale _timeManager;
		// Heartbeat counter for FsmDiag logging in SnapshotFsms.
		private int _fsmDiagTicks;

		// Per-frame game-time pin (seconds). Calibration target: trained regime
		// had gtime_baseline = 0.0424s/step at frames_per_wait=5. With
		// Time.timeScale pinned at 1f (no 3× fast-forward), each Unity frame
		// advances captureDeltaTime × 1 of game-time, so:
		//   0.0424 / 5 ≈ 0.00848 per Unity frame at fpw=5.
		// Pinned ONCE in Reset() and held constant through the episode (never
		// reset to 0). The inter-step pause uses Time.timeScale = 0, which
		// produces deltaTime = captureDeltaTime × 0 = 0 even with capture-mode
		// active — so no physics ticks between agent decisions.
		//
		// Capture mode REQUIRES Unity's render pipeline initialized — i.e.,
		// -batchmode is fine but -nographics breaks it (no renderer → capture
		// is inert and dt tracks real wall time × timeScale). InstanceManager
		// drops -nographics for this reason.
		private const float kStepDeltaTime = 0.00848f;

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
				var _bossHpStr = new StringBuilder();
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

			// Clear per-reset phase telemetry. Filled by LogResetPhase below
			// and shipped on the outgoing reset message.
			for (int i = 0; i < Net.ResetPhase.Count; i++)
			{
				_resetPhaseMs[i] = 0f;
				_resetPhaseFrames[i] = 0;
			}
			_resetBranch = Net.ResetPhase.BranchUnknown;

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
				_resetBranch = 2;
				// Stub LogResetPhase calls to keep wire alignment with the
				// 7-slot reset_phase trailer Python expects.
				LogResetPhase("pre_unload", "FAKE-PRE-UNLOAD");
				LogResetPhase("transition_out", "FAKE-NATURAL-END");
				LogResetPhase("settle", "FAKE-settle");
				LogResetPhase("load_boss_scene", "FAKE-LoadBossScene");
				LogResetPhase("recreate_reader", "FAKE-RecreateReader");
				LogResetPhase("init_boss_refs", "FAKE-InitBossRefs");
				// Fake resets keep the reader alive, so drop motion history —
				// otherwise the first step of the new episode reports a
				// displacement measured across the episode boundary.
				_hitboxObserver.ClearMotion();
				var fakeObs = _hitboxObserver.GetSplitFeatures(_bossHMs, emitTerrainDebug: _evalMode);
				var fakeGs = StateExtractor.GetGlobalState(
					fakeObs.KnightWidth, fakeObs.KnightHeight, _inputShim);
				data.combat_hitboxes = fakeObs.CombatHitboxes;
				data.combat_kinds = fakeObs.CombatKinds;
				data.combat_parents = fakeObs.CombatParents;
				data.combat_anims = fakeObs.CombatAnims;
				data.terrain_hitboxes = fakeObs.TerrainHitboxes;
				data.terrain_debug = fakeObs.TerrainDebug;
				data.global_state = fakeGs;
				data.fsm_snapshots = SnapshotFsms();
				LogResetPhase("obs_final", "obs+freeze (fake)");
				float fakeResetMs = (Time.realtimeSinceStartup - _phaseStart) * 1000f;
				Log($"[Reset-Timing] reset#{_resetCount} FAKE TOTAL {fakeResetMs:F1}ms (count={_fakeResetCount})");
				data.reset_branch = _resetBranch;
				data.reset_phase_ms = (float[])_resetPhaseMs.Clone();
				data.reset_phase_frames = (ushort[])_resetPhaseFrames.Clone();
				SendMessage(new Message { type = "reset", data = data });
				yield break;
			}

			LogResetPhase("pre_unload", "PRE-UNLOAD");

			// Release any inputs held over from the previous episode before the
			// scene transition unfreezes time — otherwise a stuck "left" or "jump"
			// runs the knight for the entire transition + intro-skip window.
			// Also clear any hard-commit lock left over from a mid-charge death.
			_inputShim.ResetCommit();
			ActionDecoder.ApplyAction(_inputShim, new int[] { 2, 2, 7, 1 },
				_frameSkipCount);

			// Unpause so scene transition and SceneHooks coroutines can proceed.
			// Pinned at 1f (no fast-forward); captureDeltaTime below makes the
			// per-frame game-time deterministic regardless of wallclock fps.
			Time.timeScale = 1f;
			// Pin per-Unity-frame game-time. Held constant from here through the
			// whole episode (never reset to 0); inter-step pause uses
			// Time.timeScale = 0, which gives deltaTime = 0 even with capture
			// mode active.
			Time.captureDeltaTime = kStepDeltaTime;

			// Two paths out of the boss arena:
			//  (a) Already out — death/win cleanup landed before reset arrived.
			//      Nothing to wait for.
			//  (b) Anywhere else — HK's natural DoDreamReturn / hero-death
			//      transition either just finished or is queued. Wait for the
			//      scene to leave preScene; the new canonical SceneHooks.LoadBossScene
			//      handles bench-leave + dream-return wake-up + statue-reinit from
			//      whatever state HK drops us in.
			var preScene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
			if (preScene == "GG_Workshop")
			{
				_resetBranch = Net.ResetPhase.BranchWorkshop;
				LogResetPhase("transition_out", "ALREADY-IN-WORKSHOP");
			}
			else
			{
				_resetBranch = Net.ResetPhase.BranchNaturalEnd;
				yield return WaitForSceneChange(preScene);
				var afterScene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
				LogResetPhase("transition_out", $"NATURAL-END (pre={preScene} post={afterScene})");
			}

			// Empty LogResetPhase preserves the 7-slot wire alignment. The
			// old WaitForFinishedEnteringScene settle here was a no-op anyway
			// (it was a PlayMaker action mis-yielded as a coroutine); the new
			// SceneHooks.LoadBossScene does its own entry-wait via
			// SceneHooks.WaitForEntryFinished after BounceThroughWorkshop.
			LogResetPhase("settle", "settle (skipped)");

			yield return SceneHooks.LoadBossScene(_level);
			LogResetPhase("load_boss_scene", "LoadBossScene");

			// Force-recreate the hitbox reader for the boss scene. The activeSceneChanged
			// event is unreliable under multi-instance load (some instances miss it), so
			// we explicitly rebuild the reader here and yield a frame for Start() to scan.
			_hitboxObserver.RecreateReader();
			yield return null;
			LogResetPhase("recreate_reader", "RecreateReader+frame");

			InitBossRefs();
			// BossSceneController.bosses can populate lazily — retry until
			// colliders enable. Replaces the realtime-settle wait that gave
			// variable sim-time advance depending on fps.
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
			LogResetPhase("init_boss_refs", "InitBossRefs+BossWake");
			Log($"[BounceCheck] reset#{_resetCount} level={_level} "
				+ $"bossAwake={bossAwake} wakeFrames={wakeFrames}");
			_knightMaxHP = PlayerData.instance.maxHealth;

			UnhookDamage();
			HookDamage();

			// TimeScale shim with multiplier=1 — its FreezeMoment IL-hooks become
			// no-ops (FreezeMoment is killed in Setup() anyway) and the
			// SetTimeScale shim passes HK's writes straight through to
			// TimeController.GenericTimeScale unchanged.
			if (_timeManager != null) _timeManager.Dispose();
			_timeManager = new Game.TimeScale(1f);

			var obs = _hitboxObserver.GetSplitFeatures(_bossHMs, emitTerrainDebug: _evalMode);
			var gs = StateExtractor.GetGlobalState(
				obs.KnightWidth, obs.KnightHeight, _inputShim);

			data.combat_hitboxes = obs.CombatHitboxes;
			data.combat_kinds = obs.CombatKinds;
			data.combat_parents = obs.CombatParents;
			data.combat_anims = obs.CombatAnims;
			data.terrain_hitboxes = obs.TerrainHitboxes;
			data.terrain_debug = obs.TerrainDebug;
			data.global_state = gs;
			data.fsm_snapshots = SnapshotFsms();

			Time.timeScale = 0;
			LogResetPhase("obs_final", "obs+freeze (final)");
			float resetTotalMs = (Time.realtimeSinceStartup - _phaseStart) * 1000f;
			Log($"[Reset-Timing] reset#{_resetCount} TOTAL {resetTotalMs:F0}ms level={_level}");
			// Ship per-phase telemetry alongside the obs. Python's TimingTracker
			// drains these via VecEnv.pop_reset_dts() and prints the per-phase
			// breakdown in the wallclock summary.
			data.reset_branch = _resetBranch;
			data.reset_phase_ms = (float[])_resetPhaseMs.Clone();
			data.reset_phase_frames = (ushort[])_resetPhaseFrames.Clone();
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
				data.global_state = new float[StateExtractor.GlobalStateDim];
				data.fsm_snapshots = new List<string>();
				data.damage_landed = 0;
				data.hits_taken = 0;
				data.hp_healed = 0;
				data.step_game_time = 0;
				data.step_real_time = 0;
				SendMessage(new Message { type = "step", data = data });
				yield break;
			}

			Time.timeScale = 1f;

			bool committedThisStep = ActionDecoder.ApplyAction(
				_inputShim, data.action_vec, _frameSkipCount);
			data.action_committed = committedThisStep;

			// Track HP at step start for heal detection
			_knightHpAtStepStart = PlayerData.instance.health;

			// captureDeltaTime was pinned in Reset and is held constant; no
			// per-step toggle. Time.timeScale = 0 between agent decisions
			// already gives deltaTime = 0 even with capture mode active.

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
					}
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
					+ $"realTime={realTimeElapsed * 1000:F0}ms done={_episodeDone}");
			}

			if (_episodeDone)
			{
				data.done = true;
				data.info = _episodeResult;
				data.combat_hitboxes = new List<float[]>();
				data.terrain_hitboxes = new List<float[]>();
				data.global_state = new float[StateExtractor.GlobalStateDim];
				data.fsm_snapshots = new List<string>();
				SendMessage(new Message { type = "step", data = data });
				yield break;
			}

			// Build observation
			var obs = _hitboxObserver.GetSplitFeatures(_bossHMs, emitTerrainDebug: _evalMode);
			var gs = StateExtractor.GetGlobalState(
				obs.KnightWidth, obs.KnightHeight, _inputShim);

			data.combat_hitboxes = obs.CombatHitboxes;
			data.combat_kinds = obs.CombatKinds;
			data.combat_parents = obs.CombatParents;
			data.combat_anims = obs.CombatAnims;
			data.terrain_hitboxes = obs.TerrainHitboxes;
			data.terrain_debug = obs.TerrainDebug;
			data.global_state = gs;
			data.fsm_snapshots = SnapshotFsms();
			data.done = false;

			SendMessage(new Message { type = "step", data = data });
			yield break;
		}

		// Snapshot every FSM relevant to the current fight: every PlayMakerFSM
		// in the subtree of each tracked boss, plus every FSM on currently-
		// active Enemy-class colliders (catches in-flight boss projectiles
		// whose pool reparented them out of the boss subtree), plus every
		// FSM on Attack-class colliders (the knight's nail / spell hitboxes,
		// for cross-referencing the agent's action against attack windows).
		// Routed to the Python visualizer / fsm_tracker only — never consumed
		// by training.
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
			}
			return result;
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
		}

		private void LogPhase(string scope, string label)
		{
			float now = Time.realtimeSinceStartup;
			float deltaMs = (now - _phaseLast) * 1000f;
			float totalMs = (now - _phaseStart) * 1000f;
			_phaseLast = now;
			_phaseLastFrame = Time.frameCount;
			Log($"[Phase-Timing] {scope}#{_resetCount} {label}: +{deltaMs:F0}ms (total {totalMs:F0}ms)");
		}

		// LogPhase variant that also accumulates the elapsed wall + frame count
		// into the per-reset phase telemetry shipped on the reset message.
		// `phaseKey` must match one of Net.ResetPhase.Keys. The human-readable
		// `label` is kept distinct so the on-disk log can still carry the
		// original branch-specific suffix (e.g. "NATURAL-END (pre=X post=Y)")
		// while the wire format uses the canonical bucket key.
		private void LogResetPhase(string phaseKey, string label)
		{
			float now = Time.realtimeSinceStartup;
			int frameNow = Time.frameCount;
			float deltaMs = (now - _phaseLast) * 1000f;
			int deltaFrames = frameNow - _phaseLastFrame;
			float totalMs = (now - _phaseStart) * 1000f;
			_phaseLast = now;
			_phaseLastFrame = frameNow;
			int idx = Net.ResetPhase.IndexOf(phaseKey);
			if (idx >= 0)
			{
				_resetPhaseMs[idx] += deltaMs;
				int frames = _resetPhaseFrames[idx] + deltaFrames;
				_resetPhaseFrames[idx] = (ushort)(frames > ushort.MaxValue ? ushort.MaxValue : frames);
			}
			Log($"[Phase-Timing] Reset#{_resetCount} {label}: +{deltaMs:F0}ms (total {totalMs:F0}ms)");
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

			// Uncap fps; rely on Time.captureDeltaTime (pinned in Reset) to
			// pin dt independently of real frame rate. Capture mode is the
			// canonical-dt facility Unity exposes — see the constants block.
			QualitySettings.vSyncCount = 0;
			Application.targetFrameRate = -1;

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

			// Fake-reset clamping prob: env-var-driven so the same C# build
			// can be used with or without it. Range [0,1]; 0 disables. Python
			// sets this when boss_rotation_period > 1 (cluster mode keeps the
			// agent on the same boss across N fake resets).
			_fakeResetProb = 0f;
			var fakeProbStr = System.Environment.GetEnvironmentVariable("FK_FAKE_RESET_PROB");
			if (!string.IsNullOrEmpty(fakeProbStr)
				&& float.TryParse(fakeProbStr, System.Globalization.NumberStyles.Float,
					System.Globalization.CultureInfo.InvariantCulture, out float parsedProb))
			{
				_fakeResetProb = Mathf.Clamp01(parsedProb);
			}
			Log($"[Setup] fakeResetProb={_fakeResetProb:F3} (FK_FAKE_RESET_PROB={fakeProbStr ?? "unset"})");

			On.GameManager.SaveGame += SaveFileProxy.DisableSaveGame;
			SaveFileProxy.LoadCompletedSave();
			GameManager.instance.ContinueGame();
			yield return new SceneHooks.WaitForSceneLoad("GG_Workshop");
			yield return new SceneHooks.WaitForEntryFinished();
			yield return new WaitForSeconds(2f);

			_hitboxObserver.Load();
			InputManager.AttachDevice(_inputShim);
			SendMessage(message);
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
			_timeManager?.Dispose();
			InputManager.DetachDevice(_inputShim);
			_hitboxObserver.Unload();
			CloseSocket();
			yield break;
		}

		private static IEnumerator EmptyCoroutine() { yield break; }

		private void HookDamage()
		{
			ModHooks.AfterTakeDamageHook += OnKnightDamaged;
			On.HealthManager.TakeDamage += OnBossDamaged;
			On.HeroController.TakeDamage += OnHeroTakeDamage;
		}

		private void UnhookDamage()
		{
			ModHooks.AfterTakeDamageHook -= OnKnightDamaged;
			On.HealthManager.TakeDamage -= OnBossDamaged;
			On.HeroController.TakeDamage -= OnHeroTakeDamage;
		}

		// AfterTakeDamageHook fires past HeroController.TakeDamage's iframe
		// short-circuit, so it only ticks on hits that actually land — using
		// TakeDamageHook here counted iframe-blocked contacts and inflated
		// hits_taken ~16×, wrecking the reward signal. Return value replaces
		// the applied damage; pass `damage` through unchanged.
		//
		// Accumulates HP lost, not hit count. hp_healed is measured in raw HP
		// (PlayerData.health delta), so counting hits here put the two halves of
		// the reward on different scales: against a boss dealing 2 masks a hit,
		// -1 for the hit against +2*heal_coef for healing it back made getting
		// hit and healing net positive. Both sides are now HP.
		private int OnKnightDamaged(int damageType, int damage)
		{
			if (!_syntheticKill && damage > 0) _hitsTakenInStep += damage;
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

			bool wouldDie = self.hp - hitInstance.DamageDealt <= 0;

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
			// _bossDied is no longer set here. Death detection moved to
			// OnBossActualDeath, subscribed to each tracked HM.OnDeath in
			// InitBossRefs. False Knight stagger restores hp via a path
			// that bypasses OnDeath, so OnDeath is the only signal that
			// aligns with HK's own "boss truly gone" judgment.
		}

		// OnDeath handler — bound per HM via a closure in InitBossRefs. Fires
		// when HK's Die() reaches SendDeathEvent (i.e., the phase-restore path
		// did NOT intercept). Matches the signal HK's own BossSceneController
		// uses to end the scene, so we can never trip _bossDied earlier than
		// HK itself.
		private void OnBossActualDeath(HealthManager hm)
		{
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

		// Wrap HeroController.TakeDamage so we can intercept lethal damage
		// before it commits. The orig() inside HC.TakeDamage handles iframes,
		// hazard types, recoil, and AfterTakeDamageHook firing — so simply
		// reducing the damage parameter and calling orig gives a "the knight
		// took N points" outcome with all hooks consistent.
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
					orig(self, go, damageSide, safeDmg, hazardType);
					RestoreFightHPs();
					_fakeResetPending = true;
					return;
				}
			}

			orig(self, go, damageSide, damageAmount, hazardType);
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
			yield return new SceneHooks.WaitForEntryFinished();
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
