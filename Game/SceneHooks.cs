using System;
using System.Collections;
using GlobalEnums;
using HutongGames.PlayMaker;
using UnityEngine;

namespace FullKnight.Game
{
	public static class SceneHooks
	{
		/// <summary>
		/// Loads a boss from the Hall of Gods given the scene name.
		///
		/// Canonical path: drive the statue's GG Boss UI FSM the way a player
		/// would. Send CONVO START -> the FSM spawns BossChallengeUI -> we call
		/// LoadBoss(level, false) on it, which fires OnLevelSelected -> FSM's
		/// DREAM event -> Take Control -> Set Facing -> Challenge -> Impact ->
		/// Dream Box Down -> Transition (BeginSceneTransition) -> Reset Player
		/// -> Change Scene. HK does all the state setup (bossSceneToLoad,
		/// RecordBossScene, bossReturnEntryGate PD, SetupEvent with statue-
		/// completion PD writes, GG visualization, HUD restore on return).
		/// </summary>
		public static IEnumerator LoadBossScene(string scene_name)
		{
			BossStatue statue = null;

			void Log(string s)
			{
				var fsmActive = statue?.bossUIControlFSM?.Fsm?.ActiveStateName ?? "(no statue)";
				FullKnight.Instance.Log($"[CanonicalLoad] {s} | scene="
					+ UnityEngine.SceneManagement.SceneManager.GetActiveScene().name
					+ $" fsmState={fsmActive}"
					+ $" t={Time.realtimeSinceStartup:F2}");
			}

			Log("ENTER target=" + scene_name);

			// Bounce only when we're not already in the workshop. After a
			// boss-kill we naturally arrive in GG_Workshop via DoDreamReturn,
			// so the original conditional is sufficient; bouncing on top of
			// dream-return causes a workshop->atrium auto-transition.
			string currentScene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
			if (currentScene != "GG_Workshop")
			{
				Log("BOUNCE-START from=" + currentScene);
				yield return BounceThroughWorkshop();
				Log("BOUNCE-END");
			}

			// Always wait for the workshop entry coroutine to finish before
			// firing CONVO START. On the bounce path BounceThroughWorkshop()
			// already waited, so this is a no-op. On the non-bounce path
			// (Reset#2+ where HK's own DoDreamReturn brought us here), the
			// lie-down/wake-up entry is still in flight when we start; the
			// FSM's Challenge state stalls because the hero's animator is
			// still mid-entry (transitionState=WAITING_TO_ENTER_LEVEL).
			yield return new WaitForEntryFinished();
			Log("entry-coroutine settled");

			// Reset#1 only: knight is sitting on RestBench (1) (from
			// save_file.json) and the workshop intro cutscene is mid-
			// progression. Sending CONVO START while the cutscene is
			// running causes the bossUI to skip Open UI -> Take Control
			// and instead auto-cancel to Close UI (the godseeker's own
			// dialogue FSM likely intercepts). Drive a jump-tap until the
			// bench FSM stands the knight up, so CONVO START fires
			// against a stable, fully-controlled hero. No-op when
			// pd.atBench is already false (every reset after the first,
			// which comes back via dream-return, not via bench).
			yield return LeaveBenchIfSitting();
			Log("bench-leave settled");

			// Drive HK's 'Return from boss' wake-up FSM to its terminal
			// state. Without this, leaving GG_Workshop while the FSM is
			// mid-sequence skips its cleanup actions (renderer re-enable,
			// anim control restart, accept-input restore), and the next
			// death's wake-up FSM restarts on dirty HC state and stalls.
			// On Reset#1 (no prior death) no FSM is in active state and
			// this returns immediately.
			yield return DriveDreamReturnWakeUp();
			Log("wake-up FSM settled");

			// Find the statue + tier (0/1/2) + dream-flag that owns scene_name.
			// Search BOTH structs (bossScene = regular variant, dreamBossScene
			// = harder dream variant), then flip the statue's
			// UsingDreamVersion to match whichever struct contained the
			// requested name. BossChallengeUI.LoadBoss reads UsingDreamVersion
			// at call time (BossChallengeUI.cs:237), so as long as the flag
			// is right just before LoadBoss runs, the correct BossScene is
			// used. Callers can pass either GG_False_Knight (regular) or
			// GG_Failed_Champion (dream) and the right variant loads.
			int level = 0;
			bool wantDream = false;
			foreach (var s in UnityEngine.Object.FindObjectsOfType<BossStatue>())
			{
				if (s.bossScene != null)
				{
					var bs = s.bossScene;
					if (bs.Tier1Scene == scene_name) { statue = s; level = 0; wantDream = false; goto found; }
					if (bs.Tier2Scene == scene_name) { statue = s; level = 1; wantDream = false; goto found; }
					if (bs.Tier3Scene == scene_name) { statue = s; level = 2; wantDream = false; goto found; }
				}
				if (s.dreamBossScene != null)
				{
					var bs = s.dreamBossScene;
					if (bs.Tier1Scene == scene_name) { statue = s; level = 0; wantDream = true; goto found; }
					if (bs.Tier2Scene == scene_name) { statue = s; level = 1; wantDream = true; goto found; }
					if (bs.Tier3Scene == scene_name) { statue = s; level = 2; wantDream = true; goto found; }
				}
			}
		found:

			// Flip the statue's dream-version flag if it doesn't match what
			// the caller asked for. UsingDreamVersion has a private set, but
			// its public getter reads StatueState.usingAltVersion and the
			// public StatueState setter writes the whole Completion struct
			// back to PlayerData; we round-trip through that.
			if (statue != null && statue.UsingDreamVersion != wantDream)
			{
				var state = statue.StatueState;
				state.usingAltVersion = wantDream;
				statue.StatueState = state;
				Log($"flipped UsingDreamVersion -> {wantDream}");
			}

			// Post-fight, this statue's bossUIControlFSM is in its terminal
			// Change Scene state with Fsm.Finished=true; global transitions
			// (CONVO START) no longer fire. Reinitialize() reloads the FSM
			// data and Start() fires the lifecycle into the Inert start state,
			// clearing Finished. One yield lets the state's OnEnter actions
			// settle before we send CONVO START.
			if (statue?.bossUIControlFSM?.Fsm?.Finished == true)
			{
				statue.bossUIControlFSM.Fsm.Reinitialize();
				statue.bossUIControlFSM.Fsm.Start();
				yield return null;
			}
			if (statue == null)
			{
				Log("ERROR: no statue matches " + scene_name);
				yield break;
			}
			Log($"FOUND statue={statue.gameObject.name} level={level}");
			LogHero("pre-CONVO");

			// One-shot topology dump of the boss UI control FSM so we can
			// read its Challenge state's actions and transition events.
			// Reset#1 advances Challenge -> Challenge Audio in 20ms (event
			// driven, not timer). Reset#2 stalls at Challenge; we need to
			// see which action's expected event isn't firing.
			DumpWakeFsm(statue.bossUIControlFSM, "bossUI");

			// Parallel diagnostic: log every FSM state change + Finished flip
			// in the boss UI control FSM. Lets us pinpoint exactly which state
			// stalls when the canonical chain doesn't reach BeginSceneTransition.
			GameManager.instance.StartCoroutine(LogFsmTransitions(statue.bossUIControlFSM, "bossUI", 30f));
			// Heartbeat HC state every second while the bossUI FSM is
			// running. If Challenge stalls we want to see whether HC is
			// still mid-animation, whether position has changed, etc.
			GameManager.instance.StartCoroutine(LogHcWhileFsmRuns(statue.bossUIControlFSM, "bossUI", 30f));

			// Suspend HC.animCtrl so the bossUI's Challenge state can play
			// 'Challenge Start' uninterrupted. After Dream Return wakes the
			// knight fully (onGround=True, accepting=True), HK's
			// HeroAnimationController writes Idle to the animator every
			// frame, racing the FSM's Tk2dPlayAnimation. Tk2dWatchAnimationEvents
			// then never sees the FINISHED frame-event of Challenge Start
			// (because Challenge Start never actually plays past frame 0)
			// and bossUI stalls at Challenge forever.
			// On the natural player path this isn't an issue: the player
			// is mid-intro-cutscene (onGround=False, anim already scripted)
			// so HeroAnimationController doesn't fight the FSM.
			// HK's boss-arena dream-warp entry (Dream Return -> Regain
			// Control -> ActivateGameObject) re-enables animCtrl on the
			// next scene, so we don't need to restart it ourselves.
			HeroController.instance?.StopAnimationControl();
			Log("StopAnimationControl called on HC");

			// CONVO START is the FSM's global transition into Open UI. Open UI
			// runs Tk2dPlayAnimation + ShowBossChallengeUI + ActivateGameObject
			// synchronously. ShowBossChallengeUI instantiates the panel and
			// hooks panel.OnCancel -> Fsm.Event("FINISHED") and
			// panel.OnLevelSelected -> Fsm.Event("DREAM").
			//
			// CRITICAL RACE: BossChallengeUI.LoadBoss(level, false) nulls
			// OnCancel BEFORE calling Hide, then fires OnLevelSelected. So
			// the FSM goes Open UI -> Take Control via DREAM only if our
			// LoadBoss runs BEFORE anything else invokes OnCancel. Under
			// multi-instance contention, the panel's own spawn coroutine
			// (Select() / StartUIInput()) can fail and auto-cancel within
			// 1-2 frames; if we yield first, we lose the race and the FSM
			// collapses Open UI -> Close UI (terminal, no further trans).
			//
			// Fix: try FindObjectOfType immediately after SendEvent (no
			// yield) — ShowBossChallengeUI Instantiates the panel synchronously
			// so it should be findable in the same frame. If not, yield once
			// and retry. Either way, call LoadBoss as soon as the panel is
			// found, before any cancel can race ahead.
			// SendEvent can throw if ShowBossChallengeUI / ActivateGameObject
			// (Open UI / Close UI actions) hits a null reference. Without
			// catching, the exception kills this coroutine before we can
			// log the post-state, leaving the FSM stranded at Close UI
			// and the reset hung forever. Retry once via Reinitialize +
			// re-send; that gives a fresh shot at the spawn race that
			// causes ShowBossChallengeUI to auto-cancel.
			BossChallengeUI panel = null;
			for (int attempt = 0; attempt < 3 && panel == null; attempt++)
			{
				string preSendState = statue.bossUIControlFSM.Fsm?.ActiveStateName ?? "(null)";
				FullKnight.Instance.Log($"[CanonicalLoad] pre-SendEvent CONVO START attempt={attempt} state={preSendState}");
				System.Exception sendEx = null;
				try
				{
					statue.bossUIControlFSM.SendEvent("CONVO START");
				}
				catch (System.Exception e)
				{
					sendEx = e;
				}
				string postSendState = statue.bossUIControlFSM.Fsm?.ActiveStateName ?? "(null)";
				bool postFinished = statue.bossUIControlFSM.Fsm?.Finished ?? false;
				if (sendEx != null)
				{
					FullKnight.Instance.Log($"[CanonicalLoad] SendEvent attempt={attempt} threw {sendEx.GetType().Name}: {sendEx.Message}");
					FullKnight.Instance.Log($"[CanonicalLoad] StackTrace: {sendEx.StackTrace}");
				}
				FullKnight.Instance.Log($"[CanonicalLoad] post-SendEvent attempt={attempt} state={postSendState} finished={postFinished}");

				panel = UnityEngine.Object.FindObjectOfType<BossChallengeUI>();
				if (panel == null)
				{
					FullKnight.Instance.Log($"[CanonicalLoad] attempt={attempt} panel not same-frame; yielding once and retrying find");
					yield return null;
					panel = UnityEngine.Object.FindObjectOfType<BossChallengeUI>();
				}
				if (panel != null) break;

				// Panel never spawned (or spawned and was destroyed before
				// we found it). bossUI almost certainly collapsed
				// Inert -> Open UI -> Close UI synchronously. Reinitialize
				// and retry.
				FullKnight.Instance.Log($"[CanonicalLoad] attempt={attempt} retry path: panel missing, fsm at {postSendState} finished={postFinished}; reinitializing");
				try
				{
					statue.bossUIControlFSM.Fsm.Reinitialize();
					statue.bossUIControlFSM.Fsm.Start();
				}
				catch (System.Exception e)
				{
					FullKnight.Instance.Log($"[CanonicalLoad] reinit threw {e.GetType().Name}: {e.Message}");
				}
				yield return null;
			}
			if (panel == null)
			{
				Log("ERROR: BossChallengeUI panel not found after 3 CONVO START attempts");
				FullKnight.Instance.Log($"[CanonicalLoad] panel-missing final-diag "
					+ $"fsmState={statue.bossUIControlFSM.Fsm?.ActiveStateName} "
					+ $"finished={statue.bossUIControlFSM.Fsm?.Finished}");
				yield break;
			}
			Log($"FOUND panel={panel.gameObject.name}");

			// LoadBoss is the canonical "press the button" function. It sets
			// bossSceneToLoad, calls RecordBossScene, swaps bossReturnEntryGate,
			// installs the full SetupEvent delegate (including OnBossesDead PD
			// writes for statue completion + OnBossSceneComplete -> DoDreamReturn),
			// then fires OnLevelSelected. doHideAnim:false skips the hide animation.
			// Call IMMEDIATELY (no yield between FindObjectOfType and LoadBoss)
			// to beat the panel's auto-cancel coroutine.
			string preLoadBossState = statue.bossUIControlFSM.Fsm?.ActiveStateName ?? "(null)";
			FullKnight.Instance.Log($"[CanonicalLoad] pre-LoadBoss state={preLoadBossState}");
			panel.LoadBoss(level, doHideAnim: false);
			string postLoadBossState = statue.bossUIControlFSM.Fsm?.ActiveStateName ?? "(null)";
			Log($"after LoadBoss return (fsmState={postLoadBossState})");
			LogHero("post-LoadBoss");

			// The FSM now drives knight animation (Take Control -> Challenge ->
			// Impact -> Dream Box Down) then BeginSceneTransition in Transition
			// state. Total ~2-3s of in-game animation. Poll for active scene.
			yield return new WaitForSceneLoad(scene_name);
			Log("SCENE LOADED");
			yield return new WaitForEntryFinished();
			Log("DONE");
		}

		// Poll the FSM every frame and emit one log line per state change
		// (and once at start). Stops when the FSM Finishes or timeoutSec
		// elapses. Runs in parallel with LoadBossScene so we don't have to
		// instrument every yield point.
		private static IEnumerator LogFsmTransitions(PlayMakerFSM pmFsm, string tag, float timeoutSec)
		{
			if (pmFsm == null || pmFsm.Fsm == null) yield break;
			float t0 = Time.realtimeSinceStartup;
			string last = "<init>";
			bool lastFinished = false;
			while (Time.realtimeSinceStartup - t0 < timeoutSec)
			{
				var fsm = pmFsm.Fsm;
				if (fsm == null) yield break;
				string cur = fsm.ActiveStateName ?? "(null)";
				bool fin = fsm.Finished;
				if (cur != last || fin != lastFinished)
				{
					FullKnight.Instance.Log(
						$"[CanonicalLoad/{tag}] state '{last}' -> '{cur}' (finished {lastFinished}->{fin})"
						+ $" t={Time.realtimeSinceStartup:F2}");
					last = cur;
					lastFinished = fin;
					if (fin) yield break;
				}
				yield return null;
			}
			FullKnight.Instance.Log($"[CanonicalLoad/{tag}] timeout after {timeoutSec}s, last state='{last}' finished={lastFinished}");
		}

		// Snapshot HeroController state at a checkpoint. Tells us whether the
		// knight is in lie-down / no-input / cutscene mode when the FSM tries
		// to take control; the prime suspect for the canonical chain stalling.
		private static void LogHero(string tag)
		{
			var hc = HeroController.instance;
			if (hc == null) { FullKnight.Instance.Log($"[CanonicalLoad/hero/{tag}] HeroController.instance=null"); return; }
			string transitionState = "?";
			try { transitionState = hc.transitionState.ToString(); } catch { }
			bool accepting = false;
			try { accepting = hc.acceptingInput; } catch { }
			bool active = hc.gameObject.activeInHierarchy;
			Vector3 pos = hc.transform.position;
			string rendererInfo = "?";
			try {
				var r = hc.GetComponentInChildren<Renderer>(includeInactive: true);
				if (r == null) rendererInfo = "noRenderer";
				else
				{
					string chain = $"{r.gameObject.name}.enabled={r.enabled} selfActive={r.gameObject.activeSelf}";
					var t = r.transform.parent;
					while (t != null && t != hc.transform.parent)
					{
						chain = $"{t.gameObject.name}.selfActive={t.gameObject.activeSelf} > " + chain;
						t = t.parent;
					}
					rendererInfo = chain;
				}
			} catch (System.Exception e) { rendererInfo = $"err:{e.Message}"; }
			var cs = hc.cState;
			string csInfo = "?";
			if (cs != null)
			{
				csInfo = $"facingRight={cs.facingRight} onGround={cs.onGround} dead={cs.dead}"
					+ $" hazardDeath={cs.hazardDeath} transitioning={cs.transitioning}"
					+ $" recoiling={cs.recoiling} invulnerable={cs.invulnerable}";
			}
			string animState = "?";
			try {
				var animCtrl = hc.GetComponentInChildren<tk2dSpriteAnimator>();
				if (animCtrl != null && animCtrl.CurrentClip != null)
					animState = animCtrl.CurrentClip.name;
			} catch { }
			FullKnight.Instance.Log(
				$"[CanonicalLoad/hero/{tag}] hcActive={active} renderer=[{rendererInfo}]"
				+ $" pos=({pos.x:F1},{pos.y:F1}) anim={animState}"
				+ $" acceptingInput={accepting} transitionState={transitionState} {csInfo}");
		}

		private static IEnumerator BounceThroughWorkshop()
		{
			var GM = GameManager.instance;
			float t0 = Time.realtimeSinceStartup;
			GM.BeginSceneTransition(new GameManager.SceneLoadInfo
			{
				SceneName = "GG_Workshop",
				EntryGateName = "door_dreamReturn",
				EntryDelay = 0,
				Visualization = GameManager.SceneLoadVisualizations.GodsAndGlory,
				PreventCameraFadeOut = true,
				WaitForSceneTransitionCameraFade = false,
			});
			float t1 = Time.realtimeSinceStartup;
			yield return new WaitForSceneLoad("GG_Workshop");
			float t2 = Time.realtimeSinceStartup;
			yield return new WaitForEntryFinished();
			float t3 = Time.realtimeSinceStartup;
			FullKnight.Instance.Log(
				$"[Phase-Timing] BounceThroughWorkshop: BeginTransition={(t1 - t0) * 1000f:F0}ms"
				+ $" WaitForSceneLoad={(t2 - t1) * 1000f:F0}ms"
				+ $" WaitForFinishedEnteringScene={(t3 - t2) * 1000f:F0}ms"
				+ $" total={(t3 - t0) * 1000f:F0}ms");
		}

		// Reset#1 entry path: save_file.json sets respawnMarkerName=
		// "RestBench (1)" and atBench=true, so the knight loads sitting on
		// the godseeker bench mid-cutscene. We need them standing on the
		// ground (accepting=True, anim=Idle) before sending CONVO START,
		// or the bossUI's Open UI auto-cancels to Close UI under the
		// godseeker's dialogue FSM. Same as a player pressing Jump to
		// leave a bench: the bench FSM listens for input, plays the
		// stand-up animation, clears atBench, restores control. We tap
		// shim Jump+Attack (3-on / 3-off so InControl sees real false->
		// true edges) until pd.atBench=false AND HC.acceptingInput=true.
		// Returns immediately when pd.atBench is already false (Reset#2+
		// arrives via dream-return, not via bench).
		private static IEnumerator LeaveBenchIfSitting(float timeoutSec = 30f)
		{
			var pd = GameManager.instance?.playerData;
			var hc = HeroController.instance;
			if (pd == null || hc == null)
			{
				FullKnight.Instance.Log("[LeaveBench] no PD/HC; skipping");
				yield break;
			}
			bool startedOnBench = false;
			try { startedOnBench = pd.atBench; } catch { }
			if (!startedOnBench)
			{
				FullKnight.Instance.Log("[LeaveBench] pd.atBench=false; nothing to leave");
				yield break;
			}
			var shim = InputDeviceShim.Attached;
			if (shim == null)
			{
				FullKnight.Instance.Log("[LeaveBench] no shim attached; cannot tap");
				yield break;
			}

			FullKnight.Instance.Log("[LeaveBench] tapping Jump+Attack until knight stands");
			float t0 = Time.realtimeSinceStartup;
			int frame = 0;
			bool keysDown = false;
			int lastSec = -1;

			while (Time.realtimeSinceStartup - t0 < timeoutSec)
			{
				bool stillBench = true;
				bool accepting = false;
				bool onGround = false;
				try { stillBench = pd.atBench; } catch { }
				try { accepting = hc.acceptingInput; } catch { }
				try { onGround = hc.cState != null && hc.cState.onGround; } catch { }

				// Standing = bench-leave anim completed (pd cleared atBench)
				// AND HC has input back AND we're grounded. All three usually
				// flip together at the tail of the bench-stand-up animation.
				if (!stillBench && accepting && onGround) break;

				int sec = (int)(Time.realtimeSinceStartup - t0);
				if (sec != lastSec)
				{
					lastSec = sec;
					string anim = "?";
					try
					{
						var ac = hc.GetComponentInChildren<tk2dSpriteAnimator>();
						if (ac != null && ac.CurrentClip != null) anim = ac.CurrentClip.name;
					}
					catch { }
					Vector3 p = hc.transform.position;
					FullKnight.Instance.Log(
						$"[LeaveBench/sec={sec}] atBench={stillBench} accepting={accepting}"
						+ $" onGround={onGround} anim={anim} pos=({p.x:F1},{p.y:F1})");
				}

				bool wantDown = (frame % 6) < 3;
				if (wantDown != keysDown)
				{
					shim.WakeTap(wantDown);
					keysDown = wantDown;
				}
				frame++;
				yield return null;
			}
			shim.WakeTap(false);

			bool finalBench = true;
			bool finalAccepting = false;
			try { finalBench = pd.atBench; } catch { }
			try { finalAccepting = hc.acceptingInput; } catch { }
			FullKnight.Instance.Log(
				$"[LeaveBench] exit atBench={finalBench} accepting={finalAccepting}"
				+ $" elapsed={(Time.realtimeSinceStartup - t0) * 1000f:F0}ms");
		}

		// Two FSMs run in parallel after a death-return to GG_Workshop:
		//   1. 'Return from boss' on door_dreamReturn_* GameObject —
		//      Pause -> Door Entry -> Wait -> Transition Wait ->
		//      Transition In -> (DREAM WAKE) -> Get Up Wait -> Gotten Up.
		//      Terminal states (Gotten Up / Not Returning) have no
		//      outgoing transitions; FSM parks there without setting
		//      Finished=true.
		//   2. 'Dream Return' on HC — Idle -> ... -> Ready -> (GET UP) ->
		//      Get Up -> Save -> Regain Control -> Idle. HC is DDOL so
		//      its FSM state persists across scenes.
		// The door's DREAM WAKE event is fired by HC.Dream Return's
		// 'Get Up' state (SendEventByName), so HC has to reach Get Up
		// for the door to advance. And HC needs input — Ready listens
		// via ListenForJump/Attack/Down/Up/L/R and fires GET UP.
		//
		// Critical: exit must wait for HC.Dream Return back to Idle.
		// Exiting earlier (e.g. when door reaches Gotten Up but HC is
		// still mid-Get-Up) interrupts HC's animation; Save and Regain
		// Control never run; HC stays parked at Get Up across the boss
		// scene. On the next death's return, LEVEL LOADED only triggers
		// from Idle, so HC.Dream Return never restarts — both FSMs
		// deadlock and the wake-up FSM stalls forever.
		private static IEnumerator DriveDreamReturnWakeUp(float timeoutSec = 6f)
		{
			var shim = InputDeviceShim.Attached;

			// Inventory ALL 'Return from boss' FSMs so we can confirm
			// only one is active (the one belonging to the boss we just
			// died to); others should sit at Pause.
			int total = 0;
			PlayMakerFSM driving = null;
			string drivingDoor = null;
			var inventoryStates = new System.Collections.Generic.List<string>();
			foreach (var fsm in UnityEngine.Object.FindObjectsOfType<PlayMakerFSM>())
			{
				if (fsm == null || fsm.Fsm == null) continue;
				if (fsm.FsmName != "Return from boss") continue;
				total++;
				string st = fsm.Fsm.ActiveStateName ?? "(null)";
				inventoryStates.Add($"{fsm.gameObject.name}:{st}");
				if (string.IsNullOrEmpty(st)) continue;
				if (st == "Pause" || st == "Not Returning") continue;
				if (driving == null)
				{
					driving = fsm;
					drivingDoor = fsm.gameObject.name;
				}
			}
			FullKnight.Instance.Log(
				$"[WakeUp] inventory total={total} active={(driving == null ? "(none)" : drivingDoor)}");
			// Only dump non-Pause door states inline; the full list is too
			// noisy for routine runs.
			foreach (var s in inventoryStates)
			{
				if (s.EndsWith(":Pause") || s.EndsWith(":Not Returning")) continue;
				FullKnight.Instance.Log($"[WakeUp/inv-nonpause] {s}");
			}

			// Also locate the 'Dream Return' FSM on HC itself. HC is DDOL,
			// so this FSM's variables (esp. Dream Returning bool, queried
			// by HC.IsDreamReturning) persist across scenes. If they leak
			// state between resets, the door's FSM can stall waiting for
			// a transition that depends on them.
			PlayMakerFSM heroDreamReturnFsm = null;
			var hc = HeroController.instance;
			if (hc != null)
			{
				heroDreamReturnFsm = PlayMakerFSM.FindFsmOnGameObject(hc.gameObject, "Dream Return");
				if (heroDreamReturnFsm != null)
				{
					string ds = heroDreamReturnFsm.Fsm?.ActiveStateName ?? "(null)";
					bool isRet = false;
					try { isRet = hc.IsDreamReturning; } catch { }
					FullKnight.Instance.Log(
						$"[WakeUp/heroFsm] 'Dream Return' on HC state={ds} IsDreamReturning={isRet}");
				}
				else
				{
					FullKnight.Instance.Log("[WakeUp/heroFsm] 'Dream Return' FSM not found on HC");
				}
			}

			if (driving == null)
			{
				FullKnight.Instance.Log("[WakeUp] no active 'Return from boss' FSM; nothing to wake");
				yield break;
			}

			FullKnight.Instance.Log(
				$"[WakeUp] driving FSM on {drivingDoor} initialState={driving.Fsm.ActiveStateName}");

			// One-shot topology dump so we can read state names + transitions
			// + action class names. Without this we're guessing which event
			// drives 'Transition In' -> 'Get Up Wait'.
			DumpWakeFsm(driving, drivingDoor);
			if (heroDreamReturnFsm != null)
			{
				DumpWakeFsm(heroDreamReturnFsm, "HC.Dream Return");
			}

			// Parallel logger for the full timeout window so we see every
			// state change on the wake FSM even after our loop exits.
			GameManager.instance.StartCoroutine(
				LogFsmTransitions(driving, "wakeFsm", timeoutSec + 1f));
			if (heroDreamReturnFsm != null)
			{
				GameManager.instance.StartCoroutine(
					LogFsmTransitions(heroDreamReturnFsm, "wakeHcFsm", timeoutSec + 1f));
			}

			float t0 = Time.realtimeSinceStartup;
			int frame = 0;
			bool keysDown = false;
			int lastHcLogSecond = -1;

			while (Time.realtimeSinceStartup - t0 < timeoutSec)
			{
				var fsm = driving.Fsm;
				if (fsm == null) break;
				string state = fsm.ActiveStateName ?? "(null)";
				bool finished = fsm.Finished;

				string heroState = heroDreamReturnFsm?.Fsm?.ActiveStateName ?? "(none)";

				// Exit only when BOTH FSMs are quiescent:
				//   door: Gotten Up / Not Returning (terminal — no outgoing
				//         transitions) or Pause / Inert (pre-active).
				//   HC.Dream Return: Idle (the start state / true terminal).
				// If the hero FSM isn't present at all, fall back to just
				// the door condition.
				bool doorTerminal = (state == "Gotten Up" || state == "Not Returning"
					|| state == "Pause" || state == "Inert");
				bool heroTerminal = (heroDreamReturnFsm == null
					|| heroState == "Idle");
				if (finished) break;
				if (doorTerminal && heroTerminal) break;

				int sec = (int)(Time.realtimeSinceStartup - t0);
				if (sec != lastHcLogSecond)
				{
					lastHcLogSecond = sec;
					LogHcDuringWake(state, finished, sec, heroState);
				}

				// Only tap Jump+Attack while HC.Dream Return is parked at
				// Ready or Ready 2 — those are the ONLY states whose
				// actions (ListenForJump/Attack/...) consume our input
				// into a GET UP event. Tapping outside those states is
				// either no-op (HC ignores input) or harmful (HC has
				// regained input and our presses trigger real actions).
				bool needInput = (heroState == "Ready" || heroState == "Ready 2");
				bool wantDown = needInput && ((frame % 6) < 3);
				if (wantDown != keysDown && shim != null)
				{
					shim.WakeTap(wantDown);
					keysDown = wantDown;
				}
				frame++;
				yield return null;
			}

			if (shim != null) shim.WakeTap(false);

			string finalDoorState = driving?.Fsm?.ActiveStateName ?? "(null)";
			bool finalFinished = driving?.Fsm?.Finished ?? false;
			string finalHeroState = heroDreamReturnFsm?.Fsm?.ActiveStateName ?? "(none)";
			FullKnight.Instance.Log(
				$"[WakeUp] exit doorState={finalDoorState} finished={finalFinished}"
				+ $" heroFsmState={finalHeroState}"
				+ $" elapsed={(Time.realtimeSinceStartup - t0) * 1000f:F0}ms");
		}

		// One-shot dump of FSM topology so logs can be read against the
		// state-transition graph: state name + outgoing transitions + the
		// .NET type names of the action components attached to each state.
		// Knowing the action types reveals what the FSM is waiting on (e.g.
		// 'WaitForCameraFade', 'GetButtonDown', 'WaitForAnimationComplete').
		private static void DumpWakeFsm(PlayMakerFSM pmFsm, string tag)
		{
			var fsm = pmFsm?.Fsm;
			if (fsm == null) return;
			try
			{
				FullKnight.Instance.Log(
					$"[WakeUpFsm/{tag}] startState={fsm.StartState} active={fsm.ActiveStateName}"
					+ $" finished={fsm.Finished} stateCount={fsm.States?.Length}");

				var evs = new System.Text.StringBuilder();
				evs.Append($"[WakeUpFsm/{tag}/events]");
				if (fsm.Events != null)
				{
					for (int i = 0; i < fsm.Events.Length; i++)
						evs.Append(" ").Append(fsm.Events[i].Name);
				}
				FullKnight.Instance.Log(evs.ToString());

				var gt = new System.Text.StringBuilder();
				gt.Append($"[WakeUpFsm/{tag}/globalTrans]");
				if (fsm.GlobalTransitions != null)
				{
					for (int i = 0; i < fsm.GlobalTransitions.Length; i++)
					{
						var t = fsm.GlobalTransitions[i];
						gt.Append($" {t.EventName}->{t.ToState}");
					}
				}
				FullKnight.Instance.Log(gt.ToString());

				if (fsm.States != null)
				{
					foreach (var state in fsm.States)
					{
						if (state == null) continue;
						var sb = new System.Text.StringBuilder();
						sb.Append($"[WakeUpFsm/{tag}/state] {state.Name} trans=[");
						if (state.Transitions != null)
						{
							for (int i = 0; i < state.Transitions.Length; i++)
							{
								if (i > 0) sb.Append(", ");
								var tr = state.Transitions[i];
								sb.Append($"{tr.EventName}->{tr.ToState}");
							}
						}
						sb.Append("] actions=[");
						var acts = state.Actions;
						if (acts != null)
						{
							for (int i = 0; i < acts.Length; i++)
							{
								if (i > 0) sb.Append(", ");
								sb.Append(acts[i]?.GetType().Name ?? "null");
							}
						}
						sb.Append("]");
						FullKnight.Instance.Log(sb.ToString());
					}
				}
			}
			catch (System.Exception e)
			{
				FullKnight.Instance.Log($"[WakeUpFsm/{tag}] dump err: {e.Message}");
			}
		}

		// Parallel HC heartbeat: every ~1s while the supplied FSM is
		// still running, log HC's animator state, position, accept-input
		// flag, transition state, and cState. Lets us see whether HC
		// drifts during a long-running FSM (e.g. bossUI stuck at Challenge)
		// or stays frozen — both tell us what the FSM is waiting on.
		private static IEnumerator LogHcWhileFsmRuns(PlayMakerFSM pmFsm, string tag, float timeoutSec)
		{
			if (pmFsm == null) yield break;
			float t0 = Time.realtimeSinceStartup;
			int lastSec = -1;
			while (Time.realtimeSinceStartup - t0 < timeoutSec)
			{
				var fsm = pmFsm.Fsm;
				if (fsm == null || fsm.Finished) yield break;
				int sec = (int)(Time.realtimeSinceStartup - t0);
				if (sec != lastSec)
				{
					lastSec = sec;
					var hc = HeroController.instance;
					if (hc != null)
					{
						string anim = "?";
						try
						{
							var animCtrl = hc.GetComponentInChildren<tk2dSpriteAnimator>();
							if (animCtrl != null && animCtrl.CurrentClip != null)
								anim = animCtrl.CurrentClip.name;
						}
						catch { }
						bool accepting = false;
						try { accepting = hc.acceptingInput; } catch { }
						string ts = "?";
						try { ts = hc.transitionState.ToString(); } catch { }
						bool onGround = false; bool facingRight = false;
						try { onGround = hc.cState != null && hc.cState.onGround; } catch { }
						try { facingRight = hc.cState != null && hc.cState.facingRight; } catch { }
						Vector3 p = hc.transform.position;
						FullKnight.Instance.Log(
							$"[{tag}/hc-sec={sec}] fsmState={fsm.ActiveStateName}"
							+ $" pos=({p.x:F1},{p.y:F1}) anim={anim} accepting={accepting}"
							+ $" onGround={onGround} facingRight={facingRight} transitionState={ts}");
					}
				}
				yield return null;
			}
		}

		// Per-second HC snapshot during the wake loop. Compared across the
		// timeout window this reveals whether HC is being driven by the FSM
		// (renderer/anim/input change over time) or frozen (all values
		// constant — the FSM is at a state but its actions are no-ops).
		private static void LogHcDuringWake(string doorState, bool doorFinished, int sec, string heroFsmState)
		{
			var hc = HeroController.instance;
			if (hc == null)
			{
				FullKnight.Instance.Log($"[WakeUp/hc-tick={sec}] HC=null doorState={doorState}");
				return;
			}
			bool rendererEnabled = false;
			try
			{
				var r = hc.GetComponentInChildren<Renderer>(includeInactive: true);
				if (r != null) rendererEnabled = r.enabled;
			}
			catch { }
			string anim = "?";
			try
			{
				var animCtrl = hc.GetComponentInChildren<tk2dSpriteAnimator>();
				if (animCtrl != null && animCtrl.CurrentClip != null)
					anim = animCtrl.CurrentClip.name;
			}
			catch { }
			bool accepting = false;
			try { accepting = hc.acceptingInput; } catch { }
			string ts = "?";
			try { ts = hc.transitionState.ToString(); } catch { }
			bool isDreamRet = false;
			try { isDreamRet = hc.IsDreamReturning; } catch { }
			FullKnight.Instance.Log(
				$"[WakeUp/hc-tick={sec}] doorState={doorState} doorFinished={doorFinished}"
				+ $" heroFsmState={heroFsmState}"
				+ $" renderer={rendererEnabled} anim={anim} accepting={accepting}"
				+ $" transitionState={ts} IsDreamReturning={isDreamRet}");
		}

		private static void DumpStatueFSM(BossStatue statue)
		{
			if (statue == null || statue.bossUIControlFSM == null)
			{
				FullKnight.Instance.Log("[StatueFSM] no statue or no bossUIControlFSM");
				return;
			}
			var fsm = statue.bossUIControlFSM.Fsm;
			var sb = new System.Text.StringBuilder();
			sb.Append($"[StatueFSM] gameObject={statue.gameObject.name} fsmName={fsm.Name} active={fsm.ActiveStateName}");
			sb.Append(" | events=[");
			for (int i = 0; i < fsm.Events.Length; i++)
			{
				if (i > 0) sb.Append(", ");
				sb.Append(fsm.Events[i].Name);
			}
			sb.Append("] | globalTransitions=[");
			for (int i = 0; i < fsm.GlobalTransitions.Length; i++)
			{
				if (i > 0) sb.Append(", ");
				var t = fsm.GlobalTransitions[i];
				sb.Append($"{t.EventName}->{t.ToState}");
			}
			sb.Append("]");
			FullKnight.Instance.Log(sb.ToString());

			foreach (var state in fsm.States)
			{
				var ts = new System.Text.StringBuilder();
				ts.Append($"[StatueFSM]   state={state.Name} transitions=[");
				for (int i = 0; i < state.Transitions.Length; i++)
				{
					if (i > 0) ts.Append(", ");
					var tr = state.Transitions[i];
					ts.Append($"{tr.EventName}->{tr.ToState}");
				}
				ts.Append("]");
				FullKnight.Instance.Log(ts.ToString());
			}
		}

		// Replacement for the global WaitForFinishedEnteringScene type; the
		// global one is a PlayMaker FsmStateAction, NOT a CustomYieldInstruction,
		// so `yield return new WaitForFinishedEnteringScene()` silently
		// degenerated to a one-frame wait. This actually polls
		// GameManager.HasFinishedEnteringScene each frame.
		public class WaitForEntryFinished : CustomYieldInstruction
		{
			private int _polls;
			private float _t0;

			public WaitForEntryFinished()
			{
				_t0 = Time.realtimeSinceStartup;
			}

			public override bool keepWaiting
			{
				get
				{
					var gm = GameManager.instance;
					var hc = HeroController.instance;
					if (gm == null || hc == null) return false;
					// Two-gate wait. GM.HasFinishedEnteringScene fires too
					// early (set right after StartCoroutine(Respawn) on the
					// dream-return path; see GameManager.cs:1614-1616), so
					// alone it lets the FSM's Challenge state run on a hero
					// whose entry coroutine is still mid-animation. The
					// truthful "hero is idle and accepting events" signal is
					// HeroController.transitionState == WAITING_TO_TRANSITION
					// with cState.transitioning == false.
					bool gmReady = gm.HasFinishedEnteringScene;
					bool hcReady = false;
					try { hcReady = hc.transitionState == HeroTransitionState.WAITING_TO_TRANSITION
						&& hc.cState != null && !hc.cState.transitioning; }
					catch { }
					if (gmReady && hcReady) return false;
					_polls++;
					if (_polls == 1 || _polls % 60 == 0)
					{
						string ts = "?"; bool transitioning = false; bool accepting = false;
						try { ts = hc.transitionState.ToString(); } catch { }
						try { transitioning = hc.cState != null && hc.cState.transitioning; } catch { }
						try { accepting = hc.acceptingInput; } catch { }
						Vector3 p = hc.transform.position;
						FullKnight.Instance.Log(
							$"[WaitForEntryFinished] polls={_polls} elapsed={(Time.realtimeSinceStartup - _t0) * 1000f:F0}ms"
							+ $" gmReady={gmReady} hcReady={hcReady}"
							+ $" gmState={gm.gameState} hcTransitionState={ts} transitioning={transitioning}"
							+ $" acceptingInput={accepting} pos=({p.x:F1},{p.y:F1})");
					}
					return true;
				}
			}
		}

		public class WaitForSceneLoad : CustomYieldInstruction, IDisposable
		{
			private string sceneName;
			// Heartbeat counter so a stuck scene transition shows up in the
			// log instead of disappearing into a silent recv on the Python
			// side. Logged via FullKnight.Instance every 60 polls of
			// keepWaiting (Unity polls once per frame).
			private int _polls;
			private float _t0;

			public WaitForSceneLoad(string sn)
			{
				sceneName = sn;
				_t0 = Time.realtimeSinceStartup;
				UnityEngine.SceneManagement.SceneManager.activeSceneChanged += OnSceneEntered;
			}

			public void OnSceneEntered(UnityEngine.SceneManagement.Scene _, UnityEngine.SceneManagement.Scene scene)
			{
				// no-op, keepWaiting checks directly
			}

			public override bool keepWaiting
			{
				get
				{
					string active = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
					var gm = GameManager.instance;
					// Both gates must clear: the scene name flips during
					// sceneLoad.ActivationComplete, but gm.sceneLoad only
					// nulls in the later Finish callback. If the caller issues
					// a fresh BeginSceneTransition in that window, HK rejects
					// it ("Cannot scene transition while a scene transition is
					// in progress") and the next load never happens. Waiting
					// on IsInSceneTransition (cleared inside Finish, after
					// sceneLoad = null) closes that race.
					bool sceneReady = (active == sceneName);
					bool transitionDone = (gm == null || !gm.IsInSceneTransition);
					if (sceneReady && transitionDone) return false;
					_polls++;
					if (_polls % 60 == 0)
					{
						string state = "?";
						try { state = gm != null ? gm.gameState.ToString() : "?"; }
						catch { }
						FullKnight.Instance.Log(
							$"[WaitForSceneLoad] target={sceneName} active={active} "
							+ $"state={state} timeScale={Time.timeScale:F2} "
							+ $"inTransition={(gm != null && gm.IsInSceneTransition)} "
							+ $"polls={_polls} elapsed={(Time.realtimeSinceStartup - _t0) * 1000f:F0}ms");
					}
					return true;
				}
			}

			public void Dispose()
			{
				UnityEngine.SceneManagement.SceneManager.activeSceneChanged -= OnSceneEntered;
			}
		}
	}
}
