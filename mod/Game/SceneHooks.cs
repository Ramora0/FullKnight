using System;
using System.Collections;
using UnityEngine;

namespace FullKnight.Game
{
	public static class SceneHooks
	{
		/// <summary>
		/// Loads a boss from the Hall of Gods given the scene name.
		/// </summary>
		public static IEnumerator LoadBossScene(string scene_name)
		{
			var GM = GameManager.instance;

			void Stage(string s)
			{
				FullKnight.Instance.Log($"[LoadBossScene] {s}: scene="
					+ UnityEngine.SceneManagement.SceneManager.GetActiveScene().name
					+ $" t={Time.realtimeSinceStartup:F2}");
			}
			Stage("ENTER target=" + scene_name);

			// FIRST-PRINCIPLES SIMPLIFICATION: stripped this method down to the
			// minimum that "moves the knight back into the arena" requires:
			//   1. Bounce through GG_Workshop if not already there.
			//   2. Set DreamReturnScene + BossSceneController.SetupEvent so HK
			//      knows which boss to load and where to land on win.
			//   3. BeginSceneTransition + WaitForFinishedEnteringScene.
			// Removed (commented inline below): FSM event broadcasts (DREAM
			// ENTER, BOX DOWN DREAM, CONVO CANCEL, GG TRANSITION OUT),
			// HC.ClearMPSendEvents/TimePasses/ResetSemiPersistentItems/
			// enterWithoutInput/AcceptInput, FixSoul (MP refresh), and the
			// 0.667s realtime FSM-wake settle. If anything breaks, restore
			// those one at a time.

			string currentScene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
			if (currentScene != "GG_Workshop")
			{
				Stage("BOUNCE-START from=" + currentScene);
				yield return BounceThroughWorkshop();
				Stage("BOUNCE-END");
			}

			// FSM event broadcasts: required to wake HK's transition FSMs.
			// Removing "GG TRANSITION OUT" leaves the workshop's exit FSM
			// stuck mid-fade and HK ends up frozen on a white screen even
			// though the boss scene is technically loading. The other three
			// (DREAM ENTER, BOX DOWN DREAM, CONVO CANCEL) clear UI state
			// that would otherwise carry over and block input.
			PlayMakerFSM.BroadcastEvent("DREAM ENTER");
			PlayerData.instance.dreamReturnScene = "GG_Workshop";
			PlayMakerFSM.BroadcastEvent("BOX DOWN DREAM");
			PlayMakerFSM.BroadcastEvent("CONVO CANCEL");
			PlayMakerFSM.BroadcastEvent("GG TRANSITION OUT");
			BossSceneController.SetupEvent = (self) =>
			{
				StaticVariableList.SetValue("bossSceneToLoad", scene_name);
				self.BossLevel = 0;
				self.DreamReturnEvent = "DREAM RETURN";
				self.OnBossSceneComplete += () => self.DoDreamReturn();
			};

			// Restored: HC.enterWithoutInput + HC.AcceptInput. Without these,
			// HK's EnterHero (called from OnNextLevelReady once the boss
			// scene loads) waits forever for player input to advance, so
			// gameState stays at ENTERING_LEVEL and FinishedEnteringScene
			// never fires. Empirical: this was the "stuck in godhome"
			// freeze with `active=GG_Workshop state=ENTERING_LEVEL` heart-
			// beats from WaitForSceneLoad.
			var HC = HeroController.instance;
			HC.enterWithoutInput = true;
			HC.AcceptInput();
			// Still commented out — MP send events, time-passes ticks, and
			// semipersistent item resets haven't produced visible breakage
			// when omitted. Restore one at a time if a new symptom shows up.
			// HC.ClearMPSendEvents();
			// GM.TimePasses();
			// GM.ResetSemiPersistentItems();

			Stage("BEGIN-TRANSITION target=" + scene_name);
			GM.BeginSceneTransition(new GameManager.SceneLoadInfo
			{
				SceneName = scene_name,
				EntryGateName = "door_dreamEnter",
				EntryDelay = 0,
				Visualization = GameManager.SceneLoadVisualizations.GodsAndGlory,
				PreventCameraFadeOut = true,
				WaitForSceneTransitionCameraFade = false,
			});
			Stage("WAIT-SCENE-LOAD target=" + scene_name);
			// `new WaitForFinishedEnteringScene()` looks like it should wait,
			// but the only class with that name in HK's assembly is an
			// FsmStateAction — yielding it from a Unity coroutine just yields
			// for one frame. WaitForSceneLoad is a real CustomYieldInstruction
			// that polls SceneManager.activeScene.name; that's what actually
			// blocks until the boss scene is active.
			yield return new WaitForSceneLoad(scene_name);
			Stage("SCENE-LOAD-DONE");
			yield return new WaitForFinishedEnteringScene();  // 1-frame yield
			// Wake settle moved to TrainingEnv.Reset() — polls for active boss
			// colliders instead of a fixed wallclock wait.
			Stage("SETTLE-DONE");
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
				// See LoadBossScene's note — same 0.5s camera fade wait.
				WaitForSceneTransitionCameraFade = false,
			});
			float t1 = Time.realtimeSinceStartup;
			yield return new WaitForSceneLoad("GG_Workshop");
			float t2 = Time.realtimeSinceStartup;
			yield return new WaitForFinishedEnteringScene();
			float t3 = Time.realtimeSinceStartup;
			FullKnight.Instance.Log(
				$"[Phase-Timing] BounceThroughWorkshop: BeginTransition={(t1 - t0) * 1000f:F0}ms"
				+ $" WaitForSceneLoad={(t2 - t1) * 1000f:F0}ms"
				+ $" WaitForFinishedEnteringScene={(t3 - t2) * 1000f:F0}ms"
				+ $" total={(t3 - t0) * 1000f:F0}ms");
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
					// sceneLoad.ActivationComplete, but `gm.sceneLoad` only
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
