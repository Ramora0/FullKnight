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
			var HC = HeroController.instance;
			var GM = GameManager.instance;

			float t0 = Time.realtimeSinceStartup;
			float tLast = t0;
			void LogStage(string stage)
			{
				float now = Time.realtimeSinceStartup;
				FullKnight.Instance.Log(
					$"[Phase-Timing] LoadBossScene {stage}: +{(now - tLast) * 1000f:F0}ms"
					+ $" (total {(now - t0) * 1000f:F0}ms)");
				tLast = now;
			}

			// Boss-to-boss transitions are unreliable: the wake FSM can miss its
			// intro event (observed: Gruz Mother's 'Big Fly Control' stuck in 'Wake'
			// with the boss sleeping on the ceiling) on both same-scene reloads AND
			// cross-boss transitions. The only reliably working path seen so far is
			// GG_Workshop → boss (Setup's initial load). Always bounce through
			// GG_Workshop first unless we're already there.
			string currentScene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
			bool willBounce = currentScene != "GG_Workshop";
			FullKnight.Instance.Log(
				$"[SceneHooks] LoadBossScene target={scene_name} current={currentScene} bounce={willBounce}");
			if (willBounce)
			{
				yield return BounceThroughWorkshop();
				string postBounce = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
				FullKnight.Instance.Log($"[SceneHooks] bounce complete, now in {postBounce}");
				LogStage("BounceThroughWorkshop");
			}

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

			HC.ClearMPSendEvents();
			GM.TimePasses();
			GM.ResetSemiPersistentItems();
			HC.enterWithoutInput = true;
			HC.AcceptInput();

			GM.BeginSceneTransition(new GameManager.SceneLoadInfo
			{
				SceneName = scene_name,
				EntryGateName = "door_dreamEnter",
				EntryDelay = 0,
				Visualization = GameManager.SceneLoadVisualizations.GodsAndGlory,
				PreventCameraFadeOut = true,
				// Skip the fixed 0.5s camera-fade wait inside
				// GameManager.BeginSceneTransitionRoutine (decomp line 596).
				// The timer decrements unscaled so timeScale=20 doesn't help.
				WaitForSceneTransitionCameraFade = false,
			});
			LogStage("BeginSceneTransition (boss)");
			yield return FixSoul();
			LogStage("FixSoul");
			// Realtime: this settle wait gives HK wallclock time for the boss
			// FSM to finish waking. WaitForSeconds scales with Time.timeScale,
			// so cranking timeScale during reset would collapse it to ~100ms
			// at timeScale=20 and intro-skip would TIMEOUT (5001 frames =
			// 13.7s wasted on step 1) — empirical from commit ed8731f.
			// 0.667s = 2f / baseline timeScale=3, the magnitude that's been
			// proven sufficient.
			yield return new WaitForSecondsRealtime(0.667f);
			LogStage("WaitForSecondsRealtime(0.667)");
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

		private static IEnumerator FixSoul()
		{
			float t0 = Time.realtimeSinceStartup;
			yield return new WaitForFinishedEnteringScene();
			float t1 = Time.realtimeSinceStartup;
			yield return null;
			// Realtime — see LoadBossScene's note. 0.333s wallclock matches
			// the prior baseline (1f / timeScale=3) and is empirically enough
			// for HeroController to be ready for the soul-charge refresh.
			yield return new WaitForSecondsRealtime(0.333f);
			float t2 = Time.realtimeSinceStartup;
			HeroController.instance.AddMPCharge(1);
			HeroController.instance.AddMPCharge(-1);
			FullKnight.Instance.Log(
				$"[Phase-Timing] FixSoul: WaitForFinishedEnteringScene={(t1 - t0) * 1000f:F0}ms"
				+ $" WaitForSecondsRealtime(0.333)+frame={(t2 - t1) * 1000f:F0}ms"
				+ $" total={(t2 - t0) * 1000f:F0}ms");
		}

		public class WaitForSceneLoad : CustomYieldInstruction, IDisposable
		{
			private string sceneName;

			public WaitForSceneLoad(string sn)
			{
				sceneName = sn;
				UnityEngine.SceneManagement.SceneManager.activeSceneChanged += OnSceneEntered;
			}

			public void OnSceneEntered(UnityEngine.SceneManagement.Scene _, UnityEngine.SceneManagement.Scene scene)
			{
				// no-op, keepWaiting checks directly
			}

			public override bool keepWaiting =>
				UnityEngine.SceneManagement.SceneManager.GetActiveScene().name != sceneName;

			public void Dispose()
			{
				UnityEngine.SceneManagement.SceneManager.activeSceneChanged -= OnSceneEntered;
			}
		}
	}
}
