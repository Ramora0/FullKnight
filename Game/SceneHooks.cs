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
			}

			PlayMakerFSM.BroadcastEvent("DREAM ENTER");
			PlayerData.instance.dreamReturnScene = "GG_Workshop";
			PlayMakerFSM.BroadcastEvent("BOX DOWN DREAM");
			PlayMakerFSM.BroadcastEvent("CONVO CANCEL");
			PlayMakerFSM.BroadcastEvent("GG TRANSITION OUT");
			BossSceneController.SetupEvent = (self) =>
			{
				StaticVariableList.SetValue("bossSceneToLoad", scene_name);
				self.BossLevel = 1;
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
				PreventCameraFadeOut = true
			});
			yield return FixSoul();
			yield return new WaitForSeconds(2f);
		}

		private static IEnumerator BounceThroughWorkshop()
		{
			var GM = GameManager.instance;
			GM.BeginSceneTransition(new GameManager.SceneLoadInfo
			{
				SceneName = "GG_Workshop",
				EntryGateName = "door_dreamReturn",
				EntryDelay = 0,
				Visualization = GameManager.SceneLoadVisualizations.GodsAndGlory,
				PreventCameraFadeOut = true
			});
			yield return new WaitForSceneLoad("GG_Workshop");
			yield return new WaitForFinishedEnteringScene();
		}

		private static IEnumerator FixSoul()
		{
			yield return new WaitForFinishedEnteringScene();
			yield return null;
			yield return new WaitForSeconds(1f);
			HeroController.instance.AddMPCharge(1);
			HeroController.instance.AddMPCharge(-1);
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
