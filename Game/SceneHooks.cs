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
