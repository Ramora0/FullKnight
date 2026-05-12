using System;
using System.IO;
using System.Reflection;
using Newtonsoft.Json;

namespace FullKnight.Game
{
	public class SaveFileProxy
	{
		public static void LoadCompletedSave()
		{
			var saveResStream = Assembly.GetExecutingAssembly()
				.GetManifestResourceStream("FullKnight.Resource.save_file.json");
			if (saveResStream == null)
			{
				FullKnight.LogS("Resource stream for save file is null");
				return;
			}

			var saveFileString = new StreamReader(saveResStream).ReadToEnd();

			SaveGameData completedSaveGameData;
			try
			{
				completedSaveGameData = JsonConvert.DeserializeObject<SaveGameData>(saveFileString);
			}
			catch (Exception e)
			{
				FullKnight.LogS($"Could not deserialize completed save file, {e.GetType()}, {e.Message}");
				return;
			}

			var gameManager = GameManager.instance;
			gameManager.playerData = PlayerData.instance = completedSaveGameData?.playerData;
			gameManager.sceneData = SceneData.instance = completedSaveGameData?.sceneData;
		}

		public static void DisableSaveGame(On.GameManager.orig_SaveGame orig, GameManager self)
		{
		}
	}
}
