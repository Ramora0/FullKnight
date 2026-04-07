using Modding;

namespace FullKnight
{
	internal class FullKnight : Mod
	{
		internal static FullKnight Instance { get; private set; }

		private string _serverUrl = "ws://localhost:8765";

		public override void Initialize()
		{
			Instance = this;
			Log("FullKnight initializing");
			var env = new Environment.TrainingEnv(_serverUrl);
			env.Start();
		}

		public override string GetVersion() => "1.0.0";
	}
}
