using System.Collections;
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
		private int _damageDoneInStep;
		private int _totalDamageTaken;
		private int _bossMaxHP;
		private int _knightMaxHP;
		private bool _bossDeadInStep;

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
			Time.timeScale = _timeScaleValue;
			SendMessage(new Message { type = "resume", data = data });
			yield break;
		}

		private IEnumerator Reset(MessageData data)
		{
			_level = data.level ?? _level;
			_frameSkipCount = data.frames_per_wait ?? _frameSkipCount;
			_timeScaleValue = data.time_scale ?? _timeScaleValue;
			_totalDamageTaken = 0;
			_hitsTakenInStep = 0;
			_damageDoneInStep = 0;
			_bossDeadInStep = false;

			yield return SceneHooks.LoadBossScene(_level);

			_bossMaxHP = GetBossMaxHP();
			_knightMaxHP = PlayerData.instance.maxHealth;

			UnhookDamage();
			HookDamage();

			if (_timeManager != null) _timeManager.Dispose();
			_timeManager = new Game.TimeScale(_timeScaleValue);

			var obs = _hitboxObserver.GetSplitFeatures();
			var gs = StateExtractor.GetGlobalState(obs.KnightWidth, obs.KnightHeight);

			data.combat_hitboxes = obs.CombatHitboxes;
			data.terrain_hitboxes = obs.TerrainHitboxes;
			data.global_state = gs;

			SendMessage(new Message { type = "reset", data = data });
			yield break;
		}

		private IEnumerator AutoReset()
		{
			_totalDamageTaken = 0;
			_hitsTakenInStep = 0;
			_damageDoneInStep = 0;
			_bossDeadInStep = false;

			yield return SceneHooks.LoadBossScene(_level);

			_bossMaxHP = GetBossMaxHP();
			_knightMaxHP = PlayerData.instance.maxHealth;

			UnhookDamage();
			HookDamage();

			if (_timeManager != null) _timeManager.Dispose();
			_timeManager = new Game.TimeScale(_timeScaleValue);

			_hitboxObserver.Load();
		}

		private IEnumerator Step(MessageData data)
		{
			ActionDecoder.ApplyAction(_inputShim, data.action_vec);

			for (int i = 0; i < _frameSkipCount; i++)
				yield return null;

			// Compute reward
			float reward = -0.001f; // step penalty
			reward += (float)_damageDoneInStep / (_bossMaxHP + 1e-8f);
			reward -= (float)_hitsTakenInStep / (_knightMaxHP + 1e-8f);

			bool done = _bossDeadInStep || _totalDamageTaken >= _knightMaxHP;

			if (_bossDeadInStep) reward += 1f;
			if (_totalDamageTaken >= _knightMaxHP) reward -= 1f;

			// Build observation
			var obs = _hitboxObserver.GetSplitFeatures();
			var gs = StateExtractor.GetGlobalState(obs.KnightWidth, obs.KnightHeight);

			data.combat_hitboxes = obs.CombatHitboxes;
			data.terrain_hitboxes = obs.TerrainHitboxes;
			data.global_state = gs;
			data.reward = reward;
			data.done = done;

			SendMessage(new Message { type = "step", data = data });

			// Reset per-step counters
			_hitsTakenInStep = 0;
			_damageDoneInStep = 0;
			_bossDeadInStep = false;

			// Auto-reset: reload boss scene for next episode (no message sent)
			if (done)
			{
				_inputShim.Reset();
				yield return AutoReset();
			}

			yield break;
		}

		protected override IEnumerator Setup()
		{
			Connect();
			yield return new Socket.WaitForMessage(socket);
			Message message = socket.UnreadMessages.Dequeue();
			if (message.type != "init")
			{
				yield return Setup();
				yield break;
			}

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

		private int OnKnightDamaged(int damageType, int _)
		{
			_hitsTakenInStep++;
			_totalDamageTaken++;
			// Return 1 so knight visually takes damage (knockback/animations)
			// but episode end is tracked via _totalDamageTaken threshold
			if (_totalDamageTaken > _knightMaxHP) return 0;
			return 1;
		}

		private void OnBossDamaged(On.HealthManager.orig_TakeDamage orig, HealthManager self, HitInstance hitInstance)
		{
			_damageDoneInStep += hitInstance.DamageDealt;
			// Check if boss would die from this hit
			if (self.hp - hitInstance.DamageDealt <= 0)
			{
				_bossDeadInStep = true;
				// Reset boss HP so the game doesn't trigger death sequence
				self.hp = _bossMaxHP;
			}
			orig(self, hitInstance);
		}

		private int GetBossMaxHP()
		{
			try
			{
				if (BossSceneController.Instance?.bosses != null
					&& BossSceneController.Instance.bosses.Length > 0)
				{
					var bossHM = BossSceneController.Instance.bosses[0]
						.gameObject.GetComponent<HealthManager>();
					if (bossHM != null) return bossHM.hp;
				}
			}
			catch { }
			return 1;
		}
	}
}
