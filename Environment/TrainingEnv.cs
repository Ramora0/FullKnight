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
		private int _timeScaleValue;
		private int _hitsTakenInStep;
		private float _damageLandedInStep;
		private int _bossMaxHP;
		private int _knightMaxHP;

		// Eval mode: real damage, real death, episode ends on kill
		private bool _evalMode;
		private bool _bossDied;
		private bool _episodeDone;
		private string _episodeResult;
		private HealthManager _bossHM;

		// Boss intro skip: keep simulating internally until combat starts
		private bool _combatStarted;

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
			_evalMode = data.eval ?? false;
			_hitsTakenInStep = 0;
			_damageLandedInStep = 0;
			_bossDied = false;
			_episodeDone = false;
			_episodeResult = null;
			_combatStarted = false;

			yield return SceneHooks.LoadBossScene(_level);

			InitBossRefs();
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

			Time.timeScale = 0;
			SendMessage(new Message { type = "reset", data = data });
			yield break;
		}

		private IEnumerator Step(MessageData data)
		{
			// If episode already ended, keep returning done
			if (_episodeDone)
			{
				data.done = true;
				data.info = _episodeResult;
				data.combat_hitboxes = new List<float[]>();
				data.terrain_hitboxes = new List<float[]>();
				data.global_state = new float[19];
				data.damage_landed = 0;
				data.hits_taken = 0;
				data.step_game_time = 0;
				data.step_real_time = 0;
				SendMessage(new Message { type = "step", data = data });
				yield break;
			}

			Time.timeScale = _timeScaleValue;

			ActionDecoder.ApplyAction(_inputShim, data.action_vec);

			// In training mode, restore knight HP each step for infinite fighting
			if (!_evalMode)
				PlayerData.instance.health = _knightMaxHP;

			float gameTimeElapsed = 0f;
			float realTimeElapsed = 0f;
			for (int i = 0; i < _frameSkipCount; i++)
			{
				yield return null;
				gameTimeElapsed += Time.deltaTime;
				realTimeElapsed += Time.unscaledDeltaTime;
				// In eval mode, break early on death
				if (_evalMode && (_bossDied || PlayerData.instance.health <= 0))
					break;
			}

			// If boss intro is still playing, fast-forward until combat starts
			if (!_combatStarted)
			{
				Time.timeScale = 20f;
				while (!HasActiveCombatHitboxes())
					yield return null;
				_combatStarted = true;
				// Clear any accidental reward signals from intro
				_hitsTakenInStep = 0;
				_damageLandedInStep = 0;
				// Run one normal frame skip at real speed so first obs is clean
				Time.timeScale = _timeScaleValue;
				for (int i = 0; i < _frameSkipCount; i++)
					yield return null;
			}

			Time.timeScale = 0;
			data.step_game_time = gameTimeElapsed;
			data.step_real_time = realTimeElapsed;

			// Check for episode end in eval mode
			if (_evalMode && !_episodeDone)
			{
				if (_bossDied)
				{
					_episodeDone = true;
					_episodeResult = "win";
				}
				else if (PlayerData.instance.health <= 0)
				{
					_episodeDone = true;
					_episodeResult = "loss";
				}
			}

			// Record reward signals
			data.damage_landed = _damageLandedInStep;
			data.hits_taken = _hitsTakenInStep;
			_hitsTakenInStep = 0;
			_damageLandedInStep = 0;

			if (_episodeDone)
			{
				data.done = true;
				data.info = _episodeResult;
				data.combat_hitboxes = new List<float[]>();
				data.terrain_hitboxes = new List<float[]>();
				data.global_state = new float[19];
				SendMessage(new Message { type = "step", data = data });
				yield break;
			}

			// Build observation
			var obs = _hitboxObserver.GetSplitFeatures();
			var gs = StateExtractor.GetGlobalState(obs.KnightWidth, obs.KnightHeight);

			data.combat_hitboxes = obs.CombatHitboxes;
			data.terrain_hitboxes = obs.TerrainHitboxes;
			data.global_state = gs;
			data.done = false;

			SendMessage(new Message { type = "step", data = data });
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

		private int OnKnightDamaged(int damageType, int damage)
		{
			_hitsTakenInStep++;
			// Eval: let real damage through. Training: minimal damage (HP restored next step)
			return _evalMode ? damage : 1;
		}

		private const float NailDamage = 21f;

		private void OnBossDamaged(On.HealthManager.orig_TakeDamage orig, HealthManager self, HitInstance hitInstance)
		{
			_damageLandedInStep += hitInstance.DamageDealt / NailDamage;

			if (!_evalMode)
			{
				// Training: prevent boss death, restore HP for infinite fighting
				if (self.hp - hitInstance.DamageDealt <= 0)
					self.hp = _bossMaxHP;
				orig(self, hitInstance);
				self.hp = _bossMaxHP;
			}
			else
			{
				// Eval: let damage through, detect boss death
				bool wouldDie = _bossHM != null && self == _bossHM
					&& self.hp - hitInstance.DamageDealt <= 0;
				orig(self, hitInstance);
				if (wouldDie)
					_bossDied = true;
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
			_bossHM = null;
			_bossMaxHP = 1;
			try
			{
				if (BossSceneController.Instance?.bosses != null
					&& BossSceneController.Instance.bosses.Length > 0)
				{
					_bossHM = BossSceneController.Instance.bosses[0]
						.gameObject.GetComponent<HealthManager>();
					if (_bossHM != null) _bossMaxHP = _bossHM.hp;
				}
			}
			catch { }
		}
	}
}
