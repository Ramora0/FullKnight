using System;
using System.Collections;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using Modding;
using Mono.Cecil.Cil;
using MonoMod.Cil;
using MonoMod.RuntimeDetour;
using UnityEngine;

namespace FullKnight.Game
{
	public class TimeScale
	{
		private float timeScale;

		/// <summary>
		/// In-place multiplier swap — avoids the cost of disposing and recreating
		/// the IL hooks (each ctor reflection-walks GameManager.FreezeMoment*
		/// methods and installs a MonoMod ILHook per coroutine, ~50–100ms).
		/// Used by the reset path to crank to 20× and back to baseline without
		/// the per-reset hook-install overhead.
		/// </summary>
		public void SetMultiplier(float newScale)
		{
			this.timeScale = newScale;
			// Push the new product through TimeController so HK's actual
			// Time.timeScale reflects it immediately. GenericTimeScale=1 ×
			// our 4-factor product effectively makes our shim's multiplier
			// the authoritative scale (HK's other 3 factors are pinned to 1).
			TimeController.GenericTimeScale = 1f;
			Time.timeScale = newScale;
		}

		public TimeScale(float TimeScale = 1f)
		{
			this.timeScale = TimeScale;
			Time.timeScale = timeScale;

			On.GameManager.SetTimeScale_float += GameManager_SetTimeScale_Shim;

			_coroutineHooks = new ILHook[FreezeCoroutines.Length];

			foreach ((MethodInfo coro, int idx) in FreezeCoroutines.Select((mi, idx) => (mi, idx)))
			{
				_coroutineHooks[idx] = new ILHook(coro, ScaleFreeze);
			}
		}

		public void Dispose()
		{
			foreach (ILHook hook in _coroutineHooks)
				hook.Dispose();

			Time.timeScale = 1f;

			On.GameManager.SetTimeScale_float -= GameManager_SetTimeScale_Shim;
		}

		private readonly MethodInfo[] FreezeCoroutines = (
			from method in typeof(GameManager).GetMethods()
			where method.Name.StartsWith("FreezeMoment")
			where method.ReturnType == typeof(IEnumerator)
			select method.GetCustomAttribute<IteratorStateMachineAttribute>() into attr
			select attr.StateMachineType into type
			select type.GetMethod("MoveNext", BindingFlags.NonPublic | BindingFlags.Instance)
		).ToArray();

		private ILHook[] _coroutineHooks;

		private void ScaleFreeze(ILContext il)
		{
			var cursor = new ILCursor(il);

			cursor.GotoNext
			(
				MoveType.After,
				x => x.MatchLdfld(out _),
				x => x.MatchCall<Time>("get_unscaledDeltaTime")
			);

			cursor.EmitDelegate<Func<float>>(() => this.timeScale);
			cursor.Emit(OpCodes.Mul);
		}

		private void GameManager_SetTimeScale_Shim(On.GameManager.orig_SetTimeScale_float orig, GameManager self, float newTimeScale)
		{
			if (ReflectionHelper.GetField<GameManager, int>(self, "timeSlowedCount") > 1)
				newTimeScale = Math.Min(newTimeScale, TimeController.GenericTimeScale);

			TimeController.GenericTimeScale = (newTimeScale <= 0.01f ? 0f : newTimeScale) * this.timeScale;
		}
	}
}
