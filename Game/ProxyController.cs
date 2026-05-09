using System.Collections.Generic;
using Modding;
using InControl;

namespace FullKnight.Game
{
	public class InputDeviceShim : InputDevice
	{
		private bool KeyUp = false;
		private bool KeyDown = false;
		private bool KeyLeft = false;
		private bool KeyRight = false;
		private bool KeyJump = false;
		private bool KeyAttack = false;
		private bool KeyDash = false;
		private bool KeyCast = false;
		private bool KeyDreamNail = false;
		private bool KeySuperDash = false;

		// When true, force one tick of key=false before pressing again (tap actions)
		private bool _retapAttack = false;
		private bool _retapCast = false;

		// Hard-commit state for hold actions. When the agent freely chooses a
		// hold (nail_charge / focus / dream_nail / super_dash), we lock action[2]
		// to that hold for a fixed number of env-steps, then force one step of
		// action[2]=none to trigger the release transition (which is what fires
		// the nail art / heal / dream nail / super dash). Other action heads
		// (movement / direction / jump) stay free throughout — HK's own physics
		// already restricts movement during heal/charge so we don't need to lock
		// them here. See ActionDecoder for the table of durations.
		public enum CommitState : byte { Idle = 0, Locked = 1, Releasing = 2 }
		public CommitState CState = CommitState.Idle;
		public int LockedAction = -1;     // 0..7: which hold is locked
		public int LockedStepsLeft = 0;   // steps remaining in Locked phase

		public void ResetCommit()
		{
			CState = CommitState.Idle;
			LockedAction = -1;
			LockedStepsLeft = 0;
		}

		public InputDeviceShim() :
			base("FullKnightInputShimDevice")
		{
			AddControl(InputControlType.DPadUp, "Up");
			AddControl(InputControlType.DPadDown, "Down");
			AddControl(InputControlType.DPadLeft, "Left");
			AddControl(InputControlType.DPadRight, "Right");
			AddControl(InputControlType.Action1, "Jump");
			AddControl(InputControlType.Action2, "Cast");
			AddControl(InputControlType.Action3, "Attack");
			AddControl(InputControlType.Action4, "DreamNail");
			AddControl(InputControlType.RightTrigger, "Dash");
			AddControl(InputControlType.LeftTrigger, "SuperDash");
			AddControl(InputControlType.RightBumper, "QuickCast");
		}

		public override void Update(ulong updateTick, float deltaTime)
		{
			// Retap: force one tick of false to create a fresh press transition
			bool effectiveAttack = KeyAttack;
			bool effectiveCast = KeyCast;
			if (_retapAttack) { effectiveAttack = false; _retapAttack = false; }
			if (_retapCast) { effectiveCast = false; _retapCast = false; }

			UpdateWithState(InputControlType.DPadUp, KeyUp, updateTick, deltaTime);
			UpdateWithState(InputControlType.DPadDown, KeyDown, updateTick, deltaTime);
			UpdateWithState(InputControlType.DPadLeft, KeyLeft, updateTick, deltaTime);
			UpdateWithState(InputControlType.DPadRight, KeyRight, updateTick, deltaTime);
			UpdateWithState(InputControlType.Action1, KeyJump, updateTick, deltaTime);
			UpdateWithState(InputControlType.RightBumper, effectiveCast, updateTick, deltaTime);
			UpdateWithState(InputControlType.Action3, effectiveAttack, updateTick, deltaTime);
			UpdateWithValue(InputControlType.RightTrigger, KeyDash ? 1 : 0, updateTick, deltaTime);
			UpdateWithState(InputControlType.Action4, KeyDreamNail, updateTick, deltaTime);
			UpdateWithValue(InputControlType.LeftTrigger, KeySuperDash ? 1 : 0, updateTick, deltaTime);
		}

		private static bool CanDash() =>
			ReflectionHelper.CallMethod<HeroController, bool>(HeroController.instance, "CanDash");

		private static bool CanAttack() =>
			ReflectionHelper.CallMethod<HeroController, bool>(HeroController.instance, "CanAttack");

		private static bool CanJump() =>
			ReflectionHelper.CallMethod<HeroController, bool>(HeroController.instance, "CanJump");

		private static bool CanDoubleJump() =>
			ReflectionHelper.CallMethod<HeroController, bool>(HeroController.instance, "CanDoubleJump");

		private static bool CanCast() =>
			ReflectionHelper.CallMethod<HeroController, bool>(HeroController.instance, "CanCast");

		private static bool CanWallJump() =>
			ReflectionHelper.CallMethod<HeroController, bool>(HeroController.instance, "CanWallJump");

		private static bool CanNailCharge() =>
			ReflectionHelper.CallMethod<HeroController, bool>(HeroController.instance, "CanNailCharge");

		private static bool CanDreamNail() =>
			HeroController.instance.CanDreamNail();

		private static bool CanSuperDash() =>
			HeroController.instance.CanSuperDash();

		public void Reset()
		{
			KeyUp = false;
			KeyDown = false;
			KeyLeft = false;
			KeyRight = false;
			KeyJump = false;
			KeyAttack = false;
			KeyDash = false;
			KeyCast = false;
			KeyDreamNail = false;
			KeySuperDash = false;
			_retapAttack = false;
			_retapCast = false;
		}

		public void Left() { KeyLeft = true; KeyRight = false; }
		public void Right() { KeyRight = true; KeyLeft = false; }
		public void Up() { KeyUp = true; KeyDown = false; }
		public void Down() { KeyDown = true; KeyUp = false; }

		public void Jump()
		{
			if (!CanJump() && !CanDoubleJump() && !CanWallJump()) return;
			KeyJump = true;
			KeyDash = false;
		}

		private void FaceDirection()
		{
			if (KeyLeft) HeroController.instance.FaceLeft();
			else if (KeyRight) HeroController.instance.FaceRight();
		}

		/// <summary>Tap attack: release-then-press to guarantee a fresh swing.</summary>
		public void AttackTap()
		{
			if (!CanAttack()) return;
			FaceDirection();
			_retapAttack = KeyAttack; // force release tick only if already held
			KeyAttack = true;
			KeyCast = false;
			KeyDreamNail = false;
			KeySuperDash = false;
		}

		/// <summary>Hold attack: keep KeyAttack held for nail art charge.</summary>
		public void NailCharge()
		{
			// Already holding — continue the charge regardless of CanNailCharge
			if (KeyAttack) return;
			if (!CanNailCharge()) return;
			FaceDirection();
			KeyAttack = true;
			KeyCast = false;
			KeyDreamNail = false;
			KeySuperDash = false;
		}

		/// <summary>Tap cast: release-then-press for spell.</summary>
		public void SpellTap()
		{
			if (!CanCast()) return;
			FaceDirection();
			_retapCast = KeyCast;
			KeyCast = true;
			KeyAttack = false;
			KeyDreamNail = false;
			KeySuperDash = false;
		}

		/// <summary>Hold cast: keep KeyCast held for focus/heal.</summary>
		public void Focus()
		{
			if (!KeyCast && !CanCast()) return;
			FaceDirection();
			KeyCast = true;
			KeyAttack = false;
			KeyDreamNail = false;
			KeySuperDash = false;
		}

		public void Dash()
		{
			if (!CanDash()) return;
			FaceDirection();
			KeyDash = true;
			KeyJump = false;
			KeyAttack = false;
			KeyCast = false;
			KeyDreamNail = false;
			KeySuperDash = false;
		}

		/// <summary>Hold dream nail.</summary>
		public void DreamNail()
		{
			if (!KeyDreamNail && !CanDreamNail()) return;
			KeyDreamNail = true;
			KeyAttack = false;
			KeyCast = false;
			KeySuperDash = false;
		}

		/// <summary>Hold super dash (crystal heart).</summary>
		public void SuperDash()
		{
			if (!KeySuperDash && !CanSuperDash()) return;
			KeySuperDash = true;
			KeyAttack = false;
			KeyCast = false;
			KeyDreamNail = false;
		}

		public void StopLR() { KeyLeft = false; KeyRight = false; }
		public void StopUD() { KeyUp = false; KeyDown = false; }
		public void StopJD() { KeyJump = false; KeyDash = false; }
		public void StopActions()
		{
			KeyAttack = false;
			KeyCast = false;
			KeyDash = false;
			KeyDreamNail = false;
			KeySuperDash = false;
			_retapAttack = false;
			_retapCast = false;
		}
	}

	public static class ActionDecoder
	{
		// Hard-commit hold durations, expressed in game-time seconds. The
		// per-env-step duration is computed at apply-time from frames_per_wait
		// and time_scale so the same wall-clock charge time holds across
		// configs. Each entry counts only the locked phase; one extra "release"
		// step (action[2]=none) is appended automatically to fire the action
		// (nail art swing / focus heal / dream nail / super dash).
		// Values are conservative — slightly above the actual HK charge times.
		private static readonly Dictionary<int, float> HoldGameSeconds = new()
		{
			{ 1, 1.5f },   // nail_charge: nail art charges in ~1.4s
			{ 3, 0.5f },   // focus: ~0.45s for one mask of healing
			{ 5, 3.0f },   // dream_nail: ~2.7s charge
			{ 6, 1.0f },   // super_dash: ~0.85s charge
		};

		private static int LockedStepsFor(int actionIdx, int framesPerWait, int timeScale)
		{
			if (!HoldGameSeconds.TryGetValue(actionIdx, out float gs)) return 0;
			// Pinned to baseline (framesPerWait=5, timeScale=3) so the locked-step
			// count is invariant to the current fpw. Otherwise trained agents
			// would see a different lock duration when we lower fpw to cut sim
			// cost — at fpw=2 the lock would balloon from 6 -> 15 steps and
			// the agent would freeze through boss attacks. Behavioral parity
			// with training matters more here than the (always-wrong) game-
			// seconds derivation; agents learned the lock count, not the wallclock.
			const float kBaselineStepGameSeconds = 5f * 3f / 60f;  // 0.25
			int n = (int)System.Math.Ceiling(gs / kBaselineStepGameSeconds);
			return n > 0 ? n : 1;
		}

		/// <summary>
		/// Decode factored action vector into InputDeviceShim calls.
		/// action[0] movement:  0=left, 1=right, 2=none
		/// action[1] direction: 0=up, 1=down, 2=none
		/// action[2] action:    0=attack(tap), 1=charge(hold), 2=spell(tap),
		///                      3=focus(hold), 4=dash, 5=dream_nail(hold),
		///                      6=super_dash(hold), 7=none
		/// action[3] jump:      0=yes, 1=no
		///
		/// Hard commit: when the agent freely picks a hold action, action[2] is
		/// locked to that hold for LockedStepsFor() env-steps, then forced to
		/// none for one release step. The action[] array is mutated in place to
		/// reflect what was actually applied. Returns true iff action[2] was
		/// overridden this step (i.e. the agent didn't make a free choice on
		/// the action head this step).
		///
		/// Apply order: movement -> direction -> jump -> action
		/// so that dash overrides jump when both are requested.
		/// </summary>
		public static bool ApplyAction(InputDeviceShim shim, int[] action,
			int framesPerWait, int timeScale)
		{
			bool committed = false;

			// Resolve action[2] against the commit state machine BEFORE the
			// shim methods are called, so the rest of this function sees the
			// actually-applied value.
			if (shim.CState == InputDeviceShim.CommitState.Releasing)
			{
				action[2] = 7;  // none — triggers StopActions and fires the held move
				committed = true;
				shim.CState = InputDeviceShim.CommitState.Idle;
				shim.LockedAction = -1;
				shim.LockedStepsLeft = 0;
			}
			else if (shim.CState == InputDeviceShim.CommitState.Locked)
			{
				action[2] = shim.LockedAction;
				committed = true;
				shim.LockedStepsLeft--;
				if (shim.LockedStepsLeft <= 0)
				{
					shim.CState = InputDeviceShim.CommitState.Releasing;
				}
			}
			else if (HoldGameSeconds.ContainsKey(action[2]))
			{
				// Idle + free hold pick: lock starting next step.
				int totalLocked = LockedStepsFor(action[2], framesPerWait, timeScale);
				shim.LockedAction = action[2];
				// This step counts toward the locked phase (the agent picked
				// it freely, KeyAttack/etc gets set true now). Subsequent
				// (totalLocked - 1) steps stay locked, then one release step.
				shim.LockedStepsLeft = totalLocked - 1;
				shim.CState = (shim.LockedStepsLeft > 0)
					? InputDeviceShim.CommitState.Locked
					: InputDeviceShim.CommitState.Releasing;
				// committed stays false — this step's action[2] is the agent's free choice.
			}

			// Movement
			switch (action[0])
			{
				case 0: shim.Left(); break;
				case 1: shim.Right(); break;
				default: shim.StopLR(); break;
			}

			// Direction
			switch (action[1])
			{
				case 0: shim.Up(); break;
				case 1: shim.Down(); break;
				default: shim.StopUD(); break;
			}

			// Jump (applied before action so dash can override)
			switch (action[3])
			{
				case 0: shim.Jump(); break;
				default: break;
			}

			// Action
			switch (action[2])
			{
				case 0: shim.AttackTap(); break;
				case 1: shim.NailCharge(); break;
				case 2: shim.SpellTap(); break;
				case 3: shim.Focus(); break;
				case 4: shim.Dash(); break;
				case 5: shim.DreamNail(); break;
				case 6: shim.SuperDash(); break;
				default:
					shim.StopActions();
					// Only stop jump/dash if no action and no jump requested
					if (action[3] != 0) shim.StopJD();
					break;
			}

			return committed;
		}
	}
}
