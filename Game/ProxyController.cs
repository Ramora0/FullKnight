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
		/// <summary>
		/// Decode factored action vector into InputDeviceShim calls.
		/// action[0] movement:  0=left, 1=right, 2=none
		/// action[1] direction: 0=up, 1=down, 2=none
		/// action[2] action:    0=attack(tap), 1=charge(hold), 2=spell(tap),
		///                      3=focus(hold), 4=dash, 5=dream_nail(hold),
		///                      6=super_dash(hold), 7=none
		/// action[3] jump:      0=yes, 1=no
		///
		/// Apply order: movement -> direction -> jump -> action
		/// so that dash overrides jump when both are requested.
		/// </summary>
		public static void ApplyAction(InputDeviceShim shim, int[] action)
		{
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
		}
	}
}
