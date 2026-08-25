using Modding;
using UnityEngine;

namespace FullKnight.Game
{
	public static class StateExtractor
	{
		/// <summary>Width of the vector GetGlobalState returns. Mirrored by
		/// config.global_state_dim and binary_protocol.GLOBAL_DIM on the Python
		/// side; all three must move together.</summary>
		public const int GlobalStateDim = 33;

		/// <summary>
		/// Returns global state vector (33 floats):
		/// [vel_x, vel_y, hp, soul, knight_w, knight_h,
		///  has_dash, has_wall_jump, has_double_jump, has_super_dash, has_dream_nail, has_acid_armour, has_nail_art,
		///  can_jump, can_double_jump, can_wall_jump, can_dash, can_attack, can_cast,
		///  can_nail_charge, can_dream_nail, can_super_dash,
		///  commit_locked, commit_releasing, commit_progress, commit_action_onehot x8]
		/// Boss HP is no longer global; it's per-hitbox via hp_raw on the combat features.
		///
		/// Ordering constraint: the first 6 entries are the only continuous ones
		/// and Python z-scores exactly that prefix (global_state_dim -
		/// n_binary_flags). Everything after index 5 is a flag or a bounded
		/// [0,1] scalar and passes through raw, so new bounded columns append at
		/// the end and bump n_binary_flags with them.
		/// </summary>
		public static float[] GetGlobalState(float knightW, float knightH,
			InputDeviceShim shim = null)
		{
			var hc = HeroController.instance;
			var pd = PlayerData.instance;
			var rb = ReflectionHelper.GetField<HeroController, Rigidbody2D>(hc, "rb2d");

			float velX = rb != null ? rb.velocity.x : 0f;
			float velY = rb != null ? rb.velocity.y : 0f;
			float hp = pd.health;
			float soul = pd.MPCharge;

			// Ability unlock flags
			float hasDash = pd.hasDash ? 1f : 0f;
			float hasWallJump = pd.canWallJump ? 1f : 0f;
			float hasDoubleJump = pd.hasDoubleJump ? 1f : 0f;
			float hasSuperDash = pd.hasSuperDash ? 1f : 0f;
			float hasDreamNail = pd.hasDreamNail ? 1f : 0f;
			float hasAcidArmour = pd.hasAcidArmour ? 1f : 0f;
			float hasNailArt = pd.GetBool("hasNailArt") ? 1f : 0f;

			// Action validity flags
			float canJump = CallCanMethod(hc, "CanJump") ? 1f : 0f;
			float canDoubleJump = CallCanMethod(hc, "CanDoubleJump") ? 1f : 0f;
			float canWallJump = CallCanMethod(hc, "CanWallJump") ? 1f : 0f;
			float canDash = CallCanMethod(hc, "CanDash") ? 1f : 0f;
			float canAttack = CallCanMethod(hc, "CanAttack") ? 1f : 0f;
			float canCast = CallCanMethod(hc, "CanCast") ? 1f : 0f;
			float canNailCharge = CallCanMethod(hc, "CanNailCharge") ? 1f : 0f;
			float canDreamNail = hc.CanDreamNail() ? 1f : 0f;
			float canSuperDash = hc.CanSuperDash() ? 1f : 0f;

			// Hard-commit proprioception. ActionDecoder's commit state machine
			// overrides action[2] for the duration of a hold — up to ~71 steps
			// for dream_nail at frames_per_wait=5 — while movement / direction /
			// jump stay free. Without these the policy has no way to know a
			// charge is in flight, which one, or how close it is to releasing.
			float commitLocked = 0f, commitReleasing = 0f, commitProgress = 0f;
			var commitAction = new float[8];
			if (shim != null)
			{
				commitLocked = shim.CState == InputDeviceShim.CommitState.Locked ? 1f : 0f;
				commitReleasing = shim.CState == InputDeviceShim.CommitState.Releasing ? 1f : 0f;
				if (shim.LockedStepsTotal > 0)
				{
					float left = shim.LockedStepsLeft > 0 ? shim.LockedStepsLeft : 0f;
					commitProgress = 1f - (left / shim.LockedStepsTotal);
				}
				else if (commitReleasing > 0f)
				{
					// Release step of a hold short enough to have no locked phase.
					commitProgress = 1f;
				}
				commitProgress = Mathf.Clamp01(commitProgress);
				if (shim.LockedAction >= 0 && shim.LockedAction < commitAction.Length)
					commitAction[shim.LockedAction] = 1f;
			}

			return new float[]
			{
				velX, velY, hp, soul,
				knightW, knightH,
				hasDash, hasWallJump, hasDoubleJump, hasSuperDash, hasDreamNail, hasAcidArmour, hasNailArt,
				canJump, canDoubleJump, canWallJump, canDash, canAttack, canCast,
				canNailCharge, canDreamNail, canSuperDash,
				commitLocked, commitReleasing, commitProgress,
				commitAction[0], commitAction[1], commitAction[2], commitAction[3],
				commitAction[4], commitAction[5], commitAction[6], commitAction[7]
			};
		}

		private static bool CallCanMethod(HeroController hc, string methodName)
		{
			try
			{
				return ReflectionHelper.CallMethod<HeroController, bool>(hc, methodName);
			}
			catch
			{
				return false;
			}
		}

	}
}
