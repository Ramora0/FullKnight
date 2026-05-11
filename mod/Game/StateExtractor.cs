using Modding;
using UnityEngine;

namespace FullKnight.Game
{
	public static class StateExtractor
	{
		/// <summary>
		/// Returns global state vector (22 floats):
		/// [vel_x, vel_y, hp, soul, knight_w, knight_h,
		///  has_dash, has_wall_jump, has_double_jump, has_super_dash, has_dream_nail, has_acid_armour, has_nail_art,
		///  can_jump, can_double_jump, can_wall_jump, can_dash, can_attack, can_cast,
		///  can_nail_charge, can_dream_nail, can_super_dash]
		/// Boss HP is no longer global; it's per-hitbox via hp_raw on the combat features.
		/// </summary>
		public static float[] GetGlobalState(float knightW, float knightH)
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

			return new float[]
			{
				velX, velY, hp, soul,
				knightW, knightH,
				hasDash, hasWallJump, hasDoubleJump, hasSuperDash, hasDreamNail, hasAcidArmour, hasNailArt,
				canJump, canDoubleJump, canWallJump, canDash, canAttack, canCast,
				canNailCharge, canDreamNail, canSuperDash
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
