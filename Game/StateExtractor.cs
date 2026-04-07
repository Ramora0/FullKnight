using Modding;
using UnityEngine;

namespace FullKnight.Game
{
	public static class StateExtractor
	{
		/// <summary>
		/// Returns global state vector (~16 floats):
		/// [vel_x, vel_y, hp, soul, abilities_bitmask, boss_hp,
		///  knight_w, knight_h,
		///  can_jump, can_double_jump, can_wall_jump, can_dash, can_attack, can_cast]
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
			float abilities = EncodeAbilities(pd);
			float bossHp = GetBossHP();

			// Action validity flags
			float canJump = CallCanMethod(hc, "CanJump") ? 1f : 0f;
			float canDoubleJump = CallCanMethod(hc, "CanDoubleJump") ? 1f : 0f;
			float canWallJump = CallCanMethod(hc, "CanWallJump") ? 1f : 0f;
			float canDash = CallCanMethod(hc, "CanDash") ? 1f : 0f;
			float canAttack = CallCanMethod(hc, "CanAttack") ? 1f : 0f;
			float canCast = CallCanMethod(hc, "CanCast") ? 1f : 0f;

			return new float[]
			{
				velX, velY, hp, soul, abilities, bossHp,
				knightW, knightH,
				canJump, canDoubleJump, canWallJump, canDash, canAttack, canCast
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

		private static float EncodeAbilities(PlayerData pd)
		{
			int mask = 0;
			if (pd.hasDash) mask |= 1;
			if (pd.canWallJump) mask |= 1 << 1;
			if (pd.hasDoubleJump) mask |= 1 << 2;
			if (pd.hasSuperDash) mask |= 1 << 3;
			if (pd.hasDreamNail) mask |= 1 << 4;
			if (pd.hasAcidArmour) mask |= 1 << 5;
			return (float)mask;
		}

		private static float GetBossHP()
		{
			try
			{
				if (BossSceneController.Instance?.bosses != null
					&& BossSceneController.Instance.bosses.Length > 0)
				{
					var bossHM = BossSceneController.Instance.bosses[0]
						.gameObject.GetComponent<HealthManager>();
					if (bossHM != null)
						return bossHM.hp;
				}
			}
			catch { }
			return 0f;
		}
	}
}
