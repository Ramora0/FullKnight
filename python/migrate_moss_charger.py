"""One-shot: migrate models/moss_charger_v1.pth to the current architecture.

Old shapes (pre boss_hp removal, pre kind embed, pre GRU):
    global_state_dim = 23  ([..., soul, boss_hp, knight_w, ...], boss_hp at idx 4)
    combat_feature_dim = 7 ([rel_x, rel_y, w, h, is_trigger, hurts_knight, is_target])

New shapes:
    global_state_dim = 22  (boss_hp removed)
    combat_feature_dim = 9 ([..., gives_damage, takes_damage, is_target, hp_raw])
    + kind_embed (512, 16) and bottleneck GRU stack — both fresh

Strategy:
  - Copy all shape-identical encoder/trunk layers verbatim.
  - Drop col 4 of global_encoder.0.weight; remap combat_encoder.phi.0.weight by column.
  - Temperature-scale actor heads by T=2.5 — preserves the rank order of the prior
    but softens the distribution so PPO has gradient room to re-learn specifics.
  - Zero gru_proj_out + gru_ln so the GRU residual starts as an exact passthrough.
  - Best-effort remap of obs/combat normalizers; terrain copies straight.
"""

import numpy as np
import torch

from config import Config
from model import FullKnightActorCritic
from ppo import PPO

OLD_PATH = "../models/moss_charger_v1.pth"
NEW_PATH = "../models/moss_charger_v2.pth"
TEMPERATURE = 2.5  # head softening factor

BOSS_HP_IDX = 4  # position of boss_hp in old global_state


def remap_global_encoder_in(W_old: torch.Tensor) -> torch.Tensor:
    # (64, 23) -> (64, 22): drop boss_hp column
    keep = [i for i in range(W_old.shape[1]) if i != BOSS_HP_IDX]
    return W_old[:, keep].clone()


def remap_combat_phi_in(W_old: torch.Tensor, new_shape) -> torch.Tensor:
    # (64, 7) -> (64, 41).
    # old cols: [rel_x, rel_y, w, h, is_trigger, hurts_knight, is_target]
    # new cols: [rel_x, rel_y, w, h, is_trigger, gives_damage, takes_damage,
    #            is_target, hp_raw, kind_emb_leaf(16), kind_emb_parent(16)]
    W_new = torch.zeros(new_shape, dtype=W_old.dtype)
    W_new[:, 0:5] = W_old[:, 0:5]   # geometry + is_trigger
    W_new[:, 5]   = W_old[:, 5]     # gives_damage <- hurts_knight
    # col 6 takes_damage: leave zero
    W_new[:, 7]   = W_old[:, 6]     # is_target -> is_target
    # col 8 hp_raw: leave zero
    # cols 9..40 kind_embed extras: leave zero (kind_embed itself is fresh)
    return W_new


def remap_obs_normalizer(old_state: dict) -> dict:
    keep = [i for i in range(old_state["mean"].shape[0]) if i != BOSS_HP_IDX]
    return {
        "mean": old_state["mean"][keep].copy(),
        "var":  old_state["var"][keep].copy(),
        "count": old_state["count"],
    }


def remap_combat_normalizer(old_state: dict) -> dict:
    # old (7,) -> new (8,). Same column logic as remap_combat_phi_in,
    # restricted to the normalized prefix (everything except hp_raw).
    new_mean = np.zeros(8, dtype=old_state["mean"].dtype)
    new_var  = np.ones(8,  dtype=old_state["var"].dtype)
    new_mean[0:5] = old_state["mean"][0:5]
    new_var[0:5]  = old_state["var"][0:5]
    new_mean[5]   = old_state["mean"][5]   # gives_damage <- hurts_knight
    new_var[5]    = old_state["var"][5]
    # col 6 takes_damage: mean=0, var=1 (booleans warm up fast)
    new_mean[7]   = old_state["mean"][6]   # is_target
    new_var[7]    = old_state["var"][6]
    return {"mean": new_mean, "var": new_var, "count": old_state["count"]}


def main():
    print(f"Loading {OLD_PATH}")
    old_ckpt = torch.load(OLD_PATH, map_location="cpu", weights_only=False)
    old_sd = old_ckpt["model"]

    config = Config()
    new_model = FullKnightActorCritic(config)
    new_sd = new_model.state_dict()

    HEAD_KEYS = {
        "head_movement.weight", "head_movement.bias",
        "head_direction.weight", "head_direction.bias",
        "head_action.weight", "head_action.bias",
        "head_jump.weight", "head_jump.bias",
    }
    SKIP_FROM_OLD = {"global_encoder.0.weight", "combat_encoder.phi.0.weight"}

    copied, softened, special, missing_in_old = [], [], [], []

    for k, target in new_sd.items():
        if k in SKIP_FROM_OLD:
            continue  # handled below
        if k.startswith(("kind_embed", "gru", "gru_proj", "gru_ln")):
            continue  # fresh init
        if k in old_sd and old_sd[k].shape == target.shape:
            new_sd[k] = old_sd[k].clone()
            if k in HEAD_KEYS:
                new_sd[k] = new_sd[k] / TEMPERATURE
                softened.append(k)
            else:
                copied.append(k)
        else:
            missing_in_old.append(k)

    new_sd["global_encoder.0.weight"] = remap_global_encoder_in(old_sd["global_encoder.0.weight"])
    special.append("global_encoder.0.weight")
    new_sd["combat_encoder.phi.0.weight"] = remap_combat_phi_in(
        old_sd["combat_encoder.phi.0.weight"], new_sd["combat_encoder.phi.0.weight"].shape
    )
    special.append("combat_encoder.phi.0.weight")

    # GRU residual starts as exact passthrough: features = trunk + LN(W_out · gru) → trunk
    with torch.no_grad():
        new_sd["gru_proj_out.weight"].zero_()
        new_sd["gru_proj_out.bias"].zero_()
        new_sd["gru_ln.weight"].zero_()
        new_sd["gru_ln.bias"].zero_()

    # Sanity-check: load into the model strict=True to confirm we covered everything.
    new_model.load_state_dict(new_sd, strict=True)

    # Normalizer remaps
    new_obs_norm = remap_obs_normalizer(old_ckpt["obs_normalizer"])
    new_combat_norm = remap_combat_normalizer(old_ckpt["combat_normalizer"])
    new_terrain_norm = old_ckpt["terrain_normalizer"]  # shape (5,) — unchanged

    # Build a fresh PPO so we can grab a clean optimizer/scheduler state matching
    # the new model's param groups. The loader assumes both keys are non-None and
    # only catches ValueError on optimizer load, so we can't just stash None.
    fresh = PPO(config)
    fresh.policy.load_state_dict(new_model.state_dict())

    out_ckpt = {
        "model": new_model.state_dict(),
        "optimizer": fresh.optimizer.state_dict(),
        "scheduler": fresh.scheduler.state_dict() if config.anneal_lr else None,
        "obs_normalizer": new_obs_norm,
        "combat_normalizer": new_combat_norm,
        "terrain_normalizer": new_terrain_norm,
        "hx": None,
        "kind_vocab": None,            # vocab is fresh; will populate from env
        "boss_state": None,            # adaptive D restarts from D_initial
        "epoch": None,                 # treat as a fresh run, not a resume
    }
    torch.save(out_ckpt, NEW_PATH)

    print(f"Wrote {NEW_PATH}")
    print(f"  copied verbatim: {len(copied)} tensors")
    print(f"  softened (T={TEMPERATURE}): {len(softened)} tensors -> {sorted(softened)}")
    print(f"  remapped: {special}")
    print(f"  fresh-init (kind_embed/gru): kept default")
    if missing_in_old:
        print(f"  WARNING — keys with no old counterpart: {missing_in_old}")


if __name__ == "__main__":
    main()
