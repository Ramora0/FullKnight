import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical


class HitboxEncoder(nn.Module):
    """Single-head attention pooling over hitboxes, queried by global state."""

    def __init__(self, input_dim=5, hidden_dim=64, output_dim=64, query_dim=14):
        super().__init__()
        self.output_dim = output_dim

        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

        self.W_q = nn.Linear(query_dim, output_dim)
        self.W_k = nn.Linear(output_dim, output_dim)
        self.W_v = nn.Linear(output_dim, output_dim)

        self.scale = output_dim ** 0.5

    def forward(self, hitboxes, mask, global_state):
        """
        Args:
            hitboxes: (B, N, input_dim) zero-padded hitboxes
            mask: (B, N) 1 for real, 0 for padding
            global_state: (B, query_dim)
        Returns:
            (B, output_dim)
        """
        B = hitboxes.shape[0]

        if hitboxes.shape[1] == 0 or mask.sum() == 0:
            return torch.zeros(B, self.output_dim, device=hitboxes.device)

        h = self.phi(hitboxes)  # (B, N, output_dim)

        q = self.W_q(global_state).unsqueeze(1)  # (B, 1, output_dim)
        k = self.W_k(h)                           # (B, N, output_dim)
        v = self.W_v(h)                           # (B, N, output_dim)

        attn_logits = (q @ k.transpose(-2, -1)) / self.scale  # (B, 1, N)
        attn_mask = mask.unsqueeze(1)  # (B, 1, N)
        attn_logits = attn_logits.masked_fill(attn_mask == 0, -1e9)

        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, 1, N)
        out = (attn_weights @ v).squeeze(1)  # (B, output_dim)

        return out


class FullKnightActorCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.global_encoder = nn.Sequential(
            nn.Linear(config.global_state_dim, config.global_hidden),
            nn.ReLU(),
            nn.Linear(config.global_hidden, config.global_output),
            nn.ReLU(),
        )

        self.combat_encoder = HitboxEncoder(
            input_dim=config.combat_feature_dim,
            hidden_dim=config.combat_hidden,
            output_dim=config.combat_output,
            query_dim=config.global_output,
        )
        self.terrain_encoder = HitboxEncoder(
            input_dim=config.terrain_feature_dim,
            hidden_dim=config.terrain_hidden,
            output_dim=config.terrain_output,
            query_dim=config.global_output,
        )

        trunk_in = config.combat_output + config.terrain_output + config.global_output
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

        # GRU for temporal memory (between trunk and heads)
        self.gru = nn.GRUCell(config.hidden_dim, config.hidden_dim)
        self.gru_ln = nn.LayerNorm(config.hidden_dim)

        # Actor heads
        self.head_movement = nn.Linear(config.hidden_dim, config.movement_n)
        self.head_direction = nn.Linear(config.hidden_dim, config.direction_n)
        self.head_action = nn.Linear(config.hidden_dim, config.action_n)
        self.head_jump = nn.Linear(config.hidden_dim, config.jump_n)

        # Decomposed critic: separate heads for attack and defense value
        self.critic_attack = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.critic_defense = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "head_" in name:
                    nn.init.orthogonal_(module.weight, gain=0.01)
                elif name.endswith(".2") and "critic_" in name:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                else:
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # GRU: small init so residual starts as near-passthrough
        nn.init.orthogonal_(self.gru.weight_ih, gain=0.1)
        nn.init.orthogonal_(self.gru.weight_hh, gain=0.1)
        nn.init.constant_(self.gru.bias_ih, 0.0)
        nn.init.constant_(self.gru.bias_hh, 0.0)

        # Bias action head toward attack_tap (idx 0) at init
        with torch.no_grad():
            self.head_action.bias[0] = 1.0

    def _encode(self, combat_hb, combat_mask, terrain_hb, terrain_mask, global_state, hx=None):
        global_emb = self.global_encoder(global_state)
        combat_emb = self.combat_encoder(combat_hb, combat_mask, global_emb)
        terrain_emb = self.terrain_encoder(terrain_hb, terrain_mask, global_emb)
        combined = torch.cat([combat_emb, terrain_emb, global_emb], dim=-1)
        trunk_out = self.trunk(combined)

        # GRU with residual connection
        if hx is None:
            hx = torch.zeros_like(trunk_out)
        gru_out = self.gru(trunk_out, hx)
        features = trunk_out + self.gru_ln(gru_out)

        return features, gru_out

    def _extract_validity(self, global_state):
        """Extract the 9 validity flags from global_state."""
        # global_state layout: [vel_x, vel_y, hp, soul, boss_hp,
        #                       knight_w, knight_h,
        #                       has_dash, has_wall_jump, has_double_jump,
        #                       has_super_dash, has_dream_nail, has_acid_armour, has_nail_art,
        #                       can_jump, can_double_jump, can_wall_jump,
        #                       can_dash, can_attack, can_cast,
        #                       can_nail_charge, can_dream_nail, can_super_dash]
        can_jump = global_state[..., 14]
        can_double_jump = global_state[..., 15]
        can_wall_jump = global_state[..., 16]
        can_dash = global_state[..., 17]
        can_attack = global_state[..., 18]
        can_cast = global_state[..., 19]
        can_nail_charge = global_state[..., 20]
        can_dream_nail = global_state[..., 21]
        can_super_dash = global_state[..., 22]
        return (can_jump, can_double_jump, can_wall_jump, can_dash,
                can_attack, can_cast, can_nail_charge, can_dream_nail, can_super_dash)

    def _mask_logits(self, logits_action, logits_jump, global_state):
        """Apply action validity masking to logits."""
        (can_jump, can_double_jump, can_wall_jump, can_dash,
         can_attack, can_cast, can_nail_charge, can_dream_nail, can_super_dash) = \
            self._extract_validity(global_state)

        # Action head: [attack_tap, nail_charge, spell_tap, focus, dash,
        #               dream_nail, super_dash, none]
        logits_action[..., 0] = logits_action[..., 0] + (can_attack - 1) * 1e4
        logits_action[..., 1] = logits_action[..., 1] + (can_nail_charge - 1) * 1e4
        logits_action[..., 2] = logits_action[..., 2] + (can_cast - 1) * 1e4
        logits_action[..., 3] = logits_action[..., 3] + (can_cast - 1) * 1e4
        logits_action[..., 4] = logits_action[..., 4] + (can_dash - 1) * 1e4
        logits_action[..., 5] = logits_action[..., 5] + (can_dream_nail - 1) * 1e4
        logits_action[..., 6] = logits_action[..., 6] + (can_super_dash - 1) * 1e4

        # Jump head: [yes, no]
        can_any_jump = torch.clamp(can_jump + can_double_jump + can_wall_jump, 0, 1)
        logits_jump[..., 0] = logits_jump[..., 0] + (can_any_jump - 1) * 1e4

        return logits_action, logits_jump

    def get_value(self, combat_hb, combat_mask, terrain_hb, terrain_mask, global_state, hx=None):
        h, hx_new = self._encode(combat_hb, combat_mask, terrain_hb, terrain_mask, global_state, hx)
        return self.critic_attack(h).squeeze(-1), self.critic_defense(h).squeeze(-1), hx_new

    def get_action_and_value(self, combat_hb, combat_mask, terrain_hb, terrain_mask,
                             global_state, hx=None, actions=None):
        """
        If actions is None: sample new actions.
        If actions is provided: compute log_probs and entropy for given actions.

        actions: dict with keys 'movement', 'direction', 'action', 'jump',
                 each (B,) LongTensor.
        Returns: actions_dict, log_prob (B,), entropy (B,), value_atk (B,), value_def (B,), hx_new (B, hidden_dim)
        """
        h, hx_new = self._encode(combat_hb, combat_mask, terrain_hb, terrain_mask, global_state, hx)

        logits_m = self.head_movement(h)
        logits_d = self.head_direction(h)
        logits_a = self.head_action(h).clone()
        logits_j = self.head_jump(h).clone()

        # Apply validity masking
        logits_a, logits_j = self._mask_logits(logits_a, logits_j, global_state)

        dist_m = Categorical(logits=logits_m)
        dist_d = Categorical(logits=logits_d)
        dist_a = Categorical(logits=logits_a)
        dist_j = Categorical(logits=logits_j)

        if actions is None:
            a_m = dist_m.sample()
            a_d = dist_d.sample()
            a_a = dist_a.sample()
            a_j = dist_j.sample()
        else:
            a_m = actions["movement"]
            a_d = actions["direction"]
            a_a = actions["action"]
            a_j = actions["jump"]

        log_prob = (
            dist_m.log_prob(a_m)
            + dist_d.log_prob(a_d)
            + dist_a.log_prob(a_a)
            + dist_j.log_prob(a_j)
        )

        entropy = (
            dist_m.entropy()
            + dist_d.entropy()
            + dist_a.entropy()
            + dist_j.entropy()
        )

        value_atk = self.critic_attack(h).squeeze(-1)
        value_def = self.critic_defense(h).squeeze(-1)

        actions_dict = {
            "movement": a_m,
            "direction": a_d,
            "action": a_a,
            "jump": a_j,
        }
        return actions_dict, log_prob, entropy, value_atk, value_def, hx_new

    def forward_sequence(self, combat_hb, combat_mask, terrain_hb, terrain_mask,
                         global_state, hx, actions):
        """Truncated BPTT over a chunk of L timesteps.

        Args:
            combat_hb:    (B, L, max_combat, feat)
            combat_mask:  (B, L, max_combat)
            terrain_hb:   (B, L, max_terrain, feat)
            terrain_mask: (B, L, max_terrain)
            global_state: (B, L, global_dim)
            hx:           (B, hidden_dim) initial hidden state
            actions:      dict of (B, L) LongTensors

        Returns: log_probs (B,L), entropies (B,L), values_atk (B,L), values_def (B,L)
        """
        L = global_state.shape[1]
        all_lp, all_ent, all_vatk, all_vdef = [], [], [], []

        for t in range(L):
            step_actions = {k: v[:, t] for k, v in actions.items()}
            _, lp, ent, vatk, vdef, hx = self.get_action_and_value(
                combat_hb[:, t], combat_mask[:, t],
                terrain_hb[:, t], terrain_mask[:, t],
                global_state[:, t], hx=hx, actions=step_actions,
            )
            all_lp.append(lp)
            all_ent.append(ent)
            all_vatk.append(vatk)
            all_vdef.append(vdef)

        return (
            torch.stack(all_lp, dim=1),
            torch.stack(all_ent, dim=1),
            torch.stack(all_vatk, dim=1),
            torch.stack(all_vdef, dim=1),
        )
