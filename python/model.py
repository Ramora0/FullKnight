import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

from observation import Observation, GS


# Readout-row indices: which row of the (B, 6, d) readout tensor each head
# consumes. Centralized so the model and any external readers (debug dumps,
# attention visualizers) stay in sync.
class RO:
    MOVEMENT = 0
    DIRECTION = 1
    ACTION = 2
    JUMP = 3
    CRITIC_ATK = 4
    CRITIC_DEF = 5
    N = 6


def _additive_mask(key_mask, dtype):
    """Build (B, 1, 1, N) additive bias from a (B, N) {0,1} key mask.

    Real keys get 0, padding gets a large negative; broadcastable over heads
    and queries. We use -1e4 (not -inf): SDPA with -inf produces NaN if
    every key in a row is padded (softmax over all-masked → 0/0). With -1e4
    the softmax is uniform over padded rows, the value-tensor for those
    rows is zero anyway (padding is zero-filled), so the output is zero —
    same behavior as the old single-query encoder.
    """
    bias = (1.0 - key_mask).to(dtype) * -1e4
    return bias.view(bias.shape[0], 1, 1, bias.shape[1])


class CrossAttnBlock(nn.Module):
    """Pre-LN cross-attention + FFN with residuals. SDPA inside.

    Queries (B, Q, d) attend to keys/values (B, N, d). If `kv_mask` is None
    we skip the additive bias and let SDPA pick the fastest kernel (flash
    on Ada); pass a (B, N) {0,1} mask when keys are padded.
    """

    def __init__(self, d_model, n_heads, ffn_expansion=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_kv = nn.Linear(d_model, 2 * d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_expansion * d_model),
            nn.GELU(),
            nn.Linear(ffn_expansion * d_model, d_model),
        )

    def forward(self, queries, kv, kv_mask=None):
        B, Q, D = queries.shape
        N = kv.shape[1]
        H, Dh = self.n_heads, self.head_dim

        q_in = self.norm_q(queries)
        kv_in = self.norm_kv(kv)

        q = self.W_q(q_in).view(B, Q, H, Dh).transpose(1, 2)
        kv_proj = self.W_kv(kv_in).view(B, N, 2, H, Dh)
        k = kv_proj[:, :, 0].transpose(1, 2).contiguous()
        v = kv_proj[:, :, 1].transpose(1, 2).contiguous()

        if kv_mask is None:
            attn_out = F.scaled_dot_product_attention(q, k, v)
        else:
            bias = _additive_mask(kv_mask, q.dtype)
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Q, D)

        x = queries + self.W_o(attn_out)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class SelfAttnBlock(nn.Module):
    """Pre-LN self-attention + FFN. Standard transformer encoder block."""

    def __init__(self, d_model, n_heads, ffn_expansion=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        self.norm_attn = nn.LayerNorm(d_model)
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_expansion * d_model),
            nn.GELU(),
            nn.Linear(ffn_expansion * d_model, d_model),
        )

    def forward(self, x):
        B, S, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        x_norm = self.norm_attn(x)
        qkv = self.W_qkv(x_norm).view(B, S, 3, H, Dh)
        q = qkv[:, :, 0].transpose(1, 2).contiguous()
        k = qkv[:, :, 1].transpose(1, 2).contiguous()
        v = qkv[:, :, 2].transpose(1, 2).contiguous()

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        x = x + self.W_o(attn_out)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class QFormer(nn.Module):
    """A set of learned queries cross-attend to a variable-length input set,
    producing a fixed-length sequence of refined query tokens. Used to
    compress (variable, padded, masked) combat / terrain hitbox sets to a
    small fixed-length sequence the trunk transformer can consume.

    Conditioning: the global state embedding is added to each query before
    the first cross-attn layer so queries are aware of the agent's current
    situation. Each query learns to specialize on a different role.
    """

    def __init__(self, n_queries, d_model, n_heads, n_layers,
                 input_dim, extra_dim=0, ffn_expansion=4):
        super().__init__()
        self.n_queries = n_queries
        self.d_model = d_model

        in_dim = input_dim + extra_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.queries = nn.Parameter(torch.zeros(n_queries, d_model))
        nn.init.normal_(self.queries, std=0.02)

        self.global_cond = nn.Linear(d_model, d_model)

        self.layers = nn.ModuleList([
            CrossAttnBlock(d_model, n_heads, ffn_expansion) for _ in range(n_layers)
        ])

    def forward(self, hitboxes, mask, global_emb, extra=None):
        B = hitboxes.shape[0]
        if extra is not None:
            hitboxes = torch.cat([hitboxes, extra], dim=-1)
        kv = self.input_proj(hitboxes)

        q = self.queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        q = q + self.global_cond(global_emb).unsqueeze(1)

        for layer in self.layers:
            q = layer(q, kv, mask)
        return q


class FullKnightActorCritic(nn.Module):
    """QFormer-compressed, transformer-trunk, GRU-recurrent actor-critic
    with per-head cross-attention readout.

    Per-timestep pipeline:

      1. Encode global state to a d-dim token; embed kind/parent/anim ids.
      2. Combat QFormer (8 learned queries × 2 cross-attn layers) compresses
         the variable-length combat hitbox set.
      3. Terrain QFormer (4 queries × 1 layer) does the same for terrain.
      4. Trunk transformer (2 self-attn layers) refines the
         (8 combat + 4 terrain + 1 global) = 13-token sequence.
      5. The GRU runs on the trunk's global-token output and emits a
         dedicated memory token (LN + learned identity) that gets
         APPENDED to the trunk sequence as a 14th slot. Keeping memory in
         its own token lets each readout head choose how much to lean on
         it — actors mostly ignoring it, critics leaning hard on it for
         trajectory-aware return estimation.
      6. Per-head readout: 6 learned queries (one per head — 4 actor +
         2 critic) cross-attend to the 14 keys (13 trunk + 1 memory)
         through one CrossAttnBlock. Each head reads its own row.
      7. Actor heads are linear; critic heads are 3-layer MLPs because
         value regression benefits from extra nonlinear depth.

    bf16 autocast wraps the heavy transformer compute (QFormers + trunk +
    readout); the GRU and Categorical math stay in fp32 (GRU's cuDNN bf16
    path is fragile, distribution softmax/log wants fp32 precision).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        d = config.model_d
        nh = config.model_n_heads
        exp = config.model_ffn_expansion

        # Global state → d-dim token.
        self.global_encoder = nn.Sequential(
            nn.Linear(config.global_state_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        # Per-hitbox semantic identity (leaf + parent), concat'd into combat
        # features before the combat QFormer's input projection.
        self.kind_embed = nn.Embedding(
            config.kind_vocab_size, config.kind_embed_dim, padding_idx=0,
        )
        # Current animation clip per hitbox. Separate table from kind_embed:
        # clip names live in their own vocab, and unlike kind/parent this id
        # changes step to step — it's what carries the attack telegraph.
        self.anim_embed = nn.Embedding(
            config.anim_vocab_size, config.anim_embed_dim, padding_idx=0,
        )

        self.combat_qformer = QFormer(
            n_queries=config.n_combat_queries,
            d_model=d, n_heads=nh,
            n_layers=config.qformer_combat_layers,
            input_dim=config.combat_feature_dim,
            extra_dim=2 * config.kind_embed_dim + config.anim_embed_dim,
            ffn_expansion=exp,
        )
        self.terrain_qformer = QFormer(
            n_queries=config.n_terrain_queries,
            d_model=d, n_heads=nh,
            n_layers=config.qformer_terrain_layers,
            input_dim=config.terrain_feature_dim,
            extra_dim=0,
            ffn_expansion=exp,
        )

        # Per-token positional + type embeddings for the trunk sequence.
        # Type ids: 0=combat-query, 1=terrain-query, 2=global-token.
        n_tokens = config.n_combat_queries + config.n_terrain_queries + 1
        self.n_tokens = n_tokens
        self.type_embed = nn.Embedding(3, d)
        type_ids = torch.zeros(n_tokens, dtype=torch.long)
        type_ids[config.n_combat_queries:config.n_combat_queries + config.n_terrain_queries] = 1
        type_ids[-1] = 2
        self.register_buffer("token_type_ids", type_ids)
        # Index of the global token within the trunk sequence (always last).
        self.global_token_idx = n_tokens - 1

        self.pos_embed = nn.Parameter(torch.zeros(n_tokens, d))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Learned identity for the GRU-produced memory token, which gets
        # appended to the trunk sequence before the readout cross-attn.
        # Same role as pos_embed[i] + type_embed[i] for trunk tokens — gives
        # the memory slot a distinct signature the readout queries can
        # recognize. Added AFTER gru_ln so the LN doesn't erase it.
        self.memory_identity = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.memory_identity, std=0.02)

        # Trunk: self-attention over the 8+4+1 = 13 mixed tokens.
        self.trunk = nn.ModuleList([
            SelfAttnBlock(d, nh, exp) for _ in range(config.trunk_n_layers)
        ])
        self.trunk_norm = nn.LayerNorm(d)

        # GRU for temporal memory on the global token. Kept in fp32.
        gru_dim = config.gru_dim
        self.gru_proj_in = nn.Linear(d, gru_dim)
        self.gru = nn.GRU(gru_dim, gru_dim, num_layers=1, batch_first=True)
        self.gru_proj_out = nn.Linear(gru_dim, d)
        self.gru_ln = nn.LayerNorm(d)

        # Per-head readout: one learned query per head, cross-attending to
        # the 14-key sequence (13 trunk tokens + 1 GRU memory token).
        self.readout_queries = nn.Parameter(torch.zeros(RO.N, d))
        nn.init.normal_(self.readout_queries, std=0.02)
        self.readout_block = CrossAttnBlock(d, nh, exp)

        # Actor heads — linear off each query's row.
        self.head_movement = nn.Linear(d, config.movement_n)
        self.head_direction = nn.Linear(d, config.direction_n)
        self.head_action = nn.Linear(d, config.action_n)
        self.head_jump = nn.Linear(d, config.jump_n)

        # Critic heads — 3-layer MLPs off each query's row. Value functions
        # benefit from extra nonlinear depth (regression), the policy heads
        # don't (the readout query + linear is enough).
        self.critic_attack = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.critic_defense = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

        self._use_bf16 = bool(getattr(config, "use_bf16", True))

        self._init_weights()

    # ---------------------------------------------------------------- init
    def _init_weights(self):
        # Residual-output projections (W_o, ffn[-1]) get a small init so the
        # transformer starts near identity through the stack.
        n_blocks = (
            self.config.qformer_combat_layers
            + self.config.qformer_terrain_layers
            + self.config.trunk_n_layers
            + 1  # readout block
        )
        attn_residual_scale = 1.0 / (2 * max(n_blocks, 1)) ** 0.5

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "head_" in name:
                    nn.init.orthogonal_(module.weight, gain=0.01)
                elif "critic_" in name and name.endswith(".4"):
                    # Final 1-D output layer of the 3-layer critic MLP.
                    nn.init.orthogonal_(module.weight, gain=1.0)
                elif name.endswith(".W_o") or (
                    "ffn" in name and name.endswith(".2")
                ):
                    nn.init.orthogonal_(module.weight, gain=attn_residual_scale)
                else:
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

        # GRU: small init so the residual into the global slot starts as
        # near-passthrough (the readout sees ~unchanged trunk tokens at t=0).
        nn.init.orthogonal_(self.gru.weight_ih_l0, gain=0.1)
        nn.init.orthogonal_(self.gru.weight_hh_l0, gain=0.1)
        nn.init.constant_(self.gru.bias_ih_l0, 0.0)
        nn.init.constant_(self.gru.bias_hh_l0, 0.0)
        nn.init.orthogonal_(self.gru_proj_out.weight, gain=0.1)
        nn.init.constant_(self.gru_proj_out.bias, 0.0)

        # Bias the action head toward attack_tap and away from the four
        # hold actions (1=nail_charge, 3=focus, 5=dream_nail, 6=super_dash).
        # See ProxyController.LockedStepsFor for why holds dominate
        # wall-clock without this bias.
        hold_bias = float(getattr(self.config, "hold_action_init_bias", -2.0))
        with torch.no_grad():
            self.head_action.bias[0] = 1.0
            if hold_bias != 0.0:
                for idx in (1, 3, 5, 6):
                    self.head_action.bias[idx] = hold_bias

    # ----------------------------------------------------------- internals
    def _trunk(self, obs):
        """QFormers + trunk self-attn. Returns (B, 13, d) sequence in the
        ambient dtype (bf16 under autocast, fp32 otherwise)."""
        global_emb = self.global_encoder(obs.global_state)

        # Per-hitbox identity: what it is (leaf kind), whose it is (parent), and
        # what it is currently doing (animation clip). The first two are static
        # per collider; the third changes every step and is what the anim_progress
        # column in combat_hb is the phase of.
        kind_emb = torch.cat(
            [self.kind_embed(obs.combat_kind_ids),
             self.kind_embed(obs.combat_parent_ids),
             self.anim_embed(obs.combat_anim_ids)],
            dim=-1,
        )

        combat_q = self.combat_qformer(
            obs.combat_hb, obs.combat_mask, global_emb, extra=kind_emb,
        )
        terrain_q = self.terrain_qformer(
            obs.terrain_hb, obs.terrain_mask, global_emb,
        )
        global_tok = global_emb.unsqueeze(1)

        seq = torch.cat([combat_q, terrain_q, global_tok], dim=1)        # (B, 13, d)
        type_emb = self.type_embed(self.token_type_ids)                  # (13, d)
        seq = seq + self.pos_embed.unsqueeze(0) + type_emb.unsqueeze(0)

        for layer in self.trunk:
            seq = layer(seq)
        return self.trunk_norm(seq)

    def _readout(self, trunk_seq):
        """Run the per-head readout cross-attn. Returns (B, RO.N, d)."""
        B = trunk_seq.shape[0]
        q = self.readout_queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        return self.readout_block(q, trunk_seq, kv_mask=None)

    def _gru_step(self, trunk_global, hx):
        """Single-step GRU on the global token. Runs in fp32.

        trunk_global: (B, d) fp32, hx: (B, gru_dim) fp32 (or None).
        Returns memory_token (B, d) fp32 — the new memory token to append
        to the trunk sequence — and hx_new (B, gru_dim).
        """
        gru_in = self.gru_proj_in(trunk_global).unsqueeze(1)             # (B, 1, gru_dim)
        if hx is None:
            hx = torch.zeros(trunk_global.shape[0], self.gru.hidden_size,
                             device=trunk_global.device, dtype=trunk_global.dtype)
        gru_seq, hx_layered = self.gru(gru_in, hx.unsqueeze(0).contiguous())
        gru_out = self.gru_proj_out(gru_seq.squeeze(1))                  # (B, d)
        # LN first (normalizes scale), then add identity bias so the
        # learned constant survives normalization.
        memory_token = self.gru_ln(gru_out) + self.memory_identity
        return memory_token, hx_layered.squeeze(0)

    def _encode(self, obs: Observation, hx=None):
        """Single timestep. Returns (readout (B, RO.N, d), hx_new (B, gru_dim))."""
        if self._use_bf16 and obs.global_state.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                trunk_seq_bf = self._trunk(obs)
            trunk_seq = trunk_seq_bf.float()
        else:
            trunk_seq = self._trunk(obs)

        trunk_global = trunk_seq[:, self.global_token_idx, :]            # (B, d) fp32
        memory_token, hx_new = self._gru_step(trunk_global, hx)

        # Append memory as the 14th token; readout cross-attends to all of it.
        readout_seq = torch.cat([trunk_seq, memory_token.unsqueeze(1)], dim=1)

        if self._use_bf16 and readout_seq.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                readout_bf = self._readout(readout_seq)
            readout = readout_bf.float()
        else:
            readout = self._readout(readout_seq)

        return readout, hx_new

    def _extract_validity(self, global_state):
        return (
            global_state[..., GS.CAN_JUMP],
            global_state[..., GS.CAN_DOUBLE_JUMP],
            global_state[..., GS.CAN_WALL_JUMP],
            global_state[..., GS.CAN_DASH],
            global_state[..., GS.CAN_ATTACK],
            global_state[..., GS.CAN_CAST],
            global_state[..., GS.CAN_NAIL_CHARGE],
            global_state[..., GS.CAN_DREAM_NAIL],
            global_state[..., GS.CAN_SUPER_DASH],
        )

    def _mask_logits(self, logits_action, logits_jump, global_state):
        (can_jump, can_double_jump, can_wall_jump, can_dash,
         can_attack, can_cast, can_nail_charge, can_dream_nail, can_super_dash) = \
            self._extract_validity(global_state)

        logits_action[..., 0] = logits_action[..., 0] + (can_attack - 1) * 1e4
        logits_action[..., 1] = logits_action[..., 1] + (can_nail_charge - 1) * 1e4
        logits_action[..., 2] = logits_action[..., 2] + (can_cast - 1) * 1e4
        logits_action[..., 3] = logits_action[..., 3] + (can_cast - 1) * 1e4
        logits_action[..., 4] = logits_action[..., 4] + (can_dash - 1) * 1e4
        logits_action[..., 5] = logits_action[..., 5] + (can_dream_nail - 1) * 1e4
        logits_action[..., 6] = logits_action[..., 6] + (can_super_dash - 1) * 1e4

        can_any_jump = torch.clamp(can_jump + can_double_jump + can_wall_jump, 0, 1)
        logits_jump[..., 0] = logits_jump[..., 0] + (can_any_jump - 1) * 1e4

        return logits_action, logits_jump

    # ----------------------------------------------------------- public API
    def get_value(self, obs: Observation, hx=None):
        readout, hx_new = self._encode(obs, hx)
        v_atk = self.critic_attack(readout[:, RO.CRITIC_ATK, :]).squeeze(-1)
        v_def = self.critic_defense(readout[:, RO.CRITIC_DEF, :]).squeeze(-1)
        return v_atk, v_def, hx_new

    def get_action_and_value(self, obs: Observation, hx=None, actions=None):
        readout, hx_new = self._encode(obs, hx)

        logits_m = self.head_movement(readout[:, RO.MOVEMENT, :])
        logits_d = self.head_direction(readout[:, RO.DIRECTION, :])
        logits_a = self.head_action(readout[:, RO.ACTION, :]).clone()
        logits_j = self.head_jump(readout[:, RO.JUMP, :]).clone()
        logits_a, logits_j = self._mask_logits(logits_a, logits_j, obs.global_state)

        dist_m = Categorical(logits=logits_m, validate_args=False)
        dist_d = Categorical(logits=logits_d, validate_args=False)
        dist_a = Categorical(logits=logits_a, validate_args=False)
        dist_j = Categorical(logits=logits_j, validate_args=False)

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

        lp_m = dist_m.log_prob(a_m)
        lp_d = dist_d.log_prob(a_d)
        lp_a = dist_a.log_prob(a_a)
        lp_j = dist_j.log_prob(a_j)
        log_prob = lp_m + lp_d + lp_a + lp_j

        ent_a = dist_a.entropy()
        entropy = dist_m.entropy() + dist_d.entropy() + ent_a + dist_j.entropy()

        value_atk = self.critic_attack(readout[:, RO.CRITIC_ATK, :]).squeeze(-1)
        value_def = self.critic_defense(readout[:, RO.CRITIC_DEF, :]).squeeze(-1)

        actions_dict = {
            "movement": a_m,
            "direction": a_d,
            "action": a_a,
            "jump": a_j,
        }
        return actions_dict, log_prob, entropy, value_atk, value_def, hx_new, lp_a, ent_a

    def forward_sequence(self, obs: Observation, hx, actions):
        """Truncated BPTT over a chunk of L timesteps. Same return tuple as
        the prior architecture so PPO / the training graph runner do not
        need to change. Trunk + readout vectorize over (B*L); only the GRU
        has a temporal dependency (handled by cuDNN's fused L-step kernel).
        """
        B, L = obs.global_state.shape[:2]

        # ---- Phase 1: flatten (B, L, ...) → (B*L, ...) and run trunk ----
        flat_obs = Observation(
            combat_hb=obs.combat_hb.reshape(B * L, *obs.combat_hb.shape[2:]),
            combat_mask=obs.combat_mask.reshape(B * L, obs.combat_mask.shape[-1]),
            combat_kind_ids=obs.combat_kind_ids.reshape(B * L, obs.combat_kind_ids.shape[-1]),
            combat_parent_ids=obs.combat_parent_ids.reshape(B * L, obs.combat_parent_ids.shape[-1]),
            combat_anim_ids=obs.combat_anim_ids.reshape(B * L, obs.combat_anim_ids.shape[-1]),
            terrain_hb=obs.terrain_hb.reshape(B * L, *obs.terrain_hb.shape[2:]),
            terrain_mask=obs.terrain_mask.reshape(B * L, obs.terrain_mask.shape[-1]),
            global_state=obs.global_state.reshape(B * L, obs.global_state.shape[-1]),
        )

        if self._use_bf16 and flat_obs.global_state.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                flat_trunk_bf = self._trunk(flat_obs)
            flat_trunk = flat_trunk_bf.float()
        else:
            flat_trunk = self._trunk(flat_obs)                            # (B*L, S, d)

        S = flat_trunk.shape[1]
        d = flat_trunk.shape[2]
        trunk_seq = flat_trunk.view(B, L, S, d)

        # ---- Phase 2: GRU over the L-step global-token sequence ----
        trunk_globals = trunk_seq[:, :, self.global_token_idx, :]        # (B, L, d)
        gru_in_seq = self.gru_proj_in(trunk_globals)                     # (B, L, gru_dim)
        gru_hidden_seq, _ = self.gru(gru_in_seq, hx.unsqueeze(0).contiguous())
        gru_out_seq = self.gru_proj_out(gru_hidden_seq)                  # (B, L, d)
        # LN + learned identity → memory token per step.
        memory_seq = self.gru_ln(gru_out_seq) + self.memory_identity     # (B, L, d)

        # Append memory as the (S+1)th token at each step.
        readout_seq_BL = torch.cat([trunk_seq, memory_seq.unsqueeze(2)], dim=2)
        flat_readout_in = readout_seq_BL.reshape(B * L, S + 1, d)

        # ---- Phase 3: readout cross-attn (B*L, RO.N, d) ----
        if self._use_bf16 and flat_readout_in.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                flat_readout_bf = self._readout(flat_readout_in)
            flat_readout = flat_readout_bf.float()
        else:
            flat_readout = self._readout(flat_readout_in)                # (B*L, RO.N, d)

        flat_global = flat_obs.global_state

        # ---- Phase 4: heads + critic per readout row ----
        logits_m = self.head_movement(flat_readout[:, RO.MOVEMENT, :])
        logits_d = self.head_direction(flat_readout[:, RO.DIRECTION, :])
        logits_a = self.head_action(flat_readout[:, RO.ACTION, :]).clone()
        logits_j = self.head_jump(flat_readout[:, RO.JUMP, :]).clone()
        logits_a, logits_j = self._mask_logits(logits_a, logits_j, flat_global)

        dist_m = Categorical(logits=logits_m, validate_args=False)
        dist_d = Categorical(logits=logits_d, validate_args=False)
        dist_a = Categorical(logits=logits_a, validate_args=False)
        dist_j = Categorical(logits=logits_j, validate_args=False)

        flat_a_m = actions["movement"].reshape(-1)
        flat_a_d = actions["direction"].reshape(-1)
        flat_a_a = actions["action"].reshape(-1)
        flat_a_j = actions["jump"].reshape(-1)

        lp_a_flat = dist_a.log_prob(flat_a_a)
        log_prob_flat = (
            dist_m.log_prob(flat_a_m)
            + dist_d.log_prob(flat_a_d)
            + lp_a_flat
            + dist_j.log_prob(flat_a_j)
        )
        ent_a_flat = dist_a.entropy()
        entropy_flat = (
            dist_m.entropy() + dist_d.entropy()
            + ent_a_flat + dist_j.entropy()
        )
        v_atk_flat = self.critic_attack(flat_readout[:, RO.CRITIC_ATK, :]).squeeze(-1)
        v_def_flat = self.critic_defense(flat_readout[:, RO.CRITIC_DEF, :]).squeeze(-1)

        log_probs = log_prob_flat.view(B, L)
        entropies = entropy_flat.view(B, L)
        values_atk = v_atk_flat.view(B, L)
        values_def = v_def_flat.view(B, L)
        log_probs_action = lp_a_flat.view(B, L)
        entropies_action = ent_a_flat.view(B, L)

        # Diagnostic: norm of the GRU residual into the global slot.
        gru_info = {'gru_norm': gru_out_seq.detach().norm(dim=-1).mean()}

        return (log_probs, entropies, values_atk, values_def, gru_info,
                log_probs_action, entropies_action)
