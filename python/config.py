import argparse
from dataclasses import dataclass, fields


@dataclass
class Config:
    @property
    def rollout_len(self) -> int:
        return self.total_steps_per_epoch // self.n_envs

    @property
    def boss_levels_list(self) -> list:
        return [s.strip() for s in self.boss_levels.split(",") if s.strip()]

    # Environment
    server_host: str = "localhost"
    server_port: int = 8765
    n_envs: int = 16
    level: str = "GG_Mega_Moss_Charger"  # used by eval
    boss_levels: str = "GG_False_Knight,GG_Mega_Moss_Charger,GG_Gruz_Mother,GG_Hornet_1"  # comma-separated pool for training
    frames_per_wait: int = 5
    time_scale: int = 3
    # Staggered reset cadence: every `steps_per_reset` accumulated env-steps,
    # schedule a reset for max(1, n_envs // envs_per_reset_div) envs. Resets
    # run as background asyncio tasks overlapping the next rollout; scheduled
    # envs sit out of the active set until their reset completes.
    envs_per_reset_div: int = 8
    steps_per_reset: int = 4096

    # Hollow Knight paths (Windows)
    hk_path: str = r"C:\Program Files (x86)\Steam\steamapps\common\Hollow Knight"
    hk_data_dir: str = "hollow_knight_Data"

    # Observation dims
    # Combat: [rel_x, rel_y, w, h, is_trigger, gives_damage, takes_damage, is_target,
    #         hp_raw, hp_max_raw]
    # Only the continuous spatial cols (rel_x, rel_y, w, h) get running-normalized.
    # Binary flags (is_trigger, gives_damage, takes_damage, is_target) pass through raw
    # so sparse flags like is_target don't get amplified by a tiny running variance.
    # hp_raw / hp_max_raw also bypass the running normalizer; PPO log1p-compresses
    # them so they enter the network as ~[0, 8] while keeping high resolution near
    # death (log1p(0)→0, log1p(21)→3.09, log1p(42)→3.76, log1p(2000)→7.6).
    combat_feature_dim: int = 10
    combat_normalized_dims: int = 4  # first N combat columns get z-scored; binary flags pass raw; hp cols get log1p
    terrain_feature_dim: int = 8 # [mx, my, hdx, hdy, npx, npy, dist, is_trigger]
    terrain_normalized_dims: int = 7  # is_trigger passes through raw
    # Terrain visibility gate — knight-relative axis-aligned box (world units)
    # roughly approximating the HK camera frame. Segments whose nearest point
    # falls outside the box are dropped before reaching the model. Trims
    # ~600 → ~30 tokens per step in typical boss arenas: speeds up the
    # terrain encoder and regularizes the input set to "what's nearby and
    # on-screen". Set either to 0 to disable.
    view_w: float = 30.0
    view_h: float = 17.0
    global_state_dim: int = 22   # vel(2), hp, soul, knight_bounds(2), 7 ability flags, 9 validity flags
    n_binary_flags: int = 16     # 7 ability unlock + 9 action validity (not normalized)

    # Model architecture (QFormer hitbox compressor + transformer trunk).
    # d_model is the working width carried through QFormers, trunk, and
    # heads. n_heads splits d_model into head_dim = d_model // n_heads
    # chunks; head_dim=64 is the sweet spot for Ada flash-attention. FFN
    # expansion sets the hidden width of the per-block MLP (d → exp*d → d).
    model_d: int = 768
    model_n_heads: int = 12         # head_dim = 64 at d=768
    model_ffn_expansion: int = 4    # FFN hidden = exp * d
    # QFormer learned-query counts. The combat QFormer cross-attends to all
    # combat hitboxes from `n_combat_queries` learned tokens; each token
    # learns a different role (target body / attack collider / projectile /
    # threat / ...). Terrain gets fewer queries because the signal is
    # mostly "what is close" rather than role-specific.
    n_combat_queries: int = 8
    n_terrain_queries: int = 4
    # Per-side depth. Combat gets more layers because identifying the right
    # combat token (boss vs projectile vs trail) is the hard part.
    qformer_combat_layers: int = 2
    qformer_terrain_layers: int = 1
    # Trunk self-attention layers over the concatenated QFormer tokens +
    # global token. After `trunk_n_layers` of refinement, the global token
    # output goes into the GRU; its hidden state is then appended as a
    # 14th memory token before the per-head readout cross-attn.
    trunk_n_layers: int = 2
    # bf16 autocast around the heavy transformer compute (encoders, QFormer,
    # trunk, readout). GRU and Categorical math stay in fp32. Big speedup on
    # Ada tensor cores (4080 Super, etc); disable to debug or for non-CUDA.
    use_bf16: bool = True

    # Hold-action init bias for the action head: adds this logit offset to
    # the four hold actions (1=nail_charge, 3=focus, 5=dream_nail, 6=super_dash)
    # at init so they're not picked ~50% of the time on random init.
    hold_action_init_bias: float = -2.0

    # GRU (temporal memory)
    gru_dim: int = 256          # bottleneck dimension for GRU (d_model -> gru_dim -> d_model)
    seq_len: int = 16           # truncated BPTT chunk length
    chunks_per_batch: int = 8   # chunks per minibatch (effective batch = chunks_per_batch * seq_len)

    # Kind-id embedding (per-hitbox semantic identity, fed into combat encoder)
    kind_vocab_size: int = 512  # cap on distinct kind strings; overflow → "unknown" (loud warning)
    kind_embed_dim: int = 16    # embedding dim concatenated into each combat hitbox feature

    # Action dims
    movement_n: int = 3   # left, right, none
    direction_n: int = 3  # up, down, none
    action_n: int = 8     # attack_tap, nail_charge, spell_tap, focus, dash, dream_nail, super_dash, none
    jump_n: int = 2       # yes, no

    # Adaptive reward scaling: D = % of boss HP dealt per hit taken.
    # D_window / D_ema / D_max_delta below are specified at the reference
    # rollout size of total_steps_per_epoch=8192 (where they were tuned);
    # train.py rescales them per-sample automatically if the actual rollout
    # size differs, so D's wallclock behavior stays consistent.
    D_min: float = 0.01       # floor (0.01% boss HP per hit) — prevents reward blowup early
    D_initial: float = 2.0    # starting difficulty (% boss HP dealt per hit taken)
    D_ema: float = 0.95       # smoothing at 8192-step epochs: D moves 5% toward new value per epoch
    D_max_delta: float = 0.05 # max relative change per 8192-step epoch (5%)
    D_window: int = 10        # rolling window size at 8192-step epochs (auto-widened for smaller rollouts)

    # Heal reward: coefficient for HP restored. Unscaled by D (like defense).
    # Pegged relative to defense penalty — 0.65 means healing undoes ~65% of
    # the penalty for having taken that damage. Creates dodge > heal > tank ordering.
    heal_coef: float = 0.65

    # Fake-reset (boss-rotation throttling). With this enabled, the C# damage
    # hooks intercept lethal hits with probability fake_reset_prob, clamp them
    # to leave 1 HP, then restore both knight and boss HPs to max. The
    # resulting "fake death" is reported to Python with info="fake_reset"; the
    # next reset takes a fast-path (no scene load) as long as Python sends the
    # same level back. boss_rotation_period controls cluster length: stays on
    # the same boss for N episodes (N-1 fake resets + 1 real on rotation).
    # Set fake_reset_prob=0 to disable entirely; set boss_rotation_period=1 to
    # re-roll the boss every death (forces real resets when prob>0 since the
    # scene mismatch in C# Reset() falls through to the full-load path).
    # FK_FAKE_RESET_PROB env var is set by train.py from this value.
    # Disabled by default; opt in by setting fake_reset_prob > 0 AND
    # boss_rotation_period > 1. period=1 re-rolls every death (matches
    # pre-fake-reset behavior even when prob>0, since the C# scene-match
    # guard falls through to a full reset on rotation).
    fake_reset_prob: float = 1.0
    boss_rotation_period: int = 10


    # PPO
    lr: float = 3e-4
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.02
    max_grad_norm: float = 0.5
    target_kl: float = 0.0

    # Training
    # total_env_steps: total env-steps collected across the whole run.
    # Replaces the old epoch-based budget; makes LR annealing, save cadence,
    # and termination independent of wall-clock epoch size.
    total_env_steps: int = 18_000_000
    total_steps_per_epoch: int = 1024
    batch_size: int = 128
    train_iters: int = 2
    anneal_lr: bool = True
    save_every_steps: int = 51_200
    # How often to hard-kill and relaunch every HK instance during training.
    # Patches the long-running env-state exploitation observed across multi-hour
    # runs (the policy starts to depend on accumulated FSM/cache/heap state
    # that doesn't exist on fresh launches; D inflates and doesn't transfer).
    # All instances restart synchronously at the same epoch boundary; cost is
    # ~30s of training pause every cadence. 0 disables.
    hard_restart_every_epochs: int = 1000
    save_path: str = "models/fullknight"
    wandb_project: str = "fullknight"

    # Resume
    resume: str = ""  # path to checkpoint to resume from

    # Reproducibility
    seed: int = 0  # 0 = non-deterministic

    # Time budget (seconds, 0 = unlimited). Disables wandb when set.
    time_budget: int = 0

    # CUDA Graphs for collect_action - captures the per-step forward (h2d +
    # forward + d2h) into a replay-only graph, slashing CPU launch overhead
    # which dominates per-step wallclock for our small model on a fast GPU.
    # Requires that batch dim B equals n_envs (we pad inactive envs with
    # zero-masked rows), and that combat/terrain dims fit the bucket caps
    # below. Caps are ceiling-pad targets; pick them above your typical
    # max observed values, but not so high that wasted compute exceeds the
    # launch-overhead savings.
    use_cuda_graphs: bool = True
    graph_combat_buckets: str = "8,16,32,64"   # comma-separated
    graph_terrain_buckets: str = "96"          # one bucket is plenty if terrain is camera-clipped

    # Debug
    visualize: bool = False
    # Persist env 0's boss FSM transition graph + state-change history to
    # state_graphs/latest.json each epoch so graph_viewer.py can render it
    # offline. Observation-only — never enters the model input. Set False
    # to skip the per-epoch JSON write entirely (no parsing, no I/O).
    save_fsm_graph: bool = True
    # When True, fsm_tracker logs every state transition it ingests. Verbose;
    # leave False unless debugging the tracker itself.
    debug_transitions: bool = False

    @classmethod
    def from_cli(cls) -> "Config":
        """Build Config from dataclass defaults, overridden by any CLI args."""
        parser = argparse.ArgumentParser()
        defaults = cls()
        for f in fields(cls):
            if f.type is bool:
                parser.add_argument(f"--{f.name}", action="store_true", default=None)
                parser.add_argument(f"--no-{f.name}", dest=f.name, action="store_false")
            else:
                parser.add_argument(f"--{f.name}", type=f.type, default=None)
        args = parser.parse_args()
        overrides = {k: v for k, v in vars(args).items() if v is not None}
        return cls(**overrides)
