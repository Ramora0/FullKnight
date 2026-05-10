import argparse
from dataclasses import dataclass, fields


# Fields exposed as `--<name>` flags by Config.from_cli. Everything else lives
# as an internal default — model/encoder dims, observation layout, D-curriculum
# tuning constants, vocab caps — and is set in code, not at the command line.
# Add a name here when a knob earns its place on the CLI; if you find yourself
# adding most fields, the whitelist isn't doing its job.
_CLI_FIELDS = frozenset({
    # Run control
    "n_envs", "level", "boss_levels",
    "time_scale", "fps_cap",
    "hk_path", "ipc",
    "seed", "resume", "time_budget", "diag_epochs", "visualize",
    "debug_transitions",
    "wandb_project", "save_path", "save_every_steps",
    "total_env_steps", "total_steps_per_epoch",
    # PPO / training knobs that get ablated
    "lr", "gamma", "gae_lambda", "clip_eps", "value_coeff",
    "entropy_coeff", "max_grad_norm", "target_kl",
    "batch_size", "train_iters", "mirror_aug", "fake_reset_prob",
    "boss_rotation_period",
    "hard_restart_every_epochs",
    "detect_glitch", "glitch_log_dir", "glitch_max_dumps",
    "debug_recoil",
})


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
    # Transport for the step() hot path. "shm" routes action/obs through a
    # named shared-memory + Win32 EventWaitHandle channel (see
    # python/shm_transport.py + Net/ShmChannel.cs); "websocket" keeps the
    # legacy path for A/B testing or rollback. init/reset/pause/resume/close
    # always go over WebSocket regardless. Validated at ~7us mean RTT vs
    # ~340us for WebSocket (python/bench_ipc_mono.py).
    ipc: str = "shm"
    level: str = "GG_Mega_Moss_Charger"  # used by eval
    boss_levels: str = "GG_Mega_Moss_Charger"  # comma-separated pool for training
    # Hardcoded to 1: the agent acts every Unity frame. Per-step game-time is
    # 0.0424s (the C# captureDeltaTime, ~24 game-fps — near the lower bound
    # where HK FSM/animator timing stays sane). Field kept for protocol
    # compat with eval/validate/test tools that still surface it as a knob;
    # train.py does not plumb it.
    frames_per_wait: int = 1
    # time_scale is currently ignored on the C# side (the timeScale mechanism
    # was ripped out so we can isolate captureDeltaTime as the only time knob).
    # Field kept for protocol compat; default 1 reflects current effective value.
    time_scale: int = 1
    # Cap Unity's frame rate (Application.targetFrameRate). 0 = uncapped (-1),
    # the default used in training. Set to e.g. 60 when watching graphical
    # runs locally, so HK doesn't burn CPU at hundreds of fps. Plumbed to C#
    # via FK_FPS_CAP env var on instance launch (see train.py / instance_manager).
    fps_cap: int = 0

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

    # Encoder dims
    global_hidden: int = 64
    global_output: int = 64
    combat_hidden: int = 64
    combat_output: int = 64
    terrain_hidden: int = 64
    terrain_output: int = 64
    hidden_dim: int = 512
    # Multi-head attention pooling in HitboxEncoder. output_dim is split
    # evenly across n_heads (head_dim = output_dim // n_heads). Same param
    # count as single-head; lets different heads specialize on different
    # hitbox roles (target body vs attack collider vs projectile).
    attn_n_heads: int = 4

    # GRU (temporal memory)
    gru_dim: int = 64           # bottleneck dimension for GRU (hidden_dim -> gru_dim -> hidden_dim)
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


    # PPO
    # 50% horizontal-mirror augmentation per minibatch (data aug exploiting
    # HK's L/R symmetry). Toggle for ablation.
    mirror_aug: bool = False
    # Fake-reset probability: when knight or boss would die, with this
    # probability we clamp the lethal damage and instantly restore both HPs
    # to max in-place — skipping the full scene-transition reset (~85% of
    # wallclock at baseline). Plumbed to C# via FK_FAKE_RESET_PROB env var.
    # 0.0 disables (full real reset every death). 1.0 = always fake.
    # Used in conjunction with boss_rotation_period: each env stays on its
    # boss for boss_rotation_period episodes (each fake-reset), then gets
    # a real reset to a new boss to flush HK FSM state.
    fake_reset_prob: float = 1.0
    # Number of consecutive same-boss episodes per env before Python rotates
    # to a new boss. With fake_reset_prob=1.0 + period=10: each cluster is
    # 9 fake-resets (instant in-place HP restore, no scene load) + 1 real
    # reset (with new boss assignment). 0 disables (re-roll every death,
    # the original behavior).
    boss_rotation_period: int = 10
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
    save_every_steps: int = 51_200
    # How often to hard-kill and relaunch every HK instance during training.
    # Patches the long-running env-state exploitation observed across multi-hour
    # runs (the policy starts to depend on accumulated FSM/cache/heap state
    # that doesn't exist on fresh launches; D inflates and doesn't transfer).
    # All instances restart synchronously at the same epoch boundary; cost is
    # ~30s of training pause every cadence. 0 disables.
    hard_restart_every_epochs: int = 0
    save_path: str = "models/fullknight"
    wandb_project: str = "fullknight"

    # Resume
    resume: str = ""  # path to checkpoint to resume from

    # Reproducibility
    seed: int = 0  # 0 = non-deterministic

    # Time budget (seconds, 0 = unlimited). Disables wandb when set.
    time_budget: int = 0

    # CUDA Graphs for collect_action — captures the per-step forward (h2d +
    # forward + d2h) into a replay-only graph, slashing CPU launch overhead
    # which dominates per-step wallclock for our small model on a fast GPU.
    # Requires that batch dim B equals n_envs (we pad inactive envs with
    # zero-masked rows), and that combat/terrain dims fit the bucket caps
    # below. Caps are ceiling-pad targets; pick them above your typical
    # max observed values, but not so high that wasted compute exceeds the
    # launch-overhead savings.
    use_cuda_graphs: bool = True
    # CUDA Graphs for the training inner loop (forward_sequence + loss +
    # backward + clip_grad_norm + optimizer.step captured per bucket pair).
    # Adam needs capturable=True; LR is baked at capture time and set_lr()
    # post-capture is a no-op (acceptable: 20-min runs only drift LR ~5%).
    # Reuses graph_combat_buckets / graph_terrain_buckets below for the
    # bucket dims; they cap per-rollout combat/terrain padded width.
    use_train_cuda_graphs: bool = True
    graph_combat_buckets: str = "8,16,32,64"   # comma-separated
    graph_terrain_buckets: str = "96"          # one bucket is plenty if terrain is camera-clipped

    # Diagnostic mode: when > 0, train.py exits after this many epochs and
    # prints an aggregated wallclock breakdown (rollout vs train, sub-phases
    # of each, reset overhead). Used to identify where time is going so we
    # can target the right bottleneck.
    diag_epochs: int = 0

    # Glitch detection: when an entire epoch passes with zero damage events
    # (no `damage_landed > 0` and no `hits_taken > 0` across every active env
    # for the full rollout), assume the knight/boss-disappeared bug fired and
    # dump the prior epoch + current epoch (per-step actions, global state,
    # combat/terrain hitboxes, kinds, diag counts, step timings) into a .log
    # file. Bounded by glitch_max_dumps so a stuck-glitched run doesn't fill
    # the disk. Logs land under glitch_log_dir; training continues.
    detect_glitch: bool = True
    glitch_log_dir: str = "glitch_logs"
    glitch_max_dumps: int = 5

    # When true, the C# mod logs `[Recoil]` lines on each rising/falling edge of
    # `cState.recoiling || cState.recoilFrozen`, with duration in agent-steps,
    # game-time ms, and real-time ms. Used to verify that knockback duration is
    # invariant to fps_cap / frames_per_wait. Plumbed via FK_DEBUG_RECOIL env var.
    debug_recoil: bool = False

    # Debug
    visualize: bool = False
    # Strip all training-loop prints (timing/perf/leak/policy/slow/epoch
    # summary) and emit only scene-transition events: reset start/done
    # with per-phase breakdown + branch, episode-end (win/loss + level),
    # connection events, hard restarts. C# mod logs ([Phase-Timing],
    # [BounceCheck], [Reset-Timing], [SceneHooks]) still go to HK's mod
    # log file — tail that alongside this output to debug freezes.
    debug_transitions: bool = False

    @classmethod
    def from_cli(cls) -> "Config":
        """Build Config from dataclass defaults, overridden by CLI args.

        Only fields listed in `_CLI_FIELDS` get a `--<name>` flag. Other
        fields are intentionally set in code only — see the whitelist's
        docstring for the rationale.
        """
        parser = argparse.ArgumentParser()
        for f in fields(cls):
            if f.name not in _CLI_FIELDS:
                continue
            if f.type is bool:
                parser.add_argument(f"--{f.name}", action="store_true", default=None)
                parser.add_argument(f"--no-{f.name}", dest=f.name, action="store_false")
            else:
                parser.add_argument(f"--{f.name}", type=f.type, default=None)
        args = parser.parse_args()
        overrides = {k: v for k, v in vars(args).items() if v is not None}
        return cls(**overrides)
