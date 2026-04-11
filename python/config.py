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
    n_envs: int = 8
    level: str = "GG_Mega_Moss_Charger"  # used by eval
    boss_levels: str = "GG_Mega_Moss_Charger"  # comma-separated pool for training
    frames_per_wait: int = 5
    time_scale: int = 3

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
    terrain_feature_dim: int = 5 # [rel_x, rel_y, w, h, is_trigger]
    terrain_normalized_dims: int = 4  # is_trigger passes through raw
    global_state_dim: int = 22   # vel(2), hp, soul, knight_bounds(2), 7 ability flags, 9 validity flags
    n_binary_flags: int = 16     # 7 ability unlock + 9 action validity (not normalized)

    # Encoder dims
    global_hidden: int = 64
    global_output: int = 64
    combat_hidden: int = 64
    combat_output: int = 64
    terrain_hidden: int = 64
    terrain_output: int = 64
    hidden_dim: int = 256

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

    # Adaptive reward scaling: D = % of boss HP dealt per hit taken
    D_min: float = 0.01       # floor (0.01% boss HP per hit) — prevents reward blowup early
    D_initial: float = 2.0    # starting difficulty (% boss HP dealt per hit taken)
    D_ema: float = 0.9        # smoothing: D moves 10% toward new value each epoch
    D_max_delta: float = 0.10 # max relative change per epoch (10%) — primary smoothing is the EMA + rolling window, not the clamp
    D_window: int = 4         # rolling window of epochs for D_raw computation


    # PPO
    lr: float = 5e-4
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_value_loss: float = 10.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.03

    # Training
    epochs: int = 2000
    total_steps_per_epoch: int = 2048
    batch_size: int = 128
    train_iters: int = 4
    anneal_lr: bool = True
    save_every: int = 50
    save_path: str = "models/fullknight"
    wandb_project: str = "fullknight"

    # Resume
    resume: str = ""  # path to checkpoint to resume from

    # Reproducibility
    seed: int = 0  # 0 = non-deterministic

    # Time budget (seconds, 0 = unlimited). Disables wandb when set.
    time_budget: int = 0

    # Debug
    visualize: bool = False

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
