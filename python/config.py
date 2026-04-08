import argparse
from dataclasses import dataclass, fields


@dataclass
class Config:
    @property
    def rollout_len(self) -> int:
        return self.total_steps_per_epoch // self.n_envs

    # Environment
    server_host: str = "localhost"
    server_port: int = 8765
    n_envs: int = 1
    level: str = "GG_Mega_Moss_Charger"
    frames_per_wait: int = 5
    time_scale: int = 3

    # Hollow Knight paths (Windows)
    hk_path: str = r"C:\Program Files (x86)\Steam\steamapps\common\Hollow Knight"
    hk_data_dir: str = "hollow_knight_Data"

    # Observation dims
    combat_feature_dim: int = 7  # [rel_x, rel_y, w, h, is_trigger, hurts_knight, is_target]
    terrain_feature_dim: int = 5 # [rel_x, rel_y, w, h, is_trigger]
    global_state_dim: int = 23   # vel, hp, soul, boss_hp, knight_bounds, 7 ability flags, 9 validity flags
    n_binary_flags: int = 16     # 7 ability unlock + 9 action validity (not normalized)

    # Encoder dims
    global_hidden: int = 64
    global_output: int = 64
    combat_hidden: int = 64
    combat_output: int = 64
    terrain_hidden: int = 64
    terrain_output: int = 64
    hidden_dim: int = 256

    # Action dims
    movement_n: int = 3   # left, right, none
    direction_n: int = 3  # up, down, none
    action_n: int = 8     # attack_tap, nail_charge, spell_tap, focus, dash, dream_nail, super_dash, none
    jump_n: int = 2       # yes, no

    # Adaptive reward scaling (nail-hit-equivalent units)
    D_min: float = 0.05       # floor
    D_max: float = 100.0      # ceiling: near-perfect play
    D_initial: float = 0.6    # starting difficulty (nail-equivalent damage landed per hit taken)
    D_ema: float = 0.8        # smoothing: D moves 20% toward new value each epoch
    D_max_delta: float = 0.1  # max relative change per epoch (10%)


    # PPO
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coeff: float = 0.25
    entropy_coeff: float = 0.05
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
