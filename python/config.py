from dataclasses import dataclass


@dataclass
class Config:
    # Environment
    server_host: str = "localhost"
    server_port: int = 8765
    n_envs: int = 4
    level: str = "GG_Mega_Moss_Charger"
    frames_per_wait: int = 5
    time_scale: int = 3

    # Hollow Knight paths (Windows)
    hk_path: str = r"F:\GOG Games\Hollow Knight"
    hk_data_dir: str = "Hollow Knight_Data"

    # Observation dims
    hitbox_feature_dim: int = 5  # [rel_x, rel_y, w, h, is_trigger]
    global_state_dim: int = 14   # vel, hp, soul, abilities, boss_hp, knight_bounds, 6 validity flags
    n_validity_flags: int = 6

    # Encoder dims
    combat_hidden: int = 64
    combat_output: int = 64
    terrain_hidden: int = 64
    terrain_output: int = 64
    hidden_dim: int = 256

    # Action dims
    movement_n: int = 3   # left, right, none
    direction_n: int = 3  # up, down, none
    action_n: int = 4     # attack, spell, dash, none
    jump_n: int = 2       # yes, no

    # PPO
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.03

    # Training
    epochs: int = 2000
    rollout_len: int = 256
    batch_size: int = 128
    train_iters: int = 4
    anneal_lr: bool = True
    save_every: int = 50
    save_path: str = "models/fullknight"
    log_path: str = "logs/fullknight"
