from dataclasses import dataclass


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

    # Debug
    visualize: bool = False
