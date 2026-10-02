"""Configuration for PPO experiments."""

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Hyperparameters and experiment metadata for one PPO run."""

    run_name: str = "ppo_default"
    scenario: str = "default"
    total_timesteps: int = 200_000
    seed: int = 0
    n_envs: int = 1
    learning_rate: float = 2e-4
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.25
    max_grad_norm: float = 0.5
    pi_layers: tuple[int, ...] = (64, 64)
    vf_layers: tuple[int, ...] = (64, 64)
    log_std_init: float = -0.5
    device: str = "auto"
    checkpoint_freq: int = 25_000
    eval_freq: int = 10_000
    eval_episodes: int = 20
    output_dir: str = "runs"

    def to_dict(self) -> dict:
        """Return a JSON-serializable copy of the configuration."""
        return dataclasses.asdict(self)
