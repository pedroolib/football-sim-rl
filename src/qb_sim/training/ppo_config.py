"""Configuration for PPO experiments."""

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Hyperparameters and experiment metadata for one PPO run."""

    run_name: str = "ppo_default"
    scenario: str = "default"
    env_kind: str = "passing"
    scenarios: tuple[str, ...] = ()
    total_timesteps: int = 200_000
    seed: int = 0
    n_envs: int = 1
    learning_rate: float = 2e-5
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.05
    target_kl: float = 0.01
    ent_coef: float = 0.0
    vf_coef: float = 0.25
    max_grad_norm: float = 0.5
    pi_layers: tuple[int, ...] = (128, 128)
    vf_layers: tuple[int, ...] = (128, 128)
    log_std_init: float = -2.0
    device: str = "auto"
    checkpoint_freq: int = 25_000
    eval_freq: int = 10_000
    eval_episodes: int = 20
    attempt_decay_steps: int = 1
    imitation_episodes: int = 300
    imitation_epochs: int = 200
    imitation_learning_rate: float = 1e-3
    output_dir: str = "runs"

    def to_dict(self) -> dict:
        """Return a JSON-serializable copy of the configuration."""
        return dataclasses.asdict(self)
