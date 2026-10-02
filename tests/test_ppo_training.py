import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO  # noqa: E402

from qb_sim.training.env import QBPassingEnv  # noqa: E402
from qb_sim.training.evaluation import evaluate  # noqa: E402
from qb_sim.training.ppo_config import PPOConfig  # noqa: E402
from qb_sim.training.train import train  # noqa: E402


def test_short_training_saves_model_config_and_checkpoints(tmp_path) -> None:
    config = PPOConfig(
        run_name="test-run",
        total_timesteps=512,
        seed=0,
        n_steps=128,
        batch_size=64,
        n_epochs=2,
        checkpoint_freq=256,
        eval_freq=256,
        eval_episodes=2,
        output_dir=str(tmp_path),
        device="cpu",
    )

    _, final_path = train(config)

    run_dir = tmp_path / "test-run"
    assert final_path.exists()
    assert (run_dir / "config.json").exists()
    assert list((run_dir / "checkpoints").glob("*.zip"))

    loaded = PPO.load(final_path, device="cpu")
    observation, _ = QBPassingEnv().reset(seed=0)
    action, _ = loaded.predict(observation, deterministic=True)
    assert np.asarray(action).shape == (6,)


def test_ppo_evaluation_is_reproducible(tmp_path) -> None:
    config = PPOConfig(
        run_name="eval-run",
        total_timesteps=256,
        seed=1,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=1,
        output_dir=str(tmp_path),
        device="cpu",
    )
    model, _ = train(config)

    def policy(env, observation):
        action, _ = model.predict(observation, deterministic=True)
        return action

    first = evaluate(policy, "default", range(3))
    second = evaluate(policy, "default", range(3))

    assert first == second
