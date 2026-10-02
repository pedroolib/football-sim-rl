import json

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
        attempt_decay_steps=128,
        imitation_episodes=0,
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


def test_attempt_shaping_is_annealed_to_zero(tmp_path) -> None:
    config = PPOConfig(
        run_name="anneal-run",
        total_timesteps=256,
        seed=0,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=1,
        attempt_decay_steps=64,
        imitation_episodes=0,
        output_dir=str(tmp_path),
        device="cpu",
    )

    model, _ = train(config)

    assert model.get_env().envs[0].unwrapped.attempt_scale == 0.0


def test_training_warm_starts_from_demonstrations(tmp_path) -> None:
    config = PPOConfig(
        run_name="bc-run",
        scenario="rushed",
        total_timesteps=64,
        seed=0,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=1,
        attempt_decay_steps=64,
        imitation_episodes=1,
        imitation_epochs=2,
        output_dir=str(tmp_path),
        device="cpu",
    )

    train(config)

    metadata = json.loads((tmp_path / "bc-run" / "config.json").read_text())
    assert metadata["imitation_samples"] > 0
    assert metadata["imitation_loss"] is not None


def test_ppo_action_policy_checks_layout_compatibility(tmp_path) -> None:
    config = PPOConfig(
        run_name="compat-run",
        scenario="rushed",
        total_timesteps=64,
        seed=0,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=1,
        attempt_decay_steps=1,
        imitation_episodes=0,
        output_dir=str(tmp_path),
        device="cpu",
    )
    _, final_path = train(config)

    from qb_sim.training.evaluation import PPOActionPolicy

    policy = PPOActionPolicy(final_path, device="cpu")
    assert policy.supports_layout()


def test_training_can_cycle_scenarios(tmp_path) -> None:
    config = PPOConfig(
        run_name="multi-run",
        scenarios=("default", "rushed"),
        total_timesteps=64,
        seed=0,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=1,
        attempt_decay_steps=1,
        imitation_episodes=0,
        output_dir=str(tmp_path),
        device="cpu",
    )

    train(config)

    metadata = json.loads((tmp_path / "multi-run" / "config.json").read_text())
    assert metadata["scenarios"] == ["default", "rushed"]


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
        imitation_episodes=0,
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


def test_drive_training_smoke(tmp_path) -> None:
    from qb_sim.training.drive_observations import DRIVE_OBSERVATION_SIZE

    config = PPOConfig(
        run_name="drive-run",
        env_kind="drive",
        scenario="rushed",
        total_timesteps=256,
        seed=0,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=1,
        attempt_decay_steps=64,
        imitation_episodes=1,
        imitation_epochs=2,
        output_dir=str(tmp_path),
        device="cpu",
    )

    _, final_path = train(config)

    from qb_sim.rules.routes import ROUTE_OBSERVATION_ORDER

    metadata = json.loads((tmp_path / "drive-run" / "config.json").read_text())
    assert metadata["env_kind"] == "drive"
    assert metadata["observation_size"] == DRIVE_OBSERVATION_SIZE
    assert metadata["route_pool"] == [route.value for route in ROUTE_OBSERVATION_ORDER]
    assert metadata["imitation_samples"] > 0

    loaded = PPO.load(final_path, device="cpu")
    assert loaded.observation_space.shape == (DRIVE_OBSERVATION_SIZE,)


def test_build_env_rejects_unknown_env_kind() -> None:
    from qb_sim.training.train import build_env

    with pytest.raises(ValueError):
        build_env(("rushed",), env_kind="driv")


def test_role_play_training_smoke(tmp_path) -> None:
    from qb_sim.training.role_context_observations import RECEIVER_CONTEXT_OBSERVATION_SIZE

    config = PPOConfig(
        run_name="role-play-run",
        env_kind="role_play",
        scenario="stage_b",
        learner="wr1",
        total_timesteps=128,
        seed=0,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=1,
        attempt_decay_steps=1,
        imitation_episodes=1,
        imitation_epochs=2,
        output_dir=str(tmp_path),
        device="cpu",
    )

    _, final_path = train(config)

    metadata = json.loads((tmp_path / "role-play-run" / "config.json").read_text())
    assert metadata["env_kind"] == "role_play"
    assert metadata["learner"] == "wr1"
    assert metadata["observation_size"] == RECEIVER_CONTEXT_OBSERVATION_SIZE
    assert metadata["fatigue"]["carrier_fatigue_per_second"] > 0.0

    loaded = PPO.load(final_path, device="cpu")
    assert loaded.observation_space.shape == (RECEIVER_CONTEXT_OBSERVATION_SIZE,)


def test_role_drive_training_smoke(tmp_path) -> None:
    config = PPOConfig(
        run_name="role-drive-run",
        env_kind="role_drive",
        scenario="stage_b",
        learner="cb1",
        total_timesteps=128,
        seed=0,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=1,
        attempt_decay_steps=1,
        imitation_episodes=1,
        imitation_epochs=2,
        output_dir=str(tmp_path),
        device="cpu",
    )

    _, final_path = train(config)

    metadata = json.loads((tmp_path / "role-drive-run" / "config.json").read_text())
    assert metadata["observation_size"] == 24
    assert metadata["learner"] == "cb1"
