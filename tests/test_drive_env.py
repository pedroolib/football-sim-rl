from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from qb_sim.config import SimulationConfig
from qb_sim.engine.entities import BallStatus
from qb_sim.engine.geometry import distance
from qb_sim.rules.drive import DriveConfig
from qb_sim.rules.routes import RouteName
from qb_sim.rules.scenarios import DEFAULT_SCENARIO, Scenario
from qb_sim.training.actions import ACTION_SIZE
from qb_sim.training.drive_env import QBDriveEnv
from qb_sim.training.drive_observations import DRIVE_OBSERVATION_SIZE
from qb_sim.training.randomization import AdaptationRanges

ZERO_DEVIATION = replace(
    SimulationConfig(),
    throw_base_sigma_degrees=0.0,
    throw_move_penalty=0.0,
    throw_pressure_penalty=0.0,
    throw_distance_penalty=0.0,
    throw_power_sigma=0.0,
)


def _noop() -> np.ndarray:
    return np.zeros(ACTION_SIZE, dtype=np.float32)


def _sack_scenario() -> Scenario:
    return Scenario(
        "test-sack",
        replace(ZERO_DEVIATION, defender_starts=((20.5, 26.65), (0.0, 0.0))),
        (RouteName.POST, RouteName.SLANT),
    )


def _open_scenario() -> Scenario:
    return Scenario(
        "test-open",
        replace(
            ZERO_DEVIATION,
            receiver_starts=((30.0, 26.65), (40.0, 0.0)),
            defender_starts=((0.0, 0.0), (0.0, 0.0)),
        ),
        (RouteName.POST, RouteName.SLANT),
    )


def _throw_at_receiver(env: QBDriveEnv) -> np.ndarray:
    state = env.play.state
    quarterback = state.player("qb")
    receiver = state.player("wr1")
    flight = distance(quarterback.position, receiver.position) / env.play.config.ball_speed
    target = receiver.position + receiver.velocity * flight
    offset = target - quarterback.position
    length = float(np.linalg.norm(offset))
    direction = offset / length
    span = env.play.config.throw_max_range - env.play.config.throw_min_range
    power = min(max((length - env.play.config.throw_min_range) / span, 0.0), 1.0)
    return np.array(
        [0.0, 0.0, direction[0], direction[1], power * 2.0 - 1.0, 1.0],
        dtype=np.float32,
    )


def test_spaces_have_the_documented_shapes() -> None:
    env = QBDriveEnv()

    assert env.action_space.shape == (ACTION_SIZE,)
    assert env.observation_space.shape == (DRIVE_OBSERVATION_SIZE,)


def test_reset_returns_observation_and_drive_info() -> None:
    env = QBDriveEnv()
    observation, info = env.reset(seed=42)

    assert env.observation_space.contains(observation)
    assert info["episode_seed"] == 42
    assert info["down"] == 1
    assert 25.0 <= info["los"] <= 50.0
    assert info["plays"] == []


def test_reset_is_deterministic_with_the_same_seed() -> None:
    first, first_info = QBDriveEnv().reset(seed=7)
    second, second_info = QBDriveEnv().reset(seed=7)

    np.testing.assert_allclose(first, second)
    assert first_info["los"] == pytest.approx(second_info["los"])


def test_environment_passes_the_gymnasium_checker() -> None:
    env = QBDriveEnv()

    check_env(env)

    env.close()


def test_environment_is_registered() -> None:
    env = gym.make("QBDrive-v0")

    assert env.unwrapped.observation_space.shape == (DRIVE_OBSERVATION_SIZE,)
    env.close()


def test_three_sacks_end_the_drive_on_downs() -> None:
    env = QBDriveEnv(_sack_scenario(), DriveConfig(los_range=(20.0, 20.0)))
    env.reset(seed=0)

    for _ in range(3):
        _, _, terminated, truncated, info = env.step(_noop())

    assert terminated
    assert not truncated
    assert info["drive_end_reason"] == "turnover_on_downs"
    assert len(info["plays"]) == 3
    assert all(play["outcome"] == "sack" for play in info["plays"])


def test_stepping_a_terminated_drive_is_rejected() -> None:
    env = QBDriveEnv(_sack_scenario(), DriveConfig(los_range=(20.0, 20.0)))
    env.reset(seed=0)

    for _ in range(3):
        _, _, terminated, _, _ = env.step(_noop())
    assert terminated

    with pytest.raises(RuntimeError):
        env.step(_noop())


def test_first_down_bonus_is_included_in_the_step_reward() -> None:
    env = QBDriveEnv(_open_scenario(), DriveConfig(los_range=(25.0, 25.0)))
    env.reset(seed=0)
    for _ in range(5):
        env.step(_noop())
    total = 0.0
    while not env.play_history:
        action = (
            _throw_at_receiver(env)
            if env.play.state.ball.status is BallStatus.HELD
            else _noop()
        )
        _, reward, _, _, _ = env.step(action)
        total += reward

    assert env.play_history[0]["first_down"] is True
    assert total > 1.0


def test_same_seed_and_actions_reproduce_a_drive() -> None:
    first_env = QBDriveEnv(_sack_scenario(), DriveConfig(los_range=(20.0, 20.0)))
    second_env = QBDriveEnv(_sack_scenario(), DriveConfig(los_range=(20.0, 20.0)))
    first_env.reset(seed=3)
    second_env.reset(seed=3)

    for _ in range(3):
        first_observation, first_reward, first_term, _, first_info = first_env.step(_noop())
        second_observation, second_reward, second_term, _, second_info = second_env.step(_noop())
        np.testing.assert_allclose(first_observation, second_observation)
        assert first_reward == pytest.approx(second_reward)
        assert first_term == second_term
        assert first_info["plays"] == second_info["plays"]


def test_truncation_at_max_episode_steps() -> None:
    env = QBDriveEnv(DEFAULT_SCENARIO, max_episode_steps=2)
    env.reset(seed=0)

    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(_noop())

    assert not terminated
    assert truncated


def test_attempt_scale_disables_the_attempt_shaping() -> None:
    env = QBDriveEnv(_open_scenario(), DriveConfig(los_range=(25.0, 25.0)))
    env.reset(seed=0)
    for _ in range(10):
        env.step(_noop())
    env.set_attempt_scale(0.0)

    _, reward, _, _, _ = env.step(_throw_at_receiver(env))

    assert reward == pytest.approx(-0.002)


def test_randomized_drive_env_applies_sampled_config() -> None:
    env = QBDriveEnv(randomization=AdaptationRanges())
    env.reset(seed=5)

    assert env.active_config == env.play.config
    assert env.active_config.throw_max_range == pytest.approx(env.play.config.throw_max_range)
    assert env.play.state.player("wr1").max_speed == pytest.approx(
        env.active_config.receiver_speed
    )


class _RecordingConfigOpponent:
    def __init__(self) -> None:
        self.configs: list = []

    def set_config(self, config) -> None:
        self.configs.append(config)

    def __call__(self, state, route, start=None):
        return None


def test_randomized_drive_env_injects_the_active_config_into_opponents() -> None:
    opponent = _RecordingConfigOpponent()
    env = QBDriveEnv(opponents={"wr1": opponent}, randomization=AdaptationRanges())
    env.reset(seed=5)

    assert opponent.configs
    assert opponent.configs[-1] is env.active_config


def test_randomized_drive_env_is_deterministic_for_a_seed() -> None:
    first = QBDriveEnv(randomization=AdaptationRanges())
    second = QBDriveEnv(randomization=AdaptationRanges())
    first_observation, _ = first.reset(seed=5)
    second_observation, _ = second.reset(seed=5)

    assert first.active_config == second.active_config
    np.testing.assert_allclose(first_observation, second_observation)


def test_training_drive_env_does_not_import_pygame() -> None:
    import subprocess
    import sys

    code = "import sys; import qb_sim.training.drive_env; assert 'pygame' not in sys.modules"

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
