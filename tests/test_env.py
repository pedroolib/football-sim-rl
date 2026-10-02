import subprocess
import sys
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from qb_sim.config import SimulationConfig
from qb_sim.engine.entities import BallStatus
from qb_sim.engine.geometry import distance
from qb_sim.rules.routes import RouteName
from qb_sim.rules.scenarios import Scenario
from qb_sim.training.env import ACTION_SIZE, QBPassingEnv
from qb_sim.training.observations import observation_size

ZERO_DEVIATION = replace(
    SimulationConfig(),
    throw_base_sigma_degrees=0.0,
    throw_move_penalty=0.0,
    throw_pressure_penalty=0.0,
    throw_distance_penalty=0.0,
    throw_power_sigma=0.0,
)


def _open_scenario() -> Scenario:
    return Scenario(
        "test-open",
        replace(
            ZERO_DEVIATION,
            quarterback_start=(20.0, 26.65),
            receiver_starts=((30.0, 26.65), (119.0, 0.0)),
            defender_starts=((0.0, 0.0), (0.0, 0.0)),
        ),
        (RouteName.GO, RouteName.GO),
    )


def _covered_scenario() -> Scenario:
    return Scenario(
        "test-covered",
        replace(
            ZERO_DEVIATION,
            quarterback_start=(20.0, 26.65),
            receiver_starts=((30.0, 43.0), (119.0, 0.0)),
            defender_starts=((28.7, 40.87), (0.0, 0.0)),
            defender_speed=1.0,
        ),
        (RouteName.POST, RouteName.POST),
    )


def _noop() -> np.ndarray:
    return np.zeros(ACTION_SIZE, dtype=np.float32)


def _throw_at_receiver(env: QBPassingEnv, player_id: str = "wr1", lead: bool = True) -> np.ndarray:
    state = env.play.state
    quarterback = state.player("qb")
    receiver = state.player(player_id)
    flight = distance(quarterback.position, receiver.position) / env.play.config.ball_speed
    target = receiver.position + (receiver.velocity * flight if lead else np.zeros(2))
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
    env = QBPassingEnv()

    assert env.action_space.shape == (ACTION_SIZE,)
    assert env.observation_space.shape == (observation_size(SimulationConfig()),)
    assert env.action_space.dtype == np.float32
    assert env.observation_space.dtype == np.float32


def test_reset_returns_observation_and_info() -> None:
    env = QBPassingEnv()

    observation, info = env.reset(seed=42)

    assert env.observation_space.contains(observation)
    assert info["episode_seed"] == 42
    assert info["outcome"] == "in_progress"


def test_reset_is_deterministic_with_the_same_seed() -> None:
    first, _ = QBPassingEnv().reset(seed=42)
    second, _ = QBPassingEnv().reset(seed=42)

    np.testing.assert_allclose(first, second)


def test_step_returns_the_gymnasium_api() -> None:
    env = QBPassingEnv()
    env.reset(seed=0)

    observation, reward, terminated, truncated, info = env.step(_noop())

    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_environment_passes_the_gymnasium_checker() -> None:
    env = gym.make("QBPassing-v0")

    check_env(env.unwrapped)

    env.close()


def test_triggered_throw_launches_the_ball() -> None:
    env = QBPassingEnv()
    env.reset(seed=0)

    _, _, _, _, info = env.step(np.array([0.0, 0.0, 1.0, 0.0, 0.5, 1.0], dtype=np.float32))

    assert env.play.state.ball.status is BallStatus.IN_FLIGHT
    assert info["throw_time"] == 0.0


def test_early_throw_attempt_gets_no_shaping() -> None:
    env = QBPassingEnv()
    env.reset(seed=0)

    _, reward, _, _, _ = env.step(np.array([0.0, 0.0, 1.0, 0.0, 0.5, 1.0], dtype=np.float32))

    assert reward == pytest.approx(-0.002)


def test_throw_attempt_after_min_time_adds_shaping() -> None:
    env = QBPassingEnv()
    env.reset(seed=0)
    for _ in range(10):
        env.step(_noop())
    assert env.play.state.elapsed_seconds == pytest.approx(0.5)

    _, reward, _, _, _ = env.step(np.array([0.0, 0.0, 1.0, 0.0, 0.5, 1.0], dtype=np.float32))

    assert reward == pytest.approx(0.1 - 0.002)


def test_attempt_scale_disables_the_shaping() -> None:
    env = QBPassingEnv()
    env.reset(seed=0)
    for _ in range(10):
        env.step(_noop())
    env.set_attempt_scale(0.0)

    _, reward, _, _, _ = env.step(np.array([0.0, 0.0, 1.0, 0.0, 0.5, 1.0], dtype=np.float32))

    assert reward == pytest.approx(-0.002)


def test_noop_step_only_pays_the_time_penalty() -> None:
    env = QBPassingEnv()
    env.reset(seed=0)

    _, reward, _, _, _ = env.step(_noop())

    assert reward == pytest.approx(-0.002)


def test_completion_episode_has_positive_total_reward() -> None:
    env = QBPassingEnv(_open_scenario())
    env.reset(seed=0)
    total = 0.0
    terminated = truncated = False
    for _ in range(5):
        _, reward, terminated, truncated, _ = env.step(_noop())
        total += reward
    assert not terminated

    _, reward, terminated, truncated, info = env.step(_throw_at_receiver(env))
    total += reward
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(_noop())
        total += reward

    assert info["outcome"] == "completed"
    assert total > 0.0


def test_interception_episode_has_negative_total_reward() -> None:
    env = QBPassingEnv(_covered_scenario())
    env.reset(seed=0)

    _, reward, terminated, truncated, info = env.step(_throw_at_receiver(env, lead=False))
    total = reward
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(_noop())
        total += reward

    assert info["outcome"] == "interception"
    assert total < 0.0


def test_time_penalty_makes_stalling_negative() -> None:
    env = QBPassingEnv(max_steps=10)
    env.reset(seed=0)

    total = 0.0
    for _ in range(10):
        _, reward, _, truncated, _ = env.step(_noop())
        total += reward
        if truncated:
            break

    assert total < 0.0


def test_truncation_at_max_steps() -> None:
    scenario = Scenario(
        "test-long",
        replace(ZERO_DEVIATION, max_play_duration=10.0),
        (RouteName.POST, RouteName.SLANT),
    )
    env = QBPassingEnv(scenario, max_steps=3)
    env.reset(seed=0)

    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(_noop())

    assert not terminated
    assert truncated


def test_random_actions_keep_observations_finite_and_bounded() -> None:
    env = QBPassingEnv()
    env.reset(seed=0)

    for _ in range(30):
        observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.isfinite(observation).all()
        assert env.observation_space.contains(observation)
        if terminated or truncated:
            break


def test_training_env_does_not_import_pygame() -> None:
    code = "import sys; import qb_sim.training.env; assert 'pygame' not in sys.modules"

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
