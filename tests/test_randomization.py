import numpy as np
import pytest

from qb_sim.config import SimulationConfig
from qb_sim.rules.scenarios import RUSHED_SCENARIO
from qb_sim.training.randomization import RandomizedEnv, SpeedRanges, sample_config


def test_sample_config_stays_within_the_ranges() -> None:
    ranges = SpeedRanges()
    base = SimulationConfig()

    for seed in range(50):
        config = sample_config(base, np.random.default_rng(seed), ranges)
        assert ranges.receiver[0] <= config.receiver_speed <= ranges.receiver[1]
        assert ranges.rusher[0] <= config.rusher_speed <= ranges.rusher[1]
        gap = config.receiver_speed - config.defender_speed
        assert ranges.defender_gap[0] - 1e-9 <= gap <= ranges.defender_gap[1] + 1e-9
        assert config.defender_speed > 0.0


def test_sample_config_is_reproducible() -> None:
    ranges = SpeedRanges()
    first = sample_config(SimulationConfig(), np.random.default_rng(7), ranges)
    second = sample_config(SimulationConfig(), np.random.default_rng(7), ranges)

    assert first == second


def test_ranges_are_validated() -> None:
    with pytest.raises(ValueError):
        SpeedRanges(receiver=(7.0, 6.0))


def test_randomized_env_reset_is_reproducible() -> None:
    env = RandomizedEnv(RUSHED_SCENARIO)

    first_observation, _ = env.reset(seed=5)
    first_config = env.config
    second_observation, _ = env.reset(seed=5)
    second_config = env.config

    np.testing.assert_allclose(first_observation, second_observation)
    assert first_config == second_config
    env.close()


def test_randomized_env_varies_speeds_across_seeds() -> None:
    env = RandomizedEnv(RUSHED_SCENARIO)
    sampled = set()

    for seed in range(5):
        env.reset(seed=seed)
        sampled.add((env.config.receiver_speed, env.config.defender_speed, env.config.rusher_speed))

    assert len(sampled) == 5
    env.close()


def test_randomized_env_persists_the_attempt_scale() -> None:
    env = RandomizedEnv(RUSHED_SCENARIO)

    env.set_attempt_scale(0.0)
    env.reset(seed=0)

    assert env._env.attempt_scale == 0.0
    env.close()
