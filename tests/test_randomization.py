import numpy as np
import pytest

from qb_sim.config import SimulationConfig
from qb_sim.rules.scenarios import RUSHED_SCENARIO, STAGE_B_CONFIG
from qb_sim.training.randomization import (
    AdaptationRanges,
    RandomizedEnv,
    SpeedRanges,
    sample_adaptation_config,
    sample_config,
)


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


def test_sample_adaptation_config_interpolates_with_progress() -> None:
    base = SimulationConfig()
    narrow = sample_adaptation_config(base, np.random.default_rng(0), 0.0)
    wide = sample_adaptation_config(base, np.random.default_rng(0), 1.0)

    # Narrow phase is each full-range center +/- 0.5 (receiver center 7.25).
    assert 6.75 <= narrow.receiver_speed <= 7.75
    assert 39.0 <= narrow.throw_max_range <= 51.0
    assert 6.0 <= wide.receiver_speed <= 8.5
    assert 30.0 <= wide.throw_max_range <= 55.0


def test_sample_adaptation_config_is_reproducible() -> None:
    first = sample_adaptation_config(SimulationConfig(), np.random.default_rng(7), 0.5)
    second = sample_adaptation_config(SimulationConfig(), np.random.default_rng(7), 0.5)

    assert first == second


def test_sample_adaptation_config_keeps_roster_and_fatigue() -> None:
    sampled = sample_adaptation_config(STAGE_B_CONFIG, np.random.default_rng(3), 1.0)

    assert sampled.receiver_starts == STAGE_B_CONFIG.receiver_starts
    assert sampled.rusher_starts == STAGE_B_CONFIG.rusher_starts
    assert sampled.carrier_fatigue_per_second == pytest.approx(
        STAGE_B_CONFIG.carrier_fatigue_per_second
    )
    assert sampled.carrier_min_speed_factor == pytest.approx(
        STAGE_B_CONFIG.carrier_min_speed_factor
    )


def test_adaptation_ranges_defaults_match_the_spec() -> None:
    ranges = AdaptationRanges()

    assert ranges.receiver_speed == (6.0, 8.5)
    assert ranges.defender_speed == (5.5, 7.5)
    assert ranges.quarterback_speed == (5.0, 8.0)
    assert ranges.rusher_speed == (5.0, 8.0)
    assert ranges.throw_max_range == (30.0, 55.0)
    assert ranges.throw_base_sigma_degrees == (0.5, 4.0)
