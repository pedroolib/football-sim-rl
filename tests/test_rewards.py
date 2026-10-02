import pytest

from qb_sim.engine.entities import PlayOutcome
from qb_sim.training.rewards import RewardConfig, terminal_reward


def test_completion_reward_includes_yardage() -> None:
    config = RewardConfig()

    assert terminal_reward(PlayOutcome.COMPLETED, yards=10.0, config=config) == pytest.approx(
        1.0 + 10.0 * 0.02
    )


def test_each_outcome_maps_to_its_coefficient() -> None:
    config = RewardConfig(
        completion=2.0,
        interception=-3.0,
        incompletion=-0.5,
        out_of_bounds=-0.6,
        time_limit=-0.7,
        sack=-2.5,
        yard=0.0,
    )

    assert terminal_reward(PlayOutcome.COMPLETED, 0.0, config) == pytest.approx(2.0)
    assert terminal_reward(PlayOutcome.INTERCEPTION, 0.0, config) == pytest.approx(-3.0)
    assert terminal_reward(PlayOutcome.INCOMPLETE, 0.0, config) == pytest.approx(-0.5)
    assert terminal_reward(PlayOutcome.OUT_OF_BOUNDS, 0.0, config) == pytest.approx(-0.6)
    assert terminal_reward(PlayOutcome.TIME_LIMIT, 0.0, config) == pytest.approx(-0.7)
    assert terminal_reward(PlayOutcome.SACK, 0.0, config) == pytest.approx(-2.5)


def test_in_progress_has_no_terminal_reward() -> None:
    assert terminal_reward(PlayOutcome.IN_PROGRESS, yards=5.0, config=RewardConfig()) == 0.0


def test_stalling_is_worse_than_an_incomplete_attempt() -> None:
    config = RewardConfig()

    stalling = config.time_limit + config.time_penalty * 120
    attempt = config.incompletion + config.time_penalty * 40

    assert stalling < attempt
