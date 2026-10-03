import pytest

from qb_sim.engine.entities import PlayOutcome
from qb_sim.rules.drive import PlayResult
from qb_sim.training.rewards import (
    DriveRewardConfig,
    ReceiverRewardConfig,
    RewardConfig,
    defender_drive_reward,
    defender_potential,
    drive_result_reward,
    receiver_result_reward,
    terminal_reward,
)


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


def test_drive_reward_includes_the_first_down_bonus() -> None:
    result = PlayResult(
        outcome=PlayOutcome.COMPLETED,
        spot=45.0,
        yards=5.0,
        first_down=True,
        touchdown=False,
        drive_over=False,
        end_reason=None,
        points=0.0,
        throw_distance=12.0,
    )

    reward = drive_result_reward(result, DriveRewardConfig())

    assert reward == pytest.approx(1.0 + 5.0 * 0.02 + 0.3)


def test_drive_reward_pays_the_touchdown_points() -> None:
    result = PlayResult(
        outcome=PlayOutcome.TOUCHDOWN,
        spot=100.0,
        yards=15.0,
        first_down=False,
        touchdown=True,
        drive_over=True,
        end_reason="touchdown",
        points=7.0,
        throw_distance=20.0,
    )

    assert drive_result_reward(result, DriveRewardConfig()) == pytest.approx(7.0)


def test_drive_reward_penalizes_turnover_on_downs() -> None:
    result = PlayResult(
        outcome=PlayOutcome.INCOMPLETE,
        spot=40.0,
        yards=0.0,
        first_down=False,
        touchdown=False,
        drive_over=True,
        end_reason="turnover_on_downs",
        points=0.0,
        throw_distance=None,
    )

    assert drive_result_reward(result, DriveRewardConfig()) == pytest.approx(-0.2 - 0.5)


def test_drive_reward_maps_a_coverage_sack_to_the_sack_coefficient() -> None:
    result = PlayResult(
        outcome=PlayOutcome.SACK,
        spot=37.0,
        yards=-3.0,
        first_down=False,
        touchdown=False,
        drive_over=False,
        end_reason=None,
        points=0.0,
        throw_distance=None,
        coverage_sack=True,
    )

    assert drive_result_reward(result, DriveRewardConfig()) == pytest.approx(-1.0)


def test_receiver_reward_config_defaults_to_dense_carry_progress() -> None:
    assert ReceiverRewardConfig().carry_yard == pytest.approx(0.05)


def test_receiver_reward_pays_the_catch_bonus_only_when_catching() -> None:
    config = ReceiverRewardConfig()

    caught = receiver_result_reward(PlayOutcome.COMPLETED, 8.0, True, config)
    not_caught = receiver_result_reward(PlayOutcome.COMPLETED, 8.0, False, config)

    assert caught == pytest.approx(1.0 + 8.0 * 0.02 + 0.3)
    assert not_caught == pytest.approx(1.0 + 8.0 * 0.02)


def test_receiver_reward_keeps_terminal_penalties() -> None:
    config = ReceiverRewardConfig()

    assert receiver_result_reward(PlayOutcome.INTERCEPTION, 0.0, False, config) == pytest.approx(
        -1.0
    )


def test_receiver_reward_pays_the_touchdown_bonus_when_catching() -> None:
    config = ReceiverRewardConfig()

    reward = receiver_result_reward(PlayOutcome.TOUCHDOWN, 0.0, True, config)

    assert reward == pytest.approx(0.0 + 0.3 + 7.0)


def test_receiver_reward_pays_the_touchdown_bonus_without_a_catch() -> None:
    config = ReceiverRewardConfig()

    reward = receiver_result_reward(PlayOutcome.TOUCHDOWN, 0.0, False, config)

    assert reward == pytest.approx(0.0 + 7.0)


def test_defender_reward_is_the_negated_drive_result_reward() -> None:
    result = PlayResult(
        outcome=PlayOutcome.COMPLETED,
        spot=52.0,
        yards=12.0,
        first_down=True,
        touchdown=False,
        drive_over=False,
        end_reason=None,
        points=0.0,
        throw_distance=12.0,
    )

    assert defender_drive_reward(result, DriveRewardConfig()) == pytest.approx(
        -(1.0 + 12.0 * 0.02 + 0.3)
    )


def test_defender_potential_is_bounded_and_monotone() -> None:
    assert defender_potential(0.0) == 0.0
    assert defender_potential(-3.0) == 0.0
    assert defender_potential(1.0) > defender_potential(2.0) > defender_potential(20.0)
    assert defender_potential(20.0) == -2.0
    assert defender_potential(1000.0) == -2.0
    with pytest.raises(ValueError):
        defender_potential(1.0, scale=0.0)
    with pytest.raises(ValueError):
        defender_potential(1.0, scale=-1.0)
