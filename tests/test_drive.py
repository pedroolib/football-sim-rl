from dataclasses import replace

import pytest

from qb_sim.config import SimulationConfig
from qb_sim.engine.entities import PlayOutcome
from qb_sim.rules.drive import Drive, DriveConfig
from qb_sim.rules.routes import DEFAULT_ROUTE_POOL
from qb_sim.rules.scenarios import PRESSURED_SCENARIO

ZERO_DEVIATION = replace(
    SimulationConfig(),
    throw_base_sigma_degrees=0.0,
    throw_move_penalty=0.0,
    throw_pressure_penalty=0.0,
    throw_distance_penalty=0.0,
    throw_power_sigma=0.0,
)


def _drive(**overrides) -> Drive:
    return Drive(ZERO_DEVIATION, DriveConfig(**overrides), seed=5)


def _finish_play(play, outcome, *, carrier_x=None, qb_x=None, throw_time=None) -> None:
    play.state.outcome = outcome
    if carrier_x is not None:
        play.state.carrier_id = "wr1"
        play.state.player("wr1").position[0] = carrier_x
    if qb_x is not None:
        play.state.player("qb").position[0] = qb_x
    play.state.throw_time = throw_time


def test_reset_samples_the_los_inside_the_range() -> None:
    drive = _drive(los_range=(30.0, 40.0))

    assert 30.0 <= drive.state.los <= 40.0
    assert drive.state.down == 1
    assert drive.state.distance == pytest.approx(10.0)


def test_same_seed_reproduces_los_and_routes() -> None:
    first = _drive(los_range=(25.0, 50.0))
    second = _drive(los_range=(25.0, 50.0))

    first_play = first.start_play()
    second_play = second.start_play()

    assert first.state.los == pytest.approx(second.state.los)
    assert first.routes == second.routes
    assert first_play.seed == second_play.seed
    assert set(DEFAULT_ROUTE_POOL).issuperset(first.routes)


def test_loss_of_down_advances_without_moving_the_los() -> None:
    drive = _drive(los_range=(40.0, 40.0))
    play = drive.start_play()
    _finish_play(play, PlayOutcome.INCOMPLETE)

    result = drive.advance(play)

    assert not result.drive_over
    assert drive.state.down == 2
    assert drive.state.los == pytest.approx(40.0)
    assert drive.state.distance == pytest.approx(10.0)


def test_third_loss_of_down_ends_the_drive() -> None:
    drive = _drive(los_range=(40.0, 40.0))
    for _ in range(3):
        play = drive.start_play()
        _finish_play(play, PlayOutcome.INCOMPLETE)
        result = drive.advance(play)

    assert result.drive_over
    assert result.end_reason == "turnover_on_downs"
    assert drive.state.end_reason == "turnover_on_downs"
    assert drive.state.terminated


def test_conversion_resets_downs_and_moves_the_los() -> None:
    drive = _drive(los_range=(40.0, 40.0))
    play = drive.start_play()
    _finish_play(play, PlayOutcome.COMPLETED, carrier_x=52.0)

    result = drive.advance(play)

    assert result.first_down
    assert drive.state.down == 1
    assert drive.state.los == pytest.approx(52.0)
    assert drive.state.distance == pytest.approx(10.0)
    assert drive.state.first_downs == 1


def test_near_the_goal_line_the_distance_is_goal_to_go() -> None:
    drive = _drive(los_range=(95.0, 95.0))
    assert drive.state.distance == pytest.approx(5.0)

    play = drive.start_play()
    _finish_play(play, PlayOutcome.COMPLETED, carrier_x=97.0)
    drive.advance(play)

    assert drive.state.down == 2
    assert drive.state.distance == pytest.approx(3.0)


def test_non_conversion_recomputes_distance_to_the_marker() -> None:
    drive = _drive(los_range=(40.0, 40.0), downs=3, first_down_yards=10.0)
    play = drive.start_play()
    _finish_play(play, PlayOutcome.COMPLETED, carrier_x=43.0)

    drive.advance(play)

    assert drive.state.down == 2
    assert drive.state.los == pytest.approx(43.0)
    assert drive.state.distance == pytest.approx(7.0)


def test_touchdown_ends_the_drive_and_scores() -> None:
    drive = _drive(los_range=(40.0, 40.0))
    play = drive.start_play()
    _finish_play(play, PlayOutcome.TOUCHDOWN, carrier_x=100.0)

    result = drive.advance(play)

    assert result.touchdown
    assert result.points == pytest.approx(7.0)
    assert result.end_reason == "touchdown"
    assert drive.state.points == pytest.approx(7.0)
    assert drive.state.end_reason == "touchdown"
    assert drive.state.terminated


def test_interception_ends_the_drive() -> None:
    drive = _drive(los_range=(40.0, 40.0))
    play = drive.start_play()
    _finish_play(play, PlayOutcome.INTERCEPTION)

    result = drive.advance(play)

    assert result.drive_over
    assert result.end_reason == "interception"
    assert drive.state.end_reason == "interception"


def test_timeout_without_a_throw_is_a_coverage_sack() -> None:
    drive = _drive(los_range=(40.0, 40.0))
    play = drive.start_play()
    _finish_play(play, PlayOutcome.TIME_LIMIT, qb_x=37.0)

    result = drive.advance(play)

    assert result.coverage_sack
    assert result.outcome is PlayOutcome.SACK
    assert result.spot == pytest.approx(37.0)
    assert drive.state.down == 2


def test_timeout_with_the_ball_in_flight_is_an_incompletion() -> None:
    drive = _drive(los_range=(40.0, 40.0))
    play = drive.start_play()
    _finish_play(play, PlayOutcome.TIME_LIMIT, throw_time=2.0)

    result = drive.advance(play)

    assert result.outcome is PlayOutcome.INCOMPLETE
    assert result.spot == pytest.approx(40.0)


def test_sack_spot_is_the_quarterback_position() -> None:
    drive = _drive(los_range=(40.0, 40.0))
    play = drive.start_play()
    _finish_play(play, PlayOutcome.SACK, qb_x=36.5)

    result = drive.advance(play)

    assert result.spot == pytest.approx(36.5)
    assert result.yards == pytest.approx(-3.5)


def test_spot_never_drops_below_the_formation_floor() -> None:
    drive = _drive(los_range=(25.0, 25.0))
    play = drive.start_play()
    _finish_play(play, PlayOutcome.SACK, qb_x=0.0)

    result = drive.advance(play)

    assert result.spot >= 2.0
    assert drive.state.los >= 2.0


def test_min_los_includes_placeholder_starts() -> None:
    drive = Drive(PRESSURED_SCENARIO.config, DriveConfig(los_range=(5.0, 5.0)))

    assert drive.state.los >= 20.0
    play = drive.start_play()

    assert play.line_of_scrimmage == pytest.approx(drive.state.los)


def test_start_play_after_a_deep_loss_does_not_raise() -> None:
    drive = _drive(los_range=(25.0, 25.0))
    play = drive.start_play()
    _finish_play(play, PlayOutcome.SACK, qb_x=0.0)
    drive.advance(play)

    next_play = drive.start_play()

    assert next_play.line_of_scrimmage == pytest.approx(drive.state.los)


def test_max_plays_ends_the_drive() -> None:
    drive = _drive(los_range=(40.0, 40.0), max_plays=2)
    for _ in range(2):
        play = drive.start_play()
        _finish_play(play, PlayOutcome.COMPLETED, carrier_x=52.0)
        result = drive.advance(play)

    assert result.drive_over
    assert result.end_reason == "max_plays"
    assert drive.state.end_reason == "max_plays"


def test_start_play_after_termination_is_rejected() -> None:
    drive = _drive(los_range=(40.0, 40.0))
    play = drive.start_play()
    _finish_play(play, PlayOutcome.INTERCEPTION)
    drive.advance(play)

    with pytest.raises(ValueError):
        drive.start_play()


def test_advance_rejects_an_in_progress_play() -> None:
    drive = _drive(los_range=(40.0, 40.0))
    play = drive.start_play()

    with pytest.raises(ValueError):
        drive.advance(play)


def test_drive_stores_the_last_throw_distance() -> None:
    drive = _drive(los_range=(40.0, 40.0))
    play = drive.start_play()
    _finish_play(play, PlayOutcome.INCOMPLETE, throw_time=0.5)
    play.state.ball.flight_distance = 22.5

    drive.advance(play)

    assert drive.state.last_throw_distance == pytest.approx(22.5)


def test_last_throw_distance_persists_across_plays_without_a_throw() -> None:
    drive = _drive(los_range=(40.0, 40.0))
    play = drive.start_play()
    _finish_play(play, PlayOutcome.INCOMPLETE, throw_time=0.5)
    play.state.ball.flight_distance = 18.0
    drive.advance(play)
    play = drive.start_play()
    _finish_play(play, PlayOutcome.SACK, qb_x=36.0)

    drive.advance(play)

    assert drive.state.last_throw_distance == pytest.approx(18.0)


def test_last_throw_distance_resets_with_the_drive() -> None:
    drive = _drive(los_range=(40.0, 40.0))
    play = drive.start_play()
    _finish_play(play, PlayOutcome.INCOMPLETE, throw_time=0.5)
    play.state.ball.flight_distance = 18.0
    drive.advance(play)

    drive.reset(seed=7)

    assert drive.state.last_throw_distance is None
