from dataclasses import replace

import pytest

from qb_sim.agents.rule_controllers import ManCoverageController
from qb_sim.config import SimulationConfig
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import BallStatus, PlayOutcome
from qb_sim.engine.geometry import vec2
from qb_sim.rules.drive import Drive, DriveConfig
from qb_sim.rules.play import Play

ZERO_DEVIATION = replace(
    SimulationConfig(),
    throw_base_sigma_degrees=0.0,
    throw_move_penalty=0.0,
    throw_pressure_penalty=0.0,
    throw_distance_penalty=0.0,
    throw_power_sigma=0.0,
)

FORWARD = Move(direction=vec2(1.0, 0.0))
BACKWARD = Move(direction=vec2(-1.0, 0.0))


def _qb_sneak(play: Play, limit: int = 200):
    for _ in range(limit):
        state = play.step({"qb": FORWARD})
        if state.outcome is not PlayOutcome.IN_PROGRESS:
            return state
    pytest.fail("QB run did not reach a terminal outcome")


def test_qb_crossing_the_los_and_being_tackled_is_a_rush() -> None:
    config = replace(ZERO_DEVIATION, defender_starts=((22.0, 26.65), (0.0, 0.0)))
    play = Play(config)

    state = play.step({"qb": FORWARD})
    assert state.carrier_id is None
    assert state.ball.status is BallStatus.HELD

    for _ in range(3):
        state = play.step({"qb": FORWARD})

    assert state.outcome is PlayOutcome.RUSH
    assert state.carrier_id == "qb"
    assert state.ball.status is BallStatus.DEAD
    assert state.player("qb").position[0] == pytest.approx(21.3)


def test_rush_spot_is_the_quarterback_position_and_gains_yards() -> None:
    config = replace(ZERO_DEVIATION, defender_starts=((22.0, 26.65), (0.0, 0.0)))
    drive = Drive(config, DriveConfig(los_range=(20.0, 20.0)), seed=1)
    play = drive.start_play()

    state = _qb_sneak(play)
    result = drive.advance(play)

    assert state.outcome is PlayOutcome.RUSH
    assert result.outcome is PlayOutcome.RUSH
    assert result.spot == pytest.approx(21.3)
    assert result.yards == pytest.approx(1.3)
    assert result.yards > 0.0
    assert result.throw_distance is None


def test_qb_running_into_the_end_zone_scores() -> None:
    config = replace(
        ZERO_DEVIATION,
        quarterback_start=(99.0, 26.65),
        receiver_starts=((20.0, 43.0), (20.0, 10.0)),
        defender_starts=((0.0, 0.0), (0.0, 0.0)),
    )
    drive = Drive(config, DriveConfig(los_range=(99.0, 99.0)), seed=2)
    play = drive.start_play()

    state = _qb_sneak(play)
    result = drive.advance(play)

    assert state.outcome is PlayOutcome.TOUCHDOWN
    assert state.carrier_id == "qb"
    assert result.touchdown
    assert result.end_reason == "touchdown"
    assert drive.state.points == pytest.approx(7.0)


def test_qb_tackled_behind_the_los_is_still_a_sack() -> None:
    config = replace(ZERO_DEVIATION, defender_starts=((20.5, 26.65), (0.0, 0.0)))
    play = Play(config)

    state = play.step()

    assert state.outcome is PlayOutcome.SACK
    assert state.carrier_id is None
    assert state.ball.status is BallStatus.DEAD


def test_rush_past_the_marker_moves_the_chains() -> None:
    config = replace(ZERO_DEVIATION, defender_starts=((32.0, 26.65), (0.0, 0.0)))
    drive = Drive(config, DriveConfig(los_range=(40.0, 40.0)), seed=3)
    play = drive.start_play()

    _qb_sneak(play)
    result = drive.advance(play)

    assert result.outcome is PlayOutcome.RUSH
    assert result.yards >= 10.0
    assert result.first_down
    assert drive.state.first_downs == 1
    assert drive.state.down == 1
    assert drive.state.los == pytest.approx(result.spot)


def test_rush_short_on_third_down_turns_it_over() -> None:
    config = replace(ZERO_DEVIATION, defender_starts=((23.0, 26.65), (0.0, 0.0)))
    drive = Drive(config, DriveConfig(los_range=(40.0, 40.0)), seed=4)

    result = None
    for _ in range(3):
        play = drive.start_play()
        _qb_sneak(play)
        result = drive.advance(play)

    assert result is not None
    assert result.outcome is PlayOutcome.RUSH
    assert result.end_reason == "turnover_on_downs"
    assert drive.state.terminated


def test_qb_retreating_behind_the_los_stays_a_runner() -> None:
    config = replace(ZERO_DEVIATION, defender_starts=((30.0, 26.65), (0.0, 0.0)))
    play = Play(config)

    for _ in range(4):
        state = play.step({"qb": FORWARD})
    assert state.carrier_id == "qb"

    for _ in range(5):
        state = play.step({"qb": BACKWARD})

    assert state.outcome is PlayOutcome.IN_PROGRESS
    assert state.carrier_id == "qb"
    assert state.ball.status is BallStatus.CARRIED
    assert state.player("qb").position[0] < play.line_of_scrimmage


def test_qb_cannot_throw_after_crossing_the_los() -> None:
    play = Play(ZERO_DEVIATION)
    for _ in range(4):
        play.step({"qb": FORWARD})

    with pytest.raises(ValueError):
        play.throw_to(vec2(60.0, 26.65))
    with pytest.raises(ValueError):
        play.step({"qb": Throw(direction=vec2(1.0, 0.0), power=0.5)})


def test_coverage_defender_chases_the_running_quarterback() -> None:
    config = replace(ZERO_DEVIATION, defender_starts=((22.5, 26.65), (18.0, 10.0)))
    play = Play(config)
    for _ in range(4):
        state = play.step({"qb": FORWARD})
    assert state.carrier_id == "qb"

    cb1 = ManCoverageController(
        "cb1",
        config.defender_speed,
        assignment_id="wr1",
        time_step=config.dt,
    )
    before = state.player("cb1").position.copy()
    action = cb1.decide(state)
    assert action is not None

    state = play.step({"qb": FORWARD, "cb1": action})

    assert state.outcome is PlayOutcome.RUSH
    assert state.player("cb1").position[1] == pytest.approx(before[1])
    assert state.player("cb1").position[0] < before[0]


def test_qb_rush_is_deterministic_for_the_same_seed_and_actions() -> None:
    config = replace(ZERO_DEVIATION, defender_starts=((30.0, 26.65), (0.0, 0.0)))

    def run() -> list[tuple[tuple[float, float], PlayOutcome, BallStatus]]:
        play = Play(config, seed=7)
        trace = []
        for _ in range(200):
            state = play.step({"qb": FORWARD})
            trace.append(
                (
                    tuple(float(value) for value in state.player("qb").position),
                    state.outcome,
                    state.ball.status,
                )
            )
            if state.outcome is not PlayOutcome.IN_PROGRESS:
                break
        return trace

    first = run()
    second = run()

    assert first == second
    assert first[-1][1] is PlayOutcome.RUSH


def test_qb_inside_the_run_margin_is_not_a_runner_and_can_throw() -> None:
    play = Play(ZERO_DEVIATION)
    play.state.player("qb").position = vec2(20.5, 26.65)

    state = play.step()

    assert state.carrier_id is None
    assert state.ball.status is BallStatus.HELD

    play.throw_to(vec2(60.0, 26.65))

    assert play.state.ball.status is BallStatus.IN_FLIGHT


def test_qb_tackled_inside_the_run_margin_is_sacked() -> None:
    config = replace(ZERO_DEVIATION, defender_starts=((21.0, 26.65), (0.0, 0.0)))
    play = Play(config)
    play.state.player("qb").position = vec2(20.5, 26.65)

    state = play.step()

    assert state.outcome is PlayOutcome.SACK
    assert state.carrier_id is None


def test_qb_beyond_the_run_margin_becomes_a_runner() -> None:
    play = Play(ZERO_DEVIATION)
    play.state.player("qb").position = vec2(21.5, 26.65)

    state = play.step()

    assert state.carrier_id == "qb"
    assert state.ball.status is BallStatus.CARRIED


def test_run_margin_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        replace(ZERO_DEVIATION, qb_run_margin_yards=-0.1)
