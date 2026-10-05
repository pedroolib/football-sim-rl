from dataclasses import asdict, replace

import pytest

from qb_sim.config import SimulationConfig
from qb_sim.engine.actions import Move
from qb_sim.engine.entities import BallStatus
from qb_sim.engine.geometry import vec2
from qb_sim.rules.drive import Drive, DriveConfig
from qb_sim.rules.play import Play
from qb_sim.rules.scenarios import STAGE_B_CONFIG

SHOTGUN = replace(
    SimulationConfig(),
    qb_snap_depth_yards=3.0,
    throw_base_sigma_degrees=0.0,
    throw_move_penalty=0.0,
    throw_pressure_penalty=0.0,
    throw_distance_penalty=0.0,
    throw_power_sigma=0.0,
)

FORWARD = Move(direction=vec2(1.0, 0.0))


def test_qb_snap_depth_defaults_to_zero() -> None:
    assert SimulationConfig().qb_snap_depth_yards == 0.0


def test_qb_snap_depth_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        replace(SimulationConfig(), qb_snap_depth_yards=-0.1)


def test_qb_snap_depth_is_serialized() -> None:
    assert asdict(SHOTGUN)["qb_snap_depth_yards"] == 3.0


def test_shotgun_qb_starts_behind_the_line_of_scrimmage() -> None:
    play = Play(SHOTGUN)

    state = play.reset()

    assert play.line_of_scrimmage == pytest.approx(20.0)
    assert state.player("qb").position.tolist() == pytest.approx([17.0, 26.65])
    assert state.player("wr1").position.tolist() == pytest.approx([20.0, 43.0])
    assert state.player("wr2").position.tolist() == pytest.approx([20.0, 10.0])
    assert state.player("cb1").position.tolist() == pytest.approx([18.0, 43.0])


def test_ball_starts_in_the_shotgun_quarterbacks_hands() -> None:
    play = Play(SHOTGUN)

    state = play.reset()

    assert state.ball.status is BallStatus.HELD
    assert state.ball.position.tolist() == pytest.approx(state.player("qb").position.tolist())


def test_shotgun_depth_survives_a_shifted_formation() -> None:
    play = Play(SHOTGUN, line_of_scrimmage=45.0)

    state = play.reset()

    assert state.player("qb").position.tolist() == pytest.approx([42.0, 26.65])
    assert state.ball.position.tolist() == pytest.approx([42.0, 26.65])
    assert state.player("wr1").position.tolist() == pytest.approx([45.0, 43.0])


def test_shotgun_qb_runs_after_the_depth_plus_margin() -> None:
    config = replace(SHOTGUN, defender_starts=((119.0, 26.65), (0.0, 0.0)))
    play = Play(config, goal_line=100.0)

    for _ in range(12):
        state = play.step({"qb": FORWARD})

    assert state.player("qb").position[0] == pytest.approx(20.9)
    assert state.carrier_id is None
    assert state.ball.status is BallStatus.HELD

    state = play.step({"qb": FORWARD})

    assert state.player("qb").position[0] == pytest.approx(21.225)
    assert state.carrier_id == "qb"
    assert state.ball.status is BallStatus.CARRIED


def test_shotgun_depth_outside_the_field_is_rejected() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(2.0, 26.65),
        qb_snap_depth_yards=3.0,
    )
    with pytest.raises(ValueError):
        Play(config)


def test_shifted_shotgun_depth_outside_the_field_is_rejected() -> None:
    with pytest.raises(ValueError):
        Play(SHOTGUN, line_of_scrimmage=2.0)


def test_drive_floor_accounts_for_the_shotgun_depth() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(5.0, 26.65),
        qb_snap_depth_yards=3.0,
        receiver_starts=((5.0, 43.0), (5.0, 10.0)),
        defender_starts=((5.0, 43.0), (5.0, 10.0)),
    )
    drive = Drive(config, DriveConfig(los_range=(0.0, 0.0)))

    assert drive.state.los == pytest.approx(3.0)
    play = drive.start_play()
    assert play.line_of_scrimmage == pytest.approx(3.0)
    assert play.state.player("qb").position[0] == pytest.approx(0.0)


def test_shotgun_positions_are_deterministic_for_a_seed() -> None:
    def run() -> list[tuple[float, ...]]:
        play = Play(SHOTGUN, seed=11)
        trace = []
        for _ in range(30):
            state = play.step({"qb": FORWARD})
            trace.append(tuple(float(value) for value in state.player("qb").position))
        return trace

    assert run() == run()


def test_stage_b_config_uses_the_balanced_run_setup() -> None:
    assert STAGE_B_CONFIG.qb_snap_depth_yards == pytest.approx(3.0)
    assert STAGE_B_CONFIG.rusher_starts == ((29.0, 29.0),)
    assert STAGE_B_CONFIG.carrier_fatigue_per_second == pytest.approx(0.12)
