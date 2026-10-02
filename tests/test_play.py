from dataclasses import replace

import numpy as np
import pytest

from qb_sim.config import SimulationConfig
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import BallStatus, PlayOutcome, Role
from qb_sim.engine.geometry import vec2
from qb_sim.rules.play import Play
from qb_sim.rules.routes import RouteName


def _run_until_terminal(simulation: Play):
    for _ in range(200):
        state = simulation.step()
        if state.outcome is not PlayOutcome.IN_PROGRESS:
            return state
    pytest.fail("play did not reach a terminal outcome")


def _run_until(simulation: Play, predicate):
    for _ in range(200):
        state = simulation.step()
        if predicate(state):
            return state
    pytest.fail("play did not reach the expected state")


def test_reset_builds_two_receivers_and_two_defenders_with_ids() -> None:
    state = Play().reset()

    assert [player.player_id for player in state.players] == ["qb", "wr1", "wr2", "cb1", "cb2"]
    assert state.player("qb").role is Role.QUARTERBACK
    assert state.player("wr1").role is Role.RECEIVER
    assert state.player("cb2").role is Role.DEFENDER


def test_each_receiver_runs_its_own_route() -> None:
    play = Play()
    for _ in range(20):
        state = play.step()

    assert state.player("wr2").position[1] > 10.5
    assert state.player("wr1").position[1] == pytest.approx(43.0)


def test_qb_move_action_displaces_him_at_quarterback_speed() -> None:
    play = Play()

    state = play.step(qb_action=Move(direction=vec2(1.0, 0.0)))

    assert state.player("qb").position.tolist() == pytest.approx([20.0 + 6.5 * 0.05, 26.65])


def test_qb_throw_action_uses_direction_and_power() -> None:
    play = Play()

    state = play.step(qb_action=Throw(direction=vec2(0.0, 1.0), power=0.5))

    assert state.ball.status is BallStatus.IN_FLIGHT
    assert state.ball.velocity.tolist() == pytest.approx([0.0, 25.0])
    assert state.ball.target.tolist() == pytest.approx([20.0, 26.65 + 25.0])


def test_throw_power_scales_range() -> None:
    soft = Play()
    soft.step(qb_action=Throw(direction=vec2(1.0, 0.0), power=0.0))

    assert soft.state.ball.velocity.tolist() == pytest.approx([25.0, 0.0])
    assert soft.state.ball.target.tolist() == pytest.approx([25.0, 26.65])

    hard = Play()
    hard.step(qb_action=Throw(direction=vec2(1.0, 0.0), power=1.0))

    assert hard.state.ball.velocity.tolist() == pytest.approx([25.0, 0.0])
    assert hard.state.ball.target.tolist() == pytest.approx([65.0, 26.65])


def test_actions_do_not_persist_between_ticks() -> None:
    play = Play()
    play.step(qb_action=Move(direction=vec2(1.0, 0.0)))
    first_x = play.state.player("qb").position[0]

    play.step()

    assert play.state.player("qb").position[0] == pytest.approx(first_x)


def test_step_moves_receivers_and_defenders_but_not_quarterback() -> None:
    play = Play()

    state = play.step()

    assert state.player("qb").position.tolist() == pytest.approx([20.0, 26.65])
    assert state.player("wr1").position[0] > 20.0
    assert state.player("wr2").position[0] > 20.0
    assert state.player("cb1").position[0] > 18.0
    assert state.player("cb2").position[0] > 18.0
    assert state.elapsed_seconds == pytest.approx(0.05)


def test_reset_restores_initial_state_and_route_progress() -> None:
    play = Play()
    play.step()
    play.step()

    state = play.reset()

    assert state.player("wr1").position.tolist() == pytest.approx([20.0, 43.0])
    assert state.elapsed_seconds == 0.0
    assert state.outcome is PlayOutcome.IN_PROGRESS
    assert state.ball.status is BallStatus.HELD


def test_throw_projects_receiver_position_using_current_velocity() -> None:
    play = Play()
    state = play.step()
    receiver = state.player("wr1")
    expected_target = receiver.position + receiver.velocity * 0.5

    play.throw_with_lead(lead_seconds=0.5)

    assert state.ball.target.tolist() == pytest.approx(expected_target.tolist())
    assert state.ball.status is BallStatus.IN_FLIGHT


@pytest.mark.parametrize("lead_seconds", [-0.1, float("nan"), float("inf")])
def test_throw_rejects_negative_or_non_finite_lead(lead_seconds: float) -> None:
    play = Play()

    with pytest.raises(ValueError):
        play.throw_with_lead(lead_seconds)


def test_throw_can_only_be_issued_once_per_play() -> None:
    play = Play()
    play.throw_with_lead(lead_seconds=0.0)

    with pytest.raises(ValueError):
        play.throw_with_lead(lead_seconds=0.0)


def test_same_config_and_action_sequence_produce_identical_traces() -> None:
    first = Play()
    second = Play()

    def run(play: Play) -> None:
        for _ in range(20):
            play.step(qb_action=Move(direction=vec2(1.0, 0.0)))
        for _ in range(10):
            play.step()
        play.step(qb_action=Throw(direction=vec2(1.0, 0.0), power=1.0))
        for _ in range(40):
            play.step()

    run(first)
    run(second)

    assert first.state.outcome is second.state.outcome
    np.testing.assert_allclose(
        first.state.player("wr1").position, second.state.player("wr1").position
    )
    np.testing.assert_allclose(
        first.state.player("wr2").position, second.state.player("wr2").position
    )
    np.testing.assert_allclose(
        first.state.player("cb1").position, second.state.player("cb1").position
    )
    np.testing.assert_allclose(first.state.ball.position, second.state.ball.position)
    assert first.state.elapsed_seconds == second.state.elapsed_seconds


def test_simulations_with_same_config_produce_identical_state_traces() -> None:
    first = Play()
    second = Play()

    for _ in range(3):
        first.step()
        second.step()
    first.throw_with_lead(lead_seconds=0.5)
    second.throw_with_lead(lead_seconds=0.5)

    first_state = _run_until_terminal(first)
    second_state = _run_until_terminal(second)

    np.testing.assert_allclose(
        first_state.player("wr1").position, second_state.player("wr1").position
    )
    np.testing.assert_allclose(
        first_state.player("cb1").position, second_state.player("cb1").position
    )
    np.testing.assert_allclose(first_state.ball.position, second_state.ball.position)
    assert first_state.elapsed_seconds == second_state.elapsed_seconds
    assert first_state.outcome is second_state.outcome


def test_catch_switches_ball_to_carried_and_play_continues() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(20.0, 26.65),
        receiver_starts=((30.0, 26.65), (119.0, 0.0)),
        defender_starts=((0.0, 0.0), (0.0, 0.0)),
    )
    play = Play(config)
    play.step()
    play.throw_with_lead(0.4)

    state = _run_until(play, lambda current: current.ball.status is BallStatus.CARRIED)

    assert state.carrier_id == "wr1"
    assert state.outcome is PlayOutcome.IN_PROGRESS


def test_tackle_ends_the_play_after_a_completion() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(20.0, 26.65),
        receiver_starts=((21.5, 26.65), (119.0, 0.0)),
        defender_starts=((23.5, 26.65), (0.0, 0.0)),
        defender_speed=8.0,
    )
    play = Play(config)
    play.step()
    play.throw_with_lead(0.0)

    carried = _run_until(play, lambda current: current.ball.status is BallStatus.CARRIED)
    assert carried.outcome is PlayOutcome.IN_PROGRESS
    assert carried.carrier_id == "wr1"

    state = _run_until_terminal(play)

    assert state.outcome is PlayOutcome.COMPLETED
    assert state.ball.status is BallStatus.DEAD


def test_second_defender_can_tackle_the_carrier() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(20.0, 26.65),
        receiver_starts=((119.0, 0.0), (21.5, 26.65)),
        defender_starts=((0.0, 0.0), (23.5, 26.65)),
        defender_speed=8.0,
    )
    play = Play(config, routes=(RouteName.GO, RouteName.GO))
    play.step()
    play.throw_with_lead(0.0, player_id="wr2")

    carried = _run_until(play, lambda current: current.ball.status is BallStatus.CARRIED)
    assert carried.carrier_id == "wr2"

    state = _run_until_terminal(play)

    assert state.outcome is PlayOutcome.COMPLETED


def test_pass_to_second_receiver_uses_its_id_as_carrier() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(20.0, 26.65),
        receiver_starts=((119.0, 0.0), (25.0, 10.0)),
        defender_starts=((0.0, 0.0), (0.0, 0.0)),
    )
    play = Play(config, routes=(RouteName.GO, RouteName.GO))
    play.step()
    play.throw_to(vec2(31.0, 10.0))

    state = _run_until(play, lambda current: current.ball.status is BallStatus.CARRIED)

    assert state.carrier_id == "wr2"


def test_carrier_out_of_bounds_ends_the_play() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(117.0, 26.65),
        receiver_starts=((118.0, 26.65), (0.0, 0.0)),
        defender_starts=((0.0, 0.0), (0.0, 0.0)),
    )
    play = Play(config)
    play.step()
    play.throw_with_lead(0.1)

    carried = _run_until(play, lambda current: current.ball.status is BallStatus.CARRIED)
    assert carried.outcome is PlayOutcome.IN_PROGRESS

    state = _run_until_terminal(play)

    assert state.outcome is PlayOutcome.COMPLETED


def test_pass_can_be_completed() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(20.0, 26.65),
        receiver_starts=((30.0, 26.65), (119.0, 0.0)),
        defender_starts=((0.0, 0.0), (0.0, 0.0)),
    )
    play = Play(config)
    play.step()
    play.throw_with_lead(0.4)

    state = _run_until_terminal(play)

    assert state.outcome is PlayOutcome.COMPLETED


def test_defender_intercepts_a_pass_crossing_its_path() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(20.0, 26.65),
        receiver_starts=((30.0, 43.0), (119.0, 0.0)),
        defender_starts=((25.0, 34.825), (0.0, 0.0)),
        defender_speed=1.0,
    )
    play = Play(config)
    play.throw_with_lead(0.0)

    state = _run_until_terminal(play)

    assert state.outcome is PlayOutcome.INTERCEPTION


def test_defender_wins_when_receiver_and_defender_contact_together() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(20.0, 40.0),
        receiver_starts=((30.0, 40.0), (119.0, 0.0)),
        defender_starts=((30.0, 40.0), (0.0, 0.0)),
        defender_speed=7.0,
    )
    play = Play(config)
    play.step()
    play.throw_with_lead(0.4)

    state = _run_until_terminal(play)

    assert state.outcome is PlayOutcome.INTERCEPTION


def test_pass_without_contact_is_incomplete_when_it_reaches_target() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(20.0, 26.65),
        receiver_starts=((30.0, 43.0), (119.0, 0.0)),
        defender_starts=((0.0, 0.0), (0.0, 0.0)),
    )
    play = Play(config)
    play.throw_with_lead(0.0)

    state = _run_until_terminal(play)

    assert state.outcome is PlayOutcome.INCOMPLETE


def test_pass_leaving_field_is_out_of_bounds() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(110.0, 20.0),
        receiver_starts=((118.0, 43.0), (0.0, 0.0)),
        defender_starts=((0.0, 0.0), (0.0, 0.0)),
    )
    play = Play(config)
    play.step()
    play.throw_with_lead(1.0)

    state = _run_until_terminal(play)

    assert state.outcome is PlayOutcome.OUT_OF_BOUNDS


def test_play_ends_at_configured_time_limit() -> None:
    config = replace(SimulationConfig(), max_play_duration=0.1)
    play = Play(config)

    play.step()
    state = play.step()

    assert state.outcome is PlayOutcome.TIME_LIMIT
    assert state.elapsed_seconds == pytest.approx(0.1)


def test_terminal_state_does_not_advance_again() -> None:
    config = replace(SimulationConfig(), max_play_duration=0.05)
    play = Play(config)
    terminal = play.step()
    receiver_position = terminal.player("wr1").position.copy()

    result = play.step()

    assert result is terminal
    assert result.player("wr1").position.tolist() == pytest.approx(receiver_position.tolist())
