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


def test_roles_are_assigned_on_reset() -> None:
    state = Play().reset()

    assert state.quarterback.role is Role.QUARTERBACK
    assert state.receiver.role is Role.RECEIVER
    assert state.defender.role is Role.DEFENDER


def test_qb_move_action_displaces_him_at_quarterback_speed() -> None:
    play = Play()

    state = play.step(qb_action=Move(direction=vec2(1.0, 0.0)))

    assert state.quarterback.position.tolist() == pytest.approx([20.0 + 6.5 * 0.05, 26.65])


def test_qb_throw_action_launches_ball_toward_target() -> None:
    play = Play()

    state = play.step(qb_action=Throw(target=vec2(40.0, 26.65)))

    assert state.ball.status is BallStatus.IN_FLIGHT
    assert state.ball.target.tolist() == pytest.approx([40.0, 26.65])


def test_actions_do_not_persist_between_ticks() -> None:
    play = Play()
    play.step(qb_action=Move(direction=vec2(1.0, 0.0)))
    first_x = play.state.quarterback.position[0]

    play.step()

    assert play.state.quarterback.position[0] == pytest.approx(first_x)


def test_step_moves_receiver_and_defender_but_not_quarterback() -> None:
    simulation = Play()

    state = simulation.step()

    assert state.quarterback.position.tolist() == pytest.approx([20.0, 26.65])
    assert state.receiver.position[0] > 20.0
    assert state.defender.position[0] > 18.0
    assert state.elapsed_seconds == pytest.approx(0.05)


def test_reset_restores_initial_state_and_route_progress() -> None:
    simulation = Play(route=RouteName.POST)
    simulation.step()
    simulation.step()

    state = simulation.reset()

    assert state.receiver.position.tolist() == pytest.approx([20.0, 43.0])
    assert state.elapsed_seconds == 0.0
    assert state.outcome is PlayOutcome.IN_PROGRESS
    assert state.ball.status is BallStatus.HELD


def test_throw_projects_receiver_position_using_current_velocity() -> None:
    simulation = Play()
    state = simulation.step()
    expected_target = state.receiver.position + state.receiver.velocity * 0.5

    simulation.throw_with_lead(lead_seconds=0.5)

    assert state.ball.target.tolist() == pytest.approx(expected_target.tolist())
    assert state.ball.status is BallStatus.IN_FLIGHT


@pytest.mark.parametrize("lead_seconds", [-0.1, float("nan"), float("inf")])
def test_throw_rejects_negative_or_non_finite_lead(lead_seconds: float) -> None:
    simulation = Play()

    with pytest.raises(ValueError):
        simulation.throw_with_lead(lead_seconds)


def test_throw_can_only_be_issued_once_per_play() -> None:
    simulation = Play()
    simulation.throw_with_lead(lead_seconds=0.0)

    with pytest.raises(ValueError):
        simulation.throw_with_lead(lead_seconds=0.0)


def test_same_config_and_action_sequence_produce_identical_traces() -> None:
    first = Play()
    second = Play()

    def run(play: Play) -> None:
        for _ in range(20):
            play.step(qb_action=Move(direction=vec2(1.0, 0.0)))
        for _ in range(10):
            play.step()
        play.step(qb_action=Throw(target=vec2(50.0, 26.65)))
        for _ in range(40):
            play.step()

    run(first)
    run(second)

    assert first.state.outcome is second.state.outcome
    np.testing.assert_allclose(first.state.receiver.position, second.state.receiver.position)
    np.testing.assert_allclose(first.state.defender.position, second.state.defender.position)
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

    np.testing.assert_allclose(first_state.receiver.position, second_state.receiver.position)
    np.testing.assert_allclose(first_state.defender.position, second_state.defender.position)
    np.testing.assert_allclose(first_state.ball.position, second_state.ball.position)
    assert first_state.elapsed_seconds == second_state.elapsed_seconds
    assert first_state.outcome is second_state.outcome


def test_catch_switches_ball_to_carried_and_play_continues() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(20.0, 26.65),
        receiver_start=(30.0, 26.65),
        defender_start=(0.0, 0.0),
    )
    play = Play(config)
    play.step()
    play.throw_with_lead(0.4)

    state = _run_until(play, lambda current: current.ball.status is BallStatus.CARRIED)

    assert state.carrier is Role.RECEIVER
    assert state.outcome is PlayOutcome.IN_PROGRESS


def test_tackle_ends_the_play_after_a_completion() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(20.0, 26.65),
        receiver_start=(21.5, 26.65),
        defender_start=(23.5, 26.65),
        defender_speed=8.0,
    )
    play = Play(config)
    play.step()
    play.throw_with_lead(0.0)

    carried = _run_until(play, lambda current: current.ball.status is BallStatus.CARRIED)
    assert carried.outcome is PlayOutcome.IN_PROGRESS

    state = _run_until_terminal(play)

    assert state.outcome is PlayOutcome.COMPLETED
    assert state.ball.status is BallStatus.DEAD


def test_carrier_out_of_bounds_ends_the_play() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(117.0, 26.65),
        receiver_start=(118.0, 26.65),
        defender_start=(0.0, 0.0),
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
        receiver_start=(30.0, 26.65),
        defender_start=(0.0, 0.0),
    )
    simulation = Play(config)
    simulation.step()
    simulation.throw_with_lead(lead_seconds=0.4)

    state = _run_until_terminal(simulation)

    assert state.outcome is PlayOutcome.COMPLETED


def test_defender_intercepts_a_pass_crossing_its_path() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(20.0, 26.65),
        receiver_start=(30.0, 43.0),
        defender_start=(25.0, 34.825),
        defender_speed=1.0,
    )
    simulation = Play(config)
    simulation.throw_with_lead(lead_seconds=0.0)

    state = _run_until_terminal(simulation)

    assert state.outcome is PlayOutcome.INTERCEPTION


def test_defender_wins_when_receiver_and_defender_contact_together() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(20.0, 40.0),
        receiver_start=(30.0, 40.0),
        defender_start=(30.0, 40.0),
        defender_speed=7.0,
    )
    simulation = Play(config)
    simulation.step()
    simulation.throw_with_lead(lead_seconds=0.4)

    state = _run_until_terminal(simulation)

    assert state.outcome is PlayOutcome.INTERCEPTION


def test_pass_without_contact_is_incomplete_when_it_reaches_target() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(20.0, 26.65),
        receiver_start=(30.0, 43.0),
        defender_start=(119.0, 0.0),
    )
    simulation = Play(config)
    simulation.throw_with_lead(lead_seconds=0.0)

    state = _run_until_terminal(simulation)

    assert state.outcome is PlayOutcome.INCOMPLETE


def test_pass_leaving_field_is_out_of_bounds() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(110.0, 20.0),
        receiver_start=(118.0, 43.0),
        defender_start=(0.0, 0.0),
    )
    simulation = Play(config)
    simulation.step()
    simulation.throw_with_lead(lead_seconds=1.0)

    state = _run_until_terminal(simulation)

    assert state.outcome is PlayOutcome.OUT_OF_BOUNDS


def test_play_ends_at_configured_time_limit() -> None:
    config = replace(SimulationConfig(), max_play_duration=0.1)
    simulation = Play(config)

    simulation.step()
    state = simulation.step()

    assert state.outcome is PlayOutcome.TIME_LIMIT
    assert state.elapsed_seconds == pytest.approx(0.1)


def test_terminal_state_does_not_advance_again() -> None:
    config = replace(SimulationConfig(), max_play_duration=0.05)
    simulation = Play(config)
    terminal = simulation.step()
    receiver_position = terminal.receiver.position.copy()

    result = simulation.step()

    assert result is terminal
    assert result.receiver.position.tolist() == pytest.approx(receiver_position.tolist())
