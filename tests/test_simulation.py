from dataclasses import replace

import numpy as np
import pytest

from qb_sim.config import SimulationConfig
from qb_sim.entities import BallStatus, PlayOutcome
from qb_sim.routes import RouteName
from qb_sim.simulation import Simulation


def _run_until_terminal(simulation: Simulation):
    for _ in range(200):
        state = simulation.step()
        if state.outcome is not PlayOutcome.IN_PROGRESS:
            return state
    pytest.fail("play did not reach a terminal outcome")


def test_step_moves_receiver_and_defender_but_not_quarterback() -> None:
    simulation = Simulation()

    state = simulation.step()

    assert state.quarterback.position.tolist() == pytest.approx([20.0, 26.65])
    assert state.receiver.position[0] > 20.0
    assert state.defender.position[0] > 18.0
    assert state.elapsed_seconds == pytest.approx(0.05)


def test_reset_restores_initial_state_and_route_progress() -> None:
    simulation = Simulation(route=RouteName.POST)
    simulation.step()
    simulation.step()

    state = simulation.reset()

    assert state.receiver.position.tolist() == pytest.approx([20.0, 43.0])
    assert state.elapsed_seconds == 0.0
    assert state.outcome is PlayOutcome.IN_PROGRESS
    assert state.ball.status is BallStatus.HELD


def test_throw_projects_receiver_position_using_current_velocity() -> None:
    simulation = Simulation()
    state = simulation.step()
    expected_target = state.receiver.position + state.receiver.velocity * 0.5

    simulation.throw(lead_seconds=0.5)

    assert state.ball.target.tolist() == pytest.approx(expected_target.tolist())
    assert state.ball.status is BallStatus.IN_FLIGHT


@pytest.mark.parametrize("lead_seconds", [-0.1, float("nan"), float("inf")])
def test_throw_rejects_negative_or_non_finite_lead(lead_seconds: float) -> None:
    simulation = Simulation()

    with pytest.raises(ValueError):
        simulation.throw(lead_seconds)


def test_throw_can_only_be_issued_once_per_play() -> None:
    simulation = Simulation()
    simulation.throw(lead_seconds=0.0)

    with pytest.raises(ValueError):
        simulation.throw(lead_seconds=0.0)


def test_simulations_with_same_config_produce_identical_state_traces() -> None:
    first = Simulation()
    second = Simulation()

    for _ in range(3):
        first.step()
        second.step()
    first.throw(lead_seconds=0.5)
    second.throw(lead_seconds=0.5)

    first_state = _run_until_terminal(first)
    second_state = _run_until_terminal(second)

    np.testing.assert_allclose(first_state.receiver.position, second_state.receiver.position)
    np.testing.assert_allclose(first_state.defender.position, second_state.defender.position)
    np.testing.assert_allclose(first_state.ball.position, second_state.ball.position)
    assert first_state.elapsed_seconds == second_state.elapsed_seconds
    assert first_state.outcome is second_state.outcome


def test_pass_can_be_completed() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(20.0, 26.65),
        receiver_start=(30.0, 26.65),
        defender_start=(0.0, 0.0),
    )
    simulation = Simulation(config)
    simulation.step()
    simulation.throw(lead_seconds=0.4)

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
    simulation = Simulation(config)
    simulation.throw(lead_seconds=0.0)

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
    simulation = Simulation(config)
    simulation.step()
    simulation.throw(lead_seconds=0.4)

    state = _run_until_terminal(simulation)

    assert state.outcome is PlayOutcome.INTERCEPTION


def test_pass_without_contact_is_incomplete_when_it_reaches_target() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(20.0, 26.65),
        receiver_start=(30.0, 43.0),
        defender_start=(119.0, 0.0),
    )
    simulation = Simulation(config)
    simulation.throw(lead_seconds=0.0)

    state = _run_until_terminal(simulation)

    assert state.outcome is PlayOutcome.INCOMPLETE


def test_pass_leaving_field_is_out_of_bounds() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(110.0, 20.0),
        receiver_start=(118.0, 43.0),
        defender_start=(0.0, 0.0),
    )
    simulation = Simulation(config)
    simulation.step()
    simulation.throw(lead_seconds=1.0)

    state = _run_until_terminal(simulation)

    assert state.outcome is PlayOutcome.OUT_OF_BOUNDS


def test_play_ends_at_configured_time_limit() -> None:
    config = replace(SimulationConfig(), max_play_duration=0.1)
    simulation = Simulation(config)

    simulation.step()
    state = simulation.step()

    assert state.outcome is PlayOutcome.TIME_LIMIT
    assert state.elapsed_seconds == pytest.approx(0.1)


def test_terminal_state_does_not_advance_again() -> None:
    config = replace(SimulationConfig(), max_play_duration=0.05)
    simulation = Simulation(config)
    terminal = simulation.step()
    receiver_position = terminal.receiver.position.copy()

    result = simulation.step()

    assert result is terminal
    assert result.receiver.position.tolist() == pytest.approx(receiver_position.tolist())
