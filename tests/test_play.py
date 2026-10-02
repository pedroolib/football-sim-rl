from dataclasses import replace

import numpy as np
import pytest

from qb_sim.agents.base import BaseAgent
from qb_sim.agents.loop import build_rule_agents
from qb_sim.config import SimulationConfig
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import BallStatus, PlayOutcome, Role
from qb_sim.engine.geometry import vec2
from qb_sim.rules.play import Play
from qb_sim.rules.routes import RouteName

ZERO_DEVIATION = replace(
    SimulationConfig(),
    throw_base_sigma_degrees=0.0,
    throw_move_penalty=0.0,
    throw_pressure_penalty=0.0,
    throw_distance_penalty=0.0,
    throw_power_sigma=0.0,
)


class _ForwardController(BaseAgent):
    """Test double: always runs forward, used to force out-of-bounds."""

    role = Role.RECEIVER

    def observe(self, state):
        return state.player(self.player_id)

    def act(self, observation):
        return Move(direction=vec2(1.0, 0.0))


def _make_agents(play: Play, routes=(RouteName.POST, RouteName.SLANT)):
    return build_rule_agents(play, routes, play.config, include_quarterback=False)


def _step(play: Play, agents, qb_action=None):
    actions = {}
    for player_id, agent in agents.items():
        action = agent.decide(play.state)
        if action is not None:
            actions[player_id] = action
    if qb_action is not None:
        actions["qb"] = qb_action
    return play.step(actions)


def _run_until_terminal(play: Play, agents):
    for _ in range(200):
        state = _step(play, agents)
        if state.outcome is not PlayOutcome.IN_PROGRESS:
            return state
    pytest.fail("play did not reach a terminal outcome")


def _run_until(play: Play, agents, predicate):
    for _ in range(200):
        state = _step(play, agents)
        if predicate(state):
            return state
    pytest.fail("play did not reach the expected state")


def test_reset_builds_two_receivers_and_two_defenders_with_ids() -> None:
    state = Play().reset()

    assert [player.player_id for player in state.players] == ["qb", "wr1", "wr2", "cb1", "cb2"]
    assert state.player("qb").role is Role.QUARTERBACK
    assert state.player("wr1").role is Role.RECEIVER
    assert state.player("cb2").role is Role.DEFENDER


def test_rushed_roster_adds_a_dedicated_rusher() -> None:
    config = replace(ZERO_DEVIATION, rusher_starts=((30.0, 29.0),))

    play = Play(config)
    state = play.reset()

    assert [player.player_id for player in state.players] == [
        "qb",
        "wr1",
        "wr2",
        "cb1",
        "cb2",
        "ru1",
    ]
    assert play.rusher_ids == ("ru1",)
    assert play.defender_ids == ("cb1", "cb2", "ru1")
    assert state.player("ru1").role is Role.DEFENDER


def test_each_receiver_runs_its_own_route() -> None:
    play = Play()
    agents = _make_agents(play)
    for _ in range(20):
        state = _step(play, agents)

    assert state.player("wr2").position[1] > 10.5
    assert state.player("wr1").position[1] == pytest.approx(43.0)


def test_qb_move_action_displaces_him_at_quarterback_speed() -> None:
    play = Play()

    state = play.step({"qb": Move(direction=vec2(1.0, 0.0))})

    assert state.player("qb").position.tolist() == pytest.approx([20.0 + 6.5 * 0.05, 26.65])


def test_qb_throw_action_uses_direction_and_power() -> None:
    play = Play(ZERO_DEVIATION)

    state = play.step({"qb": Throw(direction=vec2(0.0, 1.0), power=0.5)})

    assert state.ball.status is BallStatus.IN_FLIGHT
    assert state.ball.velocity.tolist() == pytest.approx([0.0, 25.0])
    assert state.ball.target.tolist() == pytest.approx([20.0, 26.65 + 25.0])


def test_throw_power_scales_range() -> None:
    soft = Play(ZERO_DEVIATION)
    soft.step({"qb": Throw(direction=vec2(1.0, 0.0), power=0.0)})

    assert soft.state.ball.velocity.tolist() == pytest.approx([25.0, 0.0])
    assert soft.state.ball.target.tolist() == pytest.approx([25.0, 26.65])

    hard = Play(ZERO_DEVIATION)
    hard.step({"qb": Throw(direction=vec2(1.0, 0.0), power=1.0)})

    assert hard.state.ball.velocity.tolist() == pytest.approx([25.0, 0.0])
    assert hard.state.ball.target.tolist() == pytest.approx([65.0, 26.65])


def test_actions_do_not_persist_between_ticks() -> None:
    play = Play()
    play.step({"qb": Move(direction=vec2(1.0, 0.0))})
    first_x = play.state.player("qb").position[0]

    play.step()

    assert play.state.player("qb").position[0] == pytest.approx(first_x)


def test_actions_are_rejected_for_the_wrong_role() -> None:
    play = Play()

    with pytest.raises(ValueError):
        play.step({"wr1": Throw(direction=vec2(1.0, 0.0), power=0.5)})


def test_unknown_player_actions_are_rejected() -> None:
    play = Play()

    with pytest.raises(ValueError):
        play.step({"ghost": Move(direction=vec2(1.0, 0.0))})


def test_step_moves_receivers_and_defenders_but_not_quarterback() -> None:
    play = Play()
    agents = _make_agents(play)

    state = _step(play, agents)

    assert state.player("qb").position.tolist() == pytest.approx([20.0, 26.65])
    assert state.player("wr1").position[0] > 20.0
    assert state.player("wr2").position[0] > 20.0
    assert state.player("cb1").position[0] > 18.0
    assert state.player("cb2").position[0] > 18.0
    assert state.elapsed_seconds == pytest.approx(0.05)


def test_reset_restores_initial_state_and_route_progress() -> None:
    play = Play()
    agents = _make_agents(play)
    _step(play, agents)
    _step(play, agents)

    state = play.reset()

    assert state.player("wr1").position.tolist() == pytest.approx([20.0, 43.0])
    assert state.elapsed_seconds == 0.0
    assert state.outcome is PlayOutcome.IN_PROGRESS
    assert state.ball.status is BallStatus.HELD


def test_throw_projects_receiver_position_using_current_velocity() -> None:
    play = Play(ZERO_DEVIATION)
    agents = _make_agents(play)
    state = _step(play, agents)
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
    first_agents = _make_agents(first)
    second_agents = _make_agents(second)

    def run(play: Play, agents) -> None:
        for _ in range(20):
            _step(play, agents, Move(direction=vec2(1.0, 0.0)))
        for _ in range(10):
            _step(play, agents)
        _step(play, agents, Throw(direction=vec2(1.0, 0.0), power=1.0))
        for _ in range(40):
            _step(play, agents)

    run(first, first_agents)
    run(second, second_agents)

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
    first_agents = _make_agents(first)
    second_agents = _make_agents(second)

    for _ in range(3):
        _step(first, first_agents)
        _step(second, second_agents)
    first.throw_with_lead(lead_seconds=0.5)
    second.throw_with_lead(lead_seconds=0.5)

    first_state = _run_until_terminal(first, first_agents)
    second_state = _run_until_terminal(second, second_agents)

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
        ZERO_DEVIATION,
        quarterback_start=(20.0, 26.65),
        receiver_starts=((30.0, 26.65), (119.0, 0.0)),
        defender_starts=((0.0, 0.0), (0.0, 0.0)),
    )
    play = Play(config)
    agents = _make_agents(play)
    _step(play, agents)
    play.throw_with_lead(0.4)

    state = _run_until(play, agents, lambda current: current.ball.status is BallStatus.CARRIED)

    assert state.carrier_id == "wr1"
    assert state.outcome is PlayOutcome.IN_PROGRESS


def test_tackle_ends_the_play_after_a_completion() -> None:
    config = replace(
        ZERO_DEVIATION,
        quarterback_start=(20.0, 26.65),
        receiver_starts=((21.5, 26.65), (119.0, 0.0)),
        defender_starts=((23.5, 26.65), (0.0, 0.0)),
        defender_speed=8.0,
    )
    play = Play(config)
    agents = _make_agents(play)
    _step(play, agents)
    play.throw_with_lead(0.0)

    carried = _run_until(play, agents, lambda current: current.ball.status is BallStatus.CARRIED)
    assert carried.outcome is PlayOutcome.IN_PROGRESS
    assert carried.carrier_id == "wr1"

    state = _run_until_terminal(play, agents)

    assert state.outcome is PlayOutcome.COMPLETED
    assert state.ball.status is BallStatus.DEAD


def test_second_defender_can_tackle_the_carrier() -> None:
    config = replace(
        ZERO_DEVIATION,
        quarterback_start=(20.0, 26.65),
        receiver_starts=((119.0, 0.0), (21.5, 26.65)),
        defender_starts=((0.0, 0.0), (23.5, 26.65)),
        defender_speed=8.0,
    )
    play = Play(config)
    agents = _make_agents(play, routes=(RouteName.GO, RouteName.GO))
    _step(play, agents)
    play.throw_with_lead(0.0, player_id="wr2")

    carried = _run_until(play, agents, lambda current: current.ball.status is BallStatus.CARRIED)
    assert carried.carrier_id == "wr2"

    state = _run_until_terminal(play, agents)

    assert state.outcome is PlayOutcome.COMPLETED


def test_pass_to_second_receiver_uses_its_id_as_carrier() -> None:
    config = replace(
        ZERO_DEVIATION,
        quarterback_start=(20.0, 26.65),
        receiver_starts=((119.0, 0.0), (25.0, 10.0)),
        defender_starts=((0.0, 0.0), (0.0, 0.0)),
    )
    play = Play(config)
    agents = _make_agents(play, routes=(RouteName.GO, RouteName.GO))
    _step(play, agents)
    play.throw_to(vec2(31.0, 10.0))

    state = _run_until(play, agents, lambda current: current.ball.status is BallStatus.CARRIED)

    assert state.carrier_id == "wr2"


def test_carrier_out_of_bounds_ends_the_play() -> None:
    config = replace(
        ZERO_DEVIATION,
        quarterback_start=(117.0, 26.65),
        receiver_starts=((118.0, 26.65), (0.0, 0.0)),
        defender_starts=((0.0, 0.0), (0.0, 0.0)),
    )
    play = Play(config)
    agents = _make_agents(play)
    _step(play, agents)
    play.throw_with_lead(0.1)

    carried = _run_until(play, agents, lambda current: current.ball.status is BallStatus.CARRIED)
    assert carried.outcome is PlayOutcome.IN_PROGRESS

    agents["wr1"] = _ForwardController("wr1")
    state = _run_until_terminal(play, agents)

    assert state.outcome is PlayOutcome.COMPLETED


def test_pass_can_be_completed() -> None:
    config = replace(
        ZERO_DEVIATION,
        quarterback_start=(20.0, 26.65),
        receiver_starts=((30.0, 26.65), (119.0, 0.0)),
        defender_starts=((0.0, 0.0), (0.0, 0.0)),
    )
    play = Play(config)
    agents = _make_agents(play)
    _step(play, agents)
    play.throw_with_lead(0.4)

    state = _run_until_terminal(play, agents)

    assert state.outcome is PlayOutcome.COMPLETED


def test_ball_rises_during_flight_and_lands_at_zero_height() -> None:
    config = replace(
        ZERO_DEVIATION,
        quarterback_start=(20.0, 26.65),
        receiver_starts=((30.0, 26.65), (119.0, 0.0)),
        defender_starts=((0.0, 0.0), (0.0, 0.0)),
    )
    play = Play(config)
    agents = _make_agents(play)
    play.throw_to(vec2(60.0, 26.65))

    heights: list[float] = []
    state = play.state
    for _ in range(120):
        state = _step(play, agents)
        heights.append(float(state.ball.height))
        if state.outcome is not PlayOutcome.IN_PROGRESS:
            break

    assert max(heights) > 2.0
    assert state.ball.height == pytest.approx(0.0)


def test_defender_intercepts_a_pass_crossing_its_path() -> None:
    config = replace(
        ZERO_DEVIATION,
        quarterback_start=(20.0, 26.65),
        receiver_starts=((30.0, 43.0), (119.0, 0.0)),
        defender_starts=((28.7, 40.87), (0.0, 0.0)),
        defender_speed=1.0,
    )
    play = Play(config)
    agents = _make_agents(play)
    play.throw_with_lead(0.0)

    state = _run_until_terminal(play, agents)

    assert state.outcome is PlayOutcome.INTERCEPTION


def test_ball_flying_above_a_defender_is_not_intercepted() -> None:
    config = replace(
        ZERO_DEVIATION,
        quarterback_start=(20.0, 26.65),
        receiver_starts=((30.0, 43.0), (119.0, 0.0)),
        defender_starts=((25.0, 34.825), (0.0, 0.0)),
        defender_speed=1.0,
    )
    play = Play(config)
    agents = _make_agents(play)
    play.throw_with_lead(0.0)

    state = _run_until_terminal(play, agents)

    assert state.outcome is not PlayOutcome.INTERCEPTION


def test_defender_wins_when_receiver_and_defender_contact_together() -> None:
    config = replace(
        ZERO_DEVIATION,
        quarterback_start=(20.0, 40.0),
        receiver_starts=((30.0, 40.0), (119.0, 0.0)),
        defender_starts=((30.0, 40.0), (0.0, 0.0)),
        defender_speed=7.0,
    )
    play = Play(config)
    agents = _make_agents(play)
    _step(play, agents)
    play.throw_with_lead(0.4)

    state = _run_until_terminal(play, agents)

    assert state.outcome is PlayOutcome.INTERCEPTION


def test_pass_without_contact_is_incomplete_when_it_reaches_target() -> None:
    config = replace(
        ZERO_DEVIATION,
        quarterback_start=(20.0, 26.65),
        receiver_starts=((30.0, 43.0), (119.0, 0.0)),
        defender_starts=((0.0, 0.0), (0.0, 0.0)),
    )
    play = Play(config)
    agents = _make_agents(play)
    play.throw_with_lead(0.0)

    state = _run_until_terminal(play, agents)

    assert state.outcome is PlayOutcome.INCOMPLETE


def test_pass_leaving_field_is_out_of_bounds() -> None:
    config = replace(
        ZERO_DEVIATION,
        quarterback_start=(110.0, 20.0),
        receiver_starts=((118.0, 43.0), (0.0, 0.0)),
        defender_starts=((0.0, 0.0), (0.0, 0.0)),
    )
    play = Play(config)
    agents = _make_agents(play)
    _step(play, agents)
    play.throw_with_lead(1.0)

    state = _run_until_terminal(play, agents)

    assert state.outcome is PlayOutcome.OUT_OF_BOUNDS


def test_defender_touching_the_quarterback_sacks_him() -> None:
    config = replace(ZERO_DEVIATION, defender_starts=((20.5, 26.65), (0.0, 0.0)))
    play = Play(config)

    state = play.step()

    assert state.outcome is PlayOutcome.SACK
    assert state.ball.status is BallStatus.DEAD


def test_play_ends_at_configured_time_limit() -> None:
    config = replace(ZERO_DEVIATION, max_play_duration=0.1)
    play = Play(config)

    play.step()
    state = play.step()

    assert state.outcome is PlayOutcome.TIME_LIMIT
    assert state.elapsed_seconds == pytest.approx(0.1)


def test_terminal_state_does_not_advance_again() -> None:
    config = replace(ZERO_DEVIATION, max_play_duration=0.05)
    play = Play(config)
    terminal = play.step()
    receiver_position = terminal.player("wr1").position.copy()

    result = play.step()

    assert result is terminal
    assert result.player("wr1").position.tolist() == pytest.approx(receiver_position.tolist())


def test_line_of_scrimmage_shifts_the_formation() -> None:
    play = Play(line_of_scrimmage=45.0)

    state = play.reset()

    assert play.line_of_scrimmage == pytest.approx(45.0)
    assert state.player("qb").position.tolist() == pytest.approx([45.0, 26.65])
    assert state.player("wr1").position.tolist() == pytest.approx([45.0, 43.0])
    assert state.player("cb2").position.tolist() == pytest.approx([43.0, 10.0])


def test_line_of_scrimmage_outside_the_field_is_rejected() -> None:
    with pytest.raises(ValueError):
        Play(line_of_scrimmage=-1.0)
    with pytest.raises(ValueError):
        Play(goal_line=130.0)


def test_qb_scramble_crossing_the_goal_line_scores() -> None:
    config = replace(
        ZERO_DEVIATION,
        quarterback_start=(99.9, 26.65),
        defender_starts=((100.5, 26.65), (0.0, 0.0)),
    )
    play = Play(config, goal_line=100.0)

    state = play.step({"qb": Move(direction=vec2(1.0, 0.0))})

    assert state.outcome is PlayOutcome.TOUCHDOWN


def test_carrier_crossing_the_goal_line_scores() -> None:
    config = replace(
        ZERO_DEVIATION,
        quarterback_start=(84.0, 26.65),
        receiver_starts=((95.0, 26.65), (119.0, 0.0)),
        defender_starts=((0.0, 0.0), (0.0, 0.0)),
    )
    play = Play(config, goal_line=100.0)
    agents = _make_agents(play)
    _step(play, agents)
    play.throw_with_lead(0.7)

    state = _run_until_terminal(play, agents)

    assert state.outcome is PlayOutcome.TOUCHDOWN


def test_catch_beyond_the_goal_line_scores_immediately() -> None:
    config = replace(
        ZERO_DEVIATION,
        quarterback_start=(80.0, 26.65),
        receiver_starts=((100.5, 26.65), (0.0, 0.0)),
        defender_starts=((0.0, 0.0), (0.0, 0.0)),
    )
    play = Play(config, goal_line=100.0)
    agents = _make_agents(play)
    _step(play, agents)
    play.throw_with_lead(1.2)

    state = _run_until_terminal(play, agents)

    assert state.outcome is PlayOutcome.TOUCHDOWN


def test_touchdown_has_priority_over_a_simultaneous_sack() -> None:
    config = replace(
        ZERO_DEVIATION,
        quarterback_start=(99.9, 26.65),
        defender_starts=((100.2, 26.65), (0.0, 0.0)),
    )
    play = Play(config, goal_line=100.0)

    state = play.step({"qb": Move(direction=vec2(1.0, 0.0))})

    assert state.outcome is PlayOutcome.TOUCHDOWN


def test_catch_records_when_the_carry_started() -> None:
    config = replace(
        ZERO_DEVIATION,
        quarterback_start=(20.0, 26.65),
        receiver_starts=((30.0, 26.65), (119.0, 0.0)),
        defender_starts=((0.0, 0.0), (0.0, 0.0)),
    )
    play = Play(config)
    agents = _make_agents(play)
    _step(play, agents)
    play.throw_with_lead(0.4)

    state = _run_until(play, agents, lambda current: current.ball.status is BallStatus.CARRIED)

    assert state.carry_started_at == pytest.approx(state.elapsed_seconds)


def test_carrier_fatigue_slows_the_ball_carrier_only() -> None:
    config = replace(ZERO_DEVIATION, carrier_fatigue_per_second=0.2, carrier_min_speed_factor=0.5)
    play = Play(config)
    play.state.carrier_id = "wr1"
    play.state.carry_started_at = 0.0
    play.state.elapsed_seconds = 2.0

    state = play.step({"wr1": Move(direction=vec2(1.0, 0.0))})

    assert float(np.linalg.norm(state.player("wr1").velocity)) == pytest.approx(7.0 * 0.6)

    play = Play(config)
    state = play.step({"cb1": Move(direction=vec2(1.0, 0.0))})

    assert float(np.linalg.norm(state.player("cb1").velocity)) == pytest.approx(6.5)


def test_carrier_fatigue_respects_the_floor() -> None:
    config = replace(ZERO_DEVIATION, carrier_fatigue_per_second=0.5, carrier_min_speed_factor=0.4)
    play = Play(config)
    play.state.carrier_id = "wr1"
    play.state.carry_started_at = 0.0
    play.state.elapsed_seconds = 5.0

    state = play.step({"wr1": Move(direction=vec2(1.0, 0.0))})

    assert float(np.linalg.norm(state.player("wr1").velocity)) == pytest.approx(7.0 * 0.4)


def test_fatigue_is_disabled_by_default() -> None:
    play = Play(ZERO_DEVIATION)
    play.state.carrier_id = "wr1"
    play.state.carry_started_at = 0.0
    play.state.elapsed_seconds = 3.0

    state = play.step({"wr1": Move(direction=vec2(1.0, 0.0))})

    assert float(np.linalg.norm(state.player("wr1").velocity)) == pytest.approx(7.0)
