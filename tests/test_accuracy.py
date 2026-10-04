from dataclasses import replace

import numpy as np
import pytest

from qb_sim.agents.loop import build_rule_agents
from qb_sim.config import SimulationConfig
from qb_sim.engine.actions import Move
from qb_sim.engine.entities import BallStatus, PlayOutcome
from qb_sim.engine.geometry import distance, vec2
from qb_sim.rules.accuracy import pressure_from_defenders, throw_spread
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
TERMINAL_OUTCOMES = {
    PlayOutcome.COMPLETED,
    PlayOutcome.INTERCEPTION,
    PlayOutcome.INCOMPLETE,
    PlayOutcome.OUT_OF_BOUNDS,
    PlayOutcome.TIME_LIMIT,
}


def _angle_error_degrees(play: Play, intended_direction) -> float:
    velocity = play.state.ball.velocity
    speed = float(np.linalg.norm(velocity))
    if speed == 0.0:
        return 0.0
    unit = velocity / speed
    cross = intended_direction[0] * unit[1] - intended_direction[1] * unit[0]
    dot = float(np.dot(intended_direction, unit))
    return abs(float(np.degrees(np.arctan2(cross, dot))))


def _mean_angle_error(
    config: SimulationConfig,
    seeds,
    *,
    moving: bool,
    pressured: bool,
    target: float = 60.0,
) -> float:
    scenario = config
    if pressured:
        scenario = replace(config, defender_starts=((21.0, 26.65), (0.0, 0.0)))
    errors = []
    for seed in seeds:
        play = Play(scenario, seed=seed)
        if moving:
            play.step({"qb": Move(direction=vec2(-1.0, 0.0))})
        play.throw_to(vec2(target, 26.65))
        errors.append(_angle_error_degrees(play, vec2(1.0, 0.0)))
    return float(np.mean(errors))


def test_throw_spread_grows_with_movement_pressure_and_distance() -> None:
    config = SimulationConfig()

    planted = throw_spread(0.0, 0.0, 10.0, config)[0]
    moving = throw_spread(config.quarterback_speed, 0.0, 10.0, config)[0]
    pressured = throw_spread(0.0, 1.0, 10.0, config)[0]
    distant = throw_spread(0.0, 0.0, config.throw_max_range, config)[0]

    assert planted < moving
    assert planted < pressured
    assert planted < distant


def test_throw_spread_is_zero_with_zero_sigmas() -> None:
    angle_sigma, power_sigma = throw_spread(6.5, 1.0, 40.0, ZERO_DEVIATION)

    assert angle_sigma == 0.0
    assert power_sigma == 0.0


def test_pressure_scales_from_zero_to_one() -> None:
    quarterback = vec2(0.0, 0.0)

    assert pressure_from_defenders(quarterback, [vec2(10.0, 0.0)], radius=5.0) == 0.0
    assert pressure_from_defenders(quarterback, [vec2(2.5, 0.0)], radius=5.0) == pytest.approx(0.5)
    assert pressure_from_defenders(quarterback, [vec2(0.0, 0.0)], radius=5.0) == pytest.approx(1.0)
    assert pressure_from_defenders(quarterback, [], radius=5.0) == 0.0


def test_zero_deviation_reproduces_previous_behavior() -> None:
    play = Play(ZERO_DEVIATION)

    play.throw_to(vec2(60.0, 26.65))

    assert play.state.ball.target.tolist() == pytest.approx([60.0, 26.65])
    assert play.state.ball.velocity.tolist() == pytest.approx([25.0, 0.0])


def test_power_noise_stays_within_configured_range() -> None:
    config = replace(SimulationConfig(), throw_power_sigma=0.5, throw_base_sigma_degrees=0.0)

    for seed in range(50):
        for requested_power in (0.0, 1.0):
            play = Play(config, seed=seed)
            quarterback = play.state.player("qb").position
            play.throw_to(vec2(60.0, 26.65), power=requested_power)
            launched_range = float(distance(quarterback, play.state.ball.target))
            assert config.throw_min_range <= launched_range <= config.throw_max_range + 1e-9


def test_same_seed_reproduces_the_same_deviation() -> None:
    first = Play(seed=7)
    second = Play(seed=7)

    first.throw_to(vec2(60.0, 26.65))
    second.throw_to(vec2(60.0, 26.65))

    np.testing.assert_allclose(first.state.ball.target, second.state.ball.target)


def test_different_seeds_produce_different_deviations() -> None:
    first = Play(seed=1)
    second = Play(seed=2)

    first.throw_to(vec2(60.0, 26.65))
    second.throw_to(vec2(60.0, 26.65))

    assert not np.allclose(first.state.ball.target, second.state.ball.target)


def test_reset_with_seed_reproduces_the_sequence() -> None:
    play = Play(seed=3)
    play.throw_to(vec2(60.0, 26.65))
    first_target = play.state.ball.target.copy()

    play.reset(seed=3)
    play.throw_to(vec2(60.0, 26.65))

    np.testing.assert_allclose(play.state.ball.target, first_target)


def test_reset_without_seed_advances_the_sequence() -> None:
    play = Play(seed=3)
    play.throw_to(vec2(60.0, 26.65))
    first_target = play.state.ball.target.copy()

    play.reset()
    play.throw_to(vec2(60.0, 26.65))

    assert not np.allclose(play.state.ball.target, first_target)


def test_launch_only_changes_conditions_and_not_the_outcome() -> None:
    play = Play(seed=0)

    play.throw_to(vec2(60.0, 26.65))

    assert play.state.outcome is PlayOutcome.IN_PROGRESS
    assert play.state.ball.status is BallStatus.IN_FLIGHT


def test_noise_shifts_the_target_but_resolution_still_uses_physics() -> None:
    config = replace(
        SimulationConfig(),
        quarterback_start=(20.0, 26.65),
        receiver_starts=((30.0, 26.65), (119.0, 0.0)),
        defender_starts=((0.0, 0.0), (0.0, 0.0)),
        throw_base_sigma_degrees=20.0,
    )
    play = Play(config, seed=0)
    agents = build_rule_agents(play, (RouteName.POST, RouteName.SLANT), include_quarterback=False)
    receiver = play.state.player("wr1")
    intended_target = receiver.position + receiver.velocity * 0.4

    play.throw_with_lead(0.4)

    assert not np.allclose(play.state.ball.target, intended_target)

    state = play.state
    for _ in range(200):
        actions = {}
        for player_id, agent in agents.items():
            action = agent.decide(play.state)
            if action is not None:
                actions[player_id] = action
        state = play.step(actions)
        if state.outcome is not PlayOutcome.IN_PROGRESS:
            break

    assert state.outcome in TERMINAL_OUTCOMES


def test_planted_throws_are_more_accurate_than_moving_or_pressured() -> None:
    config = SimulationConfig()
    seeds = range(200)

    planted = _mean_angle_error(config, seeds, moving=False, pressured=False)
    moving = _mean_angle_error(config, seeds, moving=True, pressured=False)
    pressured = _mean_angle_error(config, seeds, moving=False, pressured=True)

    assert planted < moving
    assert planted < pressured


def test_longer_throws_are_less_accurate_than_short_ones() -> None:
    config = SimulationConfig()
    seeds = range(200)

    short = _mean_angle_error(config, seeds, moving=False, pressured=False, target=30.0)
    long = _mean_angle_error(config, seeds, moving=False, pressured=False, target=60.0)

    assert short < long


def test_pressure_is_exposed_in_the_play_state() -> None:
    config = replace(ZERO_DEVIATION, defender_starts=((22.5, 26.65), (0.0, 0.0)))

    state = Play(config).reset()

    assert state.pressure == pytest.approx(0.5)
