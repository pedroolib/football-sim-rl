import pytest

from qb_sim.agents.rule_controllers import (
    HeuristicQuarterback,
    ManCoverageController,
    PassRushController,
    ReceiverRouteController,
)
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.geometry import vec2
from qb_sim.rules.play import Play
from qb_sim.rules.routes import RouteName, build_route


def _route_controller(play: Play, player_id: str = "wr1") -> ReceiverRouteController:
    start = play.state.player(player_id).position
    route = build_route(RouteName.GO, start, play.config.field, play.config)
    return ReceiverRouteController(player_id, route, play.config.receiver_speed, play.config.dt)


def test_receiver_controller_runs_its_route() -> None:
    play = Play()
    controller = _route_controller(play)

    action = controller.decide(play.state)

    assert isinstance(action, Move)
    assert action.direction.tolist() == pytest.approx([1.0, 0.0])


def test_receiver_controller_stops_when_the_route_is_complete() -> None:
    play = Play()
    controller = _route_controller(play)
    controller.route.advance(play.state.player("wr1").position, max_distance=1000.0)

    action = controller.decide(play.state)

    assert action is None


def test_defender_controller_chases_its_assignment() -> None:
    play = Play()
    controller = ManCoverageController("cb1", play.config.defender_speed, "wr1", play.config.dt)

    action = controller.decide(play.state)

    assert isinstance(action, Move)
    assert action.direction[0] > 0.0


def test_defender_controller_switches_to_the_carrier() -> None:
    play = Play()
    play.state.carrier_id = "wr2"
    controller = ManCoverageController("cb1", play.config.defender_speed, "wr1", play.config.dt)

    observation = controller.observe(play.state)

    assert observation.target_id == "wr2"


def test_pass_rush_controller_chases_the_quarterback() -> None:
    play = Play()
    controller = PassRushController("cb2", play.config.defender_speed, play.config.dt)

    action = controller.decide(play.state)

    assert isinstance(action, Move)
    assert action.direction[1] > 0.9


def test_pass_rush_controller_switches_to_the_carrier() -> None:
    play = Play()
    play.state.carrier_id = "wr1"
    controller = PassRushController("cb2", play.config.defender_speed, play.config.dt)

    observation = controller.observe(play.state)

    assert observation.target_id == "wr1"


def test_heuristic_quarterback_throws_to_an_open_receiver() -> None:
    play = Play()
    for defender_id in ("cb1", "cb2"):
        play.state.player(defender_id).position = vec2(119.0, 0.0)
    play.state.elapsed_seconds = 1.0
    controller = HeuristicQuarterback("qb", play.config)

    action = controller.decide(play.state)

    assert isinstance(action, Throw)


def test_heuristic_quarterback_waits_for_routes_to_develop() -> None:
    play = Play()
    for defender_id in ("cb1", "cb2"):
        play.state.player(defender_id).position = vec2(119.0, 0.0)
    controller = HeuristicQuarterback("qb", play.config)

    action = controller.decide(play.state)

    assert action is None


def test_heuristic_quarterback_waits_when_receivers_are_covered() -> None:
    play = Play()
    play.state.player("cb1").position = play.state.player("wr1").position + vec2(0.5, 0.0)
    play.state.player("cb2").position = play.state.player("wr2").position + vec2(0.5, 0.0)
    controller = HeuristicQuarterback("qb", play.config)

    action = controller.decide(play.state)

    assert action is None


def test_heuristic_quarterback_throws_when_the_deadline_expires() -> None:
    play = Play()
    play.state.player("cb1").position = play.state.player("wr1").position + vec2(0.5, 0.0)
    play.state.player("cb2").position = play.state.player("wr2").position + vec2(0.5, 0.0)
    play.state.elapsed_seconds = play.config.max_play_duration
    controller = HeuristicQuarterback("qb", play.config)

    action = controller.decide(play.state)

    assert isinstance(action, Throw)


def test_heuristic_quarterback_scrambles_away_from_pressure() -> None:
    play = Play()
    play.state.player("cb1").position = play.state.player("wr1").position + vec2(0.5, 0.0)
    play.state.player("cb2").position = play.state.player("wr2").position + vec2(0.5, 0.0)
    play.state.pressure = 1.0
    controller = HeuristicQuarterback("qb", play.config)

    action = controller.decide(play.state)

    assert isinstance(action, Move)
    assert action.direction[0] < 0.0
