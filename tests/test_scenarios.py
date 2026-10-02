import pytest

from qb_sim.config import SimulationConfig
from qb_sim.rules.routes import RouteName
from qb_sim.rules.scenarios import (
    DEFAULT_SCENARIO,
    PRESSURED_SCENARIO,
    RUSHED_SCENARIO,
    Scenario,
    get_scenario,
)


def test_default_scenario_has_one_route_per_receiver() -> None:
    assert len(DEFAULT_SCENARIO.receiver_routes) == len(DEFAULT_SCENARIO.config.receiver_starts)


def test_pressured_scenario_starts_a_defender_near_the_quarterback() -> None:
    defender_start = PRESSURED_SCENARIO.config.defender_starts[0]
    quarterback_start = PRESSURED_SCENARIO.config.quarterback_start

    distance = (
        (defender_start[0] - quarterback_start[0]) ** 2
        + (defender_start[1] - quarterback_start[1]) ** 2
    ) ** 0.5
    assert distance < PRESSURED_SCENARIO.config.pressure_radius


def test_rushed_scenario_adds_a_pass_rusher_to_the_roster() -> None:
    assert RUSHED_SCENARIO.config.rusher_starts == ((30.0, 29.0),)
    assert len(RUSHED_SCENARIO.config.defender_starts) == 2
    assert get_scenario("rushed") is RUSHED_SCENARIO


def test_get_scenario_returns_named_scenarios() -> None:
    assert get_scenario("default") is DEFAULT_SCENARIO


def test_get_scenario_rejects_unknown_names() -> None:
    with pytest.raises(ValueError):
        get_scenario("nope")


def test_scenario_rejects_route_count_mismatch() -> None:
    with pytest.raises(ValueError):
        Scenario("bad", SimulationConfig(), (RouteName.POST,))
