import pytest

from qb_sim.config import SimulationConfig
from qb_sim.geometry import vec2
from qb_sim.routes import RouteName, build_route


def test_go_route_runs_downfield_without_lateral_break() -> None:
    config = SimulationConfig()
    start = vec2(20.0, 43.0)

    route = build_route(RouteName.GO, start, config.field, config)

    assert route.waypoints[0].tolist() == pytest.approx([50.0, 43.0])


def test_slant_route_breaks_early_toward_field_center() -> None:
    config = SimulationConfig()
    start = vec2(20.0, 43.0)

    route = build_route(RouteName.SLANT, start, config.field, config)

    assert route.waypoints[0].tolist() == pytest.approx([23.0, 43.0])
    cut = route.waypoints[1] - route.waypoints[0]
    assert cut[0] == pytest.approx(20.0 / 2.0**0.5)
    assert cut[1] == pytest.approx(-20.0 / 2.0**0.5)


def test_post_route_has_ten_yard_stem_then_a_45_degree_inward_cut() -> None:
    config = SimulationConfig()
    start = vec2(20.0, 43.0)

    route = build_route(RouteName.POST, start, config.field, config)

    assert route.waypoints[0].tolist() == pytest.approx([30.0, 43.0])
    cut = route.waypoints[1] - route.waypoints[0]
    assert cut[0] == pytest.approx(20.0 / 2.0**0.5)
    assert cut[1] == pytest.approx(-20.0 / 2.0**0.5)


def test_inward_routes_mirror_when_receiver_starts_below_field_center() -> None:
    config = SimulationConfig()
    start = vec2(20.0, 10.0)

    route = build_route(RouteName.POST, start, config.field, config)

    assert route.waypoints[1][1] > route.waypoints[0][1]


def test_route_advance_caps_distance_and_finishes_at_last_waypoint() -> None:
    config = SimulationConfig()
    start = vec2(20.0, 43.0)
    route = build_route(RouteName.GO, start, config.field, config)

    moved, complete = route.advance(start, max_distance=5.0)

    assert moved.tolist() == pytest.approx([25.0, 43.0])
    assert not complete

    moved, complete = route.advance(moved, max_distance=100.0)

    assert moved.tolist() == pytest.approx([50.0, 43.0])
    assert complete
