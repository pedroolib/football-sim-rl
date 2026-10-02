from dataclasses import replace

import numpy as np
import pytest

from qb_sim.config import SimulationConfig
from qb_sim.engine.geometry import vec2
from qb_sim.rules.routes import (
    DEFAULT_ROUTE_POOL,
    ROUTE_OBSERVATION_ORDER,
    RouteName,
    build_route,
    sample_routes,
)


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


def test_route_reset_restores_progress() -> None:
    config = SimulationConfig()
    start = vec2(20.0, 43.0)
    route = build_route(RouteName.GO, start, config.field, config)
    route.advance(start, max_distance=100.0)

    route.reset()

    assert len(route.remaining_waypoints) == 1


def test_route_exposes_remaining_waypoints_after_advancing() -> None:
    config = SimulationConfig()
    start = vec2(20.0, 43.0)
    route = build_route(RouteName.GO, start, config.field, config)

    assert len(route.remaining_waypoints) == 1

    route.advance(start, max_distance=100.0)

    assert route.remaining_waypoints == ()


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


def test_out_route_breaks_outward_toward_the_sideline() -> None:
    config = replace(SimulationConfig(), cut_length_yards=5.0)
    start = vec2(20.0, 43.0)

    route = build_route(RouteName.OUT, start, config.field, config)

    break_point = route.waypoints[0]
    cut = route.waypoints[1] - break_point
    assert break_point.tolist() == pytest.approx([25.0, 43.0])
    assert cut[0] > 0.0
    assert cut[1] > 0.0


def test_curl_route_breaks_then_comes_back_toward_the_los() -> None:
    config = SimulationConfig()
    start = vec2(20.0, 43.0)

    route = build_route(RouteName.CURL, start, config.field, config)

    assert route.waypoints[0].tolist() == pytest.approx([30.0, 43.0])
    assert route.waypoints[1].tolist() == pytest.approx([27.0, 43.0])


def test_flat_route_runs_laterally_to_the_sideline() -> None:
    config = SimulationConfig()

    route = build_route(RouteName.FLAT, vec2(20.0, 10.0), config.field, config)

    assert route.waypoints[0].tolist() == pytest.approx([20.0, 4.0])


def test_wheel_route_goes_out_then_up_the_sideline() -> None:
    config = SimulationConfig()

    route = build_route(RouteName.WHEEL, vec2(20.0, 10.0), config.field, config)

    assert route.waypoints[0].tolist() == pytest.approx([20.0, 6.0])
    assert route.waypoints[1].tolist() == pytest.approx([40.0, 6.0])


def test_corner_route_breaks_outward_toward_the_sideline() -> None:
    config = SimulationConfig()

    route = build_route(RouteName.CORNER, vec2(20.0, 10.0), config.field, config)

    assert route.waypoints[0].tolist() == pytest.approx([32.0, 10.0])
    assert route.waypoints[1][1] < route.waypoints[0][1]


def test_sample_routes_is_deterministic_and_covers_the_pool() -> None:
    first = sample_routes(np.random.default_rng(7), 4)
    second = sample_routes(np.random.default_rng(7), 4)

    assert first == second
    assert len(first) == 4
    assert set(sample_routes(np.random.default_rng(3), 200)) == set(DEFAULT_ROUTE_POOL)


def test_sample_routes_validates_its_arguments() -> None:
    rng = np.random.default_rng(0)

    assert sample_routes(rng, 0) == ()
    with pytest.raises(ValueError):
        sample_routes(rng, -1)
    with pytest.raises(ValueError):
        sample_routes(rng, 1, pool=())


def test_route_observation_order_matches_the_default_pool() -> None:
    assert ROUTE_OBSERVATION_ORDER == DEFAULT_ROUTE_POOL
    assert len(ROUTE_OBSERVATION_ORDER) == 8
