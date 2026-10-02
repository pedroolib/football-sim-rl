import pytest

from qb_sim.config import SimulationConfig
from qb_sim.engine.geometry import FieldBounds, vec2
from qb_sim.ui.controls import aim_from_drag, direction_from_keys
from qb_sim.ui.view import ViewTransform, dash_segments


def test_direction_from_keys_normalizes_diagonals() -> None:
    direction = direction_from_keys(frozenset({"w", "d"}))

    assert direction.tolist() == pytest.approx([2.0**-0.5, 2.0**-0.5])


def test_direction_from_keys_cancels_opposites() -> None:
    assert direction_from_keys(frozenset({"w", "s"})).tolist() == [0.0, 0.0]


def test_direction_from_keys_ignores_unknown_keys() -> None:
    assert direction_from_keys(frozenset({"q", "shift"})).tolist() == [0.0, 0.0]


def test_view_transform_round_trips_field_coordinates() -> None:
    view = ViewTransform(FieldBounds(), screen_width=900, screen_height=500, margin=20)

    assert view.to_field(view.to_screen(vec2(60.0, 26.65))).tolist() == pytest.approx([60.0, 26.65])


def test_view_transform_maps_origin_to_bottom_left_margin() -> None:
    view = ViewTransform(FieldBounds(), screen_width=900, screen_height=500, margin=20)

    assert view.to_screen(vec2(0.0, 0.0)).tolist() == pytest.approx([20.0, 480.0])


def test_dash_segments_split_a_line_into_gaps() -> None:
    segments = dash_segments(vec2(0.0, 0.0), vec2(20.0, 0.0), dash=8.0, gap=2.0)

    assert len(segments) == 2
    assert segments[0][0].tolist() == pytest.approx([0.0, 0.0])
    assert segments[0][1].tolist() == pytest.approx([8.0, 0.0])
    assert segments[1][0].tolist() == pytest.approx([10.0, 0.0])
    assert segments[1][1].tolist() == pytest.approx([18.0, 0.0])


def test_aim_from_drag_rejects_short_drags() -> None:
    config = SimulationConfig()

    assert aim_from_drag(vec2(20.0, 26.65), vec2(1.0, 0.0), config) is None


def test_aim_from_drag_inverts_the_gesture_into_a_forward_throw() -> None:
    config = SimulationConfig()

    aim = aim_from_drag(vec2(20.0, 26.65), vec2(-20.0, 0.0), config)

    assert aim is not None
    assert aim.direction.tolist() == pytest.approx([1.0, 0.0])
    assert aim.power == pytest.approx(18.0 / 43.0)
    assert aim.target.tolist() == pytest.approx([20.0 + 935.0 / 43.0, 26.65])


def test_aim_from_drag_clamps_power_at_maximum_drag() -> None:
    config = SimulationConfig()

    aim = aim_from_drag(vec2(20.0, 26.65), vec2(-100.0, 0.0), config)

    assert aim is not None
    assert aim.direction.tolist() == pytest.approx([1.0, 0.0])
    assert aim.power == pytest.approx(1.0)
    assert aim.target.tolist() == pytest.approx([65.0, 26.65])


def test_aim_from_drag_uses_the_drag_angle() -> None:
    config = SimulationConfig()

    aim = aim_from_drag(vec2(0.0, 0.0), vec2(0.0, -3.0), config)

    assert aim is not None
    assert aim.direction.tolist() == pytest.approx([0.0, 1.0])
    assert aim.power == pytest.approx(1.0 / 43.0)
