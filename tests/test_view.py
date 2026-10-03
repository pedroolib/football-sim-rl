import numpy as np
import pytest

from qb_sim.engine.geometry import FieldBounds, vec2
from qb_sim.ui.view import ViewTransform


def _camera_view() -> ViewTransform:
    return ViewTransform(
        FieldBounds(),
        400,
        200,
        margin=0,
        center=(50.0, 26.65),
        visible_width_yards=25.0,
    )


def test_camera_scale_is_driven_by_the_visible_width() -> None:
    view = _camera_view()

    assert view.scale == pytest.approx(200.0 / 25.0)


def test_camera_maps_its_center_to_the_screen_center() -> None:
    view = _camera_view()

    screen = view.to_screen(vec2(50.0, 26.65))

    assert screen[0] == pytest.approx(200.0)
    assert screen[1] == pytest.approx(100.0)


def test_camera_round_trips_field_points() -> None:
    view = ViewTransform(
        FieldBounds(),
        400,
        200,
        margin=0,
        center=(40.0, 20.0),
        visible_width_yards=25.0,
    )
    point = vec2(45.0, 24.0)

    back = view.to_field(view.to_screen(point))

    np.testing.assert_allclose(back, point)


def test_camera_visible_length_follows_the_screen_aspect() -> None:
    view = _camera_view()

    assert view.visible_length_yards == pytest.approx(400.0 / (200.0 / 25.0))


def test_full_field_view_is_unchanged_without_a_camera() -> None:
    view = ViewTransform(FieldBounds(), 400, 200, margin=0)

    assert view.scale == pytest.approx(min(400 / 120.0, 200 / 53.3))
    assert view.to_screen(vec2(0.0, 0.0)).tolist() == pytest.approx([0.0, 200.0])
    assert view.visible_length_yards == pytest.approx(120.0)


def test_camera_rejects_a_non_positive_width() -> None:
    with pytest.raises(ValueError):
        ViewTransform(FieldBounds(), 400, 200, margin=0, visible_width_yards=0.0)
