import pytest

from qb_sim.engine.entities import BallState, BallStatus
from qb_sim.engine.geometry import FieldBounds, vec2
from qb_sim.engine.physics import (
    advance_ball,
    move_toward,
    moving_circle_contact_fraction,
    segment_field_exit_fraction,
)


def test_move_toward_caps_displacement() -> None:
    moved = move_toward(vec2(0.0, 0.0), vec2(10.0, 0.0), max_distance=2.0)

    assert moved.tolist() == pytest.approx([2.0, 0.0])


def test_move_toward_does_not_overshoot_target() -> None:
    moved = move_toward(vec2(0.0, 0.0), vec2(1.0, 0.0), max_distance=5.0)

    assert moved.tolist() == pytest.approx([1.0, 0.0])


def test_move_toward_keeps_position_when_already_at_target() -> None:
    moved = move_toward(vec2(4.0, 7.0), vec2(4.0, 7.0), max_distance=2.0)

    assert moved.tolist() == pytest.approx([4.0, 7.0])


def test_moving_circle_contact_detects_a_fast_ball_crossing() -> None:
    fraction = moving_circle_contact_fraction(
        vec2(-2.0, 0.0),
        vec2(2.0, 0.0),
        vec2(0.0, 0.0),
        vec2(0.0, 0.0),
        radius=1.0,
    )

    assert fraction == pytest.approx(0.25)


def test_moving_circle_contact_uses_both_paths() -> None:
    fraction = moving_circle_contact_fraction(
        vec2(0.0, 0.0),
        vec2(10.0, 0.0),
        vec2(5.0, 5.0),
        vec2(5.0, -5.0),
        radius=1.0,
    )

    assert fraction == pytest.approx(0.5 - 1.0 / (10.0 * 2.0**0.5))


def test_advance_ball_clamps_at_target_and_reports_tick_fraction() -> None:
    ball = BallState(
        position=vec2(0.0, 0.0),
        velocity=vec2(10.0, 0.0),
        status=BallStatus.IN_FLIGHT,
        target=vec2(1.0, 0.0),
    )

    next_position, tick_fraction = advance_ball(ball, dt=0.2)

    assert next_position.tolist() == pytest.approx([1.0, 0.0])
    assert tick_fraction == pytest.approx(0.5)


def test_segment_field_exit_fraction_returns_first_boundary_crossing() -> None:
    field = FieldBounds(length_yards=10.0, width_yards=5.0)

    fraction = segment_field_exit_fraction(vec2(9.0, 2.0), vec2(11.0, 2.0), field)

    assert fraction == pytest.approx(0.5)


def test_segment_field_exit_fraction_is_none_for_an_inside_segment() -> None:
    field = FieldBounds(length_yards=10.0, width_yards=5.0)

    fraction = segment_field_exit_fraction(vec2(1.0, 1.0), vec2(9.0, 4.0), field)

    assert fraction is None
