import pytest

from qb_sim.geometry import FieldBounds, distance, segment_circle_contact_fraction, vec2


def test_distance_uses_euclidean_coordinates() -> None:
    assert distance(vec2(0.0, 0.0), vec2(3.0, 4.0)) == pytest.approx(5.0)


def test_vec2_returns_an_independent_array() -> None:
    first = vec2(1.0, 2.0)
    second = vec2(1.0, 2.0)

    first[0] = 99.0

    assert second[0] == 1.0


def test_segment_circle_contact_returns_first_crossing_fraction() -> None:
    fraction = segment_circle_contact_fraction(
        vec2(-2.0, 0.0),
        vec2(2.0, 0.0),
        vec2(0.0, 0.0),
        radius=1.0,
    )

    assert fraction == pytest.approx(0.25)


def test_segment_circle_contact_returns_none_for_a_miss() -> None:
    fraction = segment_circle_contact_fraction(
        vec2(-2.0, 2.0),
        vec2(2.0, 2.0),
        vec2(0.0, 0.0),
        radius=1.0,
    )

    assert fraction is None


def test_segment_circle_contact_is_zero_when_starting_inside() -> None:
    fraction = segment_circle_contact_fraction(
        vec2(0.0, 0.0),
        vec2(2.0, 0.0),
        vec2(0.0, 0.0),
        radius=1.0,
    )

    assert fraction == 0.0


def test_field_bounds_include_edges_and_reject_points_outside() -> None:
    field = FieldBounds(length_yards=120.0, width_yards=53.3)

    assert field.contains(vec2(0.0, 0.0))
    assert field.contains(vec2(120.0, 53.3))
    assert not field.contains(vec2(-0.01, 25.0))
    assert not field.contains(vec2(120.01, 25.0))
    assert not field.contains(vec2(60.0, -0.01))
    assert not field.contains(vec2(60.0, 53.31))
