import pytest

pygame = pytest.importorskip("pygame")

from qb_sim.engine.geometry import FieldBounds  # noqa: E402
from qb_sim.rules.play import Play  # noqa: E402
from qb_sim.ui.rendering import (  # noqa: E402
    FOOTBALL_COLOR,
    MIN_PLAYER_RADIUS,
    PIXEL_DIGITS,
    draw_ball,
    draw_field,
    player_radius,
    yard_number_label,
)
from qb_sim.ui.view import ViewTransform  # noqa: E402


def test_yard_number_labels_mirror_around_midfield() -> None:
    assert yard_number_label(10.0, 100.0) == "10"
    assert yard_number_label(30.0, 100.0) == "30"
    assert yard_number_label(50.0, 100.0) == "50"
    assert yard_number_label(70.0, 100.0) == "30"
    assert yard_number_label(90.0, 100.0) == "10"


def test_pixel_digits_cover_every_number() -> None:
    for digit in "0123456789":
        assert digit in PIXEL_DIGITS
        assert len(PIXEL_DIGITS[digit]) == 5
        assert all(len(row) == 3 for row in PIXEL_DIGITS[digit])


def test_draw_field_with_a_goal_line_renders_numbers_and_end_zone() -> None:
    surface = pygame.Surface((480, 240))
    view = ViewTransform(FieldBounds(), 480, 240, margin=10)

    draw_field(surface, view, goal_line=100.0)

    assert surface.get_at((240, 120))[:3] != (0, 0, 0)


def test_player_radius_matches_yards_and_scales_with_the_view() -> None:
    small = ViewTransform(FieldBounds(), 200, 200, margin=10)
    large = ViewTransform(FieldBounds(), 1200, 700, margin=20)

    assert player_radius(small) >= MIN_PLAYER_RADIUS
    assert player_radius(large) > player_radius(small)
    assert player_radius(large) >= 4


def test_draw_ball_renders_a_football_in_flight() -> None:
    play = Play()
    play.throw_with_lead(0.0)
    surface = pygame.Surface((240, 240))
    surface.fill((0, 0, 0))
    view = ViewTransform(FieldBounds(), 240, 240, margin=10)

    draw_ball(surface, play.state, view)

    found = False
    for x in range(0, 240, 3):
        for y in range(0, 240, 3):
            red, green, blue, *_ = surface.get_at((x, y))
            if (
                abs(red - FOOTBALL_COLOR[0]) < 30
                and abs(green - FOOTBALL_COLOR[1]) < 30
                and abs(blue - FOOTBALL_COLOR[2]) < 30
            ):
                found = True
                break
        if found:
            break
    assert found
