import pytest

pygame = pytest.importorskip("pygame")

from qb_sim.engine.geometry import FieldBounds  # noqa: E402
from qb_sim.ui.rendering import MIN_PLAYER_RADIUS, player_radius  # noqa: E402
from qb_sim.ui.view import ViewTransform  # noqa: E402


def test_player_radius_matches_yards_and_scales_with_the_view() -> None:
    small = ViewTransform(FieldBounds(), 200, 200, margin=10)
    large = ViewTransform(FieldBounds(), 1200, 700, margin=20)

    assert player_radius(small) >= MIN_PLAYER_RADIUS
    assert player_radius(large) > player_radius(small)
    assert player_radius(large) >= 4
