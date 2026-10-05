import pytest

pygame = pytest.importorskip("pygame")

from qb_sim.ui import theme  # noqa: E402


def test_palette_tokens_are_valid_rgb() -> None:
    for name in (
        "BACKGROUND",
        "SURFACE",
        "SURFACE_ALT",
        "BORDER",
        "TEXT",
        "TEXT_MUTED",
        "PRIMARY",
        "PRIMARY_SOFT",
        "ACCENT",
        "CHART",
        "DESTRUCTIVE",
        "SUCCESS",
    ):
        color = getattr(theme, name)
        assert len(color) == 3
        assert all(0 <= channel <= 255 for channel in color)


def test_load_fonts_returns_sans_mono_and_small() -> None:
    if not pygame.font.get_init():
        pygame.font.init()
    sans, mono, small = theme.load_fonts()
    assert isinstance(sans, pygame.font.Font)
    assert isinstance(mono, pygame.font.Font)
    assert isinstance(small, pygame.font.Font)
    assert small.get_height() < sans.get_height()
