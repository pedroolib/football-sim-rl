"""Shared visual tokens for the Pygame dashboard (dark, data-dense)."""

import pygame

BACKGROUND = (11, 18, 32)
SURFACE = (18, 26, 42)
SURFACE_ALT = (24, 35, 56)
BORDER = (38, 51, 77)
TEXT = (230, 236, 245)
TEXT_MUTED = (143, 160, 184)
PRIMARY = (59, 130, 246)
PRIMARY_SOFT = (96, 165, 250)
ACCENT = (245, 158, 11)
DESTRUCTIVE = (239, 68, 68)
SUCCESS = (34, 197, 94)

SPACE_XS = 4
SPACE_SM = 8
SPACE_MD = 12
SPACE_LG = 16
SPACE_XL = 24

RADIUS_SM = 6
RADIUS_MD = 10
RADIUS_LG = 14

FONT_SIZE_SM = 14
FONT_SIZE_MD = 17
FONT_SIZE_LG = 21

_SANS_CANDIDATES = ("firasans", "dejavusans", "liberationsans", "notosans", "arial")
_MONO_CANDIDATES = ("firacode", "dejavusansmono", "liberationmono", "notosansmono", "courier")


def _load_font(candidates: tuple[str, ...], size: int, bold: bool) -> pygame.font.Font:
    for name in candidates:
        path = pygame.font.match_font(name, bold=bold)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.Font(None, size)


def load_fonts() -> tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font]:
    """Return ``(sans, mono, small)`` fonts for labels, data, and hints."""
    return (
        _load_font(_SANS_CANDIDATES, FONT_SIZE_MD, False),
        _load_font(_MONO_CANDIDATES, FONT_SIZE_MD, False),
        _load_font(_SANS_CANDIDATES, FONT_SIZE_SM, False),
    )
