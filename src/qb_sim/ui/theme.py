"""Shared visual tokens for the Pygame dashboard (shadcn/ui dark, slate)."""

import pygame

BACKGROUND = (2, 8, 23)  # hsl(222.2 84% 4.9%)
SURFACE = (15, 23, 42)  # slate-900 card lift
SURFACE_ALT = (30, 41, 59)  # slate-800: muted, accent, borders
BORDER = (30, 41, 59)
TEXT = (248, 250, 252)  # slate-50
TEXT_MUTED = (148, 163, 184)  # slate-400
PRIMARY = (248, 250, 252)  # near-white, shadcn dark primary
PRIMARY_SOFT = (203, 213, 225)  # slate-300 ring
ACCENT = (96, 165, 250)  # blue-400 interactive hint
CHART = (59, 130, 246)  # blue-500 chart line
DESTRUCTIVE = (248, 113, 113)  # red-400
SUCCESS = (74, 222, 128)  # green-400

SPACE_XS = 4
SPACE_SM = 8
SPACE_MD = 12
SPACE_LG = 16
SPACE_XL = 24

RADIUS_SM = 6
RADIUS_MD = 8
RADIUS_LG = 12

FONT_SIZE_SM = 14
FONT_SIZE_MD = 17
FONT_SIZE_LG = 21

_SANS_CANDIDATES = ("inter", "geist", "notosans", "dejavusans", "liberationsans", "arial")
_MONO_CANDIDATES = (
    "jetbrainsmononerdfont",
    "jetbrainsmono",
    "firacode",
    "dejavusansmono",
    "liberationmono",
    "courier",
)


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
