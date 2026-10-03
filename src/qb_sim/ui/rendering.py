"""Read-only drawing of simulation state with Pygame."""

from collections.abc import Mapping
from math import atan2, degrees

import numpy as np
import pygame

from qb_sim.engine.entities import BallStatus, PlayState, Role
from qb_sim.engine.geometry import Vector2, vec2
from qb_sim.rules.routes import Route
from qb_sim.ui.controls import Aim
from qb_sim.ui.view import ViewTransform, arc_screen_points, dash_segments

FIELD_COLOR = (36, 110, 54)
END_ZONE_COLOR = (26, 82, 44)
LINE_COLOR = (225, 235, 225)
ROUTE_COLOR = (245, 245, 245)
GUIDE_COLOR = (210, 210, 210)
PLAYER_COLORS = {
    Role.QUARTERBACK: (250, 205, 60),
    Role.RECEIVER: (80, 160, 255),
    Role.DEFENDER: (225, 80, 80),
}
RUSHER_COLOR = (255, 145, 40)
FOOTBALL_COLOR = (122, 62, 36)
FOOTBALL_LACE_COLOR = (242, 242, 242)
FOOTBALL_OUTLINE_COLOR = (58, 30, 16)
SHADOW_COLOR = (22, 72, 40)
SCRIMMAGE_COLOR = (70, 140, 255)
FIRST_DOWN_COLOR = (255, 220, 90)
YARD_NUMBER_COLOR = (222, 230, 214)
PLAYER_RADIUS_YARDS = 0.5
MIN_PLAYER_RADIUS = 3
BALL_RADIUS = 4
END_ZONE_YARDS = 10.0

PIXEL_DIGITS: dict[str, tuple[str, ...]] = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
}


def yard_number_label(x: float, goal_line: float = 100.0) -> str:
    """Yard-line number as the distance to the nearest goal line."""
    distance = int(round(min(x, goal_line - x)))
    return str(max(distance, 0))


def _pixel_text(
    surface: pygame.Surface,
    text: str,
    center: tuple[float, float],
    pixel: int,
    color: tuple[int, int, int],
) -> None:
    width = len(text) * 4 * pixel - pixel
    height = 5 * pixel
    start_x = int(round(center[0] - width / 2))
    start_y = int(round(center[1] - height / 2))
    for char_index, char in enumerate(text):
        rows = PIXEL_DIGITS.get(char)
        if rows is None:
            continue
        for row, line in enumerate(rows):
            for col, cell in enumerate(line):
                if cell == "1":
                    surface.fill(
                        color,
                        pygame.Rect(
                            start_x + (char_index * 4 + col) * pixel,
                            start_y + row * pixel,
                            pixel,
                            pixel,
                        ),
                    )


def draw_yard_numbers(
    surface: pygame.Surface,
    view: ViewTransform,
    goal_line: float = 100.0,
) -> None:
    """Draw pixel-art yard numbers along the top and bottom of the field."""
    pixel = max(2, int(round(view.scale * 0.45)))
    top_y = view.to_screen(vec2(0.0, view.field.width_yards - 4.0))[1]
    bottom_y = view.to_screen(vec2(0.0, 4.0))[1]
    yard = 10.0
    while yard <= goal_line - 10.0 + 1e-9:
        label = yard_number_label(yard, goal_line)
        center_x = view.to_screen(vec2(yard, 0.0))[0]
        _pixel_text(surface, label, (center_x, top_y), pixel, YARD_NUMBER_COLOR)
        _pixel_text(surface, label, (center_x, bottom_y), pixel, YARD_NUMBER_COLOR)
        yard += 10.0


def player_radius(view: ViewTransform) -> int:
    """Player draw radius in pixels, matched to yards for contact legibility."""
    return max(MIN_PLAYER_RADIUS, int(round(PLAYER_RADIUS_YARDS * view.scale)))


def _screen_int(view: ViewTransform, point: Vector2) -> tuple[int, int]:
    screen = view.to_screen(point)
    return int(round(screen[0])), int(round(screen[1]))


def _pixel(point: Vector2) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


def _field_rect(
    view: ViewTransform,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> pygame.Rect:
    top_left = view.to_screen(vec2(x0, y1))
    bottom_right = view.to_screen(vec2(x1, y0))
    return pygame.Rect(
        int(round(top_left[0])),
        int(round(top_left[1])),
        int(round(bottom_right[0] - top_left[0])),
        int(round(bottom_right[1] - top_left[1])),
    )


def draw_field(
    surface: pygame.Surface,
    view: ViewTransform,
    goal_line: float = 100.0,
) -> None:
    """Draw the field, the opponent end zone, yard lines, and yard numbers."""
    surface.fill(FIELD_COLOR)
    field = view.field
    pygame.draw.rect(
        surface,
        END_ZONE_COLOR,
        _field_rect(view, goal_line, 0.0, field.length_yards, field.width_yards),
    )
    pygame.draw.rect(
        surface,
        LINE_COLOR,
        _field_rect(view, 0.0, 0.0, field.length_yards, field.width_yards),
        width=2,
    )
    yard = 10.0
    while yard <= goal_line - 10.0 + 1e-9:
        start = _screen_int(view, vec2(yard, 0.0))
        end = _screen_int(view, vec2(yard, field.width_yards))
        pygame.draw.line(surface, LINE_COLOR, start, end, 1)
        yard += 10.0
    pygame.draw.line(
        surface,
        LINE_COLOR,
        _screen_int(view, vec2(goal_line, 0.0)),
        _screen_int(view, vec2(goal_line, field.width_yards)),
        2,
    )
    draw_yard_numbers(surface, view, goal_line)


def draw_scrimmage(
    surface: pygame.Surface,
    view: ViewTransform,
    los: float,
    first_down: float,
    goal_line: float = 100.0,
) -> None:
    """Draw the line of scrimmage and a dashed first-down marker with a flag."""
    field = view.field
    pygame.draw.line(
        surface,
        SCRIMMAGE_COLOR,
        _screen_int(view, vec2(los, 0.0)),
        _screen_int(view, vec2(los, field.width_yards)),
        2,
    )
    if 0.0 <= first_down <= goal_line:
        start = view.to_screen(vec2(first_down, 0.0))
        end = view.to_screen(vec2(first_down, field.width_yards))
        for segment_start, segment_end in dash_segments(start, end, dash=10.0, gap=7.0):
            pygame.draw.line(
                surface,
                FIRST_DOWN_COLOR,
                _pixel(segment_start),
                _pixel(segment_end),
                2,
            )
        flag_top = view.to_screen(vec2(first_down, field.width_yards))
        pygame.draw.polygon(
            surface,
            FIRST_DOWN_COLOR,
            [
                _pixel(flag_top),
                _pixel(vec2(flag_top[0] + 12.0, flag_top[1] + 7.0)),
                _pixel(vec2(flag_top[0], flag_top[1] + 14.0)),
            ],
        )


def draw_routes(
    surface: pygame.Surface,
    routes: Mapping[str, Route],
    state: PlayState,
    view: ViewTransform,
) -> None:
    """Draw each receiver's remaining route as a polyline."""
    for player_id, route in routes.items():
        points = [state.player(player_id).position, *route.remaining_waypoints]
        if len(points) < 2:
            continue
        pygame.draw.lines(
            surface,
            ROUTE_COLOR,
            False,
            [_screen_int(view, point) for point in points],
            1,
        )


def draw_players(
    surface: pygame.Surface,
    state: PlayState,
    view: ViewTransform,
    rusher_ids: frozenset[str] = frozenset(),
    highlight_ids: frozenset[str] = frozenset(),
) -> None:
    """Draw every player as a circle colored by role; rushers and learners stand out."""
    radius = player_radius(view)
    for player in state.players:
        if player.player_id in rusher_ids:
            color = RUSHER_COLOR
        else:
            color = PLAYER_COLORS.get(player.role, (200, 200, 200))
        pygame.draw.circle(surface, color, _screen_int(view, player.position), radius)
        if player.player_id in rusher_ids:
            pygame.draw.circle(
                surface,
                (120, 60, 10),
                _screen_int(view, player.position),
                radius,
                2,
            )
        if player.player_id in highlight_ids:
            pygame.draw.circle(
                surface,
                (255, 255, 255),
                _screen_int(view, player.position),
                radius + max(2, radius // 3),
                2,
            )


def _power_color(power: float) -> tuple[int, int, int]:
    low = (255, 235, 130)
    high = (255, 90, 60)
    return tuple(
        round(low[channel] + (high[channel] - low[channel]) * power) for channel in range(3)
    )


def draw_aim(
    surface: pygame.Surface,
    aim: Aim,
    view: ViewTransform,
    arc_peak: float,
) -> None:
    """Draw the dotted slingshot arc toward the landing point."""
    color = _power_color(aim.power)
    width = 2 + round(aim.power * 2)
    points = arc_screen_points(view, aim.origin, aim.target, arc_peak)
    for segment_start, segment_end in zip(points, points[1:], strict=False):
        for dash_start, dash_end in dash_segments(segment_start, segment_end):
            pygame.draw.line(surface, color, _pixel(dash_start), _pixel(dash_end), width)
    pygame.draw.circle(surface, color, _pixel(points[-1]), 7, 2)


def _ball_screen(state: PlayState, view: ViewTransform) -> Vector2:
    ground = view.to_screen(state.ball.position)
    return vec2(ground[0], ground[1] - state.ball.height * view.scale)


def draw_ball_guide(surface: pygame.Surface, state: PlayState, view: ViewTransform) -> None:
    """Draw a faint guide from an in-flight ball to its already-known target."""
    if state.ball.status is not BallStatus.IN_FLIGHT or state.ball.target is None:
        return
    pygame.draw.line(
        surface,
        GUIDE_COLOR,
        _pixel(_ball_screen(state, view)),
        _screen_int(view, state.ball.target),
        1,
    )


def _ball_velocity(state: PlayState) -> Vector2:
    velocity = state.ball.velocity
    if float(np.linalg.norm(velocity)) >= 1e-6:
        return velocity
    if state.carrier_id is not None:
        return state.player(state.carrier_id).velocity
    if state.ball.status is BallStatus.HELD:
        return state.player("qb").velocity
    return velocity


def _football_sprite(length: int, width: int) -> pygame.Surface:
    sprite = pygame.Surface((length, width), pygame.SRCALPHA)
    rect = pygame.Rect(0, 0, length, width)
    pygame.draw.ellipse(sprite, FOOTBALL_COLOR, rect)
    pygame.draw.ellipse(sprite, FOOTBALL_OUTLINE_COLOR, rect, 1)
    center_x, center_y = length // 2, width // 2
    lace_half = max(2, length // 6)
    pygame.draw.line(
        sprite,
        FOOTBALL_LACE_COLOR,
        (center_x - lace_half, center_y),
        (center_x + lace_half, center_y),
        1,
    )
    step = max(2, lace_half // 2)
    for offset in range(-lace_half + 1, lace_half, step):
        pygame.draw.line(
            sprite,
            FOOTBALL_LACE_COLOR,
            (center_x + offset, center_y - max(1, width // 4)),
            (center_x + offset, center_y + max(1, width // 4)),
            1,
        )
    return sprite


def draw_ball(surface: pygame.Surface, state: PlayState, view: ViewTransform) -> None:
    """Draw the football as a rotating sprite with laces and an airborne shadow."""
    ball_screen = _ball_screen(state, view)
    if state.ball.height <= 0.0 and state.ball.status in (
        BallStatus.HELD,
        BallStatus.CARRIED,
    ):
        ball_screen = vec2(ball_screen[0] + player_radius(view), ball_screen[1] - 2)
    else:
        pygame.draw.circle(
            surface,
            SHADOW_COLOR,
            _screen_int(view, state.ball.position),
            BALL_RADIUS,
        )
    length = max(12, int(round(1.8 * view.scale)))
    width = max(7, int(round(length * 0.62)))
    velocity = _ball_velocity(state)
    if float(np.linalg.norm(velocity)) < 1e-6:
        angle = 0.0
    else:
        angle = -degrees(atan2(float(velocity[1]), float(velocity[0])))
    sprite = pygame.transform.rotate(_football_sprite(length, width), angle)
    surface.blit(sprite, sprite.get_rect(center=_pixel(ball_screen)))


def draw_hud(
    surface: pygame.Surface,
    state: PlayState,
    aiming_power: float | None,
    paused: bool,
    font: pygame.font.Font,
) -> None:
    """Draw time, outcome, aiming power, and control hints."""
    lines = [f"Time: {state.elapsed_seconds:0.2f}s   Outcome: {state.outcome.value}"]
    if aiming_power is not None:
        lines.append(f"AIMING  power: {aiming_power * 100:0.0f}%")
    lines.append(
        "WASD/arrows move | Drag & release to throw | Esc/right click cancel | P pause | R restart"
    )
    if paused:
        lines.append("PAUSED")
    for index, text in enumerate(lines):
        image = font.render(text, True, (255, 255, 255))
        surface.blit(image, (14, 12 + index * 28))
