"""Read-only drawing of simulation state with Pygame."""

import pygame

from qb_sim.engine.entities import BallStatus, PlayState, Role
from qb_sim.engine.geometry import Vector2, vec2
from qb_sim.rules.play import Play
from qb_sim.ui.controls import Aim
from qb_sim.ui.view import ViewTransform, dash_segments

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
BALL_COLOR = (150, 75, 35)
PLAYER_RADIUS = 8
BALL_RADIUS = 4
END_ZONE_YARDS = 10.0


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


def draw_field(surface: pygame.Surface, view: ViewTransform) -> None:
    """Draw the field, end zones, boundary, and yard lines."""
    surface.fill(FIELD_COLOR)
    field = view.field
    pygame.draw.rect(
        surface, END_ZONE_COLOR, _field_rect(view, 0.0, 0.0, END_ZONE_YARDS, field.width_yards)
    )
    pygame.draw.rect(
        surface,
        END_ZONE_COLOR,
        _field_rect(
            view,
            field.length_yards - END_ZONE_YARDS,
            0.0,
            field.length_yards,
            field.width_yards,
        ),
    )
    pygame.draw.rect(
        surface,
        LINE_COLOR,
        _field_rect(view, 0.0, 0.0, field.length_yards, field.width_yards),
        width=2,
    )
    yard = END_ZONE_YARDS
    while yard <= field.length_yards - END_ZONE_YARDS:
        start = _screen_int(view, vec2(yard, 0.0))
        end = _screen_int(view, vec2(yard, field.width_yards))
        pygame.draw.line(surface, LINE_COLOR, start, end, 1)
        yard += 10.0


def draw_routes(surface: pygame.Surface, play: Play, view: ViewTransform) -> None:
    """Draw each receiver's remaining route as a polyline."""
    for receiver_id, route in zip(play.receiver_ids, play.receiver_routes, strict=True):
        points = [play.state.player(receiver_id).position, *route.remaining_waypoints]
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
) -> None:
    """Draw every player as a circle colored by role."""
    for player in state.players:
        color = PLAYER_COLORS.get(player.role, (200, 200, 200))
        pygame.draw.circle(surface, color, _screen_int(view, player.position), PLAYER_RADIUS)


def _power_color(power: float) -> tuple[int, int, int]:
    low = (255, 235, 130)
    high = (255, 90, 60)
    return tuple(
        round(low[channel] + (high[channel] - low[channel]) * power) for channel in range(3)
    )


def draw_aim(surface: pygame.Surface, aim: Aim, view: ViewTransform) -> None:
    """Draw a dotted slingshot aim line toward the landing point."""
    color = _power_color(aim.power)
    width = 2 + round(aim.power * 2)
    start = view.to_screen(aim.origin)
    end = view.to_screen(aim.target)
    for segment_start, segment_end in dash_segments(start, end):
        pygame.draw.line(surface, color, _pixel(segment_start), _pixel(segment_end), width)
    pygame.draw.circle(surface, color, _screen_int(view, aim.target), 7, 2)


def draw_ball_guide(surface: pygame.Surface, state: PlayState, view: ViewTransform) -> None:
    """Draw a faint guide from an in-flight ball to its already-known target."""
    if state.ball.status is not BallStatus.IN_FLIGHT or state.ball.target is None:
        return
    pygame.draw.line(
        surface,
        GUIDE_COLOR,
        _screen_int(view, state.ball.position),
        _screen_int(view, state.ball.target),
        1,
    )


def draw_ball(surface: pygame.Surface, state: PlayState, view: ViewTransform) -> None:
    """Draw the football at its current position."""
    pygame.draw.circle(surface, BALL_COLOR, _screen_int(view, state.ball.position), BALL_RADIUS)


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
