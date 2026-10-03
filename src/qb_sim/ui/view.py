"""Field-to-screen coordinate transforms for the visual layer."""

from dataclasses import dataclass

import numpy as np

from qb_sim.engine.geometry import FieldBounds, Vector2, vec2


@dataclass(frozen=True, slots=True)
class ViewTransform:
    """Uniform-scale mapping between field yards and screen pixels.

    Without a camera (``visible_width_yards=None``) the whole field fits inside
    the margins. With a camera, the view shows a ``visible_width_yards`` band
    centered on ``center`` and the length implied by the screen aspect, which
    gives the broadcast-style follow view.
    """

    field: FieldBounds
    screen_width: int
    screen_height: int
    margin: int = 20
    center: tuple[float, float] | None = None
    visible_width_yards: float | None = None

    def __post_init__(self) -> None:
        if self.screen_width <= 2 * self.margin or self.screen_height <= 2 * self.margin:
            raise ValueError("screen must be larger than twice the margin")
        if self.margin < 0:
            raise ValueError("margin must be non-negative")
        if self.visible_width_yards is not None and self.visible_width_yards <= 0.0:
            raise ValueError("visible_width_yards must be positive")

    @property
    def scale(self) -> float:
        """Pixels per yard for the largest uniform fit inside the margins."""
        usable_width = self.screen_width - 2 * self.margin
        usable_height = self.screen_height - 2 * self.margin
        if self.visible_width_yards is not None:
            return usable_height / self.visible_width_yards
        return min(
            usable_width / self.field.length_yards,
            usable_height / self.field.width_yards,
        )

    @property
    def visible_length_yards(self) -> float:
        """Field length shown on screen, in yards."""
        if self.visible_width_yards is None:
            return self.field.length_yards
        usable_width = self.screen_width - 2 * self.margin
        return usable_width / self.scale

    def _origin(self) -> tuple[float, float]:
        if self.visible_width_yards is None:
            return 0.0, 0.0
        center_x, center_y = (
            self.center
            if self.center is not None
            else (self.field.length_yards / 2.0, self.field.width_yards / 2.0)
        )
        return (
            center_x - self.visible_length_yards / 2.0,
            center_y - self.visible_width_yards / 2.0,
        )

    def to_screen(self, point: Vector2) -> Vector2:
        """Map a field point in yards to screen pixels (y grows downward)."""
        origin_x, origin_y = self._origin()
        return vec2(
            self.margin + (point[0] - origin_x) * self.scale,
            self.screen_height - self.margin - (point[1] - origin_y) * self.scale,
        )

    def to_field(self, point: Vector2) -> Vector2:
        """Map a screen pixel back to field yards."""
        origin_x, origin_y = self._origin()
        return vec2(
            origin_x + (point[0] - self.margin) / self.scale,
            origin_y + (self.screen_height - self.margin - point[1]) / self.scale,
        )


def dash_segments(
    start: Vector2,
    end: Vector2,
    dash: float = 8.0,
    gap: float = 6.0,
) -> list[tuple[Vector2, Vector2]]:
    """Split a screen-space segment into dash sub-segments for a dotted line."""
    start_point = np.asarray(start, dtype=np.float64)
    end_point = np.asarray(end, dtype=np.float64)
    total_length = float(np.linalg.norm(end_point - start_point))
    if total_length == 0.0 or dash <= 0.0 or gap < 0.0:
        return []

    direction = (end_point - start_point) / total_length
    segments: list[tuple[Vector2, Vector2]] = []
    distance = 0.0
    while distance < total_length:
        segment_end = min(distance + dash, total_length)
        segments.append(
            (
                start_point + direction * distance,
                start_point + direction * segment_end,
            )
        )
        distance += dash + gap
    return segments


def arc_screen_points(
    view: ViewTransform,
    start: Vector2,
    end: Vector2,
    peak_height: float,
    samples: int = 24,
) -> list[Vector2]:
    """Sample a pass arc as screen points, lifting each ground point by height."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    points: list[Vector2] = []
    for index in range(samples + 1):
        progress = index / samples
        ground = start + (end - start) * progress
        height = 4.0 * peak_height * progress * (1.0 - progress)
        screen = view.to_screen(ground)
        points.append(vec2(screen[0], screen[1] - height * view.scale))
    return points
