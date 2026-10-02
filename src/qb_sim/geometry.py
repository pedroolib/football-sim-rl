"""Small 2D geometry helpers used by the simulation core."""

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import NDArray

Vector2 = NDArray[np.float64]


def vec2(x: float, y: float) -> Vector2:
    """Create an independent two-component vector."""
    return np.array((x, y), dtype=np.float64)


def distance(a: Vector2, b: Vector2) -> float:
    """Return the Euclidean distance between two points."""
    return float(np.linalg.norm(a - b))


def segment_circle_contact_fraction(
    start: Vector2,
    end: Vector2,
    center: Vector2,
    radius: float,
) -> float | None:
    """Return the first segment fraction that touches a circle, if any."""
    if radius < 0.0:
        raise ValueError("radius must be non-negative")

    offset = start - center
    radius_squared = radius * radius
    if float(np.dot(offset, offset)) <= radius_squared:
        return 0.0

    direction = end - start
    quadratic = float(np.dot(direction, direction))
    if quadratic == 0.0:
        return None

    linear = 2.0 * float(np.dot(offset, direction))
    constant = float(np.dot(offset, offset)) - radius_squared
    discriminant = linear * linear - 4.0 * quadratic * constant
    if discriminant < 0.0:
        return None

    first_root = (-linear - sqrt(discriminant)) / (2.0 * quadratic)
    if 0.0 <= first_root <= 1.0:
        return first_root
    return None


@dataclass(frozen=True, slots=True)
class FieldBounds:
    """Axis-aligned field boundaries, expressed in yards."""

    length_yards: float = 120.0
    width_yards: float = 53.3

    def __post_init__(self) -> None:
        if self.length_yards <= 0.0 or self.width_yards <= 0.0:
            raise ValueError("field dimensions must be positive")

    def contains(self, position: Vector2) -> bool:
        """Return whether a point lies on or inside the field boundaries."""
        return bool(
            0.0 <= position[0] <= self.length_yards and 0.0 <= position[1] <= self.width_yards
        )
