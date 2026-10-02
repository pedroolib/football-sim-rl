"""Validated parameters for a deterministic simulation play."""

from dataclasses import dataclass, field

from qb_sim.engine.geometry import FieldBounds


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Field, timing, movement, passing, and route parameters."""

    field: FieldBounds = field(default_factory=FieldBounds)
    dt: float = 0.05
    max_play_duration: float = 6.0
    quarterback_start: tuple[float, float] = (20.0, 26.65)
    receiver_start: tuple[float, float] = (20.0, 43.0)
    defender_start: tuple[float, float] = (18.0, 43.0)
    receiver_speed: float = 7.0
    defender_speed: float = 6.5
    quarterback_speed: float = 6.5
    ball_speed: float = 25.0
    catch_radius: float = 1.0
    tackle_radius: float = 1.0
    go_length_yards: float = 30.0
    slant_break_yards: float = 3.0
    slant_angle_degrees: float = 45.0
    post_break_yards: float = 10.0
    post_angle_degrees: float = 45.0
    cut_length_yards: float = 20.0

    def __post_init__(self) -> None:
        positive_values = {
            "dt": self.dt,
            "max_play_duration": self.max_play_duration,
            "receiver_speed": self.receiver_speed,
            "defender_speed": self.defender_speed,
            "quarterback_speed": self.quarterback_speed,
            "ball_speed": self.ball_speed,
            "go_length_yards": self.go_length_yards,
            "slant_break_yards": self.slant_break_yards,
            "post_break_yards": self.post_break_yards,
            "cut_length_yards": self.cut_length_yards,
        }
        for name, value in positive_values.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.catch_radius < 0.0:
            raise ValueError("catch_radius must be non-negative")
        if self.tackle_radius < 0.0:
            raise ValueError("tackle_radius must be non-negative")
        if not 0.0 <= self.slant_angle_degrees <= 90.0:
            raise ValueError("slant_angle_degrees must be between 0 and 90")
        if not 0.0 <= self.post_angle_degrees <= 90.0:
            raise ValueError("post_angle_degrees must be between 0 and 90")

        for name, position in (
            ("quarterback_start", self.quarterback_start),
            ("receiver_start", self.receiver_start),
            ("defender_start", self.defender_start),
        ):
            if len(position) != 2 or not self.field.contains(position):
                raise ValueError(f"{name} must be a two-dimensional point inside the field")
