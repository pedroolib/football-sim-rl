"""Validated parameters for a deterministic simulation play."""

from dataclasses import dataclass, field

from qb_sim.engine.geometry import FieldBounds

MAX_RECEIVERS = 2
MAX_DEFENDERS = 3


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Field, timing, movement, passing, and route parameters."""

    field: FieldBounds = field(default_factory=FieldBounds)
    dt: float = 0.05
    max_play_duration: float = 6.0
    quarterback_start: tuple[float, float] = (20.0, 26.65)
    receiver_starts: tuple[tuple[float, float], ...] = ((20.0, 43.0), (20.0, 10.0))
    defender_starts: tuple[tuple[float, float], ...] = ((18.0, 43.0), (18.0, 10.0))
    rusher_starts: tuple[tuple[float, float], ...] = ()
    receiver_speed: float = 7.0
    defender_speed: float = 6.5
    rusher_speed: float = 6.5
    quarterback_speed: float = 6.5
    ball_speed: float = 25.0
    throw_min_range: float = 5.0
    throw_max_range: float = 45.0
    throw_min_drag: float = 2.0
    throw_max_drag: float = 45.0
    throw_base_sigma_degrees: float = 1.0
    throw_move_penalty: float = 0.75
    throw_pressure_penalty: float = 0.75
    throw_distance_penalty: float = 0.40
    throw_power_sigma: float = 0.015
    pressure_radius: float = 5.0
    catch_radius: float = 1.0
    catch_height: float = 3.0
    arc_ratio: float = 0.2
    max_arc_height: float = 6.0
    carrier_fatigue_per_second: float = 0.0
    carrier_min_speed_factor: float = 0.55
    qb_run_margin_yards: float = 1.0
    qb_snap_depth_yards: float = 0.0
    tackle_radius: float = 1.0
    go_length_yards: float = 30.0
    slant_break_yards: float = 3.0
    slant_angle_degrees: float = 45.0
    post_break_yards: float = 10.0
    post_angle_degrees: float = 45.0
    cut_length_yards: float = 20.0
    out_break_yards: float = 5.0
    out_angle_degrees: float = 75.0
    curl_depth_yards: float = 10.0
    curl_back_yards: float = 3.0
    flat_length_yards: float = 6.0
    wheel_break_yards: float = 4.0
    wheel_length_yards: float = 20.0
    corner_break_yards: float = 12.0
    corner_angle_degrees: float = 45.0

    def __post_init__(self) -> None:
        positive_values = {
            "dt": self.dt,
            "max_play_duration": self.max_play_duration,
            "receiver_speed": self.receiver_speed,
            "defender_speed": self.defender_speed,
            "rusher_speed": self.rusher_speed,
            "quarterback_speed": self.quarterback_speed,
            "ball_speed": self.ball_speed,
            "throw_min_range": self.throw_min_range,
            "throw_max_range": self.throw_max_range,
            "throw_min_drag": self.throw_min_drag,
            "throw_max_drag": self.throw_max_drag,
            "go_length_yards": self.go_length_yards,
            "slant_break_yards": self.slant_break_yards,
            "post_break_yards": self.post_break_yards,
            "cut_length_yards": self.cut_length_yards,
            "out_break_yards": self.out_break_yards,
            "curl_depth_yards": self.curl_depth_yards,
            "curl_back_yards": self.curl_back_yards,
            "flat_length_yards": self.flat_length_yards,
            "wheel_break_yards": self.wheel_break_yards,
            "wheel_length_yards": self.wheel_length_yards,
            "corner_break_yards": self.corner_break_yards,
        }
        for name, value in positive_values.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.catch_radius < 0.0:
            raise ValueError("catch_radius must be non-negative")
        if self.catch_height < 0.0:
            raise ValueError("catch_height must be non-negative")
        if self.arc_ratio < 0.0:
            raise ValueError("arc_ratio must be non-negative")
        if self.max_arc_height <= 0.0:
            raise ValueError("max_arc_height must be positive")
        if self.tackle_radius < 0.0:
            raise ValueError("tackle_radius must be non-negative")
        if not 0.0 <= self.slant_angle_degrees <= 90.0:
            raise ValueError("slant_angle_degrees must be between 0 and 90")
        if not 0.0 <= self.post_angle_degrees <= 90.0:
            raise ValueError("post_angle_degrees must be between 0 and 90")
        if not 0.0 <= self.out_angle_degrees <= 90.0:
            raise ValueError("out_angle_degrees must be between 0 and 90")
        if not 0.0 <= self.corner_angle_degrees <= 90.0:
            raise ValueError("corner_angle_degrees must be between 0 and 90")
        if self.throw_max_range <= self.throw_min_range:
            raise ValueError("throw_max_range must exceed throw_min_range")
        if self.throw_max_drag <= self.throw_min_drag:
            raise ValueError("throw_max_drag must exceed throw_min_drag")
        non_negative_spread = {
            "throw_base_sigma_degrees": self.throw_base_sigma_degrees,
            "throw_move_penalty": self.throw_move_penalty,
            "throw_pressure_penalty": self.throw_pressure_penalty,
            "throw_distance_penalty": self.throw_distance_penalty,
            "throw_power_sigma": self.throw_power_sigma,
            "carrier_fatigue_per_second": self.carrier_fatigue_per_second,
        }
        for name, value in non_negative_spread.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.carrier_min_speed_factor <= 1.0:
            raise ValueError("carrier_min_speed_factor must be between 0 and 1")
        if self.qb_run_margin_yards < 0.0:
            raise ValueError("qb_run_margin_yards must be non-negative")
        if self.qb_snap_depth_yards < 0.0:
            raise ValueError("qb_snap_depth_yards must be non-negative")
        if self.pressure_radius <= 0.0:
            raise ValueError("pressure_radius must be positive")

        if len(self.receiver_starts) == 0 or len(self.defender_starts) == 0:
            raise ValueError("receiver_starts and defender_starts must not be empty")
        if len(self.receiver_starts) > MAX_RECEIVERS:
            raise ValueError(f"at most {MAX_RECEIVERS} receivers are supported")
        if len(self.defender_starts) + len(self.rusher_starts) > MAX_DEFENDERS:
            raise ValueError(f"at most {MAX_DEFENDERS} defenders (including rushers) are supported")

        start_groups = (
            ("quarterback_start", (self.quarterback_start,)),
            ("receiver_starts", self.receiver_starts),
            ("defender_starts", self.defender_starts),
            ("rusher_starts", self.rusher_starts),
        )
        for group_name, starts in start_groups:
            for position in starts:
                if len(position) != 2 or not self.field.contains(position):
                    raise ValueError(
                        f"{group_name} must contain two-dimensional points inside the field"
                    )
