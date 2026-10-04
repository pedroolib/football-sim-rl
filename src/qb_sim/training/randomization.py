"""Per-episode speed randomization within viable ranges (M7)."""

from dataclasses import dataclass, replace
from typing import Any

import gymnasium as gym
import numpy as np

from qb_sim.config import SimulationConfig
from qb_sim.rules.scenarios import Scenario
from qb_sim.training.env import QBPassingEnv


@dataclass(frozen=True, slots=True)
class SpeedRanges:
    """Sampling ranges kept inside viable play.

    Man coverage is suffocating when the corner is as fast as the receiver, so
    the defender speed is sampled as ``receiver_speed - gap`` with the gap
    bounded below; the rusher speed is independent.
    """

    receiver: tuple[float, float] = (6.5, 8.0)
    defender_gap: tuple[float, float] = (0.3, 1.0)
    rusher: tuple[float, float] = (4.5, 8.0)

    def __post_init__(self) -> None:
        for name, (low, high) in (
            ("receiver", self.receiver),
            ("defender_gap", self.defender_gap),
            ("rusher", self.rusher),
        ):
            if low <= 0.0 or high < low:
                raise ValueError(f"{name} range must be positive and ordered")
        if self.receiver[0] - self.defender_gap[1] <= 0.0:
            raise ValueError("defender speeds must stay positive")


def sample_config(
    base: SimulationConfig,
    rng: np.random.Generator,
    ranges: SpeedRanges,
) -> SimulationConfig:
    """Sample a configuration with randomized speeds inside the ranges."""
    receiver = float(rng.uniform(*ranges.receiver))
    gap = float(rng.uniform(*ranges.defender_gap))
    rusher = float(rng.uniform(*ranges.rusher))
    return replace(
        base,
        receiver_speed=receiver,
        defender_speed=receiver - gap,
        rusher_speed=rusher,
    )


@dataclass(frozen=True, slots=True)
class AdaptationRanges:
    """Full capability ranges; the curriculum interpolates from a narrow phase."""

    receiver_speed: tuple[float, float] = (6.0, 8.5)
    defender_speed: tuple[float, float] = (5.5, 7.5)
    quarterback_speed: tuple[float, float] = (5.0, 8.0)
    rusher_speed: tuple[float, float] = (5.0, 8.0)
    throw_max_range: tuple[float, float] = (30.0, 55.0)
    throw_base_sigma_degrees: tuple[float, float] = (0.5, 4.0)


_NARROW_SPEED_HALF_WIDTH = 0.5
_NARROW_THROW_MAX_RANGE = (40.0, 50.0)
_NARROW_THROW_SIGMA_DEGREES = (1.0, 2.0)


def _narrow_speed_range(full: tuple[float, float]) -> tuple[float, float]:
    center = (full[0] + full[1]) / 2.0
    return (center - _NARROW_SPEED_HALF_WIDTH, center + _NARROW_SPEED_HALF_WIDTH)


def _interpolated_range(
    narrow: tuple[float, float],
    full: tuple[float, float],
    progress: float,
) -> tuple[float, float]:
    return (
        narrow[0] + (full[0] - narrow[0]) * progress,
        narrow[1] + (full[1] - narrow[1]) * progress,
    )


def sample_adaptation_config(
    base: SimulationConfig,
    rng: np.random.Generator,
    progress: float,
    ranges: AdaptationRanges | None = None,
) -> SimulationConfig:
    """Sample capability fields, widening from the narrow phase to full by progress.

    Only speed, arm, and accuracy-sigma fields are replaced; the roster and
    fatigue settings of ``base`` are preserved.
    """
    chosen = ranges if ranges is not None else AdaptationRanges()
    phase = min(max(float(progress), 0.0), 1.0)
    receiver_speed = float(
        rng.uniform(
            *_interpolated_range(
                _narrow_speed_range(chosen.receiver_speed), chosen.receiver_speed, phase
            )
        )
    )
    defender_speed = float(
        rng.uniform(
            *_interpolated_range(
                _narrow_speed_range(chosen.defender_speed), chosen.defender_speed, phase
            )
        )
    )
    quarterback_speed = float(
        rng.uniform(
            *_interpolated_range(
                _narrow_speed_range(chosen.quarterback_speed),
                chosen.quarterback_speed,
                phase,
            )
        )
    )
    rusher_speed = float(
        rng.uniform(
            *_interpolated_range(
                _narrow_speed_range(chosen.rusher_speed), chosen.rusher_speed, phase
            )
        )
    )
    throw_max_range = float(
        rng.uniform(
            *_interpolated_range(
                _NARROW_THROW_MAX_RANGE, chosen.throw_max_range, phase
            )
        )
    )
    throw_base_sigma_degrees = float(
        rng.uniform(
            *_interpolated_range(
                _NARROW_THROW_SIGMA_DEGREES, chosen.throw_base_sigma_degrees, phase
            )
        )
    )
    return replace(
        base,
        receiver_speed=receiver_speed,
        defender_speed=defender_speed,
        quarterback_speed=quarterback_speed,
        rusher_speed=rusher_speed,
        throw_max_range=throw_max_range,
        throw_base_sigma_degrees=throw_base_sigma_degrees,
    )


class RandomizedEnv(gym.Env):
    """QBPassingEnv with fresh speed samples on every reset."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: Scenario,
        ranges: SpeedRanges | None = None,
        max_steps: int = 240,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.ranges = ranges if ranges is not None else SpeedRanges()
        self.max_steps = int(max_steps)
        self._attempt_scale = 1.0
        self._env = QBPassingEnv(scenario, max_steps=self.max_steps)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    @property
    def config(self) -> SimulationConfig:
        """Configuration of the current episode."""
        return self._env.play.config

    @property
    def play(self):
        """Current play, for evaluation helpers."""
        return self._env.play

    def set_attempt_scale(self, scale: float) -> None:
        """Persist and forward the annealing scale to the current episode."""
        self._attempt_scale = max(0.0, float(scale))
        self._env.set_attempt_scale(self._attempt_scale)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        rng = np.random.default_rng(seed)
        config = sample_config(self.scenario.config, rng, self.ranges)
        randomized = Scenario(self.scenario.name, config, self.scenario.receiver_routes)
        self._env = QBPassingEnv(randomized, max_steps=self.max_steps)
        self._env.set_attempt_scale(self._attempt_scale)
        return self._env.reset(seed=seed, options=options)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self._env.step(action)

    def render(self) -> None:
        """No rendering for training environments."""
        return None

    def close(self) -> None:
        self._env.close()
