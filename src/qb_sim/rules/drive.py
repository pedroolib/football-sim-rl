"""Down-and-distance drive rules layered on top of single plays."""

from dataclasses import dataclass

import numpy as np

from qb_sim.config import SimulationConfig
from qb_sim.engine.entities import PlayOutcome
from qb_sim.rules.play import Play
from qb_sim.rules.routes import DEFAULT_ROUTE_POOL, RouteName, sample_routes


@dataclass(frozen=True, slots=True)
class DriveConfig:
    """Downs, field-position range, and arcade scoring for a drive."""

    downs: int = 3
    first_down_yards: float = 10.0
    los_range: tuple[float, float] = (25.0, 50.0)
    max_plays: int = 20
    touchdown_points: float = 7.0
    goal_line: float = 100.0

    def __post_init__(self) -> None:
        if self.downs < 1:
            raise ValueError("downs must be at least 1")
        if self.first_down_yards <= 0.0:
            raise ValueError("first_down_yards must be positive")
        low, high = self.los_range
        if not 0.0 <= low <= high < self.goal_line:
            raise ValueError("los_range must be inside the field")
        if self.max_plays < 1:
            raise ValueError("max_plays must be at least 1")
        if self.touchdown_points <= 0.0:
            raise ValueError("touchdown_points must be positive")


@dataclass(slots=True)
class DriveState:
    """Mutable down, distance, and field-position state for one drive."""

    los: float
    down: int = 1
    distance: float = 10.0
    play_number: int = 0
    first_downs: int = 0
    points: float = 0.0
    terminated: bool = False
    end_reason: str | None = None


@dataclass(frozen=True, slots=True)
class PlayResult:
    """Effective result of one play, already interpreted by the drive rules."""

    outcome: PlayOutcome
    spot: float
    yards: float
    first_down: bool
    touchdown: bool
    drive_over: bool
    end_reason: str | None
    points: float
    throw_distance: float | None
    coverage_sack: bool = False


class Drive:
    """Deterministic 3-down drive: LOS, routes, and play outcomes."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
        drive_config: DriveConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.config = config if config is not None else SimulationConfig()
        self.drive_config = drive_config if drive_config is not None else DriveConfig()
        starts = (
            (self.config.quarterback_start,)
            + tuple(self.config.receiver_starts)
            + tuple(self.config.defender_starts)
            + tuple(self.config.rusher_starts)
        )
        self._min_los = max(0.0, self.config.quarterback_start[0] - min(s[0] for s in starts))
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._routes: tuple[RouteName, ...] = ()
        self.state: DriveState
        self.reset()

    @property
    def routes(self) -> tuple[RouteName, ...]:
        """Routes sampled for the current play."""
        return self._routes

    def reset(self, seed: int | None = None) -> DriveState:
        """Start a new drive, sampling a fresh line of scrimmage."""
        if seed is not None:
            self.seed = int(seed)
            self._rng = np.random.default_rng(self.seed)
        low, high = self.drive_config.los_range
        los = float(self._rng.uniform(low, high))
        los = max(los, self._min_los)
        distance = min(self.drive_config.first_down_yards, self.drive_config.goal_line - los)
        self.state = DriveState(los=los, distance=distance)
        self._routes = ()
        return self.state

    def start_play(self) -> Play:
        """Sample routes and return the next mechanical play at the current LOS."""
        if self.state.terminated:
            raise ValueError("the drive is already over")
        self.state.play_number += 1
        self._routes = sample_routes(
            self._rng,
            len(self.config.receiver_starts),
            DEFAULT_ROUTE_POOL,
        )
        play_seed = int(self._rng.integers(0, 2**31 - 1))
        return Play(
            self.config,
            seed=play_seed,
            line_of_scrimmage=self.state.los,
            goal_line=self.drive_config.goal_line,
        )

    def advance(self, play: Play) -> PlayResult:
        """Apply a finished play to the drive state and return its result."""
        if play.state.outcome is PlayOutcome.IN_PROGRESS:
            raise ValueError("cannot advance a play that is still in progress")
        state = play.state
        old_los = self.state.los
        outcome = state.outcome
        coverage_sack = False
        if outcome is PlayOutcome.TIME_LIMIT:
            if state.throw_time is None and state.carrier_id is None:
                outcome = PlayOutcome.SACK
                coverage_sack = True
            else:
                outcome = PlayOutcome.INCOMPLETE
        spot = self._spot(state, outcome, old_los)
        spot = max(spot, self._min_los)
        yards = spot - old_los
        touchdown = outcome is PlayOutcome.TOUCHDOWN
        first_down = False
        points = 0.0
        end_reason: str | None = None

        if touchdown:
            points = self.drive_config.touchdown_points
            self.state.points += points
            self.state.terminated = True
            end_reason = "touchdown"
        elif outcome is PlayOutcome.INTERCEPTION:
            self.state.terminated = True
            end_reason = "interception"
        elif spot - old_los >= self.state.distance:
            first_down = True
            self.state.first_downs += 1
            self.state.down = 1
            self.state.los = spot
            self.state.distance = min(
                self.drive_config.first_down_yards,
                self.drive_config.goal_line - spot,
            )
        else:
            marker = old_los + self.state.distance
            self.state.los = spot
            self.state.down += 1
            self.state.distance = marker - spot
            if self.state.down > self.drive_config.downs:
                self.state.terminated = True
                end_reason = "turnover_on_downs"

        if not self.state.terminated and self.state.play_number >= self.drive_config.max_plays:
            self.state.terminated = True
            end_reason = "max_plays"

        self.state.end_reason = end_reason
        throw_distance = (
            float(state.ball.flight_distance) if state.throw_time is not None else None
        )
        return PlayResult(
            outcome=outcome,
            spot=spot,
            yards=yards,
            first_down=first_down,
            touchdown=touchdown,
            drive_over=self.state.terminated,
            end_reason=end_reason,
            points=points,
            throw_distance=throw_distance,
            coverage_sack=coverage_sack,
        )

    def _spot(self, state, outcome: PlayOutcome, los: float) -> float:
        goal_line = self.drive_config.goal_line
        if outcome is PlayOutcome.TOUCHDOWN:
            return goal_line
        if outcome is PlayOutcome.COMPLETED and state.carrier_id is not None:
            carrier_x = float(state.player(state.carrier_id).position[0])
            return float(min(max(carrier_x, 0.0), goal_line))
        if outcome is PlayOutcome.SACK:
            quarterback_x = float(state.player("qb").position[0])
            return float(min(max(quarterback_x, 0.0), goal_line))
        return float(los)
