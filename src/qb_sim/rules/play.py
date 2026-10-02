"""Orchestration for one deterministic, fixed-step passing play."""

from math import isfinite
from typing import NamedTuple

import numpy as np

from qb_sim.config import SimulationConfig
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import (
    BallState,
    BallStatus,
    PlayerState,
    PlayOutcome,
    PlayState,
    Role,
)
from qb_sim.engine.geometry import Vector2, distance, rotate, vec2
from qb_sim.engine.physics import (
    advance_ball,
    arc_peak,
    ball_height_at,
    move_toward,
    moving_circle_contact_fraction,
    segment_field_exit_fraction,
)
from qb_sim.rules.accuracy import pressure_from_defenders, throw_spread
from qb_sim.rules.routes import Route, RouteName, build_route

_QB_ID = "qb"


class _PassEvent(NamedTuple):
    tick_fraction: float
    priority: int
    order: int
    outcome: PlayOutcome
    segment_fraction: float
    player_id: str | None


class Play:
    """Advance a human or scripted offense against a basic man defense."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
        routes: tuple[RouteName, ...] = (RouteName.POST, RouteName.SLANT),
        seed: int = 0,
    ) -> None:
        self.config = config if config is not None else SimulationConfig()
        self.routes = tuple(RouteName(route) for route in routes)
        if len(self.routes) != len(self.config.receiver_starts):
            raise ValueError("a route is required for each receiver")
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._player_routes: list[Route] = []
        self._receiver_ids: list[str] = []
        self._defender_ids: list[str] = []
        self._coverage: list[str] = []
        self.state: PlayState
        self.reset()

    def reset(self, seed: int | None = None) -> PlayState:
        """Restore the roster and routes; re-seed only when ``seed`` is given."""
        if seed is not None:
            self.seed = int(seed)
            self._rng = np.random.default_rng(self.seed)
        quarterback_position = vec2(*self.config.quarterback_start)
        players = [
            PlayerState(quarterback_position.copy(), vec2(0.0, 0.0), Role.QUARTERBACK, _QB_ID)
        ]
        receiver_positions: list[Vector2] = []
        for index, start in enumerate(self.config.receiver_starts, start=1):
            position = vec2(*start)
            receiver_positions.append(position)
            players.append(PlayerState(position, vec2(0.0, 0.0), Role.RECEIVER, f"wr{index}"))
        for index, start in enumerate(self.config.defender_starts, start=1):
            players.append(PlayerState(vec2(*start), vec2(0.0, 0.0), Role.DEFENDER, f"cb{index}"))

        self.state = PlayState(
            players=players,
            ball=BallState(position=quarterback_position.copy(), velocity=vec2(0.0, 0.0)),
        )
        self._receiver_ids = [f"wr{index}" for index in range(1, len(receiver_positions) + 1)]
        self._defender_ids = [
            f"cb{index}" for index in range(1, len(self.config.defender_starts) + 1)
        ]
        self._coverage = [
            self._receiver_ids[index % len(self._receiver_ids)]
            for index in range(len(self._defender_ids))
        ]
        self._player_routes = [
            build_route(route, position, self.config.field, self.config)
            for route, position in zip(self.routes, receiver_positions, strict=True)
        ]
        self.state.pressure = self.pressure_on_quarterback()
        return self.state

    def pressure_on_quarterback(self) -> float:
        """Return 0..1 pressure from the nearest defender within ``pressure_radius``."""
        defender_positions = [
            self.state.player(defender_id).position for defender_id in self._defender_ids
        ]
        return pressure_from_defenders(
            self.state.player(_QB_ID).position,
            defender_positions,
            self.config.pressure_radius,
        )

    def throw_to(self, target: Vector2, power: float | None = None) -> None:
        """Launch the ball toward a target point, deriving power from distance."""
        target_point = np.asarray(target, dtype=np.float64)
        quarterback = self.state.player(_QB_ID)
        offset = target_point - quarterback.position
        target_distance = float(np.linalg.norm(offset))
        if target_distance == 0.0:
            raise ValueError("target must differ from the quarterback")
        if power is None:
            span = self.config.throw_max_range - self.config.throw_min_range
            power = (target_distance - self.config.throw_min_range) / span
            power = min(max(power, 0.0), 1.0)
        self._launch_throw(offset / target_distance, power)

    @property
    def receiver_ids(self) -> tuple[str, ...]:
        """Stable receiver identifiers in roster order."""
        return tuple(self._receiver_ids)

    @property
    def receiver_routes(self) -> tuple[Route, ...]:
        """Route objects in the same order as :attr:`receiver_ids`."""
        return tuple(self._player_routes)

    def throw_with_lead(
        self,
        lead_seconds: float,
        player_id: str = "wr1",
        power: float | None = None,
    ) -> None:
        """Launch the ball toward a receiver's projected lead position."""
        if not isfinite(lead_seconds) or lead_seconds < 0.0:
            raise ValueError("lead_seconds must be finite and non-negative")
        receiver = self.state.player(player_id)
        target = receiver.position + receiver.velocity * lead_seconds
        self.throw_to(target, power)

    def _launch_throw(self, direction: Vector2, power: float) -> None:
        if self.state.outcome is not PlayOutcome.IN_PROGRESS:
            raise ValueError("cannot throw after the play has ended")
        if self.state.ball.status is not BallStatus.HELD:
            raise ValueError("the ball has already been thrown")

        action = Throw(direction=direction, power=power)
        quarterback = self.state.player(_QB_ID)
        range_yards = self.config.throw_min_range + action.power * (
            self.config.throw_max_range - self.config.throw_min_range
        )
        angle_sigma, power_sigma = throw_spread(
            float(np.linalg.norm(quarterback.velocity)),
            self.pressure_on_quarterback(),
            range_yards,
            self.config,
        )
        angle_error = float(self._rng.normal(0.0, angle_sigma))
        power_error = float(self._rng.normal(0.0, power_sigma))
        actual_direction = rotate(action.direction, angle_error)
        actual_power = min(max(action.power + power_error, 0.0), 1.0)
        actual_range = self.config.throw_min_range + actual_power * (
            self.config.throw_max_range - self.config.throw_min_range
        )
        target = quarterback.position + actual_direction * actual_range

        self.state.ball.position = quarterback.position.copy()
        self.state.ball.velocity = actual_direction * self.config.ball_speed
        self.state.ball.target = np.array(target, dtype=np.float64, copy=True)
        self.state.ball.launch_position = quarterback.position.copy()
        self.state.ball.flight_distance = actual_range
        self.state.ball.arc_peak = arc_peak(
            actual_range,
            self.config.arc_ratio,
            self.config.max_arc_height,
        )
        self.state.ball.height = 0.0
        self.state.ball.status = BallStatus.IN_FLIGHT

    def step(self, qb_action: Move | Throw | None = None) -> PlayState:
        """Advance one fixed time step and resolve any terminal play event."""
        if self.state.outcome is not PlayOutcome.IN_PROGRESS:
            return self.state

        remaining_time = self.config.max_play_duration - self.state.elapsed_seconds
        if remaining_time <= 0.0:
            self._finish_expired_play()
            return self.state
        dt = min(self.config.dt, remaining_time)

        if qb_action is not None:
            self._apply_quarterback_action(qb_action, dt)

        previous_positions = {
            player.player_id: player.position.copy() for player in self.state.players
        }
        tick_fraction = 1.0
        if self.state.carrier_id is not None:
            self._advance_carrier(dt)
            self._advance_defenders(dt)
            self._resolve_carrier_events()
        else:
            self._advance_receivers(dt)
            self._advance_defenders(dt)
            if self.state.ball.status is BallStatus.IN_FLIGHT:
                tick_fraction = self._advance_pass(dt, previous_positions)

        self.state.elapsed_seconds += dt * tick_fraction
        self.state.pressure = self.pressure_on_quarterback()
        if (
            self.state.outcome is PlayOutcome.IN_PROGRESS
            and self.state.elapsed_seconds >= self.config.max_play_duration
        ):
            self._finish_expired_play()
        return self.state

    def _apply_quarterback_action(self, action: Move | Throw, dt: float) -> None:
        if isinstance(action, Throw):
            self._launch_throw(action.direction, action.power)
            return
        if isinstance(action, Move):
            quarterback = self.state.player(_QB_ID)
            previous = quarterback.position.copy()
            new_position = self.config.field.clamp(
                previous + action.direction * (self.config.quarterback_speed * dt)
            )
            quarterback.position = new_position
            quarterback.velocity = (new_position - previous) / dt
            return
        raise TypeError(f"unsupported quarterback action: {type(action).__name__}")

    def _advance_receivers(self, dt: float) -> None:
        for player_id, route in zip(self._receiver_ids, self._player_routes, strict=True):
            receiver = self.state.player(player_id)
            previous = receiver.position.copy()
            new_position, _ = route.advance(previous, max_distance=self.config.receiver_speed * dt)
            receiver.position = new_position
            receiver.velocity = (new_position - previous) / dt

    def _advance_defenders(self, dt: float) -> None:
        for player_id, coverage_id in zip(self._defender_ids, self._coverage, strict=True):
            defender = self.state.player(player_id)
            target_id = self.state.carrier_id if self.state.carrier_id is not None else coverage_id
            previous = defender.position.copy()
            new_position = move_toward(
                previous,
                self.state.player(target_id).position,
                max_distance=self.config.defender_speed * dt,
            )
            defender.position = new_position
            defender.velocity = (new_position - previous) / dt

    def _advance_carrier(self, dt: float) -> None:
        carrier = self.state.player(self.state.carrier_id)
        previous = carrier.position.copy()
        new_position = previous + vec2(1.0, 0.0) * (self.config.receiver_speed * dt)
        carrier.position = new_position
        carrier.velocity = (new_position - previous) / dt
        self.state.ball.position = new_position.copy()

    def _resolve_carrier_events(self) -> None:
        carrier = self.state.player(self.state.carrier_id)
        for defender_id in self._defender_ids:
            defender = self.state.player(defender_id)
            if distance(defender.position, carrier.position) <= self.config.tackle_radius:
                self._finish(PlayOutcome.COMPLETED)
                return
        if not self.config.field.contains(carrier.position):
            self._finish(PlayOutcome.COMPLETED)

    def _advance_pass(self, dt: float, previous_positions: dict[str, Vector2]) -> float:
        """Advance the ball and return the fraction of ``dt`` the play consumed."""
        ball_start = self.state.ball.position.copy()
        ball_end, ball_travel_fraction = advance_ball(self.state.ball, dt)
        events: list[_PassEvent] = []

        for order, player_id in enumerate(self._receiver_ids):
            contact = self._contact_fraction(
                player_id, previous_positions, ball_start, ball_end, ball_travel_fraction
            )
            if contact is not None:
                events.append(
                    _PassEvent(
                        contact * ball_travel_fraction,
                        1,
                        order,
                        PlayOutcome.COMPLETED,
                        contact,
                        player_id,
                    )
                )

        for order, player_id in enumerate(self._defender_ids):
            contact = self._contact_fraction(
                player_id, previous_positions, ball_start, ball_end, ball_travel_fraction
            )
            if contact is not None:
                events.append(
                    _PassEvent(
                        contact * ball_travel_fraction,
                        0,
                        order,
                        PlayOutcome.INTERCEPTION,
                        contact,
                        player_id,
                    )
                )

        field_exit = segment_field_exit_fraction(ball_start, ball_end, self.config.field)
        if field_exit is not None:
            events.append(
                _PassEvent(
                    field_exit * ball_travel_fraction,
                    2,
                    0,
                    PlayOutcome.OUT_OF_BOUNDS,
                    field_exit,
                    None,
                )
            )

        target = self.state.ball.target
        if target is not None and np.array_equal(ball_end, target):
            events.append(_PassEvent(ball_travel_fraction, 3, 0, PlayOutcome.INCOMPLETE, 1.0, None))

        if not events:
            self.state.ball.position = ball_end
            self.state.ball.height = ball_height_at(self.state.ball, ball_end)
            return 1.0

        event = min(events, key=lambda item: (item.tick_fraction, item.priority, item.order))
        self.state.ball.position = ball_start + (ball_end - ball_start) * event.segment_fraction
        self.state.ball.height = ball_height_at(self.state.ball, self.state.ball.position)
        for player in self.state.players:
            previous = previous_positions[player.player_id]
            player.position = previous + (player.position - previous) * event.tick_fraction
        if event.outcome is PlayOutcome.COMPLETED:
            self._complete_catch(event.player_id)
        else:
            self._finish(event.outcome)
        return event.tick_fraction

    def _contact_fraction(
        self,
        player_id: str,
        previous_positions: dict[str, Vector2],
        ball_start: Vector2,
        ball_end: Vector2,
        ball_travel_fraction: float,
    ) -> float | None:
        previous = previous_positions[player_id]
        current = self.state.player(player_id).position
        window_end = previous + (current - previous) * ball_travel_fraction
        fraction = moving_circle_contact_fraction(
            ball_start,
            ball_end,
            previous,
            window_end,
            self.config.catch_radius,
        )
        if fraction is None:
            return None
        contact_point = ball_start + (ball_end - ball_start) * fraction
        if ball_height_at(self.state.ball, contact_point) > self.config.catch_height:
            return None
        return fraction

    def _complete_catch(self, player_id: str | None) -> None:
        if player_id is None:
            raise ValueError("a catch requires a receiver")
        self.state.carrier_id = player_id
        self.state.ball.status = BallStatus.CARRIED
        self.state.ball.velocity = vec2(0.0, 0.0)
        self.state.ball.target = None
        self.state.ball.height = 0.0
        self.state.ball.position = self.state.player(player_id).position.copy()

    def _finish_expired_play(self) -> None:
        if self.state.carrier_id is not None:
            self._finish(PlayOutcome.COMPLETED)
        else:
            self._finish(PlayOutcome.TIME_LIMIT)

    def _finish(self, outcome: PlayOutcome) -> None:
        self.state.outcome = outcome
        self.state.ball.status = BallStatus.DEAD
        self.state.ball.height = 0.0
