"""Mechanical play rules: apply per-player actions and resolve physics events."""

from collections.abc import Mapping
from math import isfinite
from typing import NamedTuple

import numpy as np

from qb_sim.config import SimulationConfig
from qb_sim.engine.actions import Move, Throw, allowed_actions
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
    moving_circle_contact_fraction,
    segment_field_exit_fraction,
)
from qb_sim.rules.accuracy import pressure_from_defenders, throw_spread

_QB_ID = "qb"


class _PassEvent(NamedTuple):
    tick_fraction: float
    priority: int
    order: int
    outcome: PlayOutcome
    segment_fraction: float
    player_id: str | None


class Play:
    """Apply per-player actions and resolve the rules of one passing play."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.config = config if config is not None else SimulationConfig()
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._receiver_ids: list[str] = []
        self._defender_ids: list[str] = []
        self.state: PlayState
        self.reset()

    def reset(self, seed: int | None = None) -> PlayState:
        """Restore the roster; re-seed the generator only when ``seed`` is given."""
        if seed is not None:
            self.seed = int(seed)
            self._rng = np.random.default_rng(self.seed)
        quarterback_position = vec2(*self.config.quarterback_start)
        players = [
            PlayerState(quarterback_position.copy(), vec2(0.0, 0.0), Role.QUARTERBACK, _QB_ID)
        ]
        for index, start in enumerate(self.config.receiver_starts, start=1):
            players.append(PlayerState(vec2(*start), vec2(0.0, 0.0), Role.RECEIVER, f"wr{index}"))
        for index, start in enumerate(self.config.defender_starts, start=1):
            players.append(PlayerState(vec2(*start), vec2(0.0, 0.0), Role.DEFENDER, f"cb{index}"))

        self.state = PlayState(
            players=players,
            ball=BallState(position=quarterback_position.copy(), velocity=vec2(0.0, 0.0)),
        )
        self._receiver_ids = [
            f"wr{index}" for index in range(1, len(self.config.receiver_starts) + 1)
        ]
        self._defender_ids = [
            f"cb{index}" for index in range(1, len(self.config.defender_starts) + 1)
        ]
        self.state.pressure = self.pressure_on_quarterback()
        return self.state

    @property
    def receiver_ids(self) -> tuple[str, ...]:
        """Stable receiver identifiers in roster order."""
        return tuple(self._receiver_ids)

    @property
    def defender_ids(self) -> tuple[str, ...]:
        """Stable defender identifiers in roster order."""
        return tuple(self._defender_ids)

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
        self.state.throw_time = self.state.elapsed_seconds

    def step(self, actions: Mapping[str, Move | Throw] | None = None) -> PlayState:
        """Apply one tick of per-player actions and resolve terminal events."""
        if self.state.outcome is not PlayOutcome.IN_PROGRESS:
            return self.state

        remaining_time = self.config.max_play_duration - self.state.elapsed_seconds
        if remaining_time <= 0.0:
            self._finish_expired_play()
            return self.state
        dt = min(self.config.dt, remaining_time)

        previous_positions = {
            player.player_id: player.position.copy() for player in self.state.players
        }
        self._apply_actions(actions or {}, dt)

        tick_fraction = 1.0
        if self.state.carrier_id is not None:
            carrier = self.state.player(self.state.carrier_id)
            self.state.ball.position = carrier.position.copy()
            self.state.ball.height = 0.0
            self._resolve_carrier_events()
        elif self.state.ball.status is BallStatus.IN_FLIGHT:
            tick_fraction = self._advance_pass(dt, previous_positions)

        self.state.elapsed_seconds += dt * tick_fraction
        self.state.pressure = self.pressure_on_quarterback()
        if (
            self.state.outcome is PlayOutcome.IN_PROGRESS
            and self.state.elapsed_seconds >= self.config.max_play_duration
        ):
            self._finish_expired_play()
        return self.state

    def _apply_actions(self, actions: Mapping[str, Move | Throw], dt: float) -> None:
        known_ids = {player.player_id for player in self.state.players}
        unknown = sorted(set(actions) - known_ids)
        if unknown:
            raise ValueError(f"unknown players: {unknown}")
        for player in self.state.players:
            action = actions.get(player.player_id)
            if action is not None:
                self._apply_action(player, action, dt)

    def _apply_action(self, player: PlayerState, action: Move | Throw, dt: float) -> None:
        if not isinstance(action, allowed_actions(player.role)):
            raise ValueError(f"{player.player_id} cannot use {type(action).__name__}")
        if isinstance(action, Throw):
            self._launch_throw(action.direction, action.power)
            return
        speed = self._speed_for(player.role)
        previous = player.position.copy()
        new_position = previous + action.direction * (speed * dt)
        if player.role is Role.QUARTERBACK:
            new_position = self.config.field.clamp(new_position)
        player.position = new_position
        player.velocity = (new_position - previous) / dt

    def _speed_for(self, role: Role) -> float:
        if role is Role.QUARTERBACK:
            return self.config.quarterback_speed
        if role is Role.RECEIVER:
            return self.config.receiver_speed
        if role is Role.DEFENDER:
            return self.config.defender_speed
        raise ValueError(f"unsupported role: {role!r}")

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
