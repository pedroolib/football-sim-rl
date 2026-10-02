"""Orchestration for one deterministic, fixed-step passing play."""

from math import isfinite

import numpy as np

from qb_sim.config import SimulationConfig
from qb_sim.entities import BallState, BallStatus, PlayerState, PlayOutcome, PlayState
from qb_sim.geometry import Vector2, vec2
from qb_sim.physics import (
    advance_ball,
    move_toward,
    moving_circle_contact_fraction,
    segment_field_exit_fraction,
)
from qb_sim.routes import Route, RouteName, build_route


class Simulation:
    """Advance a quarterback, receiver, defender, and football through one play."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
        route: RouteName = RouteName.GO,
    ) -> None:
        self.config = config if config is not None else SimulationConfig()
        self.route_name = RouteName(route)
        self._route: Route
        self.state: PlayState
        self.reset()

    def reset(self) -> PlayState:
        """Restore the configured starting positions and route progress."""
        quarterback_position = vec2(*self.config.quarterback_start)
        receiver_position = vec2(*self.config.receiver_start)
        defender_position = vec2(*self.config.defender_start)
        self.state = PlayState(
            quarterback=PlayerState(quarterback_position, vec2(0.0, 0.0)),
            receiver=PlayerState(receiver_position, vec2(0.0, 0.0)),
            defender=PlayerState(defender_position, vec2(0.0, 0.0)),
            ball=BallState(
                position=quarterback_position.copy(),
                velocity=vec2(0.0, 0.0),
            ),
        )
        self._route = build_route(
            self.route_name,
            receiver_position,
            self.config.field,
            self.config,
        )
        return self.state

    def throw(self, lead_seconds: float) -> None:
        """Launch the ball toward the receiver's lead position."""
        if not isfinite(lead_seconds) or lead_seconds < 0.0:
            raise ValueError("lead_seconds must be finite and non-negative")
        if self.state.outcome is not PlayOutcome.IN_PROGRESS:
            raise ValueError("cannot throw after the play has ended")
        if self.state.ball.status is not BallStatus.HELD:
            raise ValueError("the ball has already been thrown")

        target = self.state.receiver.position + self.state.receiver.velocity * lead_seconds
        offset = target - self.state.quarterback.position
        target_distance = float(np.linalg.norm(offset))
        if target_distance == 0.0:
            velocity = vec2(0.0, 0.0)
        else:
            velocity = offset * (self.config.ball_speed / target_distance)

        self.state.ball.position = self.state.quarterback.position.copy()
        self.state.ball.velocity = velocity
        self.state.ball.target = np.array(target, dtype=np.float64, copy=True)
        self.state.ball.status = BallStatus.IN_FLIGHT

    def step(self) -> PlayState:
        """Advance one fixed time step and resolve any terminal play event."""
        if self.state.outcome is not PlayOutcome.IN_PROGRESS:
            return self.state

        remaining_time = self.config.max_play_duration - self.state.elapsed_seconds
        if remaining_time <= 0.0:
            self._finish(PlayOutcome.TIME_LIMIT)
            return self.state
        dt = min(self.config.dt, remaining_time)

        previous_receiver_position = self.state.receiver.position.copy()
        previous_defender_position = self.state.defender.position.copy()
        self._advance_receiver(dt)
        self._advance_defender(dt)

        tick_fraction = 1.0
        if self.state.ball.status is BallStatus.IN_FLIGHT:
            tick_fraction = self._advance_pass(
                dt,
                previous_receiver_position,
                previous_defender_position,
            )

        self.state.elapsed_seconds += dt * tick_fraction
        if (
            self.state.outcome is PlayOutcome.IN_PROGRESS
            and self.state.elapsed_seconds >= self.config.max_play_duration
        ):
            self._finish(PlayOutcome.TIME_LIMIT)
        return self.state

    def _advance_receiver(self, dt: float) -> None:
        previous = self.state.receiver.position.copy()
        new_position, _ = self._route.advance(
            previous,
            max_distance=self.config.receiver_speed * dt,
        )
        self.state.receiver.position = new_position
        self.state.receiver.velocity = (new_position - previous) / dt

    def _advance_defender(self, dt: float) -> None:
        previous = self.state.defender.position.copy()
        new_position = move_toward(
            previous,
            self.state.receiver.position,
            max_distance=self.config.defender_speed * dt,
        )
        self.state.defender.position = new_position
        self.state.defender.velocity = (new_position - previous) / dt

    def _advance_pass(
        self,
        dt: float,
        previous_receiver_position: Vector2,
        previous_defender_position: Vector2,
    ) -> float:
        """Advance the ball and return the fraction of ``dt`` the play consumed."""
        ball_start = self.state.ball.position.copy()
        ball_end, ball_travel_fraction = advance_ball(self.state.ball, dt)
        receiver_window_end = (
            previous_receiver_position
            + (self.state.receiver.position - previous_receiver_position) * ball_travel_fraction
        )
        defender_window_end = (
            previous_defender_position
            + (self.state.defender.position - previous_defender_position) * ball_travel_fraction
        )
        events: list[tuple[float, int, PlayOutcome, float]] = []

        receiver_contact = moving_circle_contact_fraction(
            ball_start,
            ball_end,
            previous_receiver_position,
            receiver_window_end,
            self.config.catch_radius,
        )
        if receiver_contact is not None:
            events.append(
                (
                    receiver_contact * ball_travel_fraction,
                    1,
                    PlayOutcome.COMPLETED,
                    receiver_contact,
                )
            )

        defender_contact = moving_circle_contact_fraction(
            ball_start,
            ball_end,
            previous_defender_position,
            defender_window_end,
            self.config.catch_radius,
        )
        if defender_contact is not None:
            events.append(
                (
                    defender_contact * ball_travel_fraction,
                    0,
                    PlayOutcome.INTERCEPTION,
                    defender_contact,
                )
            )

        field_exit = segment_field_exit_fraction(ball_start, ball_end, self.config.field)
        if field_exit is not None:
            events.append(
                (
                    field_exit * ball_travel_fraction,
                    2,
                    PlayOutcome.OUT_OF_BOUNDS,
                    field_exit,
                )
            )

        target = self.state.ball.target
        target_reached = target is not None and np.array_equal(ball_end, target)
        if target_reached:
            events.append(
                (
                    ball_travel_fraction,
                    3,
                    PlayOutcome.INCOMPLETE,
                    1.0,
                )
            )

        if not events:
            self.state.ball.position = ball_end
            return 1.0

        tick_fraction, _, outcome, segment_fraction = min(
            events, key=lambda event: (event[0], event[1])
        )
        self.state.ball.position = ball_start + (ball_end - ball_start) * segment_fraction
        self.state.receiver.position = (
            previous_receiver_position
            + (self.state.receiver.position - previous_receiver_position) * tick_fraction
        )
        self.state.defender.position = (
            previous_defender_position
            + (self.state.defender.position - previous_defender_position) * tick_fraction
        )
        self._finish(outcome)
        return tick_fraction

    def _finish(self, outcome: PlayOutcome) -> None:
        self.state.outcome = outcome
        self.state.ball.status = BallStatus.DEAD
