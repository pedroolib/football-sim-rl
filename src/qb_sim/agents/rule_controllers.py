"""Rule-based controllers implementing the common agent interface."""

from math import inf

import numpy as np

from qb_sim.agents.base import BaseAgent
from qb_sim.agents.observations import (
    DefenderObservation,
    PlayerView,
    QuarterbackObservation,
    ReceiverObservation,
)
from qb_sim.config import SimulationConfig
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import BallStatus, PlayState, Role
from qb_sim.engine.geometry import FieldBounds, Vector2, distance, vec2
from qb_sim.rules.routes import Route


class ReceiverRouteController(BaseAgent):
    """Follow a scripted route, or run to the end zone while carrying."""

    role = Role.RECEIVER

    def __init__(
        self,
        player_id: str,
        route: Route,
        speed: float,
        time_step: float,
        field: FieldBounds | None = None,
    ) -> None:
        super().__init__(player_id)
        self._route = route
        self._speed = speed
        self._time_step = time_step
        self._field = field if field is not None else FieldBounds()

    @property
    def route(self) -> Route:
        """The route this receiver is running, for visualization."""
        return self._route

    def reset(self) -> None:
        self._route.reset()

    def observe(self, state: PlayState) -> ReceiverObservation:
        me = state.player(self.player_id)
        nearest: PlayerView | None = None
        nearest_distance = inf
        for player in state.players:
            if player.role is not Role.DEFENDER:
                continue
            current_distance = distance(player.position, me.position)
            if current_distance < nearest_distance:
                nearest_distance = current_distance
                nearest = PlayerView(
                    player.player_id,
                    player.position.copy(),
                    player.velocity.copy(),
                )
        return ReceiverObservation(
            player_id=self.player_id,
            role=self.role,
            position=me.position.copy(),
            velocity=me.velocity.copy(),
            quarterback_position=state.player("qb").position.copy(),
            ball_status=state.ball.status,
            ball_position=state.ball.position.copy(),
            ball_height=state.ball.height,
            nearest_defender=nearest,
            time_step=self._time_step,
            elapsed_seconds=state.elapsed_seconds,
            carrier_id=state.carrier_id,
        )

    def act(self, observation: ReceiverObservation) -> Move | None:
        if observation.carrier_id == self.player_id:
            return self._carry_action(observation)
        max_distance = self._speed * observation.time_step
        new_position, _ = self._route.advance(observation.position, max_distance)
        offset = new_position - observation.position
        if float(np.linalg.norm(offset)) == 0.0:
            return None
        return Move(direction=offset / max_distance)

    def _carry_action(self, observation: ReceiverObservation) -> Move:
        direction = vec2(1.0, 0.0)
        if observation.nearest_defender is not None:
            offset = observation.position - observation.nearest_defender.position
            distance = float(np.linalg.norm(offset))
            if 0.0 < distance < 3.0:
                direction = direction + offset / distance * 0.75
        margin = 3.0
        if observation.position[1] < margin:
            direction = direction + vec2(0.0, 0.5)
        elif observation.position[1] > self._field.width_yards - margin:
            direction = direction + vec2(0.0, -0.5)
        length = float(np.linalg.norm(direction))
        if length == 0.0:
            return Move(direction=vec2(1.0, 0.0))
        return Move(direction=direction / length)


class ManCoverageController(BaseAgent):
    """Cover the assigned receiver, or chase the ball carrier after a catch."""

    role = Role.DEFENDER

    def __init__(
        self,
        player_id: str,
        speed: float,
        assignment_id: str,
        time_step: float,
    ) -> None:
        super().__init__(player_id)
        self._speed = speed
        self._assignment_id = assignment_id
        self._time_step = time_step

    def observe(self, state: PlayState) -> DefenderObservation:
        me = state.player(self.player_id)
        target_id = state.carrier_id if state.carrier_id is not None else self._assignment_id
        target: PlayerView | None = None
        if target_id is not None:
            target_player = state.player(target_id)
            target = PlayerView(
                target_player.player_id,
                target_player.position.copy(),
                target_player.velocity.copy(),
            )
        return DefenderObservation(
            player_id=self.player_id,
            role=self.role,
            position=me.position.copy(),
            velocity=me.velocity.copy(),
            target_id=target_id,
            target=target,
            ball_status=state.ball.status,
            ball_position=state.ball.position.copy(),
            ball_height=state.ball.height,
            carrier_id=state.carrier_id,
            time_step=self._time_step,
            elapsed_seconds=state.elapsed_seconds,
        )

    def act(self, observation: DefenderObservation) -> Move | None:
        if observation.target is None:
            return None
        offset = observation.target.position - observation.position
        length = float(np.linalg.norm(offset))
        if length == 0.0:
            return None
        max_distance = self._speed * observation.time_step
        if length <= max_distance:
            return Move(direction=offset / max_distance)
        return Move(direction=offset / length)


class PassRushController(BaseAgent):
    """Rush the quarterback and try to sack him before the throw."""

    role = Role.DEFENDER

    def __init__(self, player_id: str, speed: float, time_step: float) -> None:
        super().__init__(player_id)
        self._speed = speed
        self._time_step = time_step

    def observe(self, state: PlayState) -> DefenderObservation:
        me = state.player(self.player_id)
        target_id = state.carrier_id if state.carrier_id is not None else "qb"
        target_player = state.player(target_id)
        return DefenderObservation(
            player_id=self.player_id,
            role=self.role,
            position=me.position.copy(),
            velocity=me.velocity.copy(),
            target_id=target_id,
            target=PlayerView(
                target_player.player_id,
                target_player.position.copy(),
                target_player.velocity.copy(),
            ),
            ball_status=state.ball.status,
            ball_position=state.ball.position.copy(),
            ball_height=state.ball.height,
            carrier_id=state.carrier_id,
            time_step=self._time_step,
            elapsed_seconds=state.elapsed_seconds,
        )

    def act(self, observation: DefenderObservation) -> Move | None:
        if observation.target is None:
            return None
        offset = observation.target.position - observation.position
        length = float(np.linalg.norm(offset))
        if length == 0.0:
            return None
        max_distance = self._speed * observation.time_step
        if length <= max_distance:
            return Move(direction=offset / max_distance)
        return Move(direction=offset / length)


class HeuristicQuarterback(BaseAgent):
    """Throw to the openest receiver, scramble under pressure, or wait."""

    role = Role.QUARTERBACK

    def __init__(
        self,
        player_id: str,
        config: SimulationConfig,
        open_threshold: float = 2.5,
        throw_after_seconds: float = 2.0,
        min_throw_seconds: float = 0.5,
    ) -> None:
        super().__init__(player_id)
        self._config = config
        self._open_threshold = open_threshold
        self._throw_after_seconds = throw_after_seconds
        self._min_throw_seconds = min_throw_seconds

    def observe(self, state: PlayState) -> QuarterbackObservation:
        me = state.player(self.player_id)
        receivers = tuple(
            PlayerView(player.player_id, player.position.copy(), player.velocity.copy())
            for player in state.players
            if player.role is Role.RECEIVER
        )
        defenders = tuple(
            PlayerView(player.player_id, player.position.copy(), player.velocity.copy())
            for player in state.players
            if player.role is Role.DEFENDER
        )
        return QuarterbackObservation(
            player_id=self.player_id,
            role=self.role,
            position=me.position.copy(),
            velocity=me.velocity.copy(),
            pressure=state.pressure,
            ball_status=state.ball.status,
            receivers=receivers,
            defenders=defenders,
            time_step=self._config.dt,
            elapsed_seconds=state.elapsed_seconds,
            time_remaining=max(0.0, self._config.max_play_duration - state.elapsed_seconds),
        )

    def act(self, observation: QuarterbackObservation) -> Move | Throw | None:
        if observation.ball_status is not BallStatus.HELD:
            return None
        routes_developed = observation.elapsed_seconds >= self._min_throw_seconds
        best = self._openest_receiver(observation)
        if routes_developed and best is not None:
            receiver, separation = best
            deadline_expired = observation.elapsed_seconds >= self._throw_after_seconds
            if separation >= self._open_threshold or deadline_expired:
                return self._throw_to(observation, receiver)
        return self._scramble_or_none(observation)

    def _openest_receiver(
        self,
        observation: QuarterbackObservation,
    ) -> tuple[PlayerView, float] | None:
        max_distance = 0.95 * self._config.throw_max_range
        best: tuple[PlayerView, float] | None = None
        for receiver in observation.receivers:
            target = self._lead_target(observation, receiver)
            if float(np.linalg.norm(target - observation.position)) > max_distance:
                continue
            nearest = min(
                (
                    distance(receiver.position, defender.position)
                    for defender in observation.defenders
                ),
                default=inf,
            )
            if best is None or nearest > best[1]:
                best = (receiver, nearest)
        return best

    def _lead_target(self, observation: QuarterbackObservation, receiver: PlayerView) -> Vector2:
        offset = receiver.position - observation.position
        flight = float(np.linalg.norm(offset)) / self._config.ball_speed
        return receiver.position + receiver.velocity * flight

    def _throw_to(self, observation: QuarterbackObservation, receiver: PlayerView) -> Throw | None:
        target = self._lead_target(observation, receiver)
        direction = target - observation.position
        if float(np.linalg.norm(direction)) == 0.0:
            return None
        span = self._config.throw_max_range - self._config.throw_min_range
        power = (float(np.linalg.norm(direction)) - self._config.throw_min_range) / span
        power = min(max(power, 0.0), 1.0)
        return Throw(direction=direction, power=power)

    def _scramble_or_none(self, observation: QuarterbackObservation) -> Move | None:
        if observation.pressure <= 0.0 or not observation.defenders:
            return None
        nearest = min(
            observation.defenders,
            key=lambda defender: distance(defender.position, observation.position),
        )
        away = observation.position - nearest.position
        length = float(np.linalg.norm(away))
        if length == 0.0:
            return Move(direction=vec2(1.0, 0.0))
        return Move(direction=away / length)
