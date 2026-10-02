"""Headless play session shared by the Pygame app and tests."""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from qb_sim.agents.base import BaseAgent
from qb_sim.agents.loop import build_rule_agents
from qb_sim.agents.rule_controllers import ReceiverRouteController
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import BallStatus, PlayOutcome, PlayState
from qb_sim.engine.geometry import Vector2
from qb_sim.rules.play import Play
from qb_sim.rules.routes import Route, RouteName
from qb_sim.ui.controls import Aim, aim_from_drag, direction_from_keys

QuarterbackPolicy = Callable[[PlayState], Move | Throw | None]


@dataclass(frozen=True, slots=True)
class FrameInput:
    """One frame of raw human input, already translated to field coordinates."""

    held: frozenset[str] = field(default_factory=frozenset)
    pressed: frozenset[str] = field(default_factory=frozenset)
    mouse_field: Vector2 | None = None
    mouse_down: bool = False
    mouse_released: bool = False
    cancel: bool = False


class PlaySession:
    """Drive a play from human input without importing any window library.

    Input processing is decoupled from simulation time: the graphical loop
    calls :meth:`handle_frame` on every rendered frame and :meth:`tick` only
    when a fixed simulation step is due, so one-shot inputs are never dropped.
    """

    def __init__(
        self,
        play: Play | None = None,
        routes: tuple[RouteName, ...] = (RouteName.POST, RouteName.SLANT),
        quarterback_policy: QuarterbackPolicy | None = None,
    ) -> None:
        self.play = play if play is not None else Play()
        self._routes = tuple(RouteName(route) for route in routes)
        self._quarterback_policy = quarterback_policy
        self._agents: dict[str, BaseAgent] = {}
        self._receiver_routes: dict[str, Route] = {}
        self._build_agents()
        self.paused = False
        self.aim_preview: Aim | None = None
        self._press_field: Vector2 | None = None
        self._held: frozenset[str] = frozenset()
        self._pending_action: Move | Throw | None = None

    def _build_agents(self) -> None:
        self._agents = build_rule_agents(
            self.play,
            self._routes,
            self.play.config,
            include_quarterback=False,
        )
        self._receiver_routes = {
            player_id: agent.route
            for player_id, agent in self._agents.items()
            if isinstance(agent, ReceiverRouteController)
        }

    @property
    def routes(self) -> dict[str, Route]:
        """Receiver routes in roster order, for visualization."""
        return self._receiver_routes

    def handle_frame(self, frame: FrameInput) -> None:
        """Process one frame of raw input without advancing the simulation."""
        if "r" in frame.pressed:
            self.restart()
        if "p" in frame.pressed:
            self.toggle_pause()
        self._held = frame.held
        if self._quarterback_policy is None:
            self._update_aim(frame)
            self._queue_release(frame)

    def tick(self) -> PlayState:
        """Advance the simulation one fixed step using the pending input."""
        if self.paused:
            return self.play.state
        actions: dict[str, Move | Throw] = {}
        for player_id, agent in self._agents.items():
            action = agent.decide(self.play.state)
            if action is not None:
                actions[player_id] = action
        if self._quarterback_policy is not None:
            quarterback_action = self._quarterback_policy(self.play.state)
        else:
            quarterback_action = self._pending_action
            self._pending_action = None
            if quarterback_action is None:
                quarterback_action = self._movement_action()
        if quarterback_action is not None:
            actions["qb"] = quarterback_action
        return self.play.step(actions)

    def step_frame(self, frame: FrameInput) -> PlayState:
        """Process one frame and advance one tick; convenient for tests."""
        self.handle_frame(frame)
        return self.tick()

    def toggle_pause(self) -> None:
        """Pause or resume the session."""
        self.paused = not self.paused

    def restart(self) -> PlayState:
        """Reset the play, the aim gesture, the pause state, and the agents."""
        self.paused = False
        self.aim_preview = None
        self._press_field = None
        self._pending_action = None
        state = self.play.reset()
        self._build_agents()
        return state

    def _can_aim(self) -> bool:
        state = self.play.state
        return state.outcome is PlayOutcome.IN_PROGRESS and state.ball.status is BallStatus.HELD

    def _update_aim(self, frame: FrameInput) -> None:
        if frame.cancel or not self._can_aim():
            self._press_field = None
            self.aim_preview = None
            return
        if frame.mouse_down and frame.mouse_field is not None:
            if self._press_field is None:
                self._press_field = np.array(frame.mouse_field, dtype=np.float64, copy=True)
            drag = frame.mouse_field - self._press_field
            self.aim_preview = aim_from_drag(
                self.play.state.player("qb").position,
                drag,
                self.play.config,
            )

    def _queue_release(self, frame: FrameInput) -> None:
        if not frame.mouse_released:
            return
        aim: Aim | None = None
        if self._press_field is not None and frame.mouse_field is not None:
            drag = frame.mouse_field - self._press_field
            aim = aim_from_drag(
                self.play.state.player("qb").position,
                drag,
                self.play.config,
            )
        self._press_field = None
        self.aim_preview = None
        if aim is not None and self._can_aim():
            self._pending_action = Throw(direction=aim.direction, power=aim.power)

    def _movement_action(self) -> Move | None:
        direction = direction_from_keys(self._held)
        if float(np.linalg.norm(direction)) == 0.0:
            return None
        return Move(direction=direction)
