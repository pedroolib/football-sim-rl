"""Headless play session shared by the Pygame app and tests."""

from dataclasses import dataclass, field

import numpy as np

from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import BallStatus, PlayOutcome, PlayState
from qb_sim.engine.geometry import Vector2
from qb_sim.rules.play import Play
from qb_sim.ui.controls import Aim, aim_from_drag, direction_from_keys


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

    def __init__(self, play: Play | None = None) -> None:
        self.play = play if play is not None else Play()
        self.paused = False
        self.aim_preview: Aim | None = None
        self._press_field: Vector2 | None = None
        self._held: frozenset[str] = frozenset()
        self._pending_action: Move | Throw | None = None

    def handle_frame(self, frame: FrameInput) -> None:
        """Process one frame of raw input without advancing the simulation."""
        if "r" in frame.pressed:
            self.restart()
        if "p" in frame.pressed:
            self.toggle_pause()
        self._held = frame.held
        self._update_aim(frame)
        self._queue_release(frame)

    def tick(self) -> PlayState:
        """Advance the simulation one fixed step using the pending input."""
        if self.paused:
            return self.play.state
        action = self._pending_action
        self._pending_action = None
        if action is None:
            action = self._movement_action()
        return self.play.step(action)

    def step_frame(self, frame: FrameInput) -> PlayState:
        """Process one frame and advance one tick; convenient for tests."""
        self.handle_frame(frame)
        return self.tick()

    def toggle_pause(self) -> None:
        """Pause or resume the session."""
        self.paused = not self.paused

    def restart(self) -> PlayState:
        """Reset the play, the aim gesture, and the pause state."""
        self.paused = False
        self.aim_preview = None
        self._press_field = None
        self._pending_action = None
        return self.play.reset()

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
