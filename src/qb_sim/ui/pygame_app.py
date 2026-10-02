"""Interactive Pygame shell over PlaySession."""

import argparse
from collections.abc import Mapping, Sequence

import pygame

from qb_sim.engine.geometry import distance, vec2
from qb_sim.engine.physics import arc_peak
from qb_sim.rules.play import Play
from qb_sim.rules.scenarios import SCENARIOS, get_scenario
from qb_sim.ui.rendering import (
    draw_aim,
    draw_ball,
    draw_ball_guide,
    draw_field,
    draw_hud,
    draw_players,
    draw_routes,
)
from qb_sim.ui.session import FrameInput, PlaySession
from qb_sim.ui.view import ViewTransform

DEFAULT_SCREEN_SIZE = (1280, 720)
FPS = 60
_PRESSED_KEYS = {"p", "r"}
_KEY_NAMES = {
    pygame.K_w: "w",
    pygame.K_a: "a",
    pygame.K_s: "s",
    pygame.K_d: "d",
    pygame.K_UP: "up",
    pygame.K_DOWN: "down",
    pygame.K_LEFT: "left",
    pygame.K_RIGHT: "right",
    pygame.K_p: "p",
    pygame.K_r: "r",
}


class PygameApp:
    """Window loop that translates events into engine actions; no game logic here."""

    def __init__(
        self,
        session: PlaySession | None = None,
        screen_size: tuple[int, int] = DEFAULT_SCREEN_SIZE,
    ) -> None:
        pygame.init()
        self.session = session if session is not None else PlaySession(Play())
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("QB Lab")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.view = ViewTransform(
            self.session.play.config.field,
            screen_width=screen_size[0],
            screen_height=screen_size[1],
        )
        self.running = True
        self._mouse_held = False

    def step(self, frame: FrameInput) -> None:
        """Advance the session one simulation tick with the given input."""
        self.session.step_frame(frame)

    def draw_frame(self) -> None:
        """Render the current state; never mutates the simulation."""
        state = self.session.play.state
        aim = self.session.aim_preview
        draw_field(self.screen, self.view)
        draw_routes(self.screen, self.session.routes, state, self.view)
        draw_players(self.screen, state, self.view, self.session.rusher_ids)
        if aim is not None:
            peak = arc_peak(
                distance(aim.origin, aim.target),
                self.session.play.config.arc_ratio,
                self.session.play.config.max_arc_height,
            )
            draw_aim(self.screen, aim, self.view, peak)
        draw_ball_guide(self.screen, state, self.view)
        draw_ball(self.screen, state, self.view)
        draw_hud(
            self.screen,
            state,
            aim.power if aim is not None else None,
            self.session.paused,
            self.font,
        )
        pygame.display.flip()

    def frame_input(
        self,
        events: list[pygame.event.Event],
        keys_pressed: Mapping[int, bool] | Sequence[bool],
    ) -> FrameInput:
        """Translate raw Pygame events into a :class:`FrameInput`."""
        pressed: set[str] = set()
        released = False
        cancel = False
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    cancel = True
                name = _KEY_NAMES.get(event.key)
                if name in _PRESSED_KEYS:
                    pressed.add(name)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self._mouse_held = True
                elif event.button == 3:
                    cancel = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._mouse_held = False
                released = True
        held = frozenset(
            name
            for key, name in _KEY_NAMES.items()
            if keys_pressed[key] and name not in _PRESSED_KEYS
        )
        mouse_field = self.view.to_field(vec2(*pygame.mouse.get_pos()))
        return FrameInput(
            held=held,
            pressed=frozenset(pressed),
            mouse_field=mouse_field,
            mouse_down=self._mouse_held,
            mouse_released=released,
            cancel=cancel,
        )

    def run(self) -> None:
        """Run the interactive loop at a fixed simulation timestep.

        Input is processed on every rendered frame while the simulation only
        advances when a fixed step is due, so short clicks and one-shot keys
        are never dropped between ticks.
        """
        accumulator = 0.0
        while self.running:
            accumulator += self.clock.tick(FPS) / 1000.0
            frame = self.frame_input(pygame.event.get(), pygame.key.get_pressed())
            self.session.handle_frame(frame)
            while accumulator >= self.session.play.config.dt:
                self.session.tick()
                accumulator -= self.session.play.config.dt
            self.draw_frame()
        self.close()

    def close(self) -> None:
        """Release Pygame resources."""
        pygame.quit()


def build_session(scenario_name: str = "rushed", model_path: str | None = None) -> PlaySession:
    """Build a session for the chosen scenario, optionally with a PPO QB."""
    scenario = get_scenario(scenario_name)
    policy = None
    if model_path is not None:
        from qb_sim.training.policy import load_policy

        policy = load_policy(model_path, scenario_name)
    return PlaySession(
        Play(scenario.config),
        routes=scenario.receiver_routes,
        quarterback_policy=policy,
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m qb_sim.ui.pygame_app``."""
    parser = argparse.ArgumentParser(description="QB Lab visualizer")
    parser.add_argument("--model", default=None, help="Path to a trained PPO model (.zip)")
    parser.add_argument("--scenario", default="rushed", choices=sorted(SCENARIOS))
    args = parser.parse_args(argv)

    PygameApp(build_session(args.scenario, args.model)).run()


if __name__ == "__main__":
    main()
