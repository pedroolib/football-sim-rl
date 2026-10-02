import os
import subprocess
import sys
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

pygame = pytest.importorskip("pygame")

from qb_sim.engine.entities import BallStatus  # noqa: E402
from qb_sim.rules.play import Play  # noqa: E402
from qb_sim.ui.pygame_app import PygameApp  # noqa: E402
from qb_sim.ui.session import FrameInput, PlaySession  # noqa: E402


@pytest.fixture
def dummy_video(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")


def test_graphical_loop_matches_headless_session(dummy_video) -> None:
    frames = [FrameInput(held=frozenset({"w"})) for _ in range(5)]

    headless = PlaySession(Play())
    for frame in frames:
        headless.step_frame(frame)

    app = PygameApp(PlaySession(Play()))
    try:
        for frame in frames:
            app.step(frame)

        assert app.session.play.state.elapsed_seconds == headless.play.state.elapsed_seconds
        np.testing.assert_allclose(
            app.session.play.state.player("wr1").position,
            headless.play.state.player("wr1").position,
        )
        np.testing.assert_allclose(
            app.session.play.state.ball.position,
            headless.play.state.ball.position,
        )
    finally:
        app.close()


def test_draw_frame_does_not_mutate_simulation_state(dummy_video) -> None:
    app = PygameApp(PlaySession(Play()))
    try:
        app.step(FrameInput(held=frozenset({"w"})))
        before = np.array([player.position.copy() for player in app.session.play.state.players])

        app.draw_frame()

        after = np.array([player.position for player in app.session.play.state.players])
        np.testing.assert_allclose(before, after)
    finally:
        app.close()


def test_frame_input_translates_drag_and_cancel(dummy_video, monkeypatch) -> None:
    app = PygameApp(PlaySession(Play()))
    try:
        monkeypatch.setattr(pygame.mouse, "get_pos", lambda: (450, 250))
        keys: defaultdict[int, bool] = defaultdict(bool)

        down = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(450, 250))
        frame = app.frame_input([down], keys)
        assert frame.mouse_down
        assert not frame.mouse_released
        assert not frame.cancel

        up = pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(450, 250))
        frame = app.frame_input([up], keys)
        assert not frame.mouse_down
        assert frame.mouse_released

        right_click = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=(450, 250))
        assert app.frame_input([right_click], keys).cancel

        escape = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)
        assert app.frame_input([escape], keys).cancel
    finally:
        app.close()


def test_run_loop_does_not_drop_input_between_ticks(dummy_video, monkeypatch) -> None:
    app = PygameApp(PlaySession(Play()))
    try:
        positions = [(700, 360), (600, 360), (600, 360), (600, 360)]
        position_index = {"value": 0}

        def fake_get_pos():
            index = min(position_index["value"], len(positions) - 1)
            position_index["value"] += 1
            return positions[index]

        monkeypatch.setattr(pygame.mouse, "get_pos", fake_get_pos)
        scripted = [
            [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(700, 360))],
            [],
            [pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(600, 360))],
            [],
            [],
            [],
        ]
        calls = {"count": 0}

        def fake_events():
            index = calls["count"]
            calls["count"] += 1
            if calls["count"] >= len(scripted):
                app.running = False
            return scripted[index] if index < len(scripted) else []

        monkeypatch.setattr(pygame.event, "get", fake_events)
        monkeypatch.setattr(app, "clock", SimpleNamespace(tick=lambda fps: 16))
        monkeypatch.setattr(app, "draw_frame", lambda: None)

        app.run()

        assert app.session.play.state.ball.status is BallStatus.IN_FLIGHT
    finally:
        app.close()


def test_module_entry_point_runs_the_loop(dummy_video) -> None:
    env = os.environ.copy()
    env["SDL_VIDEODRIVER"] = "dummy"
    env["SDL_AUDIODRIVER"] = "dummy"
    process = subprocess.Popen(
        [sys.executable, "-m", "qb_sim.ui.pygame_app"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(1.0)
        assert process.poll() is None, process.stderr.read().decode()
    finally:
        process.terminate()
        process.wait(timeout=5)
