import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from qb_sim.config import SimulationConfig
from qb_sim.engine.entities import BallStatus, PlayOutcome
from qb_sim.engine.geometry import distance, vec2
from qb_sim.rules.play import Play
from qb_sim.rules.scenarios import RUSHED_SCENARIO
from qb_sim.ui.session import FrameInput, PlaySession

ZERO_DEVIATION = replace(
    SimulationConfig(),
    throw_base_sigma_degrees=0.0,
    throw_move_penalty=0.0,
    throw_pressure_penalty=0.0,
    throw_distance_penalty=0.0,
    throw_power_sigma=0.0,
)


def _frame(**kwargs) -> FrameInput:
    defaults = {
        "held": frozenset(),
        "pressed": frozenset(),
        "mouse_field": None,
        "mouse_down": False,
        "mouse_released": False,
        "cancel": False,
    }
    defaults.update(kwargs)
    return FrameInput(**defaults)


def test_headless_modules_do_not_import_pygame() -> None:
    code = (
        "import sys; import qb_sim; import qb_sim.ui.session; import qb_sim.ui.controls; "
        "assert 'pygame' not in sys.modules, sorted(sys.modules)"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_held_move_keys_move_the_quarterback() -> None:
    play = Play()
    session = PlaySession(play)

    session.step_frame(_frame(held=frozenset({"w"})))

    assert play.state.player("qb").position.tolist() == pytest.approx([20.0, 26.65 + 6.5 * 0.05])


def test_qb_can_move_while_aiming() -> None:
    play = Play()
    session = PlaySession(play)
    session.step_frame(_frame(mouse_field=vec2(50.0, 26.65), mouse_down=True))

    session.step_frame(
        _frame(held=frozenset({"w"}), mouse_field=vec2(30.0, 26.65), mouse_down=True)
    )

    assert play.state.player("qb").position[1] == pytest.approx(26.65 + 6.5 * 0.05)
    assert play.state.ball.status is BallStatus.HELD
    assert session.aim_preview is not None


def test_drag_release_throws_with_angle_and_power() -> None:
    play = Play(ZERO_DEVIATION)
    session = PlaySession(play)
    expected_power = 18.0 / 43.0
    expected_target_x = 20.0 + 935.0 / 43.0

    session.step_frame(_frame(mouse_field=vec2(50.0, 26.65), mouse_down=True))
    session.step_frame(_frame(mouse_field=vec2(30.0, 26.65), mouse_down=True))

    preview = session.aim_preview
    assert preview is not None
    assert preview.direction.tolist() == pytest.approx([1.0, 0.0])
    assert preview.power == pytest.approx(expected_power)
    assert preview.target.tolist() == pytest.approx([expected_target_x, 26.65])

    session.step_frame(_frame(mouse_field=vec2(30.0, 26.65), mouse_released=True))

    assert play.state.ball.status is BallStatus.IN_FLIGHT
    assert play.state.ball.target.tolist() == pytest.approx([expected_target_x, 26.65])
    assert play.state.ball.velocity.tolist() == pytest.approx([25.0, 0.0])
    assert session.aim_preview is None


def test_release_below_minimum_drag_does_not_throw() -> None:
    play = Play()
    session = PlaySession(play)

    session.step_frame(_frame(mouse_field=vec2(30.0, 26.65), mouse_down=True))
    session.step_frame(_frame(mouse_field=vec2(31.0, 26.65), mouse_released=True))

    assert play.state.ball.status is BallStatus.HELD
    assert session.aim_preview is None


def test_cancel_prevents_the_throw() -> None:
    play = Play()
    session = PlaySession(play)
    session.step_frame(_frame(mouse_field=vec2(30.0, 26.65), mouse_down=True))
    session.step_frame(_frame(mouse_field=vec2(50.0, 26.65), mouse_down=True))

    session.step_frame(_frame(mouse_field=vec2(50.0, 26.65), cancel=True))

    assert session.aim_preview is None

    session.step_frame(_frame(mouse_field=vec2(50.0, 26.65), mouse_released=True))

    assert play.state.ball.status is BallStatus.HELD


def test_release_between_ticks_is_applied_on_the_next_tick() -> None:
    play = Play()
    session = PlaySession(play)
    session.handle_frame(_frame(mouse_field=vec2(50.0, 26.65), mouse_down=True))
    session.handle_frame(_frame(mouse_field=vec2(30.0, 26.65), mouse_down=True))
    session.handle_frame(_frame(mouse_field=vec2(30.0, 26.65), mouse_released=True))

    assert play.state.ball.status is BallStatus.HELD

    session.tick()

    assert play.state.ball.status is BallStatus.IN_FLIGHT


def test_pause_and_restart_apply_without_a_tick() -> None:
    session = PlaySession(Play())

    session.handle_frame(_frame(pressed=frozenset({"p"})))
    assert session.paused

    session.handle_frame(_frame(pressed=frozenset({"p"})))
    assert not session.paused

    session.step_frame(_frame())
    session.handle_frame(_frame(pressed=frozenset({"r"})))
    assert session.play.state.elapsed_seconds == 0.0


def test_session_with_the_rushed_scenario_uses_a_pass_rusher() -> None:
    scenario = RUSHED_SCENARIO
    session = PlaySession(Play(scenario.config), routes=scenario.receiver_routes)
    initial_distance = distance(
        session.play.state.player("ru1").position,
        session.play.state.player("qb").position,
    )

    for _ in range(20):
        session.tick()

    final_distance = distance(
        session.play.state.player("ru1").position,
        session.play.state.player("qb").position,
    )
    assert session.rusher_ids == frozenset({"ru1"})
    assert final_distance < initial_distance - 2.0


def test_pause_stops_the_clock_and_restart_resets() -> None:
    session = PlaySession(Play())
    session.step_frame(_frame(pressed=frozenset({"p"})))

    session.step_frame(_frame())

    assert session.play.state.elapsed_seconds == 0.0

    session.step_frame(_frame(pressed=frozenset({"r"})))

    assert session.play.state.outcome is PlayOutcome.IN_PROGRESS
    assert not session.paused
    assert session.aim_preview is None


def test_same_frame_sequence_produces_identical_traces() -> None:
    frames = [
        _frame(held=frozenset({"w"})),
        _frame(mouse_field=vec2(30.0, 26.65), mouse_down=True),
        _frame(mouse_field=vec2(45.0, 30.0), mouse_down=True),
        _frame(mouse_field=vec2(45.0, 30.0), mouse_released=True),
        _frame(held=frozenset({"s"})),
        _frame(),
    ]

    def run() -> PlaySession:
        session = PlaySession(Play())
        for frame in frames:
            session.step_frame(frame)
        return session

    first = run()
    second = run()

    assert first.play.state.outcome is second.play.state.outcome
    np.testing.assert_allclose(
        first.play.state.player("qb").position, second.play.state.player("qb").position
    )
    np.testing.assert_allclose(first.play.state.ball.position, second.play.state.ball.position)
    assert first.play.state.elapsed_seconds == second.play.state.elapsed_seconds
