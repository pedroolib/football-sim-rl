import io
import json

import numpy as np
import pytest

from qb_sim.agents.loop import run_play
from qb_sim.config import SimulationConfig
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.geometry import vec2
from qb_sim.recording import PlayRecorder, replay
from qb_sim.rules.routes import RouteName


def _recorded_play() -> str:
    recorder = PlayRecorder()
    run_play(seed=3, recorder=recorder)
    return recorder.getvalue()


def test_recorded_play_is_jsonl_with_header_ticks_and_result() -> None:
    payloads = [json.loads(line) for line in _recorded_play().splitlines()]

    assert payloads[0]["type"] == "header"
    assert payloads[0]["seed"] == 3
    assert payloads[0]["scenario"] == "default"
    assert payloads[-1]["type"] == "result"
    assert any(payload["type"] == "tick" for payload in payloads[1:-1])


def test_replay_reproduces_the_recorded_play() -> None:
    record = _recorded_play()
    expected = json.loads(record.splitlines()[-1])

    state = replay(record)

    assert state.outcome.value == expected["outcome"]
    np.testing.assert_allclose(state.ball.position, expected["ball_position"])


def test_replay_detects_divergence() -> None:
    lines = _recorded_play().splitlines()
    result = json.loads(lines[-1])
    result["outcome"] = "interception" if result["outcome"] != "interception" else "completed"
    tampered = "\n".join([*lines[:-1], json.dumps(result)])

    with pytest.raises(ValueError):
        replay(tampered)


def test_action_serialization_round_trips() -> None:
    recorder = PlayRecorder()
    recorder.record_header(
        seed=0,
        scenario="default",
        config=SimulationConfig(),
        routes=(RouteName.POST, RouteName.SLANT),
    )
    recorder.record_tick(
        {
            "qb": Throw(direction=vec2(1.0, 0.0), power=0.5),
            "wr1": Move(direction=vec2(0.0, 1.0)),
        }
    )

    tick = json.loads(recorder.getvalue().splitlines()[1])

    assert tick["actions"]["qb"] == {"type": "throw", "direction": [1.0, 0.0], "power": 0.5}
    assert tick["actions"]["wr1"] == {"type": "move", "direction": [0.0, 1.0]}


def test_recorder_can_write_to_a_stream() -> None:
    stream = io.StringIO()
    recorder = PlayRecorder(stream)

    run_play(seed=1, recorder=recorder)

    assert stream.getvalue().startswith('{"type": "header"')
