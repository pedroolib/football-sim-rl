"""JSONL recording and deterministic replay of plays."""

import dataclasses
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TextIO

import numpy as np

from qb_sim.config import SimulationConfig
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import PlayState
from qb_sim.engine.geometry import FieldBounds, Vector2, vec2
from qb_sim.rules.play import Play
from qb_sim.rules.routes import RouteName


class PlayRecorder:
    """Write a play as JSONL: a header, one line per tick, and the result."""

    def __init__(self, stream: TextIO | str | Path | None = None) -> None:
        if stream is None:
            self._stream: TextIO = io.StringIO()
            self._close_stream = False
        elif isinstance(stream, (str, Path)):
            self._stream = Path(stream).open("w", encoding="utf-8")
            self._close_stream = True
        else:
            self._stream = stream
            self._close_stream = False

    def record_header(
        self,
        *,
        seed: int,
        scenario: str,
        config: SimulationConfig,
        routes: Sequence[RouteName],
    ) -> None:
        """Write the seed, scenario, configuration, and routes."""
        self._write(
            {
                "type": "header",
                "seed": seed,
                "scenario": scenario,
                "config": dataclasses.asdict(config),
                "routes": [str(route) for route in routes],
            }
        )

    def record_tick(self, actions: Mapping[str, Move | Throw]) -> None:
        """Write one tick of per-player actions in roster-independent order."""
        self._write(
            {
                "type": "tick",
                "actions": {
                    player_id: _action_to_dict(actions[player_id]) for player_id in sorted(actions)
                },
            }
        )

    def record_result(self, state: PlayState) -> None:
        """Write the final outcome, ball position, and carrier."""
        self._write(
            {
                "type": "result",
                "outcome": state.outcome.value,
                "elapsed_seconds": state.elapsed_seconds,
                "ball_position": [float(state.ball.position[0]), float(state.ball.position[1])],
                "carrier_id": state.carrier_id,
            }
        )

    def getvalue(self) -> str:
        """Return the in-memory record; only valid for stream recorders."""
        if isinstance(self._stream, io.StringIO):
            return self._stream.getvalue()
        raise ValueError("recorder is writing to a file or external stream")

    def close(self) -> None:
        """Close the file when the recorder opened it."""
        if self._close_stream:
            self._stream.close()

    def _write(self, payload: dict) -> None:
        self._stream.write(json.dumps(payload) + "\n")


def replay(record: str | Iterable[str]) -> PlayState:
    """Rebuild a play from a JSONL record and verify it matches the result."""
    lines = record.splitlines() if isinstance(record, str) else list(record)
    lines = [line for line in lines if line.strip()]
    if not lines:
        raise ValueError("empty record")

    header = json.loads(lines[0])
    if header.get("type") != "header":
        raise ValueError("record must start with a header")

    play = Play(_config_from_dict(header["config"]), seed=int(header["seed"]))
    expected: dict | None = None
    for line in lines[1:]:
        payload = json.loads(line)
        kind = payload.get("type")
        if kind == "tick":
            actions = {
                player_id: _action_from_dict(data) for player_id, data in payload["actions"].items()
            }
            play.step(actions)
        elif kind == "result":
            expected = payload
        else:
            raise ValueError(f"unknown record entry: {kind!r}")

    if expected is None:
        raise ValueError("record has no result")

    state = play.state
    if state.outcome.value != expected["outcome"]:
        raise ValueError("replay outcome diverged from the record")
    if not np.allclose(state.ball.position, expected["ball_position"]):
        raise ValueError("replay ball position diverged from the record")
    return state


def _action_to_dict(action: Move | Throw) -> dict:
    if isinstance(action, Throw):
        return {
            "type": "throw",
            "direction": _vector_list(action.direction),
            "power": action.power,
        }
    if isinstance(action, Move):
        return {"type": "move", "direction": _vector_list(action.direction)}
    raise TypeError(f"unsupported action: {type(action).__name__}")


def _action_from_dict(data: Mapping) -> Move | Throw:
    direction = vec2(float(data["direction"][0]), float(data["direction"][1]))
    if data["type"] == "throw":
        return Throw(direction=direction, power=float(data["power"]))
    if data["type"] == "move":
        return Move(direction=direction)
    raise ValueError(f"unknown action type: {data['type']!r}")


def _vector_list(vector: Vector2) -> list[float]:
    return [float(vector[0]), float(vector[1])]


def _config_from_dict(data: Mapping) -> SimulationConfig:
    values = {**data, "field": FieldBounds(**data["field"])}
    return SimulationConfig(**values)
