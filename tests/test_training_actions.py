import numpy as np
import pytest

from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import BallStatus
from qb_sim.engine.geometry import vec2
from qb_sim.training.actions import decode_action, encode_engine_action


def test_noop_action_decodes_to_nothing() -> None:
    assert decode_action(np.zeros(6, dtype=np.float32), BallStatus.HELD) == {}


def test_move_action_decodes_to_move() -> None:
    action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

    decoded = decode_action(action, BallStatus.HELD)

    assert isinstance(decoded["qb"], Move)
    assert decoded["qb"].direction.tolist() == pytest.approx([1.0, 0.0])


def test_trigger_throws_with_power_mapping() -> None:
    action = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.1], dtype=np.float32)

    decoded = decode_action(action, BallStatus.HELD)

    assert isinstance(decoded["qb"], Throw)
    assert decoded["qb"].direction.tolist() == pytest.approx([0.0, 1.0])
    assert decoded["qb"].power == pytest.approx(0.5)


def test_trigger_with_zero_aim_defaults_forward() -> None:
    action = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    decoded = decode_action(action, BallStatus.HELD)

    assert isinstance(decoded["qb"], Throw)
    assert decoded["qb"].direction.tolist() == pytest.approx([1.0, 0.0])
    assert decoded["qb"].power == pytest.approx(1.0)


def test_trigger_is_ignored_when_the_ball_is_not_held() -> None:
    action = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)

    decoded = decode_action(action, BallStatus.IN_FLIGHT)

    assert isinstance(decoded["qb"], Move)


def test_action_with_wrong_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        decode_action(np.zeros(5, dtype=np.float32), BallStatus.HELD)


def test_encode_round_trips_throw_and_move() -> None:
    throw = Throw(direction=vec2(0.0, 1.0), power=0.25)
    decoded_throw = decode_action(encode_engine_action(throw), BallStatus.HELD)
    assert isinstance(decoded_throw["qb"], Throw)
    assert decoded_throw["qb"].direction.tolist() == pytest.approx([0.0, 1.0])
    assert decoded_throw["qb"].power == pytest.approx(0.25)

    move = Move(direction=vec2(1.0, 0.0))
    decoded_move = decode_action(encode_engine_action(move), BallStatus.HELD)
    assert isinstance(decoded_move["qb"], Move)
    assert decoded_move["qb"].direction.tolist() == pytest.approx([1.0, 0.0])
