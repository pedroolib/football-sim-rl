import pytest

from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.geometry import vec2


def test_move_rejects_direction_above_unit_magnitude() -> None:
    with pytest.raises(ValueError):
        Move(direction=vec2(1.0, 1.0))


def test_move_rejects_non_finite_direction() -> None:
    with pytest.raises(ValueError):
        Move(direction=vec2(float("nan"), 0.0))


def test_move_accepts_zero_and_unit_directions() -> None:
    assert Move(direction=vec2(0.0, 0.0)).direction.tolist() == [0.0, 0.0]
    assert Move(direction=vec2(1.0, 0.0)).direction.tolist() == [1.0, 0.0]


def test_throw_rejects_non_finite_target() -> None:
    with pytest.raises(ValueError):
        Throw(target=vec2(0.0, float("inf")))
