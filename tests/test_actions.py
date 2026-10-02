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


def test_throw_normalizes_direction() -> None:
    action = Throw(direction=vec2(2.0, 0.0))

    assert action.direction.tolist() == pytest.approx([1.0, 0.0])


def test_throw_defaults_to_full_power() -> None:
    assert Throw(direction=vec2(1.0, 0.0)).power == pytest.approx(1.0)


def test_throw_rejects_zero_direction() -> None:
    with pytest.raises(ValueError):
        Throw(direction=vec2(0.0, 0.0))


def test_throw_rejects_non_finite_direction() -> None:
    with pytest.raises(ValueError):
        Throw(direction=vec2(0.0, float("inf")))


@pytest.mark.parametrize("power", [-0.1, 1.1, float("nan"), float("inf")])
def test_throw_rejects_power_outside_unit_interval(power: float) -> None:
    with pytest.raises(ValueError):
        Throw(direction=vec2(1.0, 0.0), power=power)
