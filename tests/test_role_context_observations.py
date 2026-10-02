import numpy as np
import pytest

from qb_sim.rules.drive import DriveState
from qb_sim.rules.play import Play
from qb_sim.rules.routes import ROUTE_OBSERVATION_ORDER, RouteName
from qb_sim.training.role_context_observations import (
    DEFENDER_CONTEXT_OBSERVATION_SIZE,
    RECEIVER_CONTEXT_OBSERVATION_SIZE,
    defender_context_observation,
    receiver_context_observation,
)


def test_receiver_context_observation_has_the_documented_size_and_range() -> None:
    play = Play()
    observation = receiver_context_observation(
        play.state, "wr1", play.config, RouteName.POST
    )

    assert observation.shape == (RECEIVER_CONTEXT_OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert np.all(observation >= -1.0)
    assert np.all(observation <= 1.0)


def test_receiver_context_route_one_hot_uses_the_fixed_order() -> None:
    play = Play()
    observation = receiver_context_observation(
        play.state, "wr1", play.config, RouteName.CURL
    )

    index = 19 + ROUTE_OBSERVATION_ORDER.index(RouteName.CURL)
    assert observation[index] == pytest.approx(1.0)
    assert float(observation[19:27].sum()) == pytest.approx(1.0)


def test_receiver_context_encodes_yards_to_the_goal_line() -> None:
    play = Play()
    play.state.player("wr1").position[0] = 40.0

    observation = receiver_context_observation(
        play.state, "wr1", play.config, RouteName.GO, goal_line=100.0
    )

    assert observation[27] == pytest.approx(0.6)


def test_defender_context_observation_has_the_documented_size_and_range() -> None:
    play = Play()
    observation = defender_context_observation(
        play.state, "cb1", "wr1", play.config, DriveState(los=40.0)
    )

    assert observation.shape == (DEFENDER_CONTEXT_OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert np.all(observation >= -1.0)
    assert np.all(observation <= 1.0)


def test_defender_context_encodes_down_distance_and_goal() -> None:
    play = Play()
    observation = defender_context_observation(
        play.state, "cb1", "wr1", play.config, DriveState(los=40.0, down=3, distance=20.0)
    )

    assert observation[19:22].tolist() == [0.0, 0.0, 1.0]
    assert observation[22] == pytest.approx(0.5)
    assert observation[23] == pytest.approx(0.6)
