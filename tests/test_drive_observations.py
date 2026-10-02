from dataclasses import replace

import numpy as np
import pytest

from qb_sim.config import SimulationConfig
from qb_sim.rules.drive import DriveState
from qb_sim.rules.play import Play
from qb_sim.rules.routes import ROUTE_OBSERVATION_ORDER, RouteName
from qb_sim.training.drive_observations import (
    DRIVE_OBSERVATION_SIZE,
    drive_observation_from_state,
)


def _observation(
    drive_state: DriveState | None = None,
    routes: tuple[RouteName, ...] = (RouteName.POST, RouteName.GO),
) -> np.ndarray:
    play = Play()
    state = drive_state if drive_state is not None else DriveState(los=40.0)
    return drive_observation_from_state(
        state=play.state, routes=routes, drive_state=state, config=play.config
    )


def test_observation_has_the_documented_size_and_range() -> None:
    observation = _observation()

    assert observation.shape == (DRIVE_OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert np.all(observation >= -1.0)
    assert np.all(observation <= 1.0)


def test_receiver_route_one_hot_uses_the_fixed_order() -> None:
    observation = _observation(routes=(RouteName.POST, RouteName.GO))

    post_index = 36 + ROUTE_OBSERVATION_ORDER.index(RouteName.POST)
    go_index = 36 + len(ROUTE_OBSERVATION_ORDER) + ROUTE_OBSERVATION_ORDER.index(RouteName.GO)
    assert observation[post_index] == pytest.approx(1.0)
    assert observation[go_index] == pytest.approx(1.0)
    assert float(observation[36:52].sum()) == pytest.approx(2.0)


def test_down_distance_and_yards_to_goal_are_encoded() -> None:
    observation = _observation(DriveState(los=40.0, down=2, distance=20.0))

    assert observation[52:55].tolist() == [0.0, 1.0, 0.0]
    assert observation[55] == pytest.approx(20.0 / 40.0)
    assert observation[56] == pytest.approx(0.6)


def test_missing_receiver_slots_are_zeroed() -> None:
    config = replace(
        SimulationConfig(),
        receiver_starts=((20.0, 43.0),),
        defender_starts=((18.0, 43.0),),
    )
    play = Play(config)

    observation = drive_observation_from_state(
        state=play.state,
        routes=(RouteName.FLAT,),
        drive_state=DriveState(los=20.0),
        config=play.config,
    )

    assert float(observation[36:44].sum()) == pytest.approx(1.0)
    assert observation[44:52].tolist() == [0.0] * 8
