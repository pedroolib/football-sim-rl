from dataclasses import replace

import numpy as np

from qb_sim.config import SimulationConfig
from qb_sim.rules.play import Play
from qb_sim.rules.scenarios import RUSHED_SCENARIO
from qb_sim.training.env import QBPassingEnv
from qb_sim.training.observations import OBSERVATION_SIZE, observation_from_state


def test_observation_has_a_fixed_size_across_rosters() -> None:
    default_observation = observation_from_state(Play().state, Play().config)
    rushed_play = Play(RUSHED_SCENARIO.config)
    rushed_observation = observation_from_state(rushed_play.state, rushed_play.config)

    assert default_observation.shape == (OBSERVATION_SIZE,)
    assert rushed_observation.shape == (OBSERVATION_SIZE,)
    assert default_observation.dtype == np.float32
    assert np.all(default_observation >= -1.0)
    assert np.all(default_observation <= 1.0)


def test_presence_flags_reflect_the_roster() -> None:
    default_observation = observation_from_state(Play().state, Play().config)
    assert default_observation[17:19].tolist() == [1.0, 1.0]
    assert default_observation[31:34].tolist() == [1.0, 1.0, 0.0]

    rushed_play = Play(RUSHED_SCENARIO.config)
    rushed_observation = observation_from_state(rushed_play.state, rushed_play.config)
    assert rushed_observation[17:19].tolist() == [1.0, 1.0]
    assert rushed_observation[31:34].tolist() == [1.0, 1.0, 1.0]


def test_empty_slots_are_zeroed_and_flagged() -> None:
    config = replace(
        SimulationConfig(),
        receiver_starts=((20.0, 43.0),),
        defender_starts=((18.0, 43.0),),
    )
    play = Play(config)

    observation = observation_from_state(play.state, play.config)

    assert observation.shape == (OBSERVATION_SIZE,)
    assert observation[17:19].tolist() == [1.0, 0.0]
    assert observation[31:34].tolist() == [1.0, 0.0, 0.0]
    assert observation[13:17].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert observation[27:31].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_environment_uses_the_shared_observation_builder() -> None:
    env = QBPassingEnv()

    observation, _ = env.reset(seed=3)

    np.testing.assert_allclose(
        observation,
        observation_from_state(env.play.state, env.play.config),
    )
