import numpy as np

from qb_sim.rules.play import Play
from qb_sim.rules.scenarios import RUSHED_SCENARIO
from qb_sim.training.env import QBPassingEnv
from qb_sim.training.observations import observation_from_state, observation_size


def test_observation_shape_bounds_and_dtype() -> None:
    play = Play()

    observation = observation_from_state(play.state, play.config)

    assert observation.shape == (observation_size(play.config),)
    assert observation.dtype == np.float32
    assert np.all(observation >= -1.0)
    assert np.all(observation <= 1.0)


def test_observation_grows_with_the_rusher_roster() -> None:
    play = Play(RUSHED_SCENARIO.config)

    observation = observation_from_state(play.state, play.config)

    assert observation.shape == (observation_size(play.config),)
    assert observation.shape[0] > observation_size(Play().config)


def test_environment_uses_the_shared_observation_builder() -> None:
    env = QBPassingEnv()

    observation, _ = env.reset(seed=3)

    np.testing.assert_allclose(
        observation,
        observation_from_state(env.play.state, env.play.config),
    )
