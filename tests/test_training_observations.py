import numpy as np

from qb_sim.rules.play import Play
from qb_sim.training.env import QBPassingEnv
from qb_sim.training.observations import OBSERVATION_SIZE, observation_from_state


def test_observation_shape_bounds_and_dtype() -> None:
    play = Play()

    observation = observation_from_state(play.state, play.config)

    assert observation.shape == (OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert np.all(observation >= -1.0)
    assert np.all(observation <= 1.0)


def test_environment_uses_the_shared_observation_builder() -> None:
    env = QBPassingEnv()

    observation, _ = env.reset(seed=3)

    np.testing.assert_allclose(
        observation,
        observation_from_state(env.play.state, env.play.config),
    )
