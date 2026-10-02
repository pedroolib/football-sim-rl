import numpy as np
import pytest

from qb_sim.training.env import OBSERVATION_SIZE
from qb_sim.training.multiscenario import MultiScenarioEnv


def test_multi_scenario_env_cycles_scenarios() -> None:
    env = MultiScenarioEnv(("default", "rushed"))

    env.reset(seed=0)
    assert env.scenario.name == "default"
    env.reset(seed=1)
    assert env.scenario.name == "rushed"
    env.reset(seed=2)
    assert env.scenario.name == "default"

    env.close()


def test_multi_scenario_env_shares_spaces_and_steps() -> None:
    env = MultiScenarioEnv(("default", "rushed"))

    observation, _ = env.reset(seed=0)
    assert env.observation_space.shape == (OBSERVATION_SIZE,)
    assert env.observation_space.contains(observation)

    observation, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))

    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.close()


def test_multi_scenario_env_persists_the_attempt_scale() -> None:
    env = MultiScenarioEnv(("default", "rushed"))

    env.set_attempt_scale(0.0)
    env.reset(seed=0)

    assert env.attempt_scale == 0.0
    assert env._env.attempt_scale == 0.0
    env.close()


def test_multi_scenario_env_requires_scenarios() -> None:
    with pytest.raises(ValueError):
        MultiScenarioEnv(())
