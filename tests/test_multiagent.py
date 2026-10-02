import numpy as np
import pytest

pytest.importorskip("pettingzoo")

from pettingzoo.test import parallel_api_test  # noqa: E402

from qb_sim.rules.scenarios import DEFAULT_SCENARIO, RUSHED_SCENARIO  # noqa: E402
from qb_sim.training.multiagent import (  # noqa: E402
    QBPassingMultiAgentEnv,
    run_rule_episode,
)
from qb_sim.training.observations import OBSERVATION_SIZE  # noqa: E402
from qb_sim.training.role_observations import ROLE_OBSERVATION_SIZE  # noqa: E402


def test_possible_agents_match_the_roster() -> None:
    rushed = QBPassingMultiAgentEnv(RUSHED_SCENARIO)
    default = QBPassingMultiAgentEnv(DEFAULT_SCENARIO)

    assert rushed.possible_agents == ["qb", "wr1", "wr2", "cb1", "cb2", "ru1"]
    assert default.possible_agents == ["qb", "wr1", "wr2", "cb1", "cb2"]


def test_observation_and_action_spaces_per_role() -> None:
    env = QBPassingMultiAgentEnv(RUSHED_SCENARIO)

    assert env.observation_space("qb").shape == (OBSERVATION_SIZE,)
    assert env.observation_space("wr1").shape == (ROLE_OBSERVATION_SIZE,)
    assert env.observation_space("ru1").shape == (ROLE_OBSERVATION_SIZE,)
    assert env.action_space("qb").shape == (6,)
    assert env.action_space("cb1").shape == (2,)


def test_rule_episode_is_deterministic() -> None:
    first = run_rule_episode(QBPassingMultiAgentEnv(RUSHED_SCENARIO), seed=7)
    second = run_rule_episode(QBPassingMultiAgentEnv(RUSHED_SCENARIO), seed=7)

    assert first == second
    assert first["outcome"] in {"completed", "incomplete", "interception", "sack", "time_limit"}


def test_rewards_are_zero_sum_at_the_terminal_step() -> None:
    env = QBPassingMultiAgentEnv(RUSHED_SCENARIO)
    env.reset(seed=3)
    rewards: dict[str, float] = {}

    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        rewards = env.step(actions)[1]

    assert rewards["qb"] == pytest.approx(env.reward_config.time_penalty - rewards["cb1"])


def test_mover_action_shape_is_validated() -> None:
    env = QBPassingMultiAgentEnv(DEFAULT_SCENARIO)
    env.reset(seed=0)

    with pytest.raises(ValueError):
        env.step({"wr1": np.zeros(6, dtype=np.float32)})


def test_pettingzoo_parallel_api_compliance() -> None:
    env = QBPassingMultiAgentEnv(DEFAULT_SCENARIO)

    parallel_api_test(env, num_cycles=5)


def test_headless_core_does_not_import_pettingzoo() -> None:
    import subprocess
    import sys

    code = (
        "import sys; import qb_sim; import qb_sim.training.env; "
        "assert 'pettingzoo' not in sys.modules"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
