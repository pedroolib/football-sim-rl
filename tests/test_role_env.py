import numpy as np
from gymnasium.utils.env_checker import check_env

from qb_sim.engine.entities import BallStatus
from qb_sim.training.role_context_observations import RECEIVER_CONTEXT_OBSERVATION_SIZE
from qb_sim.training.role_env import RolePlayEnv


def _noop() -> np.ndarray:
    return np.zeros(2, dtype=np.float32)


def test_role_play_spaces_and_registration_contract() -> None:
    env = RolePlayEnv()

    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (RECEIVER_CONTEXT_OBSERVATION_SIZE,)
    assert env.observation_space.contains(env.reset(seed=0)[0])


def test_role_play_passes_the_gymnasium_checker() -> None:
    env = RolePlayEnv()

    check_env(env)

    env.close()


def test_role_play_reset_is_deterministic_for_a_seed() -> None:
    first, _ = RolePlayEnv().reset(seed=7)
    second, _ = RolePlayEnv().reset(seed=7)

    np.testing.assert_allclose(first, second)


def test_role_play_scores_the_catch_bonus_for_the_learner() -> None:
    env = RolePlayEnv(learner_id="wr1")
    env.reset(seed=0)
    state = env.play.state
    state.carrier_id = "wr1"
    state.ball.status = BallStatus.CARRIED
    state.carry_started_at = 0.0
    total = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        _, reward, terminated, truncated, _ = env.step(_noop())
        total += reward

    assert total >= 1.2  # completion (+1.0) plus the +0.3 catch bonus


def test_role_play_rule_agent_is_exposed_for_demonstrations() -> None:
    env = RolePlayEnv(learner_id="wr2")

    agent = env.rule_agent("wr2")

    assert agent.player_id == "wr2"
