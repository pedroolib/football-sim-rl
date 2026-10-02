import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from qb_sim.engine.actions import Throw
from qb_sim.engine.entities import BallStatus
from qb_sim.rules.drive import DriveConfig
from qb_sim.training.role_context_observations import RECEIVER_CONTEXT_OBSERVATION_SIZE
from qb_sim.training.role_env import RoleDriveEnv, RolePlayEnv


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


def test_role_play_invokes_a_supplied_qb_opponent() -> None:
    calls: list[int] = []

    def qb(state):
        calls.append(1)
        if state.ball.status is not BallStatus.HELD:
            return None
        return Throw(direction=np.array([1.0, 0.0]), power=1.0)

    env = RolePlayEnv(opponents={"qb": qb})
    env.reset(seed=0)
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(_noop())

    assert calls
    assert info["outcome"] != "sack"


def test_role_play_ignores_an_opponent_for_the_learner() -> None:
    calls: list[int] = []

    def learner_opponent(state, route):
        calls.append(1)
        return None

    env = RolePlayEnv(learner_id="wr1", opponents={"wr1": learner_opponent})
    env.reset(seed=0)
    env.step(np.array([1.0, 0.0], dtype=np.float32))

    assert not calls


def test_role_play_trajectory_is_deterministic() -> None:
    first = RolePlayEnv()
    second = RolePlayEnv()
    first_observation, _ = first.reset(seed=11)
    second_observation, _ = second.reset(seed=11)
    actions = [np.array([1.0, 0.0], dtype=np.float32)] * 5
    for action in actions:
        first_observation, first_reward, first_term, first_trunc, _ = first.step(action)
        second_observation, second_reward, second_term, second_trunc, _ = second.step(action)
        np.testing.assert_allclose(first_observation, second_observation)
        assert first_reward == pytest.approx(second_reward)
        assert (first_term, first_trunc) == (second_term, second_trunc)


def _sack_scenario():
    from dataclasses import replace

    from qb_sim.config import SimulationConfig
    from qb_sim.rules.routes import RouteName
    from qb_sim.rules.scenarios import Scenario

    zero = replace(
        SimulationConfig(),
        throw_base_sigma_degrees=0.0,
        throw_move_penalty=0.0,
        throw_pressure_penalty=0.0,
        throw_distance_penalty=0.0,
        throw_power_sigma=0.0,
        defender_starts=((20.5, 26.65), (0.0, 0.0)),
        rusher_starts=((0.0, 0.0),),
    )
    return Scenario("test-sack", zero, (RouteName.POST, RouteName.SLANT))


def test_role_drive_spaces_and_checker() -> None:
    env = RoleDriveEnv(
        _sack_scenario(),
        learner_id="cb1",
        drive_config=DriveConfig(los_range=(20.0, 20.0)),
    )
    observation, info = env.reset(seed=0)

    assert env.observation_space.shape == (24,)
    assert env.observation_space.contains(observation)
    assert info["down"] == 1

    check_env(env)


def test_role_drive_defender_reward_is_zero_sum() -> None:
    env = RoleDriveEnv(
        _sack_scenario(),
        learner_id="cb1",
        drive_config=DriveConfig(los_range=(20.0, 20.0)),
    )
    env.reset(seed=0)
    rewards: list[float] = []
    terminated = truncated = False
    while not (terminated or truncated):
        _, reward, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        rewards.append(reward)
    # Sacks pay the defense +1 each; the final play also pays the turnover penalty.
    assert rewards == pytest.approx([1.0, 1.0, 1.5])
    assert sum(rewards) == pytest.approx(3.5)

    ongoing = RoleDriveEnv(learner_id="cb1")
    ongoing.reset(seed=0)
    _, reward, terminated, truncated, _ = ongoing.step(np.zeros(2, dtype=np.float32))
    assert not (terminated or truncated)
    assert reward == 0.0


def test_role_drive_step_after_termination_raises() -> None:
    env = RoleDriveEnv(
        _sack_scenario(),
        learner_id="cb1",
        drive_config=DriveConfig(los_range=(20.0, 20.0)),
    )
    env.reset(seed=0)
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))

    with pytest.raises(RuntimeError):
        env.step(np.zeros(2, dtype=np.float32))


def test_role_drive_records_separation_allowed_at_throw() -> None:
    def qb(state, routes, drive_state):
        if state.ball.status is not BallStatus.HELD or state.elapsed_seconds == 0.0:
            return None
        return Throw(direction=np.array([1.0, 0.0]), power=1.0)

    env = RoleDriveEnv(learner_id="cb1", opponents={"qb": qb})
    env.reset(seed=4)
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))

    separations = [
        play["separation_allowed_throw"]
        for play in info["plays"]
        if play["separation_allowed_throw"] is not None
    ]
    assert separations
    assert all(np.isfinite(value) for value in separations)


def test_role_drive_is_deterministic_for_a_seed() -> None:
    config = DriveConfig(los_range=(20.0, 20.0))
    first = RoleDriveEnv(_sack_scenario(), learner_id="ru1", drive_config=config)
    second = RoleDriveEnv(_sack_scenario(), learner_id="ru1", drive_config=config)
    first_observation, _ = first.reset(seed=3)
    second_observation, _ = second.reset(seed=3)

    np.testing.assert_allclose(first_observation, second_observation)


def test_role_drive_trajectory_is_deterministic() -> None:
    first = RoleDriveEnv(learner_id="cb1")
    second = RoleDriveEnv(learner_id="cb1")
    first_observation, _ = first.reset(seed=11)
    second_observation, _ = second.reset(seed=11)
    actions = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ] * 5
    for action in actions:
        first_observation, first_reward, first_term, first_trunc, _ = first.step(action)
        second_observation, second_reward, second_term, second_trunc, _ = second.step(action)
        np.testing.assert_allclose(first_observation, second_observation)
        assert first_reward == pytest.approx(second_reward)
        assert (first_term, first_trunc) == (second_term, second_trunc)
