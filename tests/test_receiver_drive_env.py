import numpy as np
import pytest

from qb_sim.agents.rule_controllers import HeuristicQuarterback
from qb_sim.engine.entities import BallStatus, PlayOutcome
from qb_sim.rules.drive import DriveConfig, DriveState
from qb_sim.rules.play import Play
from qb_sim.rules.routes import RouteName
from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.receiver_drive_env import ReceiverDriveEnv
from qb_sim.training.rewards import (
    ReceiverDriveRewardConfig,
    progress_shaping,
    receiver_drive_reward,
    receiver_result_reward,
    receiver_separation_potential,
)
from qb_sim.training.role_context_observations import (
    RECEIVER_CONTEXT_OBSERVATION_SIZE,
    RECEIVER_DRIVE_OBSERVATION_SIZE,
    receiver_context_observation,
    receiver_drive_observation,
)
from qb_sim.training.role_env import decide_player_action
from qb_sim.training.train import build_env


def test_receiver_drive_reward_config_defaults() -> None:
    config = ReceiverDriveRewardConfig()

    assert config.separation_bonus == pytest.approx(0.3)
    assert config.separation_threshold == pytest.approx(3.0)
    assert config.separation_shaping is True
    assert config.shaping_gamma == pytest.approx(0.99)
    assert config.shaping_scale == pytest.approx(5.0)
    assert config.progress_scale == pytest.approx(1.0)
    assert config.progress_gamma == pytest.approx(0.99)


def test_receiver_drive_reward_adds_the_bonus_only_when_targeted_open() -> None:
    config = ReceiverDriveRewardConfig()
    base = receiver_result_reward(PlayOutcome.COMPLETED, 8.0, True, config.receiver)

    closed = receiver_drive_reward(PlayOutcome.COMPLETED, 8.0, True, False, config)
    open_ = receiver_drive_reward(PlayOutcome.COMPLETED, 8.0, True, True, config)

    assert closed == pytest.approx(base)
    assert open_ == pytest.approx(base + 0.3)


def test_receiver_drive_reward_keeps_terminal_penalties_and_pays_the_open_bonus() -> None:
    config = ReceiverDriveRewardConfig()

    reward = receiver_drive_reward(PlayOutcome.INTERCEPTION, 0.0, False, True, config)

    assert reward == pytest.approx(-2.0 + 0.3)


def test_receiver_separation_potential_is_bounded_and_monotone() -> None:
    assert receiver_separation_potential(0.0, 5.0) == 0.0
    assert receiver_separation_potential(-2.0, 5.0) == 0.0
    assert receiver_separation_potential(2.5, 5.0) == pytest.approx(0.5)
    assert receiver_separation_potential(5.0, 5.0) == pytest.approx(1.0)
    assert receiver_separation_potential(50.0, 5.0) == pytest.approx(1.0)
    assert receiver_separation_potential(1.0, 5.0) < receiver_separation_potential(2.0, 5.0)
    with pytest.raises(ValueError):
        receiver_separation_potential(1.0, 0.0)


def test_receiver_drive_observation_extends_the_context_layout() -> None:
    config = get_scenario("stage_b").config
    play = Play(config)
    drive_state = DriveState(los=40.0, down=3, distance=20.0)

    observation = receiver_drive_observation(
        play.state, "wr1", config, RouteName.POST, drive_state
    )
    context = receiver_context_observation(
        play.state, "wr1", config, RouteName.POST
    )

    assert RECEIVER_DRIVE_OBSERVATION_SIZE == RECEIVER_CONTEXT_OBSERVATION_SIZE + 5 == 40
    assert observation.shape == (RECEIVER_DRIVE_OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert np.all(observation >= -1.0)
    assert np.all(observation <= 1.0)
    np.testing.assert_allclose(observation[:RECEIVER_CONTEXT_OBSERVATION_SIZE], context)
    assert observation[35:38].tolist() == [0.0, 0.0, 1.0]
    assert observation[38] == pytest.approx(0.5)
    assert observation[39] == pytest.approx(0.6)


def test_receiver_drive_spaces_and_reset_contract() -> None:
    env = ReceiverDriveEnv()
    observation, info = env.reset(seed=0)

    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (RECEIVER_DRIVE_OBSERVATION_SIZE,)
    assert env.observation_space.contains(observation)
    assert info["down"] == 1
    assert info["plays"] == []


def test_receiver_drive_uses_the_rule_qb_by_default() -> None:
    env = ReceiverDriveEnv()

    assert isinstance(env.opponents["qb"], HeuristicQuarterback)


def test_receiver_drive_rule_agent_is_exposed_for_demonstrations() -> None:
    env = ReceiverDriveEnv(learner_id="wr2")

    assert env.rule_agent("wr2").player_id == "wr2"


def test_receiver_drive_trajectory_is_deterministic() -> None:
    first = ReceiverDriveEnv()
    second = ReceiverDriveEnv()
    first_observation, _ = first.reset(seed=11)
    second_observation, _ = second.reset(seed=11)
    np.testing.assert_allclose(first_observation, second_observation)
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


def test_receiver_drive_pays_dense_progress_while_carrying() -> None:
    env = ReceiverDriveEnv(drive_config=DriveConfig(los_range=(20.0, 20.0)))
    env.reset(seed=0)
    state = env.play.state
    state.carrier_id = "wr1"
    state.ball.status = BallStatus.CARRIED
    state.carry_started_at = 0.0
    x_before = float(state.player("wr1").position[0])

    _, reward, terminated, truncated, _ = env.step(
        np.array([1.0, 0.0], dtype=np.float32)
    )

    assert not (terminated or truncated)
    delta_x = float(state.player("wr1").position[0]) - x_before
    assert delta_x > 0.0
    assert reward == pytest.approx(delta_x * 0.05)


def test_receiver_drive_pays_no_progress_bonus_while_standing_still() -> None:
    env = ReceiverDriveEnv(drive_config=DriveConfig(los_range=(20.0, 20.0)))
    env.reset(seed=0)
    state = env.play.state
    state.carrier_id = "wr1"
    state.ball.status = BallStatus.CARRIED
    state.carry_started_at = 0.0

    _, reward, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))

    assert not (terminated or truncated)
    assert reward == pytest.approx(0.0)


def test_receiver_drive_short_episode_truncates_with_well_formed_info() -> None:
    env = ReceiverDriveEnv(
        max_steps=5,
        drive_config=DriveConfig(los_range=(20.0, 20.0)),
    )
    env.reset(seed=0)
    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(
            np.array([1.0, 1.0], dtype=np.float32)
        )

    assert truncated
    assert not terminated
    assert info["steps"] == 5
    assert isinstance(info["plays"], list)
    for key in ("down", "distance", "los", "outcome", "drive_end_reason", "steps", "plays"):
        assert key in info


def test_decide_player_action_passes_drive_state_only_to_drive_aware_receivers() -> None:
    config = get_scenario("stage_b").config
    play = Play(config)
    drive_state = DriveState(los=30.0)
    seen: list[tuple] = []

    def plain(state, route, start=None):
        seen.append(("plain", None))
        return None

    class DriveAwareReceiver:
        drive_aware = True

        def __call__(self, state, route, start=None, drive_state=None):
            seen.append(("aware", drive_state))
            return None

    decide_player_action(play, (RouteName.POST, RouteName.SLANT), "wr1", plain, drive_state)
    decide_player_action(
        play, (RouteName.POST, RouteName.SLANT), "wr1", DriveAwareReceiver(), drive_state
    )

    assert seen[0] == ("plain", None)
    assert seen[1][0] == "aware"
    assert seen[1][1] is drive_state


def test_build_env_returns_the_receiver_drive_env() -> None:
    env = build_env(("stage_b",), "receiver_drive", "wr1")

    assert isinstance(env, ReceiverDriveEnv)
    assert env.observation_space.shape == (RECEIVER_DRIVE_OBSERVATION_SIZE,)

    env.close()


def _run_receiver_drive(
    reward_config: ReceiverDriveRewardConfig,
) -> tuple[float, dict]:
    env = ReceiverDriveEnv(
        reward_config=reward_config,
        drive_config=DriveConfig(los_range=(25.0, 25.0)),
    )
    env.reset(seed=3)
    total = 0.0
    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(
            np.array([1.0, 0.0], dtype=np.float32)
        )
        total += reward
    return total, info


def test_receiver_drive_applies_progress_shaping_on_play_end() -> None:
    shaped_total, info = _run_receiver_drive(ReceiverDriveRewardConfig())
    plain_total, _ = _run_receiver_drive(
        ReceiverDriveRewardConfig(progress_scale=0.0)
    )

    expected = sum(
        progress_shaping(
            play["spot"] - play["yards"], play["spot"], 100.0, 1.0, 0.99
        )
        for play in info["plays"]
    )

    assert expected != pytest.approx(0.0)
    assert shaped_total - plain_total == pytest.approx(expected)
