import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from qb_sim.agents.rule_controllers import HeuristicQuarterback
from qb_sim.engine.actions import Throw
from qb_sim.engine.entities import BallStatus
from qb_sim.engine.geometry import vec2
from qb_sim.rules.drive import DriveConfig
from qb_sim.rules.routes import RouteName
from qb_sim.training.randomization import AdaptationRanges
from qb_sim.training.rewards import DefenderRewardConfig, DriveRewardConfig, defender_potential
from qb_sim.training.role_context_observations import (
    DEFENDER_CONTEXT_OBSERVATION_SIZE,
    RECEIVER_CONTEXT_OBSERVATION_SIZE,
    receiver_context_observation,
)
from qb_sim.training.role_env import RoleDriveEnv, RolePlayEnv, defender_target


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
    first_env = RolePlayEnv()
    second_env = RolePlayEnv()
    first, _ = first_env.reset(seed=7)
    second, _ = second_env.reset(seed=7)

    np.testing.assert_allclose(first, second)
    assert first_env.play.line_of_scrimmage == pytest.approx(second_env.play.line_of_scrimmage)


def test_role_play_reset_samples_the_los_inside_the_range() -> None:
    env = RolePlayEnv(los_range=(30.0, 40.0))
    env.reset(seed=0)
    first = env.play.line_of_scrimmage
    env.reset(seed=1)
    second = env.play.line_of_scrimmage

    assert 30.0 <= first <= 40.0
    assert 30.0 <= second <= 40.0
    assert first != second
    assert env.play.state.player("qb").position[0] == pytest.approx(second)


def test_randomized_env_applies_sampled_speeds() -> None:
    env = RolePlayEnv(randomization=AdaptationRanges())
    env.reset(seed=5)

    assert env.active_config.receiver_speed == pytest.approx(
        env.play.config.receiver_speed
    )
    assert env.play.state.player("wr1").max_speed == pytest.approx(
        env.active_config.receiver_speed
    )


def test_role_play_randomization_is_deterministic_for_a_seed() -> None:
    first = RolePlayEnv(randomization=AdaptationRanges())
    second = RolePlayEnv(randomization=AdaptationRanges())
    first_observation, _ = first.reset(seed=5)
    second_observation, _ = second.reset(seed=5)

    assert first.active_config == second.active_config
    np.testing.assert_allclose(first_observation, second_observation)


def test_role_play_curriculum_progress_clamps_and_narrows() -> None:
    env = RolePlayEnv(randomization=AdaptationRanges())

    env.set_curriculum_progress(2.0)
    env.reset(seed=0)
    assert 6.0 <= env.active_config.receiver_speed <= 8.5

    env.set_curriculum_progress(0.0)
    env.reset(seed=0)
    assert 6.75 <= env.active_config.receiver_speed <= 7.75


def test_role_play_without_randomization_keeps_the_scenario_config() -> None:
    env = RolePlayEnv()
    env.reset(seed=0)

    assert env.active_config is env.scenario.config
    assert env.play.config is env.scenario.config


def test_role_drive_randomization_applies_sampled_config() -> None:
    env = RoleDriveEnv(randomization=AdaptationRanges())
    env.reset(seed=5)

    assert env.active_config == env.play.config
    assert env.active_config.receiver_speed == pytest.approx(env.drive.config.receiver_speed)
    assert env.play.state.player("wr1").max_speed == pytest.approx(
        env.active_config.receiver_speed
    )


def test_sampled_capabilities_do_not_follow_the_los() -> None:
    env = RolePlayEnv(randomization=AdaptationRanges())
    pairs = []
    for seed in range(8):
        env.reset(seed=seed)
        pairs.append((env.play.line_of_scrimmage, env.active_config.receiver_speed))
    speeds_by_los = [speed for _, speed in sorted(pairs)]
    assert speeds_by_los != sorted(speeds_by_los)  # not a monotone function of the LOS


class _RecordingConfigOpponent:
    def __init__(self) -> None:
        self.configs: list = []

    def set_config(self, config) -> None:
        self.configs.append(config)

    def __call__(self, state, route, start=None):
        return None


def test_randomized_env_injects_the_active_config_into_opponents() -> None:
    opponent = _RecordingConfigOpponent()
    env = RolePlayEnv(opponents={"wr2": opponent}, randomization=AdaptationRanges())
    env.reset(seed=5)

    assert opponent.configs
    assert opponent.configs[-1] is env.active_config


def test_randomized_env_updates_an_explicit_rule_qb_config() -> None:
    qb = HeuristicQuarterback("qb", RolePlayEnv().scenario.config)
    env = RolePlayEnv(opponents={"qb": qb}, randomization=AdaptationRanges())
    env.reset(seed=5)

    assert qb._config is env.active_config


def test_role_drive_injects_the_active_config_into_opponents() -> None:
    opponent = _RecordingConfigOpponent()
    env = RoleDriveEnv(opponents={"wr1": opponent}, randomization=AdaptationRanges())
    env.reset(seed=5)

    assert opponent.configs
    assert opponent.configs[-1] is env.active_config


def test_role_play_rejects_an_los_range_below_the_formation_floor() -> None:
    with pytest.raises(ValueError, match="formation floor"):
        RolePlayEnv(los_range=(0.0, 0.0))


def test_role_play_observation_uses_the_shifted_start() -> None:
    env = RolePlayEnv(los_range=(45.0, 45.0))
    env.reset(seed=0)
    env.routes = (RouteName.GO, env.routes[1])

    observation = env.observation()

    # wr1 starts at (45,43); GO breaks at (75,43), so the delta is 30 yards.
    assert observation[28] == pytest.approx(30.0 / 120.0)
    assert observation[29] == pytest.approx(0.0)
    assert observation[30] == pytest.approx(1.0)


def test_role_play_fixed_los_matches_the_unshifted_setup() -> None:
    env = RolePlayEnv(los_range=(20.0, 20.0))
    observation, _ = env.reset(seed=0)

    assert env.play.line_of_scrimmage == pytest.approx(20.0)
    assert env.play.state.player("qb").position[0] == pytest.approx(20.0)
    assert env.play.state.player("wr1").position.tolist() == pytest.approx([20.0, 43.0])
    expected = receiver_context_observation(
        env.play.state, "wr1", env.scenario.config, env.routes[0]
    )
    np.testing.assert_allclose(observation, expected)


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


def test_role_play_pays_dense_progress_while_carrying() -> None:
    env = RolePlayEnv(learner_id="wr1")
    env.reset(seed=0)
    state = env.play.state
    state.carrier_id = "wr1"
    state.ball.status = BallStatus.CARRIED
    state.carry_started_at = 0.0
    x_before = float(state.player("wr1").position[0])

    _, reward, terminated, truncated, _ = env.step(np.array([1.0, 0.0], dtype=np.float32))

    assert not (terminated or truncated)
    delta_x = float(state.player("wr1").position[0]) - x_before
    assert delta_x > 0.0
    assert reward == pytest.approx(delta_x * 0.05)


def test_role_play_pays_no_progress_bonus_while_standing_still() -> None:
    env = RolePlayEnv(learner_id="wr1")
    env.reset(seed=0)
    state = env.play.state
    state.carrier_id = "wr1"
    state.ball.status = BallStatus.CARRIED
    state.carry_started_at = 0.0

    _, reward, terminated, truncated, _ = env.step(_noop())

    assert not (terminated or truncated)
    assert reward == pytest.approx(0.0)


def test_role_play_reports_yards_after_catch() -> None:
    env = RolePlayEnv(learner_id="wr1")
    env.reset(seed=0)
    state = env.play.state
    catch_x = env.play.line_of_scrimmage + 12.0
    state.player("wr1").position = vec2(catch_x, 43.0)
    state.carrier_id = "wr1"
    state.ball.status = BallStatus.CARRIED
    state.carry_started_at = state.elapsed_seconds
    _, _, terminated, truncated, info = env.step(_noop())
    assert not (terminated or truncated)
    assert info["yards_after_catch"] == pytest.approx(0.0)
    expected_yards = catch_x - env.play.line_of_scrimmage

    _, _, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))

    assert not (terminated or truncated)
    assert info["yards"] > expected_yards
    assert info["yards_after_catch"] == pytest.approx(
        info["yards"] - expected_yards
    )
    assert info["yards_after_catch"] > 0.0


def test_role_play_rule_agent_is_exposed_for_demonstrations() -> None:
    env = RolePlayEnv(learner_id="wr2")

    agent = env.rule_agent("wr2")

    assert agent.player_id == "wr2"


def test_role_play_invokes_a_supplied_qb_opponent() -> None:
    calls: list[int] = []

    def qb(state, routes, drive_state):
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


def test_role_play_passes_drive_context_to_the_qb_opponent() -> None:
    contexts = []

    def qb(state, routes, drive_state):
        contexts.append(drive_state)
        return None

    env = RolePlayEnv(opponents={"qb": qb})
    env.reset(seed=0)
    env.step(np.zeros(2, dtype=np.float32))

    assert contexts
    assert contexts[0].down == 1
    assert contexts[0].distance == 10.0


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

    assert env.observation_space.shape == (DEFENDER_CONTEXT_OBSERVATION_SIZE,)
    assert env.observation_space.contains(observation)
    assert info["down"] == 1

    check_env(env)


def test_role_drive_defender_reward_is_zero_sum() -> None:
    env = RoleDriveEnv(
        _sack_scenario(),
        learner_id="cb1",
        reward_config=DefenderRewardConfig(
            terminal=DriveRewardConfig(progress_scale=0.0), shaping=False
        ),
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

    ongoing = RoleDriveEnv(
        learner_id="cb1",
        reward_config=DefenderRewardConfig(shaping=False),
    )
    ongoing.reset(seed=0)
    _, reward, terminated, truncated, _ = ongoing.step(np.zeros(2, dtype=np.float32))
    assert not (terminated or truncated)
    assert reward == 0.0


def _learner_potential(env: RoleDriveEnv) -> float:
    target_id = defender_target(env.play, env.learner_id)
    state = env.play.state
    distance = float(
        np.linalg.norm(
            state.player(env.learner_id).position - state.player(target_id).position
        )
    )
    return defender_potential(distance, env.reward_config.shaping_scale)


def test_role_drive_shaping_is_potential_based() -> None:
    shaped = RoleDriveEnv(learner_id="cb1")
    plain = RoleDriveEnv(
        learner_id="cb1",
        reward_config=DefenderRewardConfig(shaping=False),
    )
    shaped.reset(seed=0)
    plain.reset(seed=0)
    action = np.zeros(2, dtype=np.float32)
    terminated = truncated = False
    terminal_terms: list[float] = []
    while not (terminated or truncated):
        potential_before = _learner_potential(shaped)
        _, shaped_reward, terminated, truncated, _ = shaped.step(action)
        _, plain_reward, _, _, _ = plain.step(action)
        terminal_terms.append(plain_reward)
        # Phi(s') is 0 once the drive is over; otherwise it is the potential of
        # the current (or newly started) play state.
        phi_after = 0.0 if terminated else _learner_potential(shaped)
        expected = (
            plain_reward
            + shaped.reward_config.shaping_gamma * phi_after
            - potential_before
        )
        assert shaped_reward == pytest.approx(expected)
    assert any(term != 0.0 for term in terminal_terms)


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


def test_role_drive_play_history_records_the_sampled_routes() -> None:
    env = RoleDriveEnv(
        _sack_scenario(),
        learner_id="cb1",
        drive_config=DriveConfig(los_range=(20.0, 20.0)),
    )
    env.reset(seed=0)
    expected: list[list[str]] = []
    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        routes_before = [route.value for route in env.drive.routes]
        plays_before = len(env.play_history)
        _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        if len(env.play_history) > plays_before:
            expected.append(routes_before)

    assert [play["routes"] for play in info["plays"]] == expected


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
