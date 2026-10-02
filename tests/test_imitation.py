import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qb_sim.rules.scenarios import get_scenario  # noqa: E402
from qb_sim.training.actions import ACTION_SIZE  # noqa: E402
from qb_sim.training.imitation import (  # noqa: E402
    collect_demonstrations,
    train_behavior_cloning,
    warm_start_policy,
)
from qb_sim.training.observations import OBSERVATION_SIZE  # noqa: E402


def test_demonstrations_have_expected_shapes_and_range() -> None:
    dataset = collect_demonstrations(get_scenario("rushed"), episodes=2, seed_start=0)

    assert dataset.observations.shape[1] == OBSERVATION_SIZE
    assert dataset.actions.shape[1] == ACTION_SIZE
    assert dataset.observations.dtype == np.float32
    assert dataset.actions.dtype == np.float32
    assert np.all(np.abs(dataset.actions) <= 1.0)
    assert len(dataset.observations) == len(dataset.actions)


def test_stand_demonstrations_keep_the_trigger_below_zero() -> None:
    from qb_sim.engine.actions import Throw
    from qb_sim.engine.entities import BallStatus
    from qb_sim.training.actions import decode_action

    dataset = collect_demonstrations(get_scenario("rushed"), episodes=1, seed_start=0)
    stand_rows = 0
    for action in dataset.actions:
        if float(action[5]) < 0.0:
            stand_rows += 1
            assert not isinstance(decode_action(action, BallStatus.HELD).get("qb"), Throw)
    assert stand_rows > 0


def test_demonstrations_are_reproducible() -> None:
    first = collect_demonstrations(get_scenario("rushed"), episodes=1, seed_start=5)
    second = collect_demonstrations(get_scenario("rushed"), episodes=1, seed_start=5)

    np.testing.assert_allclose(first.observations, second.observations)
    np.testing.assert_allclose(first.actions, second.actions)


def test_behavior_cloning_reduces_the_loss() -> None:
    dataset = collect_demonstrations(get_scenario("rushed"), episodes=3, seed_start=0)

    _, initial = train_behavior_cloning(dataset, epochs=0, seed=0)
    _, final = train_behavior_cloning(dataset, epochs=40, seed=0)

    assert final < initial


def _throw_direction_mse(means: torch.Tensor, actions: np.ndarray) -> float:
    """Normalized-direction MSE between raw policy means and demonstrated aims."""
    throw_rows = torch.as_tensor(actions[:, 5] > 0.0)
    mean_unit = means[:, 2:4] / means[:, 2:4].norm(dim=1, keepdim=True).clamp_min(1e-9)
    action = torch.as_tensor(actions[:, 2:4])
    action_unit = action / action.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return float((mean_unit[throw_rows] - action_unit[throw_rows]).pow(2).mean())


def _trigger_agreement(means: torch.Tensor, actions: np.ndarray) -> float:
    """Fraction of rows where the raw trigger sign matches the demonstration."""
    throw_rows = torch.as_tensor(actions[:, 5] > 0.0)
    return float(((means[:, 5] > 0.0) == throw_rows).float().mean())


def test_warm_start_imitates_the_rule_quarterback() -> None:
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    from qb_sim.training.env import QBPassingEnv

    scenario = get_scenario("rushed")
    dataset = collect_demonstrations(scenario, episodes=5, seed_start=0)
    model = PPO(
        "MlpPolicy",
        QBPassingEnv(scenario),
        n_steps=32,
        batch_size=16,
        seed=0,
        device="cpu",
    )

    loss = warm_start_policy(model, dataset, epochs=150, seed=0)

    assert np.isfinite(loss)
    observations = torch.as_tensor(dataset.observations)
    with torch.no_grad():
        means = model.policy.action_net(model.policy.mlp_extractor.policy_net(observations))
    assert _throw_direction_mse(means, dataset.actions) < 0.01
    assert _trigger_agreement(means, dataset.actions) > 0.9


from qb_sim.training.drive_observations import DRIVE_OBSERVATION_SIZE  # noqa: E402
from qb_sim.training.imitation import collect_drive_demonstrations  # noqa: E402


def test_drive_demonstrations_have_expected_shapes() -> None:
    dataset = collect_drive_demonstrations(get_scenario("rushed"), episodes=1, seed_start=0)

    assert dataset.observations.shape[1] == DRIVE_OBSERVATION_SIZE
    assert dataset.actions.shape[1] == ACTION_SIZE
    assert len(dataset.observations) == len(dataset.actions)
    assert len(dataset.observations) > 0


def test_drive_warm_start_loads_the_behavior_cloning_weights() -> None:
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    from qb_sim.training.drive_env import QBDriveEnv

    scenario = get_scenario("rushed")
    dataset = collect_drive_demonstrations(scenario, episodes=1, seed_start=0)
    network, _ = train_behavior_cloning(dataset, epochs=2, seed=0)
    model = PPO(
        "MlpPolicy",
        QBDriveEnv(scenario),
        n_steps=32,
        batch_size=16,
        seed=0,
        device="cpu",
    )

    warm_start_policy(model, dataset, epochs=2, seed=0)

    observations = torch.as_tensor(dataset.observations[:16])
    with torch.no_grad():
        expected = network(observations)
        loaded = model.policy.action_net(model.policy.mlp_extractor.policy_net(observations))
    assert torch.allclose(expected, loaded, atol=1e-6)


def test_warm_start_targets_raw_actions_not_tanh() -> None:
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    from qb_sim.training.drive_env import QBDriveEnv

    scenario = get_scenario("rushed")
    dataset = collect_drive_demonstrations(scenario, episodes=2, seed_start=0)
    model = PPO(
        "MlpPolicy",
        QBDriveEnv(scenario),
        n_steps=32,
        batch_size=16,
        seed=0,
        device="cpu",
    )

    warm_start_policy(model, dataset, epochs=400, seed=0)

    observations = torch.as_tensor(dataset.observations)
    with torch.no_grad():
        means = model.policy.action_net(model.policy.mlp_extractor.policy_net(observations))
    assert _throw_direction_mse(means, dataset.actions) < 0.01
    assert _trigger_agreement(means, dataset.actions) > 0.9
