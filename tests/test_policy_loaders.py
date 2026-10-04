from types import SimpleNamespace

import pytest

sb3 = pytest.importorskip("stable_baselines3")

from qb_sim.training import policy as policy_module  # noqa: E402
from qb_sim.training.drive_observations import DRIVE_OBSERVATION_SIZE  # noqa: E402
from qb_sim.training.observations import OBSERVATION_SIZE  # noqa: E402
from qb_sim.training.role_context_observations import (  # noqa: E402
    DEFENDER_CONTEXT_OBSERVATION_SIZE,
    RECEIVER_CONTEXT_OBSERVATION_SIZE,
)


def _patch_load(monkeypatch, size: int):
    stub = SimpleNamespace(observation_space=SimpleNamespace(shape=(size,)))
    monkeypatch.setattr(sb3.PPO, "load", lambda path, device=None: stub)
    return stub


def test_receiver_loader_rejects_a_stale_model(monkeypatch) -> None:
    _patch_load(monkeypatch, 31)
    with pytest.raises(ValueError, match="31"):
        policy_module.load_receiver_policy("old.zip", "stage_b", "wr1")


def test_receiver_loader_accepts_the_current_observation(monkeypatch) -> None:
    _patch_load(monkeypatch, RECEIVER_CONTEXT_OBSERVATION_SIZE)
    policy = policy_module.load_receiver_policy("new.zip", "stage_b", "wr1")
    assert isinstance(policy, policy_module.ReceiverPolicy)


def test_defender_loader_rejects_a_stale_model(monkeypatch) -> None:
    _patch_load(monkeypatch, 24)
    with pytest.raises(ValueError, match="24"):
        policy_module.load_defender_policy("old.zip", "stage_b", "cb1")


def test_drive_loader_rejects_a_stale_model(monkeypatch) -> None:
    _patch_load(monkeypatch, 57)
    with pytest.raises(ValueError, match="57"):
        policy_module.load_drive_policy("old.zip", "stage_b")


def test_play_loader_rejects_a_stale_model(monkeypatch) -> None:
    _patch_load(monkeypatch, 27)
    with pytest.raises(ValueError, match="27"):
        policy_module.load_policy("old.zip", "default")


def test_expected_sizes_match_the_current_builders() -> None:
    assert OBSERVATION_SIZE == 36
    assert DRIVE_OBSERVATION_SIZE == 61
    assert RECEIVER_CONTEXT_OBSERVATION_SIZE == 35
    assert DEFENDER_CONTEXT_OBSERVATION_SIZE == 27
