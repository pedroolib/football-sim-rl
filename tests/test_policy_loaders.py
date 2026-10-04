from types import SimpleNamespace

import numpy as np
import pytest

sb3 = pytest.importorskip("stable_baselines3")

from qb_sim.rules.drive import Drive, DriveConfig  # noqa: E402
from qb_sim.rules.play import Play  # noqa: E402
from qb_sim.rules.routes import ROUTE_OBSERVATION_ORDER  # noqa: E402
from qb_sim.rules.scenarios import get_scenario  # noqa: E402
from qb_sim.training import policy as policy_module  # noqa: E402
from qb_sim.training.drive_observations import DRIVE_OBSERVATION_SIZE  # noqa: E402
from qb_sim.training.observations import OBSERVATION_SIZE  # noqa: E402
from qb_sim.training.role_context_observations import (  # noqa: E402
    DEFENDER_CONTEXT_OBSERVATION_SIZE,
    RECEIVER_CONTEXT_OBSERVATION_SIZE,
    RECEIVER_DRIVE_OBSERVATION_SIZE,
    receiver_context_observation,
    receiver_drive_observation,
)


class _RecordingModel:
    def __init__(self, size: int, action_size: int = 2) -> None:
        self.observation_space = SimpleNamespace(shape=(size,))
        self.seen: np.ndarray | None = None
        self._action_size = action_size

    def predict(self, observation, deterministic=True):
        self.seen = np.asarray(observation)
        return np.zeros(self._action_size, dtype=np.float32), None


def _patch_load(monkeypatch, size: int):
    stub = SimpleNamespace(observation_space=SimpleNamespace(shape=(size,)))
    monkeypatch.setattr(sb3.PPO, "load", lambda path, device=None: stub)
    return stub


def test_receiver_loader_accepts_legacy_and_current_models(monkeypatch) -> None:
    _patch_load(monkeypatch, 31)
    legacy = policy_module.load_receiver_policy("old.zip", "stage_b", "wr1")
    assert isinstance(legacy, policy_module.ReceiverPolicy)

    _patch_load(monkeypatch, RECEIVER_CONTEXT_OBSERVATION_SIZE)
    current = policy_module.load_receiver_policy("new.zip", "stage_b", "wr1")
    assert isinstance(current, policy_module.ReceiverPolicy)


def test_receiver_loader_accepts_drive_models(monkeypatch) -> None:
    _patch_load(monkeypatch, RECEIVER_DRIVE_OBSERVATION_SIZE)
    drive = policy_module.load_receiver_policy("drive.zip", "stage_b", "wr1")
    assert isinstance(drive, policy_module.ReceiverPolicy)

    _patch_load(monkeypatch, 40)
    legacy_drive = policy_module.load_receiver_policy("drive_old.zip", "stage_b", "wr1")
    assert isinstance(legacy_drive, policy_module.ReceiverPolicy)
    assert legacy_drive.drive_aware is True


def test_receiver_loader_rejects_an_unknown_size(monkeypatch) -> None:
    _patch_load(monkeypatch, 20)
    with pytest.raises(ValueError, match="20"):
        policy_module.load_receiver_policy("odd.zip", "stage_b", "wr1")


def test_defender_loader_accepts_legacy_and_current_models(monkeypatch) -> None:
    _patch_load(monkeypatch, 24)
    legacy = policy_module.load_defender_policy("old.zip", "stage_b", "cb1")
    assert isinstance(legacy, policy_module.DefenderPolicy)

    _patch_load(monkeypatch, DEFENDER_CONTEXT_OBSERVATION_SIZE)
    current = policy_module.load_defender_policy("new.zip", "stage_b", "cb1")
    assert isinstance(current, policy_module.DefenderPolicy)


def test_drive_loader_accepts_legacy_and_current_models(monkeypatch) -> None:
    _patch_load(monkeypatch, 57)
    legacy = policy_module.load_drive_policy("old.zip", "stage_b")
    assert isinstance(legacy, policy_module.DrivePolicyQuarterback)

    _patch_load(monkeypatch, DRIVE_OBSERVATION_SIZE)
    current = policy_module.load_drive_policy("new.zip", "stage_b")
    assert isinstance(current, policy_module.DrivePolicyQuarterback)


def test_play_loader_rejects_a_stale_model(monkeypatch) -> None:
    _patch_load(monkeypatch, 27)
    with pytest.raises(ValueError, match="27"):
        policy_module.load_policy("old.zip", "default")


def test_receiver_policy_slices_for_a_legacy_model() -> None:
    config = get_scenario("stage_b").config
    state = Play(config, seed=0).state
    model = _RecordingModel(31)
    policy = policy_module.ReceiverPolicy(model, config, "wr1")

    policy(state, ROUTE_OBSERVATION_ORDER[0])

    assert model.seen is not None
    assert model.seen.shape == (31,)
    full = receiver_context_observation(state, "wr1", config, ROUTE_OBSERVATION_ORDER[0])
    assert np.allclose(model.seen, full[:31])


def test_receiver_policy_uses_the_drive_layout_for_a_drive_model() -> None:
    config = get_scenario("stage_b").config
    state = Play(config, seed=0).state
    drive_state = Drive(config, DriveConfig(), seed=0).state
    model = _RecordingModel(RECEIVER_DRIVE_OBSERVATION_SIZE)
    policy = policy_module.ReceiverPolicy(model, config, "wr1")

    policy(state, ROUTE_OBSERVATION_ORDER[0], None, drive_state)

    assert policy.drive_aware is True
    assert model.seen is not None
    assert model.seen.shape == (RECEIVER_DRIVE_OBSERVATION_SIZE,)
    expected = receiver_drive_observation(
        state, "wr1", config, ROUTE_OBSERVATION_ORDER[0], drive_state
    )
    assert np.allclose(model.seen, expected)


def test_receiver_policy_slices_for_a_legacy_drive_model() -> None:
    config = get_scenario("stage_b").config
    state = Play(config, seed=0).state
    drive_state = Drive(config, DriveConfig(), seed=0).state
    model = _RecordingModel(40)
    policy = policy_module.ReceiverPolicy(model, config, "wr1")

    policy(state, ROUTE_OBSERVATION_ORDER[0], None, drive_state)

    assert policy.drive_aware is True
    assert model.seen is not None
    assert model.seen.shape == (40,)
    full = receiver_drive_observation(
        state, "wr1", config, ROUTE_OBSERVATION_ORDER[0], drive_state
    )
    assert np.allclose(model.seen, full[:40])


def test_receiver_policy_slices_the_context_layout_for_a_legacy_model() -> None:
    config = get_scenario("stage_b").config
    state = Play(config, seed=0).state
    model = _RecordingModel(35)
    policy = policy_module.ReceiverPolicy(model, config, "wr1")

    policy(state, ROUTE_OBSERVATION_ORDER[0])

    assert policy.drive_aware is False
    assert model.seen is not None
    assert model.seen.shape == (35,)
    full = receiver_context_observation(state, "wr1", config, ROUTE_OBSERVATION_ORDER[0])
    assert np.allclose(model.seen, full[:35])


def test_defender_policy_slices_for_a_legacy_model() -> None:
    config = get_scenario("stage_b").config
    state = Play(config, seed=0).state
    drive_state = Drive(config, DriveConfig(), seed=0).state
    model = _RecordingModel(24)
    policy = policy_module.DefenderPolicy(model, config, "cb1")

    policy(state, "wr1", drive_state)

    assert model.seen is not None
    assert model.seen.shape == (24,)


def test_drive_policy_slices_for_a_legacy_model() -> None:
    config = get_scenario("stage_b").config
    state = Play(config, seed=0).state
    drive_state = Drive(config, DriveConfig(), seed=0).state
    model = _RecordingModel(57, action_size=6)
    policy = policy_module.DrivePolicyQuarterback(model, config)

    policy(state, ("go", "slant"), drive_state)

    assert model.seen is not None
    assert model.seen.shape == (57,)


def test_expected_sizes_match_the_current_builders() -> None:
    assert OBSERVATION_SIZE == 36
    assert DRIVE_OBSERVATION_SIZE == 62
    assert RECEIVER_CONTEXT_OBSERVATION_SIZE == 38
    assert RECEIVER_DRIVE_OBSERVATION_SIZE == 43
    assert DEFENDER_CONTEXT_OBSERVATION_SIZE == 27
    assert policy_module.SUPPORTED_RECEIVER_SIZES == (43, 40, 38, 35, 31)
    assert policy_module.SUPPORTED_DEFENDER_SIZES == (27, 24)
    assert policy_module.SUPPORTED_DRIVE_SIZES == (62, 61, 57)
