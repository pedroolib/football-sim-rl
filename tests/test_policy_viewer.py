import subprocess
import sys

import numpy as np
import pytest

from qb_sim.engine.actions import Throw
from qb_sim.engine.entities import BallStatus
from qb_sim.engine.geometry import vec2
from qb_sim.rules.drive import DriveState
from qb_sim.rules.play import Play
from qb_sim.rules.routes import RouteName
from qb_sim.training.policy import (
    DefenderPolicy,
    DrivePolicyQuarterback,
    PolicyQuarterback,
    ReceiverPolicy,
)
from qb_sim.ui.session import FrameInput, PlaySession


class _StubModel:
    def __init__(self, action: list[float]) -> None:
        self._action = np.array(action, dtype=np.float32)

    def predict(self, observation, deterministic: bool = True):
        return self._action, None


def test_policy_quarterback_maps_the_action_to_the_engine() -> None:
    play = Play()
    policy = PolicyQuarterback(_StubModel([0.0, 0.0, 1.0, 0.0, 1.0, 1.0]), play.config)

    action = policy(play.state)

    assert isinstance(action, Throw)


def test_drive_policy_quarterback_maps_the_action_to_the_engine() -> None:
    play = Play()
    policy = DrivePolicyQuarterback(_StubModel([0.0, 0.0, 1.0, 0.0, 1.0, 1.0]), play.config)

    action = policy(play.state, (RouteName.POST, RouteName.SLANT), DriveState(los=20.0))

    assert isinstance(action, Throw)


def test_session_uses_the_policy_instead_of_human_input() -> None:
    play = Play()
    policy = PolicyQuarterback(_StubModel([0.0, 0.0, 1.0, 0.0, 1.0, 1.0]), play.config)
    session = PlaySession(play, quarterback_policy=policy)

    session.step_frame(FrameInput())

    assert play.state.ball.status is BallStatus.IN_FLIGHT


def test_importing_the_policy_module_does_not_import_torch() -> None:
    code = (
        "import sys; import qb_sim.training.policy; "
        "assert 'torch' not in sys.modules; assert 'stable_baselines3' not in sys.modules"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


class _RecordingModel:
    def __init__(self, action: list[float]) -> None:
        self._action = np.array(action, dtype=np.float32)
        self.observation_shape: tuple[int, ...] | None = None
        self.observation: np.ndarray | None = None

    def predict(self, observation, deterministic: bool = True):
        self.observation = np.array(observation, dtype=np.float32, copy=True)
        self.observation_shape = tuple(self.observation.shape)
        return self._action, None


def test_receiver_policy_builds_the_31_float_observation() -> None:
    play = Play()
    model = _RecordingModel([1.0, 0.0])
    policy = ReceiverPolicy(model, play.config, "wr1")

    action = policy(play.state, RouteName.POST)

    assert model.observation_shape == (31,)
    assert action is not None
    assert action.direction.tolist() == pytest.approx([1.0, 0.0])


def test_receiver_policy_forwards_the_start_override() -> None:
    play = Play()
    play.state.player("wr1").position = vec2(45.0, 43.0)
    model = _RecordingModel([1.0, 0.0])
    policy = ReceiverPolicy(model, play.config, "wr1")

    policy(play.state, RouteName.GO, vec2(40.0, 43.0))

    assert model.observation is not None
    assert model.observation[28] == pytest.approx(25.0 / 120.0)
    assert model.observation[30] == pytest.approx(1.0)


def test_defender_policy_builds_the_24_float_observation() -> None:
    play = Play()
    model = _RecordingModel([0.0, 1.0])
    policy = DefenderPolicy(model, play.config, "cb1")

    action = policy(play.state, "wr1", DriveState(los=40.0))

    assert model.observation_shape == (24,)
    assert action is not None
    assert action.direction.tolist() == pytest.approx([0.0, 1.0])
