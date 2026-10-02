import subprocess
import sys

import numpy as np

from qb_sim.engine.actions import Throw
from qb_sim.engine.entities import BallStatus
from qb_sim.rules.play import Play
from qb_sim.training.policy import PolicyQuarterback
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
