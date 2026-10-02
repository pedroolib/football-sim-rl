from qb_sim.agents.base import BaseAgent
from qb_sim.engine.actions import Move, Throw, allowed_actions
from qb_sim.engine.entities import PlayState, Role
from qb_sim.engine.geometry import vec2
from qb_sim.rules.play import Play


class _RecordingAgent(BaseAgent):
    role = Role.RECEIVER

    def __init__(self) -> None:
        super().__init__("wr1")
        self.seen: PlayState | None = None

    def observe(self, state: PlayState) -> str:
        self.seen = state
        return "observation"

    def act(self, observation: str) -> Move:
        assert observation == "observation"
        return Move(direction=vec2(1.0, 0.0))


def test_allowed_actions_depend_on_role() -> None:
    assert allowed_actions(Role.QUARTERBACK) == (Move, Throw)
    assert allowed_actions(Role.RECEIVER) == (Move,)
    assert allowed_actions(Role.DEFENDER) == (Move,)


def test_decide_composes_observe_and_act() -> None:
    play = Play()
    agent = _RecordingAgent()

    action = agent.decide(play.state)

    assert agent.seen is play.state
    assert isinstance(action, Move)
