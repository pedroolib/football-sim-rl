import numpy as np
import pytest

from qb_sim.agents.rule_controllers import ScramblingQuarterback
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import BallStatus, PlayOutcome
from qb_sim.engine.geometry import vec2
from qb_sim.rules.play import Play


def _cover_receivers(play: Play) -> None:
    play.state.player("cb1").position = play.state.player("wr1").position + vec2(0.5, 0.0)
    play.state.player("cb2").position = play.state.player("wr2").position + vec2(0.5, 0.0)


def test_scrambling_quarterback_runs_when_receivers_are_covered() -> None:
    play = Play()
    _cover_receivers(play)
    play.state.elapsed_seconds = 1.5
    controller = ScramblingQuarterback("qb", play.config)

    action = controller.decide(play.state)

    assert isinstance(action, Move)
    assert action.direction[0] > 0.5
    assert float(np.linalg.norm(action.direction)) == pytest.approx(1.0)


def test_scrambling_quarterback_throws_to_an_open_receiver() -> None:
    play = Play()
    for defender_id in ("cb1", "cb2"):
        play.state.player(defender_id).position = vec2(119.0, 0.0)
    play.state.elapsed_seconds = 1.0
    controller = ScramblingQuarterback("qb", play.config)

    action = controller.decide(play.state)

    assert isinstance(action, Throw)


def test_scrambling_quarterback_throws_when_the_deadline_expires() -> None:
    play = Play()
    _cover_receivers(play)
    play.state.elapsed_seconds = play.config.max_play_duration
    controller = ScramblingQuarterback("qb", play.config)

    action = controller.decide(play.state)

    assert isinstance(action, Throw)


def test_scrambling_quarterback_scrambles_before_the_run_threshold() -> None:
    play = Play()
    _cover_receivers(play)
    play.state.pressure = 1.0
    play.state.elapsed_seconds = 1.0
    controller = ScramblingQuarterback("qb", play.config)

    action = controller.decide(play.state)

    assert isinstance(action, Move)
    assert action.direction[0] < 0.0


def test_scrambling_quarterback_runs_before_scrambling_under_pressure() -> None:
    play = Play()
    _cover_receivers(play)
    play.state.pressure = 1.0
    play.state.elapsed_seconds = 1.5
    controller = ScramblingQuarterback("qb", play.config)

    action = controller.decide(play.state)

    assert isinstance(action, Move)
    assert action.direction[0] > 0.5


def test_scrambling_quarterback_bends_away_from_a_close_defender() -> None:
    play = Play()
    play.state.player("wr2").position = vec2(20.5, 26.5)
    play.state.player("cb2").position = vec2(21.0, 27.0)
    play.state.player("cb1").position = play.state.player("wr1").position + vec2(0.5, 0.0)
    play.state.elapsed_seconds = 1.5
    controller = ScramblingQuarterback("qb", play.config)

    action = controller.decide(play.state)

    assert isinstance(action, Move)
    assert action.direction[1] < 0.0


def test_scrambling_quarterback_steers_away_from_the_sideline() -> None:
    play = Play()
    _cover_receivers(play)
    play.state.player("qb").position = vec2(20.0, 1.0)
    play.state.elapsed_seconds = 1.5
    controller = ScramblingQuarterback("qb", play.config)

    action = controller.decide(play.state)

    assert isinstance(action, Move)
    assert action.direction[1] > 0.0


@pytest.mark.parametrize("elapsed", [0.0, 0.5, 3.0])
def test_scrambling_quarterback_keeps_running_as_the_carrier(elapsed: float) -> None:
    play = Play()
    play.state.ball.status = BallStatus.CARRIED
    play.state.carrier_id = "qb"
    play.state.elapsed_seconds = elapsed
    controller = ScramblingQuarterback("qb", play.config)

    action = controller.decide(play.state)

    assert isinstance(action, Move)
    assert action.direction[0] > 0.5
    assert float(np.linalg.norm(action.direction)) == pytest.approx(1.0)


def test_scrambler_env_drive_produces_a_positive_rush() -> None:
    from qb_sim.rules.scenarios import get_scenario
    from qb_sim.training.actions import encode_engine_action
    from qb_sim.training.drive_env import QBDriveEnv

    scenario = get_scenario("default")
    env = QBDriveEnv(scenario)
    try:
        for seed in range(20):
            env.reset(seed=seed)
            scrambler = ScramblingQuarterback("qb", scenario.config)
            scrambler.set_config(env.active_config)
            terminated = truncated = False
            for _ in range(400):
                engine_action = scrambler.decide(env.play.state)
                action = (
                    encode_engine_action(engine_action)
                    if engine_action is not None
                    else np.zeros(6, dtype=np.float32)
                )
                _, _, terminated, truncated, info = env.step(action)
                rush = next(
                    (
                        play
                        for play in info["plays"]
                        if play["outcome"] == PlayOutcome.RUSH.value
                    ),
                    None,
                )
                if rush is not None and rush["yards"] > 0.0:
                    return
                if terminated or truncated:
                    break
        pytest.fail("the scrambling QB never produced a positive rush")
    finally:
        env.close()


class _RecordingScrambler(ScramblingQuarterback):
    def __init__(self, player_id: str, config) -> None:
        super().__init__(player_id, config)
        self.statuses: list[BallStatus] = []
        self.carriers: list[str | None] = []

    def decide(self, state):
        self.statuses.append(state.ball.status)
        self.carriers.append(state.carrier_id)
        return super().decide(state)


def test_scrambling_quarterback_ignores_a_receiver_carrier() -> None:
    play = Play()
    play.state.ball.status = BallStatus.CARRIED
    play.state.carrier_id = "wr1"
    play.state.elapsed_seconds = 0.5
    controller = ScramblingQuarterback("qb", play.config)

    action = controller.decide(play.state)

    assert action is None


def test_scrambler_demos_do_not_run_while_a_receiver_carries() -> None:
    from qb_sim.rules.scenarios import get_scenario
    from qb_sim.training.imitation import collect_drive_demonstrations

    scenario = get_scenario("rushed")
    scrambler = _RecordingScrambler("qb", scenario.config)

    dataset = collect_drive_demonstrations(
        scenario, episodes=10, seed_start=0, quarterback=scrambler
    )

    receiver_carries = 0
    for status, carrier, action in zip(
        scrambler.statuses, scrambler.carriers, dataset.actions, strict=True
    ):
        if status is BallStatus.CARRIED and carrier not in (None, "qb"):
            receiver_carries += 1
            assert not (action[0] > 0.5 and action[5] < 0.0)
    assert receiver_carries > 0


def test_scrambler_drive_demonstrations_sustain_carried_runs() -> None:
    from qb_sim.rules.scenarios import get_scenario
    from qb_sim.training.imitation import collect_drive_demonstrations

    scenario = get_scenario("default")
    scrambler = _RecordingScrambler("qb", scenario.config)

    dataset = collect_drive_demonstrations(
        scenario, episodes=3, seed_start=0, quarterback=scrambler
    )

    assert len(scrambler.statuses) == len(dataset.actions)
    qb_carries = np.array(
        [
            status is BallStatus.CARRIED and carrier == "qb"
            for status, carrier in zip(
                scrambler.statuses, scrambler.carriers, strict=True
            )
        ],
        dtype=bool,
    )
    forward_runs = (dataset.actions[:, 0] > 0.5) & (dataset.actions[:, 5] < 0.0)
    sustained = qb_carries & forward_runs
    consecutive = sustained[:-1] & sustained[1:]
    assert consecutive.any()


def test_scrambling_quarterback_is_deterministic() -> None:
    play = Play()
    _cover_receivers(play)
    play.state.elapsed_seconds = 1.5
    controller = ScramblingQuarterback("qb", play.config)

    first = controller.decide(play.state)
    second = controller.decide(play.state)

    assert isinstance(first, Move)
    assert isinstance(second, Move)
    assert first.direction.tolist() == second.direction.tolist()
