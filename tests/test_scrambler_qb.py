import numpy as np
import pytest

from qb_sim.agents.rule_controllers import ScramblingQuarterback
from qb_sim.engine.actions import Move, Throw
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
