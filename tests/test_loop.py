import numpy as np
import pytest

from qb_sim.agents.loop import build_rule_agents, run_play, run_series
from qb_sim.engine.entities import PlayOutcome
from qb_sim.rules.play import Play
from qb_sim.rules.scenarios import DEFAULT_SCENARIO


def test_build_rule_agents_creates_one_agent_per_player() -> None:
    play = Play()

    agents = build_rule_agents(play, DEFAULT_SCENARIO.receiver_routes)

    assert set(agents) == {"qb", "wr1", "wr2", "cb1", "cb2"}


def test_build_rule_agents_can_exclude_the_quarterback() -> None:
    play = Play()

    agents = build_rule_agents(
        play,
        DEFAULT_SCENARIO.receiver_routes,
        include_quarterback=False,
    )

    assert "qb" not in agents


def test_run_play_reaches_a_terminal_outcome() -> None:
    state = run_play(seed=0)

    assert state.outcome is not PlayOutcome.IN_PROGRESS


def test_run_play_resets_provided_controllers() -> None:
    play = Play()
    agents = build_rule_agents(play, DEFAULT_SCENARIO.receiver_routes)

    first = run_play(controllers=agents, seed=5)
    second = run_play(controllers=agents, seed=5)

    assert first.outcome is second.outcome
    np.testing.assert_allclose(first.ball.position, second.ball.position)


def test_run_series_is_reproducible_with_the_same_seeds() -> None:
    first = run_series(DEFAULT_SCENARIO, count=3, seed_start=10)
    second = run_series(DEFAULT_SCENARIO, count=3, seed_start=10)

    assert [state.outcome for state in first] == [state.outcome for state in second]
    for left, right in zip(first, second, strict=True):
        np.testing.assert_allclose(left.ball.position, right.ball.position)


def test_run_series_rejects_non_positive_count() -> None:
    with pytest.raises(ValueError):
        run_series(count=0)
