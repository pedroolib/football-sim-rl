import pytest

from qb_sim.engine.entities import PlayOutcome, PlayState
from qb_sim.engine.geometry import vec2
from qb_sim.metrics import summarize
from qb_sim.rules.play import Play


def _state(
    outcome: PlayOutcome,
    *,
    carrier_x: float | None = None,
    throw_time: float | None = None,
    elapsed: float = 1.0,
) -> PlayState:
    play = Play()
    state = play.state
    state.outcome = outcome
    state.elapsed_seconds = elapsed
    state.throw_time = throw_time
    if carrier_x is not None:
        state.carrier_id = "wr1"
        state.player("wr1").position = vec2(carrier_x, 26.65)
    return state


def test_summarize_computes_outcome_percentages() -> None:
    results = [
        _state(PlayOutcome.COMPLETED, carrier_x=30.0, throw_time=1.0),
        _state(PlayOutcome.INCOMPLETE, throw_time=1.5),
        _state(PlayOutcome.INTERCEPTION, throw_time=0.8),
        _state(PlayOutcome.COMPLETED, carrier_x=25.0, throw_time=1.2),
    ]

    summary = summarize(results, line_of_scrimmage=20.0)

    assert summary["plays"] == 4
    assert summary["completion_pct"] == pytest.approx(50.0)
    assert summary["interception_pct"] == pytest.approx(25.0)
    assert summary["incompletion_pct"] == pytest.approx(25.0)


def test_summarize_computes_yards_and_throw_time() -> None:
    results = [
        _state(PlayOutcome.COMPLETED, carrier_x=30.0, throw_time=1.0),
        _state(PlayOutcome.COMPLETED, carrier_x=25.0, throw_time=1.5),
        _state(PlayOutcome.INCOMPLETE, throw_time=2.0),
    ]

    summary = summarize(results, line_of_scrimmage=20.0)

    assert summary["yards_per_attempt"] == pytest.approx(5.0)
    assert summary["yards_per_completion"] == pytest.approx(7.5)
    assert summary["average_time_to_throw"] == pytest.approx(1.5)
    assert summary["average_duration"] == pytest.approx(1.0)


def test_summarize_requires_at_least_one_result() -> None:
    with pytest.raises(ValueError):
        summarize([], line_of_scrimmage=20.0)
