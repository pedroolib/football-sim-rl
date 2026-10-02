"""Aggregate metrics for comparing rule-based and learned policies."""

from collections.abc import Sequence

from qb_sim.engine.entities import PlayOutcome, PlayState


def summarize(results: Sequence[PlayState], line_of_scrimmage: float) -> dict[str, float]:
    """Aggregate outcomes, yards, and timing over a series of plays."""
    if not results:
        raise ValueError("at least one result is required")

    total = len(results)
    completed = [state for state in results if state.outcome is PlayOutcome.COMPLETED]
    yards = [_yards(state, line_of_scrimmage) for state in results]
    completion_yards = [_yards(state, line_of_scrimmage) for state in completed]
    throw_times = [state.throw_time for state in results if state.throw_time is not None]

    return {
        "plays": float(total),
        "completion_pct": 100.0 * len(completed) / total,
        "interception_pct": 100.0 * _count(results, PlayOutcome.INTERCEPTION) / total,
        "incompletion_pct": 100.0 * _count(results, PlayOutcome.INCOMPLETE) / total,
        "out_of_bounds_pct": 100.0 * _count(results, PlayOutcome.OUT_OF_BOUNDS) / total,
        "time_limit_pct": 100.0 * _count(results, PlayOutcome.TIME_LIMIT) / total,
        "sack_pct": 100.0 * _count(results, PlayOutcome.SACK) / total,
        "yards_per_attempt": sum(yards) / total,
        "yards_per_completion": (sum(completion_yards) / len(completed)) if completed else 0.0,
        "average_time_to_throw": (sum(throw_times) / len(throw_times)) if throw_times else 0.0,
        "average_duration": sum(state.elapsed_seconds for state in results) / total,
    }


def _count(results: Sequence[PlayState], outcome: PlayOutcome) -> int:
    return sum(state.outcome is outcome for state in results)


def _yards(state: PlayState, line_of_scrimmage: float) -> float:
    if state.carrier_id is None:
        return 0.0
    return float(state.player(state.carrier_id).position[0] - line_of_scrimmage)
