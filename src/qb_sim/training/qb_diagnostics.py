"""Throw-level diagnostics for drive quarterbacks (Fase 0).

Pure instrumentation on top of the existing drive environment: it records where
and when the QB threw, how open the chosen receiver was, and against which
coverage, without changing engine or rules behavior.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from qb_sim.config import SimulationConfig
from qb_sim.engine.entities import PlayerState, PlayOutcome, PlayState, Role

if TYPE_CHECKING:
    from qb_sim.training.drive_env import QBDriveEnv

ActionFn = Callable[["QBDriveEnv", np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class ThrowRecord:
    """Everything observed at the exact tick the quarterback released the ball."""

    play_index: int
    down: int
    distance: float
    throw_time: float
    qb_speed: float
    throw_distance: float
    target_id: str
    target_advantage: float
    best_advantage: float
    routes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InstrumentedDrive:
    """One drive outcome plus its throw-level records and play history."""

    outcome: str
    points: float
    first_downs: int
    records: tuple[ThrowRecord, ...]
    plays: tuple[dict, ...]


def _active_config(env: "QBDriveEnv") -> SimulationConfig:
    config = getattr(env, "config", None)
    if config is not None:
        return config
    config = getattr(env, "_active_config", None)
    if config is not None:
        return config
    return env.scenario.config


def _player_speed(player: PlayerState, config: SimulationConfig) -> float:
    """Roster capability speed, falling back to the config value by role."""
    if player.max_speed > 0.0:
        return player.max_speed
    if player.role is Role.QUARTERBACK:
        return config.quarterback_speed
    if player.role is Role.RECEIVER:
        return config.receiver_speed
    return config.defender_speed


def _throw_record(
    env: "QBDriveEnv",
    state: PlayState,
    config: SimulationConfig,
    play_index: int,
) -> ThrowRecord | None:
    ball = state.ball
    if ball.target is None:
        return None
    receivers = [player for player in state.players if player.role is Role.RECEIVER]
    defenders = [player for player in state.players if player.role is Role.DEFENDER]
    if not receivers:
        return None

    advantages: dict[str, float] = {}
    for receiver in receivers:
        if defenders:
            nearest = min(
                defenders,
                key=lambda defender: float(
                    np.linalg.norm(defender.position - receiver.position)
                ),
            )
            advantages[receiver.player_id] = _player_speed(
                receiver, config
            ) - _player_speed(nearest, config)
        else:
            advantages[receiver.player_id] = 0.0

    target = min(
        receivers,
        key=lambda receiver: float(np.linalg.norm(receiver.position - ball.target)),
    )
    quarterback = state.player("qb")
    return ThrowRecord(
        play_index=play_index,
        down=int(env.drive.state.down),
        distance=float(env.drive.state.distance),
        throw_time=float(state.throw_time) if state.throw_time is not None else 0.0,
        qb_speed=float(np.linalg.norm(quarterback.velocity)),
        throw_distance=float(np.linalg.norm(ball.target - quarterback.position)),
        target_id=target.player_id,
        target_advantage=float(advantages[target.player_id]),
        best_advantage=float(max(0.0, max(advantages.values()))),
        routes=tuple(route.value for route in env.drive.routes),
    )


def run_instrumented_drive(
    env: "QBDriveEnv",
    action_fn: ActionFn,
    seed: int,
) -> InstrumentedDrive:
    """Run one seeded drive, recording each throw at the tick it is released."""
    observation, _ = env.reset(seed=seed)
    config = _active_config(env)
    recorded_plays: set[int] = set()
    records: list[ThrowRecord] = []
    terminated = truncated = False
    while not (terminated or truncated):
        action = np.asarray(action_fn(env, observation), dtype=np.float32)
        observation, _, terminated, truncated, _ = env.step(action)
        state = env.play.state
        play_index = int(env.drive.state.play_number)
        if (
            state.outcome is PlayOutcome.IN_PROGRESS
            and state.throw_time is not None
            and play_index not in recorded_plays
        ):
            record = _throw_record(env, state, config, play_index)
            if record is not None:
                records.append(record)
                recorded_plays.add(play_index)

    drive_state = env.drive.state
    return InstrumentedDrive(
        outcome=drive_state.end_reason or "truncated",
        points=float(drive_state.points),
        first_downs=int(drive_state.first_downs),
        records=tuple(records),
        plays=tuple(env.play_history),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize_diagnostics(drives: list[InstrumentedDrive]) -> dict[str, Any]:
    """Aggregate drive outcomes, play results, and throw decisions."""
    total = len(drives)
    if total == 0:
        raise ValueError("at least one drive is required")

    def pct(count: int, denominator: int) -> float:
        return 100.0 * count / denominator if denominator else 0.0

    touchdowns = sum(drive.outcome == "touchdown" for drive in drives)
    interceptions = sum(drive.outcome == "interception" for drive in drives)
    turnovers = sum(drive.outcome == "turnover_on_downs" for drive in drives)
    plays = [play for drive in drives for play in drive.plays]
    records = [record for drive in drives for record in drive.records]
    throw_times = [record.throw_time for record in records]
    qb_speeds = [record.qb_speed for record in records]
    throw_distances = [record.throw_distance for record in records]
    best_advantages = [record.best_advantage for record in records]
    target_advantages = [record.target_advantage for record in records]
    decision_gaps = [
        record.best_advantage - record.target_advantage for record in records
    ]
    deep_open = sum(
        record.throw_distance >= 15.0 and record.target_advantage > 0.0
        for record in records
    )

    return {
        "drives": float(total),
        "touchdown_pct": pct(touchdowns, total),
        "interception_pct": pct(interceptions, total),
        "turnover_on_downs_pct": pct(turnovers, total),
        "points_per_drive": sum(drive.points for drive in drives) / total,
        "first_downs_per_drive": sum(drive.first_downs for drive in drives) / total,
        "plays_per_drive": len(plays) / total,
        "sack_pct": pct(sum(bool(play["sack"]) for play in plays), len(plays)),
        "coverage_sack_pct": pct(
            sum(bool(play["coverage_sack"]) for play in plays), len(plays)
        ),
        "incomplete_pct": pct(
            sum(play["outcome"] == "incomplete" for play in plays), len(plays)
        ),
        "throw_rate_pct": pct(
            sum(play["throw_distance"] is not None for play in plays), len(plays)
        ),
        "mean_throw_time": _mean(throw_times),
        "median_throw_time": float(np.median(throw_times)) if throw_times else 0.0,
        "moving_throw_pct": pct(sum(speed > 0.5 for speed in qb_speeds), len(records)),
        "mean_qb_speed": _mean(qb_speeds),
        "mean_throw_distance": _mean(throw_distances),
        "mean_best_advantage": _mean(best_advantages),
        "mean_target_advantage": _mean(target_advantages),
        "decision_gap": _mean(decision_gaps),
        "pct_target_max_advantage": pct(
            sum(gap < 0.01 for gap in decision_gaps), len(records)
        ),
        "pct_deep_open_throws": pct(deep_open, len(records)),
        "depth_advantage_corr": _correlation(throw_distances, best_advantages),
    }
