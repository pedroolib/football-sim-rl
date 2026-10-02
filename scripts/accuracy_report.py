"""Reproducible calibration report for contextual throw accuracy.

Run from the repository root:

    python scripts/accuracy_report.py
"""

from dataclasses import replace

import numpy as np

from qb_sim.config import SimulationConfig
from qb_sim.engine.actions import Move
from qb_sim.engine.entities import PlayOutcome
from qb_sim.engine.geometry import distance, vec2
from qb_sim.rules.accuracy import throw_spread
from qb_sim.rules.play import Play

SEEDS = range(200)
INTENDED_DIRECTION = vec2(1.0, 0.0)
PRESSURE_DEFENDERS = ((18.5, 26.65), (0.0, 0.0))


def _angle_error_degrees(play: Play) -> float:
    velocity = play.state.ball.velocity
    speed = float(np.linalg.norm(velocity))
    unit = velocity / speed
    cross = INTENDED_DIRECTION[0] * unit[1] - INTENDED_DIRECTION[1] * unit[0]
    dot = float(np.dot(INTENDED_DIRECTION, unit))
    return abs(float(np.degrees(np.arctan2(cross, dot))))


def angle_stats(
    label: str,
    config: SimulationConfig,
    *,
    moving: bool,
    pressured: bool,
    target: float,
) -> None:
    scenario = config
    if pressured:
        scenario = replace(config, defender_starts=PRESSURE_DEFENDERS)
    errors = []
    for seed in SEEDS:
        play = Play(scenario, seed=seed)
        if moving:
            play.step(qb_action=Move(direction=vec2(1.0, 0.0)))
        play.throw_to(vec2(target, 26.65))
        errors.append(_angle_error_degrees(play))
    print(f"{label:<28} media={np.mean(errors):6.3f}°  p95={np.percentile(errors, 95):6.3f}°")


def outcome_distribution(
    label: str,
    config: SimulationConfig,
    *,
    moving: bool,
    pressured: bool,
) -> None:
    scenario = config
    if pressured:
        scenario = replace(config, defender_starts=PRESSURE_DEFENDERS)
    counts: dict[str, int] = {}
    for seed in SEEDS:
        play = Play(scenario, seed=seed)
        if moving:
            play.step(qb_action=Move(direction=vec2(1.0, 0.0)))
        play.step()
        quarterback = play.state.player("qb")
        receiver = play.state.player("wr1")
        lead = distance(quarterback.position, receiver.position) / config.ball_speed
        play.throw_with_lead(lead)
        state = play.state
        for _ in range(200):
            state = play.step()
            if state.outcome is not PlayOutcome.IN_PROGRESS:
                break
        counts[state.outcome.value] = counts.get(state.outcome.value, 0) + 1
    summary = "  ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    print(f"{label:<28} {summary}")


def main() -> None:
    config = SimulationConfig()
    print("Desviación angular (media y p95 sobre 200 semillas fijas)")
    angle_stats("plantado", config, moving=False, pressured=False, target=60.0)
    angle_stats("en movimiento", config, moving=True, pressured=False, target=60.0)
    angle_stats("bajo presión", config, moving=False, pressured=True, target=60.0)
    angle_stats("pase corto", config, moving=False, pressured=False, target=30.0)
    print()
    print("Sigmas del modelo")
    for label, speed, pressure, distance_yards in (
        ("plantado 10 yd", 0.0, 0.0, 10.0),
        ("movimiento 10 yd", config.quarterback_speed, 0.0, 10.0),
        ("presión 10 yd", 0.0, 1.0, 10.0),
        ("distancia 45 yd", 0.0, 0.0, 45.0),
    ):
        angle_sigma, power_sigma = throw_spread(speed, pressure, distance_yards, config)
        print(f"{label:<28} ángulo={np.degrees(angle_sigma):5.2f}°  potencia={power_sigma:5.4f}")
    print()
    print("Distribución de resultados (pase con lead a wr1, 200 semillas)")
    outcome_distribution("plantado", config, moving=False, pressured=False)
    outcome_distribution("en movimiento", config, moving=True, pressured=False)
    outcome_distribution("bajo presión", config, moving=False, pressured=True)


if __name__ == "__main__":
    main()
