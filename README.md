# QB Simulation Core

A small, deterministic, headless 2D American football passing simulator for reinforcement-learning research. Milestone 1 contains the simulation core only; it does not depend on Gymnasium or a rendering library.

## Coordinates

Positions are measured in yards. The origin is the lower-left corner of the field, `x` increases downfield toward the opponent's end zone, and `y` increases across the width. The default field bounds are 120 yards long by 53.3 yards wide.

## Setup

```bash
python -m pip install -e ".[dev]"
```

## Checks

```bash
python -m pytest
ruff check .
ruff format --check .
```

## Public API

- `Simulation(config=None, route=RouteName.GO)`: creates one repeatable play with a stationary quarterback, a route-running receiver, and a pursuing defender.
- `Simulation.reset()`: restores the configured starting positions and route progress.
- `Simulation.throw(lead_seconds)`: aims at `receiver.position + receiver.velocity * lead_seconds`; rejects a second throw or a negative/non-finite lead.
- `Simulation.step()`: advances one fixed tick and returns the `PlayState`; terminal states do not advance again.
- `PlayOutcome`: `IN_PROGRESS`, `COMPLETED`, `INTERCEPTION`, `INCOMPLETE`, `OUT_OF_BOUNDS`, or `TIME_LIMIT`.

Routes: `GO` runs straight downfield, `SLANT` breaks early toward field center, and `POST` runs a 10-yard stem before a 45-degree cut toward field center.

## Headless play

```python
from qb_sim import BallStatus, PlayOutcome, RouteName, Simulation

simulation = Simulation(route=RouteName.POST)
state = simulation.reset()
while state.outcome is PlayOutcome.IN_PROGRESS:
    state = simulation.step()
    if state.elapsed_seconds >= 1.0 and state.ball.status is BallStatus.HELD:
        simulation.throw(lead_seconds=0.5)

print(state.outcome, state.ball.position)
```

The simulator is designed to run without a display. Gymnasium integration, agent training, and visualization belong to later milestones.

