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

- `Play(config=None, route=RouteName.GO)`: one repeatable play with a quarterback, a route-running receiver, and a pursuing defender.
- `Play.reset()`: restores the configured starting positions and route progress.
- `Play.step(qb_action=None)`: advances one fixed tick and returns the `PlayState`; actions do not persist between ticks and terminal states do not advance again.
- `Move(direction)`: moves the quarterback in a direction of magnitude at most 1 (magnitude 0 stands still).
- `Throw(target)`: throws the ball toward a field-coordinate point. `Play.throw_to(target)` and `Play.throw_with_lead(lead_seconds)` are immediate shortcuts that share the same logic.
- `PlayOutcome`: `IN_PROGRESS`, `COMPLETED`, `INTERCEPTION`, `INCOMPLETE`, `OUT_OF_BOUNDS`, or `TIME_LIMIT`.

Play rules: after a catch the ball becomes `CARRIED`, the receiver runs downfield, and the play ends when the defender touches the carrier (`tackle_radius`), the carrier leaves the field, or time expires (still `COMPLETED`). An interception ends the play immediately.

Routes: `GO` runs straight downfield, `SLANT` breaks early toward field center, and `POST` runs a 10-yard stem before a 45-degree cut toward field center.

## Headless play

```python
from qb_sim import BallStatus, Play, PlayOutcome, RouteName, Throw, vec2

play = Play(route=RouteName.POST)
state = play.reset()
while state.outcome is PlayOutcome.IN_PROGRESS:
    action = None
    if state.ball.status is BallStatus.HELD and state.elapsed_seconds >= 1.0:
        action = Throw(target=vec2(50.0, 26.65))
    state = play.step(qb_action=action)

print(state.outcome, state.ball.position)
```

The simulator is designed to run without a display. Gymnasium integration, agent training, and visualization belong to later milestones.
