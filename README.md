# QB Simulation Core

A small, deterministic, headless-first 2D American football passing simulator for reinforcement-learning research. The current prototype (M2) is playable by a human: 1 quarterback, 2 receivers running scripted routes, and 2 defenders in man coverage.

## Coordinates

Positions are measured in yards. The origin is the lower-left corner of the field, `x` increases downfield toward the opponent's end zone, and `y` increases across the width. The default field bounds are 120 yards long by 53.3 yards wide.

## Setup

```bash
python -m pip install -e ".[dev]"

# Optional visualizer. `pygame` has no wheel for Python 3.14, so the project uses
# the API-compatible `pygame-ce`.
python -m pip install -e ".[dev,viz]"
```

## Checks

```bash
python -m pytest
ruff check .
ruff format --check .
```

## Public API

- `Play(config=None, routes=(RouteName.POST, RouteName.SLANT))`: one repeatable play with a quarterback (`qb`), two receivers (`wr1`, `wr2`) with one route each, and two defenders (`cb1`, `cb2`) covering the receiver with the same index.
- `Play.reset()`: restores the roster, starting positions, and route progress.
- `Play.step(qb_action=None)`: advances one fixed tick and returns the `PlayState`; actions do not persist between ticks and terminal states do not advance again.
- `Move(direction)`: moves the quarterback in a direction of magnitude at most 1.
- `Throw(direction, power=1.0)`: throws at an angle with a force fraction in `[0, 1]`; `power` maps to a landing range between `throw_min_range` and `throw_max_range`. This `(angle, force)` representation is what a human drag gesture produces and what an RL policy can learn directly.
- `Play.throw_to(target, power=None)` and `Play.throw_with_lead(lead_seconds, player_id="wr1", power=None)` are immediate shortcuts that derive the angle from a field point (and the power from its distance when omitted).
- `PlayState.players` + `PlayState.player(player_id)`: roster state by stable id; `carrier_id` identifies the ball carrier.
- `PlayOutcome`: `IN_PROGRESS`, `COMPLETED`, `INTERCEPTION`, `INCOMPLETE`, `OUT_OF_BOUNDS`, or `TIME_LIMIT`.

Play rules: after a catch the ball becomes `CARRIED`, the receiver runs downfield, and the play ends when any defender touches the carrier (`tackle_radius`), the carrier leaves the field, or time expires (still `COMPLETED`). An interception ends the play immediately. There is no automatic receiver selection or catch assistance.

## Controls (visualizer)

Drag-and-throw, Retro Bowl style (slingshot):

| Input | Action |
| --- | --- |
| `WASD` or arrow keys | Move the quarterback |
| Hold left mouse and pull back | Aim: pull away from where you want the ball to go; drag length is the power |
| Release left mouse | Throw to the landing point shown by the dotted line |
| `Esc` or right click | Cancel the aim without throwing |
| `P` | Pause / resume |
| `R` | Restart the play |

The dotted aim line and landing marker update in real time and change color with power; a drag shorter than `throw_min_drag` is ignored to avoid accidental throws. Human input is translated into the same `Move`/`Throw` actions used by scripted and future learned policies; the visual layer contains no game logic.

## Headless session

`PlaySession` drives a play from translated human input without importing Pygame:

```python
from qb_sim import Play
from qb_sim.engine.geometry import vec2
from qb_sim.ui.session import FrameInput, PlaySession

session = PlaySession(Play())
session.step_frame(FrameInput(held=frozenset({"w"})))
session.step_frame(FrameInput(mouse_field=vec2(50.0, 26.65), mouse_down=True))
session.step_frame(FrameInput(mouse_field=vec2(30.0, 26.65), mouse_down=True))
session.step_frame(FrameInput(mouse_field=vec2(30.0, 26.65), mouse_released=True))

print(session.play.state.outcome, session.play.state.ball.position)
```

## Visualizer

```bash
python -m qb_sim.ui.pygame_app
```

The simulator itself never depends on a display. Gymnasium integration, agent training, and multi-agent play belong to later milestones.
