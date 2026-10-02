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

- `Play(config=None, routes=(RouteName.POST, RouteName.SLANT), seed=0)`: one repeatable play with a quarterback (`qb`), two receivers (`wr1`, `wr2`) with one route each, and two defenders (`cb1`, `cb2`) covering the receiver with the same index. The local seeded NumPy generator drives contextual throw deviation.
- `Play.reset(seed=None)`: restores the roster, starting positions, and route progress; passing a seed re-seeds the generator, omitting it keeps advancing the sequence so episodes vary.
- `Play.step(qb_action=None)`: advances one fixed tick and returns the `PlayState`; actions do not persist between ticks and terminal states do not advance again.
- `Move(direction)`: moves the quarterback in a direction of magnitude at most 1.
- `Throw(direction, power=1.0)`: throws at an angle with a force fraction in `[0, 1]`; `power` maps to a landing range between `throw_min_range` and `throw_max_range`. This `(angle, force)` representation is what a human drag gesture produces and what an RL policy can learn directly.
- `Play.throw_to(target, power=None)` and `Play.throw_with_lead(lead_seconds, player_id="wr1", power=None)` are immediate shortcuts that derive the angle from a field point (and the power from its distance when omitted).
- `PlayState.players` + `PlayState.player(player_id)`: roster state by stable id; `carrier_id` identifies the ball carrier; `pressure` exposes the 0..1 pressure from the nearest defender.
- `PlayOutcome`: `IN_PROGRESS`, `COMPLETED`, `INTERCEPTION`, `INCOMPLETE`, `OUT_OF_BOUNDS`, or `TIME_LIMIT`.

Play rules: after a catch the ball becomes `CARRIED`, the receiver runs downfield, and the play ends when any defender touches the carrier (`tackle_radius`), the carrier leaves the field, or time expires (still `COMPLETED`). An interception ends the play immediately. There is no automatic receiver selection or catch assistance.

Pass flight is a parabolic arc: `BallState.height` follows `4 · peak · p · (1 − p)` with `peak = min(max_arc_height, distance · arc_ratio)`. A player can only touch the ball when its height is at most `catch_height`, so well-thrown passes can clear defenders; interceptions happen near the launch or landing points or on flat throws.

## Contextual accuracy

Throws leave with a deviation drawn from a local seeded NumPy generator. The angular sigma (configured in degrees) grows with quarterback speed, defensive pressure (nearest defender inside `pressure_radius`), and pass distance; power noise is applied in the normalized `power` range and clipped to `[0, 1]`. Parameters: `throw_base_sigma_degrees`, `throw_move_penalty`, `throw_pressure_penalty`, `throw_distance_penalty`, `throw_power_sigma`. Deviation changes the ball's launch direction and range, so catches, interceptions, and incompletions still come from physics and player interactions — never from an artificial incompletion probability. The aim preview shows the intended throw without noise; the error is drawn only when the throw executes.

Run `python scripts/accuracy_report.py` for a reproducible calibration report over fixed seeds (spread statistics and outcome distributions for planted, moving, and pressured throws).

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

The dotted aim line is a parabola sampled from the same arc the ball will fly, and its landing marker updates in real time with a color that changes with power; a drag shorter than `throw_min_drag` is ignored to avoid accidental throws. Human input is translated into the same `Move`/`Throw` actions used by scripted and future learned policies; the visual layer contains no game logic.

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
