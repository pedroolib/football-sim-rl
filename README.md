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

- `Play(config=None, seed=0)`: the mechanical rules engine for one play with a quarterback (`qb`), two receivers (`wr1`, `wr2`), and two defenders (`cb1`, `cb2`). It only applies actions and resolves physics; decisions live in agents. The local seeded NumPy generator drives contextual throw deviation.
- `Play.reset(seed=None)`: restores the roster and starting positions; passing a seed re-seeds the generator, omitting it keeps advancing the sequence so episodes vary.
- `Play.step(actions=None)`: advances one fixed tick applying a `Mapping[str, Move | Throw]` keyed by player id. Actions do not persist between ticks, a missing action means standing still, unknown players and actions not allowed for a role raise `ValueError`, and terminal states do not advance again.
- `Move(direction)`: moves a player in a direction of magnitude at most 1 (speed depends on the role).
- `Throw(direction, power=1.0)`: quarterback-only throw at an angle with a force fraction in `[0, 1]`; `power` maps to a landing range between `throw_min_range` and `throw_max_range`. This `(angle, force)` representation is what a human drag gesture produces and what an RL policy can learn directly.
- `Play.throw_to(target, power=None)` and `Play.throw_with_lead(lead_seconds, player_id="wr1", power=None)` are immediate shortcuts that derive the angle from a field point (and the power from its distance when omitted).
- `PlayState.players` + `PlayState.player(player_id)`: roster state by stable id; `carrier_id` identifies the ball carrier; `pressure` exposes the 0..1 pressure from the nearest defender; `throw_time` records when the ball was released.
- `PlayOutcome`: `IN_PROGRESS`, `COMPLETED`, `INTERCEPTION`, `INCOMPLETE`, `OUT_OF_BOUNDS`, or `TIME_LIMIT`.

## Agents and scenarios

- `agents.base.BaseAgent` is the common interface: one instance per player with `observe(state)`, `act(observation)`, and `decide(state)`. Observations are frozen dataclasses with role-local information only.
- `agents.rule_controllers` provides `ReceiverRouteController`, `ManCoverageController`, and `HeuristicQuarterback`; learned policies will implement the same interface.
- `rules.scenarios.Scenario` bundles config, routes, and a base seed; `default`, `pressured`, and `deep` are available through `get_scenario`.
- `agents.loop.run_play(...)` and `run_series(...)` run reproducible headless plays; `run_play(..., recorder=PlayRecorder())` writes a JSONL trace.

## Recording, replay, and metrics

- `recording.PlayRecorder` writes a JSONL header (seed, scenario, config, routes), one line per tick with per-player actions, and a final result line; `recording.replay(record)` rebuilds the play from the seed and actions and raises `ValueError` if the trace diverges.
- `metrics.summarize(results, line_of_scrimmage)` aggregates completion/interception/incompletion percentages, yards per attempt and per completion, average time to throw, and average duration.
- `python scripts/rule_baseline_report.py` prints reproducible baseline metrics per scenario; `python scripts/accuracy_report.py` covers throw-accuracy calibration.

Play rules: after a catch the ball becomes `CARRIED`, the receiver runs downfield, and the play ends when any defender touches the carrier (`tackle_radius`), the carrier leaves the field, or time expires (still `COMPLETED`). An interception ends the play immediately. There is no automatic receiver selection or catch assistance.

Pass flight is a parabolic arc: `BallState.height` follows `4 · peak · p · (1 − p)` with `peak = min(max_arc_height, distance · arc_ratio)`. A player can only touch the ball when its height is at most `catch_height`, so well-thrown passes can clear defenders; interceptions happen near the launch or landing points or on flat throws.

## Contextual accuracy

Throws leave with a deviation drawn from a local seeded NumPy generator. The angular sigma (configured in degrees) grows with quarterback speed, defensive pressure (nearest defender inside `pressure_radius`), and pass distance; power noise is applied in the normalized `power` range and clipped to `[0, 1]`. Parameters: `throw_base_sigma_degrees`, `throw_move_penalty`, `throw_pressure_penalty`, `throw_distance_penalty`, `throw_power_sigma`. Deviation changes the ball's launch direction and range, so catches, interceptions, and incompletions still come from physics and player interactions — never from an artificial incompletion probability. The aim preview shows the intended throw without noise; the error is drawn only when the throw executes.

Run `python scripts/accuracy_report.py` for a reproducible calibration report over fixed seeds (spread statistics and outcome distributions for planted, moving, and pressured throws).

## Gymnasium environment (M4)

`qb_sim.training.QBPassingEnv` trains the quarterback while receivers and defenders use the M3 rule controllers. Importing `qb_sim.training` registers it as `QBPassing-v0`:

```python
import gymnasium as gym
import qb_sim.training  # noqa: F401 - registers QBPassing-v0

env = gym.make("QBPassing-v0")
```

- **Observation** (27 floats in `[-1, 1]`): QB position/velocity, ball status one-hot + height, relative position and velocity of each receiver and defender, pressure, and remaining time.
- **Action** (6 floats in `[-1, 1]`): `[move_x, move_y, aim_x, aim_y, power_raw, throw_trigger]`; a positive trigger throws toward `(aim_x, aim_y)` with `power = (power_raw + 1) / 2`, otherwise the QB moves. Human and learned policies share the same engine actions.
- **Rewards** (`training.rewards.RewardConfig`): completion plus yardage, penalties for interception/incompletion/out-of-bounds/time limit, and a small per-tick time penalty; every coefficient is configurable.
- **Episodes**: terminate when the play outcome is terminal, truncate at `max_steps`; `reset(seed=...)` is reproducible and `info` exposes the episode seed, outcome, pressure, throw time, and carrier.

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
