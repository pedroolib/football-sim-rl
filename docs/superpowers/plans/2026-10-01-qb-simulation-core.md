# QB Simulation Core Implementation Plan

> **For agentic workers:** Use the inline `executing-plans` workflow to implement this plan task by task. Steps use checkbox syntax for tracking.

**Goal:** Build a deterministic, headless 2D passing simulation core with a stationary quarterback, a route-running receiver, a pursuing defender, and Go, Slant, and Post routes.

**Architecture:** A small `src/qb_sim` package separates configuration, geometry, entities, route progression, physics, and orchestration. The core exposes simulation operations directly and has no Gymnasium or rendering dependency.

**Tech Stack:** Python 3.11+, NumPy, pytest, Ruff.

**Spec:** [`agents.md`](../../../agents.md), Milestone 1 and the approved scope in this conversation.

## Global Constraints

- Use Python 3.11+ and NumPy for numerical calculations.
- Keep the simulation core independent of Gymnasium and rendering libraries.
- Use fixed simulation time steps and deterministic updates.
- Include one quarterback, one wide receiver, one cornerback, and one football.
- Implement Go, Slant, and Post routes; Post breaks after 10 yards at 45 degrees toward field center.
- Do not implement RL observations/actions, rewards, training, baselines, or rendering in this milestone.
- Use type hints, pytest, and Ruff.

---

### Task 1: Package setup and geometry primitives

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/qb_sim/__init__.py`
- Create: `src/qb_sim/geometry.py`
- Create: `tests/test_geometry.py`

**Interfaces:**
- `vec2(x: float, y: float) -> Vector2` returns an independent NumPy vector of shape `(2,)`.
- `distance(a: Vector2, b: Vector2) -> float` returns Euclidean distance.
- `segment_circle_contact_fraction(start, end, center, radius) -> float | None` returns the first contact fraction in `[0, 1]`, or `None` when the segment misses.
- `FieldBounds(length_yards=120.0, width_yards=53.3)` treats points on the boundary as in bounds.

- [x] Add a 3-4-5 distance test and a test that mutating one vector does not mutate another.
- [x] Add segment-circle tests for a crossing, a miss, and a segment starting inside the radius.
- [x] Add field tests for inclusive edges and a point just outside each axis.
- [x] Run `python -m pytest tests/test_geometry.py -q`; confirm failures are due to the missing module or functions.
- [x] Add the minimal geometry implementation and rerun the focused tests.
- [x] Configure Python 3.11+, NumPy runtime dependency, pytest/Ruff development dependencies, Ruff rules, and pytest's `src` import path in `pyproject.toml`.
- [x] Document the yard-based coordinate convention, setup commands, and headless scope in `README.md`.
- [x] Run `python -m pytest tests/test_geometry.py -q` and `ruff check .`.

### Task 2: Configuration, entities, and route paths

**Files:**
- Create: `src/qb_sim/config.py`
- Create: `src/qb_sim/entities.py`
- Create: `src/qb_sim/routes.py`
- Create: `tests/test_routes.py`

**Interfaces:**
- `SimulationConfig` owns the field bounds, fixed `dt=0.05`, maximum play duration, movement speeds, contact radii, initial positions, ball speed, and route parameters.
- `PlayerState` stores position and velocity; `BallState` stores position, velocity, target, and ball status.
- `RouteName` has `GO`, `SLANT`, and `POST` values.
- `build_route(name, start, field, config) -> Route` builds deterministic waypoints relative to the receiver's starting side.
- `Route.advance(position, max_distance) -> tuple[Vector2, bool]` moves toward the active waypoint without overshooting and reports whether the route is complete.

- [x] Add a Go test asserting its first waypoint advances downfield without changing lateral position.
- [x] Add a Slant test asserting an early break toward field center.
- [x] Add a Post test asserting a 10-yard vertical stem followed by a 45-degree inward segment.
- [x] Add a movement test asserting a route step stops exactly at a waypoint when its movement budget exceeds the remaining distance.
- [x] Run `python -m pytest tests/test_routes.py -q`; confirm the route API is absent.
- [x] Implement configuration validation, state dataclasses/enums, and route waypoint construction/movement.
- [x] Rerun the route tests and `ruff check .`.

### Task 3: Player movement and ball physics

**Files:**
- Create: `src/qb_sim/physics.py`
- Create: `tests/test_physics.py`

**Interfaces:**
- `move_toward(position, target, max_distance) -> Vector2` returns the new position without exceeding the movement limit; callers derive velocity from displacement and `dt`.
- `advance_ball(ball, dt) -> tuple[Vector2, float]` returns the tentative next position and fraction of the tick used when clamped at the pass target.
- `moving_circle_contact_fraction(first_start, first_end, second_start, second_end, radius) -> float | None` checks two moving points using relative swept segments.
- `segment_field_exit_fraction(start, end, field) -> float | None` returns the first fraction at which a segment leaves the inclusive field bounds.
- Contact checks use relative swept segments for the ball and moving players so a fast pass cannot skip a player between ticks.

- [x] Test movement toward a target, no overshoot, and a zero-distance target.
- [x] Test a ball segment that crosses a stationary player's catch radius and one that misses.
- [x] Test relative swept collision when both the ball and player move during the same tick.
- [x] Test the first field-boundary crossing fraction and an entirely in-bounds segment.
- [x] Run `python -m pytest tests/test_physics.py -q`; confirm the physics API is absent.
- [x] Implement only the geometry and motion needed by these behaviors.
- [x] Rerun focused geometry and physics tests, then `ruff check .`.

### Task 4: Simulation orchestration and outcome resolution

**Files:**
- Create: `src/qb_sim/simulation.py`
- Modify: `src/qb_sim/__init__.py`
- Create: `tests/test_simulation.py`
- Modify: `README.md`

**Interfaces:**
- `Simulation(config: SimulationConfig, route: RouteName = RouteName.GO)` creates a repeatable play.
- `reset() -> PlayState` restores the configured initial positions and route progress.
- `throw(lead_seconds: float) -> None` targets `receiver.position + receiver.velocity * lead_seconds`; it rejects negative/non-finite leads and a second throw.
- `step() -> PlayState` advances one fixed tick and returns the state, including the play outcome.
- `PlayOutcome` distinguishes `IN_PROGRESS`, `COMPLETED`, `INTERCEPTION`, `INCOMPLETE`, `OUT_OF_BOUNDS`, and `TIME_LIMIT`.
- Resolve events by the earliest contact/boundary/target fraction in the tick; if receiver and defender contact at the same fraction, the defender wins.

- [x] Test that a tick moves the receiver along its route and the defender toward the receiver while the quarterback remains stationary.
- [x] Test that two simulations with the same configuration and commands produce identical state traces and outcomes.
- [x] Add controlled integration fixtures for a completion, interception, incomplete pass, out-of-bounds pass, and play time limit.
- [x] Test that `step()` after a terminal outcome leaves the terminal state unchanged.
- [x] Run `python -m pytest tests/test_simulation.py -q`; confirm failures are caused by missing simulation behavior.
- [x] Implement reset, throw validation/lead targeting, fixed-step update order, swept collision resolution, and terminal outcomes.
- [x] Document the public API and one headless play sequence in `README.md`.
- [x] Run the complete suite with `python -m pytest`.

### Task 5: Final verification

**Files:**
- Verify all created source, tests, and documentation files.

- [x] Run `python -m pip install -e ".[dev]"` when dependency access is available.
- [x] Run `python -m pytest` and confirm all tests pass.
- [x] Run `ruff check .` and `ruff format --check .`.
- [x] Review that importing `qb_sim` does not import Gymnasium or a rendering library.
- [x] Report the actual test and lint results; do not claim unrun checks passed.
