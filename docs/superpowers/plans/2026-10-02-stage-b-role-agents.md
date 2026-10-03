# Stage B — Role Agents (WR, CB, Rusher) and QB v12: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn receivers, cornerbacks, and the rusher into trained policies (BC + PPO), add carrier-run and fatigue mechanics, compose them with the drive QB, and re-fine-tune the QB as `ppo_drive_qb_v12`.

**Architecture:** Engine/rules gain carrier fatigue and end-zone running; `training/role_context_observations.py` defines the WR (28) and defender (24) layouts; `training/role_env.py` provides `RolePlayEnv` (WR, play level) and `RoleDriveEnv` (CB/rusher, drives); `training/policy.py` gains context-aware adapters; the existing BC/PPO pipeline is extended with `role_play`/`role_drive` modes and frozen opponents.

**Tech Stack:** Python 3.11+ (env 3.14), NumPy, Gymnasium, Stable-Baselines3 + PyTorch, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-10-02-stage-b-role-agents-design.md` (approved 2026-10-02).

## Global Constraints

- Frozen baselines: `runs/ppo_rushed_v10` and `runs/ppo_drive_qb_v11` are never overwritten or retrained in place.
- `SimulationConfig` defaults keep every M5/M7 trajectory unchanged: `carrier_fatigue_per_second=0.0` by default; fatigue is only enabled by `STAGE_B_CONFIG`.
- `engine`/`rules` never import `training`, gym, torch, or pygame.
- Observations are clipped to `[-1, 1]`, dtype float32, fixed sizes (28 WR / 24 defenders); actions are finite and bounded by the env decode.
- Determinism for a fixed seed and action sequence is mandatory in every env.
- Commit each task with a focused `feat:`/`fix:` message once tests and `ruff check .` pass; never stage `docs/superpowers/` (the user handles those).
- Run tests per task with `python -m pytest <files> -q`, then the full suite `python -m pytest -q`, then `ruff check .`.
- Repository docs and code are in English.

---

### Task 1: Carrier runs to the end zone + carrier fatigue

**Files:**
- Modify: `src/qb_sim/config.py`, `src/qb_sim/engine/entities.py`, `src/qb_sim/rules/play.py`, `src/qb_sim/agents/observations.py`, `src/qb_sim/agents/rule_controllers.py`, `src/qb_sim/agents/loop.py`
- Test: `tests/test_play.py`, `tests/test_rule_controllers.py`

**Interfaces:**
- Produces: `SimulationConfig.carrier_fatigue_per_second`, `SimulationConfig.carrier_min_speed_factor`; `PlayState.carry_started_at`; `ReceiverObservation.carrier_id`; `ReceiverRouteController(player_id, route, speed, time_step, field=None)` that runs to the end zone while carrying.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_play.py`:

```python
def test_catch_records_when_the_carry_started() -> None:
    config = replace(
        ZERO_DEVIATION,
        quarterback_start=(20.0, 26.65),
        receiver_starts=((30.0, 26.65), (119.0, 0.0)),
        defender_starts=((0.0, 0.0), (0.0, 0.0)),
    )
    play = Play(config)
    agents = _make_agents(play)
    _step(play, agents)
    play.throw_with_lead(0.4)

    state = _run_until(play, agents, lambda current: current.ball.status is BallStatus.CARRIED)

    assert state.carry_started_at == pytest.approx(state.elapsed_seconds)


def test_carrier_fatigue_slows_the_ball_carrier_only() -> None:
    config = replace(ZERO_DEVIATION, carrier_fatigue_per_second=0.2, carrier_min_speed_factor=0.5)
    play = Play(config)
    play.state.carrier_id = "wr1"
    play.state.carry_started_at = 0.0
    play.state.elapsed_seconds = 2.0

    state = play.step({"wr1": Move(direction=vec2(1.0, 0.0))})

    assert float(np.linalg.norm(state.player("wr1").velocity)) == pytest.approx(7.0 * 0.6)

    play = Play(config)
    state = play.step({"cb1": Move(direction=vec2(1.0, 0.0))})

    assert float(np.linalg.norm(state.player("cb1").velocity)) == pytest.approx(6.5)


def test_carrier_fatigue_respects_the_floor() -> None:
    config = replace(ZERO_DEVIATION, carrier_fatigue_per_second=0.5, carrier_min_speed_factor=0.4)
    play = Play(config)
    play.state.carrier_id = "wr1"
    play.state.carry_started_at = 0.0
    play.state.elapsed_seconds = 5.0

    state = play.step({"wr1": Move(direction=vec2(1.0, 0.0))})

    assert float(np.linalg.norm(state.player("wr1").velocity)) == pytest.approx(7.0 * 0.4)


def test_fatigue_is_disabled_by_default() -> None:
    play = Play(ZERO_DEVIATION)
    play.state.carrier_id = "wr1"
    play.state.carry_started_at = 0.0
    play.state.elapsed_seconds = 3.0

    state = play.step({"wr1": Move(direction=vec2(1.0, 0.0))})

    assert float(np.linalg.norm(state.player("wr1").velocity)) == pytest.approx(7.0)
```

Append to `tests/test_rule_controllers.py`:

```python
def test_receiver_carrier_runs_toward_the_end_zone_after_the_route_ends() -> None:
    play = Play()
    play.state.carrier_id = "wr1"
    controller = _route_controller(play)
    controller.route.advance(play.state.player("wr1").position, max_distance=1000.0)

    action = controller.decide(play.state)

    assert isinstance(action, Move)
    assert action.direction[0] > 0.5


def test_receiver_carrier_steers_away_from_the_sideline() -> None:
    play = Play()
    play.state.carrier_id = "wr1"
    play.state.player("wr1").position = vec2(40.0, 1.0)
    controller = _route_controller(play)

    action = controller.decide(play.state)

    assert isinstance(action, Move)
    assert action.direction[1] > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_play.py tests/test_rule_controllers.py -q`
Expected: FAIL with `TypeError`/`AttributeError` for the new fields/behavior.

- [ ] **Step 3: Add the config fields**

In `src/qb_sim/config.py`, after `max_arc_height: float = 6.0`:

```python
    carrier_fatigue_per_second: float = 0.0
    carrier_min_speed_factor: float = 0.55
```

In `__post_init__` add `"carrier_fatigue_per_second"` to the non-negative spread checks and:

```python
        if not 0.0 <= self.carrier_min_speed_factor <= 1.0:
            raise ValueError("carrier_min_speed_factor must be between 0 and 1")
```

- [ ] **Step 4: Track the carry start and apply fatigue**

In `src/qb_sim/engine/entities.py`, append to `PlayState`:

```python
    carry_started_at: float | None = None
```

In `src/qb_sim/rules/play.py`:
- In `step`, immediately after `self.state.elapsed_seconds += dt * tick_fraction`, record the carry start once:

```python
        if self.state.carrier_id is not None and self.state.carry_started_at is None:
            self.state.carry_started_at = self.state.elapsed_seconds
```

- In `_apply_action`, replace the speed line:

```python
        speed = player.max_speed if player.max_speed > 0.0 else self._speed_for(player.role)
        if player.player_id == self.state.carrier_id:
            speed *= self._fatigue_factor()
```

- Add the method:

```python
    def _fatigue_factor(self) -> float:
        if self.state.carry_started_at is None or self.config.carrier_fatigue_per_second <= 0.0:
            return 1.0
        carried = self.state.elapsed_seconds - self.state.carry_started_at
        return max(
            self.config.carrier_min_speed_factor,
            1.0 - self.config.carrier_fatigue_per_second * carried,
        )
```

- [ ] **Step 5: Carrier-run behavior in the rule controller**

In `src/qb_sim/agents/observations.py`, append to `ReceiverObservation`:

```python
    carrier_id: str | None = None
```

In `src/qb_sim/agents/rule_controllers.py`:
- Import `FieldBounds` from `qb_sim.engine.geometry`.
- Extend the constructor:

```python
    def __init__(
        self,
        player_id: str,
        route: Route,
        speed: float,
        time_step: float,
        field: FieldBounds | None = None,
    ) -> None:
        super().__init__(player_id)
        self._route = route
        self._speed = speed
        self._time_step = time_step
        self._field = field if field is not None else FieldBounds()
```

- In `observe`, pass `carrier_id=state.carrier_id`.
- In `act`, add at the top:

```python
        if observation.carrier_id == self.player_id:
            return self._carry_action(observation)
```

- Add:

```python
    def _carry_action(self, observation: ReceiverObservation) -> Move:
        direction = vec2(1.0, 0.0)
        if observation.nearest_defender is not None:
            offset = observation.position - observation.nearest_defender.position
            distance = float(np.linalg.norm(offset))
            if 0.0 < distance < 3.0:
                direction = direction + offset / distance * 0.75
        margin = 3.0
        if observation.position[1] < margin:
            direction = direction + vec2(0.0, 0.5)
        elif observation.position[1] > self._field.width_yards - margin:
            direction = direction + vec2(0.0, -0.5)
        length = float(np.linalg.norm(direction))
        if length == 0.0:
            return Move(direction=vec2(1.0, 0.0))
        return Move(direction=direction / length)
```

In `src/qb_sim/agents/loop.py`, pass the field to the controller:

```python
        agents[player_id] = ReceiverRouteController(
            player_id,
            route,
            settings.receiver_speed,
            settings.dt,
            settings.field,
        )
```

- [ ] **Step 6: Run tests and full suite**

Run: `python -m pytest tests/test_play.py tests/test_rule_controllers.py -q` → PASS
Run: `python -m pytest -q` → PASS (the carrier-run change may adjust trajectories; if an existing test asserts a stale position, update the assertion to the new intended behavior and note it in the report)
Run: `ruff check .` → clean

---

### Task 2: Fatigue calibration + `STAGE_B_CONFIG`

**Files:**
- Create: `scripts/carrier_feasibility.py`
- Modify: `src/qb_sim/rules/scenarios.py`
- Test: `tests/test_scenarios.py`

**Interfaces:**
- Produces: `STAGE_B_CONFIG`, `STAGE_B_SCENARIO` (name `"stage_b"`) in `SCENARIOS`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scenarios.py`:

```python
def test_stage_b_scenario_enables_carrier_fatigue() -> None:
    from qb_sim.rules.scenarios import STAGE_B_SCENARIO

    config = STAGE_B_SCENARIO.config
    assert config.carrier_fatigue_per_second > 0.0
    assert 0.0 < config.carrier_min_speed_factor < 1.0
    assert STAGE_B_SCENARIO.name in SCENARIOS
    assert STAGE_B_SCENARIO.config.rusher_starts == ((30.0, 29.0),)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenarios.py -q -k stage_b`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add the scenario**

In `src/qb_sim/rules/scenarios.py`:

```python
STAGE_B_CONFIG = replace(
    SimulationConfig(),
    rusher_starts=((30.0, 29.0),),
    carrier_fatigue_per_second=0.08,
    carrier_min_speed_factor=0.65,
)
STAGE_B_SCENARIO = Scenario(
    "stage_b",
    STAGE_B_CONFIG,
    (RouteName.POST, RouteName.SLANT),
)
```

Add `STAGE_B_SCENARIO` to the `SCENARIOS` tuple.

- [ ] **Step 4: Write the calibration script**

Create `scripts/carrier_feasibility.py`:

```python
"""Calibrate carrier fatigue: TD rate and YAC by receiver-corner speed gap.

Example:

    python scripts/carrier_feasibility.py --fatigue 0.08 --floor 0.65 --drives 100
"""

import argparse
from dataclasses import replace

import numpy as np

from qb_sim.agents.loop import build_rule_agents
from qb_sim.config import SimulationConfig
from qb_sim.engine.entities import PlayOutcome
from qb_sim.rules.drive import Drive, DriveConfig
from qb_sim.rules.scenarios import get_scenario

DEFAULT_SEEDS = 100
GAPS = ((7.0, 7.0), (7.0, 6.7), (7.5, 7.0), (7.5, 6.8), (8.0, 7.0))


def run_drive(config, seed, drive_config):
    drive = Drive(config, drive_config, seed=seed)
    touchdowns = 0
    catches = 0
    yac = []
    while not drive.state.terminated:
        play = drive.start_play()
        agents = build_rule_agents(play, drive.routes, config, include_quarterback=True)
        while play.state.outcome is PlayOutcome.IN_PROGRESS:
            actions = {}
            for player_id, agent in agents.items():
                action = agent.decide(play.state)
                if action is not None:
                    actions[player_id] = action
            play.step(actions)
        carrier = play.state.carrier_id
        if carrier is not None:
            catches += 1
            yac.append(float(play.state.player(carrier).position[0] - drive.state.los))
        result = drive.advance(play)
        touchdowns += int(result.touchdown)
    return touchdowns, catches, yac


def main() -> None:
    parser = argparse.ArgumentParser(description="Carrier fatigue feasibility")
    parser.add_argument("--fatigue", type=float, default=0.08)
    parser.add_argument("--floor", type=float, default=0.65)
    parser.add_argument("--drives", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--seed-start", type=int, default=10_000)
    args = parser.parse_args()

    drive_config = DriveConfig(los_range=(40.0, 40.0))
    print(f"fatigue={args.fatigue}/s floor={args.floor} drives/gap={args.drives}")
    print(f"{'recv':>5} {'cb':>5} {'td%':>7} {'catches':>8} {'yac':>7}")
    for receiver_speed, defender_speed in GAPS:
        config = replace(
            get_scenario("stage_b").config,
            receiver_speed=receiver_speed,
            defender_speed=defender_speed,
            carrier_fatigue_per_second=args.fatigue,
            carrier_min_speed_factor=args.floor,
        )
        touchdowns = 0
        catches = 0
        yac: list[float] = []
        for seed in range(args.seed_start, args.seed_start + args.drives):
            td, caught, yards = run_drive(config, seed, drive_config)
            touchdowns += td
            catches += caught
            yac.extend(yards)
        mean_yac = float(np.mean(yac)) if yac else 0.0
        print(
            f"{receiver_speed:>5.1f} {defender_speed:>5.1f} "
            f"{100.0 * touchdowns / args.drives:>6.1f}% {catches:>8} {mean_yac:>7.2f}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the script and lock the values**

Run: `python scripts/carrier_feasibility.py --fatigue 0.08 --floor 0.65 --drives 100`
Expected: TD% well below 100% at small speed gaps and clearly higher at a 1.0-yard gap (the Retro Bowl behavior). If the separation is too weak or too strong, adjust `carrier_fatigue_per_second` in `STAGE_B_CONFIG` (try 0.08–0.2) and re-run until the gap curve is monotone and the 1.0-yard gap still scores often. Record the final table in the task report.

- [ ] **Step 6: Run tests and full suite**

Run: `python -m pytest tests/test_scenarios.py -q` → PASS
Run: `python -m pytest -q` → PASS
Run: `ruff check .` → clean

---

### Task 3: Per-role observations (28 / 24)

**Files:**
- Create: `src/qb_sim/training/role_context_observations.py`
- Test: `tests/test_role_context_observations.py`

**Interfaces:**
- Consumes: `receiver_observation`/`defender_observation` (19), `ROUTE_OBSERVATION_ORDER`, `DriveState`.
- Produces: `RECEIVER_CONTEXT_OBSERVATION_SIZE = 31`, `DEFENDER_CONTEXT_OBSERVATION_SIZE = 24`, `receiver_context_observation(state, player_id, config, route, goal_line=100.0)`, `defender_context_observation(state, player_id, target_id, config, drive_state, goal_line=100.0)`.

Post-review correction (2026-10-02, Task 11 debugging): the WR layout grows
from 28 to 31 with route-progress features (next waypoint relative position and
distance), because a stateless BC of the route controller otherwise compounds
~1.2° per-tick direction errors and underperforms the rule baseline
(measured 10–12% vs 17% catches; with the features, BC-only reaches 22%).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_role_context_observations.py`:

```python
import numpy as np
import pytest

from qb_sim.rules.drive import DriveState
from qb_sim.rules.play import Play
from qb_sim.rules.routes import ROUTE_OBSERVATION_ORDER, RouteName
from qb_sim.training.role_context_observations import (
    DEFENDER_CONTEXT_OBSERVATION_SIZE,
    RECEIVER_CONTEXT_OBSERVATION_SIZE,
    defender_context_observation,
    receiver_context_observation,
)


def test_receiver_context_observation_has_the_documented_size_and_range() -> None:
    play = Play()
    observation = receiver_context_observation(
        play.state, "wr1", play.config, RouteName.POST
    )

    assert observation.shape == (RECEIVER_CONTEXT_OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert np.all(observation >= -1.0)
    assert np.all(observation <= 1.0)


def test_receiver_context_route_one_hot_uses_the_fixed_order() -> None:
    play = Play()
    observation = receiver_context_observation(
        play.state, "wr1", play.config, RouteName.CURL
    )

    index = 19 + ROUTE_OBSERVATION_ORDER.index(RouteName.CURL)
    assert observation[index] == pytest.approx(1.0)
    assert float(observation[19:27].sum()) == pytest.approx(1.0)


def test_receiver_context_encodes_yards_to_the_goal_line() -> None:
    play = Play()
    play.state.player("wr1").position[0] = 40.0

    observation = receiver_context_observation(
        play.state, "wr1", play.config, RouteName.GO, goal_line=100.0
    )

    assert observation[27] == pytest.approx(0.6)


def test_defender_context_observation_has_the_documented_size_and_range() -> None:
    play = Play()
    observation = defender_context_observation(
        play.state, "cb1", "wr1", play.config, DriveState(los=40.0)
    )

    assert observation.shape == (DEFENDER_CONTEXT_OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert np.all(observation >= -1.0)
    assert np.all(observation <= 1.0)


def test_defender_context_encodes_down_distance_and_goal() -> None:
    play = Play()
    observation = defender_context_observation(
        play.state, "cb1", "wr1", play.config, DriveState(los=40.0, down=3, distance=20.0)
    )

    assert observation[19:22].tolist() == [0.0, 0.0, 1.0]
    assert observation[22] == pytest.approx(0.5)
    assert observation[23] == pytest.approx(0.6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_role_context_observations.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the module**

Create `src/qb_sim/training/role_context_observations.py`:

```python
"""Context-enriched role observations for Stage B (WR 28, defenders 24).

WR layout: 19 base + route one-hot (8) + yards to the goal line.
Defender layout: 19 base + down one-hot (3) + distance/40 + yards to the goal line.
"""

import numpy as np

from qb_sim.config import SimulationConfig
from qb_sim.engine.entities import PlayState
from qb_sim.rules.drive import DriveState
from qb_sim.rules.routes import ROUTE_OBSERVATION_ORDER, RouteName
from qb_sim.training.role_observations import (
    defender_observation,
    receiver_observation,
)

RECEIVER_CONTEXT_OBSERVATION_SIZE = 19 + len(ROUTE_OBSERVATION_ORDER) + 1
DEFENDER_CONTEXT_OBSERVATION_SIZE = 19 + 3 + 2
DISTANCE_SCALE = 40.0


def receiver_context_observation(
    state: PlayState,
    player_id: str,
    config: SimulationConfig,
    route: RouteName | None,
    goal_line: float = 100.0,
) -> np.ndarray:
    """Receiver observation plus its called route and yards to the end zone."""
    values = [float(value) for value in receiver_observation(state, player_id, config)]
    values.extend(1.0 if route is route_name else 0.0 for route_name in ROUTE_OBSERVATION_ORDER)
    position_x = float(state.player(player_id).position[0])
    values.append((float(goal_line) - position_x) / float(goal_line))
    return np.clip(np.array(values, dtype=np.float32), -1.0, 1.0)


def defender_context_observation(
    state: PlayState,
    player_id: str,
    target_id: str | None,
    config: SimulationConfig,
    drive_state: DriveState,
    goal_line: float = 100.0,
) -> np.ndarray:
    """Defender observation plus down, distance, and field position."""
    values = [
        float(value)
        for value in defender_observation(state, player_id, target_id, config)
    ]
    down = int(drive_state.down)
    values.extend(1.0 if down == index else 0.0 for index in range(1, 4))
    values.append(min(float(drive_state.distance), DISTANCE_SCALE) / DISTANCE_SCALE)
    values.append((float(goal_line) - float(drive_state.los)) / float(goal_line))
    return np.clip(np.array(values, dtype=np.float32), -1.0, 1.0)
```

- [ ] **Step 4: Run tests and full suite**

Run: `python -m pytest tests/test_role_context_observations.py -q` → PASS
Run: `python -m pytest -q` → PASS
Run: `ruff check .` → clean

---

### Task 4: Role rewards

**Files:**
- Modify: `src/qb_sim/training/rewards.py`
- Test: `tests/test_rewards.py`

**Interfaces:**
- Produces: `ReceiverRewardConfig(play, catch_bonus=0.3)`, `receiver_result_reward(outcome, yards, caught, config)`, `defender_drive_reward(result, config)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rewards.py`:

```python
from qb_sim.training.rewards import (
    ReceiverRewardConfig,
    defender_drive_reward,
    receiver_result_reward,
)


def test_receiver_reward_pays_the_catch_bonus_only_when_catching() -> None:
    config = ReceiverRewardConfig()

    caught = receiver_result_reward(PlayOutcome.COMPLETED, 8.0, True, config)
    not_caught = receiver_result_reward(PlayOutcome.COMPLETED, 8.0, False, config)

    assert caught == pytest.approx(1.0 + 8.0 * 0.02 + 0.3)
    assert not_caught == pytest.approx(1.0 + 8.0 * 0.02)


def test_receiver_reward_keeps_terminal_penalties() -> None:
    config = ReceiverRewardConfig()

    assert receiver_result_reward(PlayOutcome.INTERCEPTION, 0.0, False, config) == pytest.approx(-1.0)


def test_defender_reward_is_the_negated_drive_result_reward() -> None:
    result = PlayResult(
        outcome=PlayOutcome.COMPLETED,
        spot=52.0,
        yards=12.0,
        first_down=True,
        touchdown=False,
        drive_over=False,
        end_reason=None,
        points=0.0,
        throw_distance=12.0,
    )

    assert defender_drive_reward(result, DriveRewardConfig()) == pytest.approx(
        -(1.0 + 12.0 * 0.02 + 0.3)
    )
```

Add `PlayResult` and `DriveRewardConfig` to the imports already present in that file.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rewards.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the rewards**

In `src/qb_sim/training/rewards.py` add:

```python
@dataclass(frozen=True, slots=True)
class ReceiverRewardConfig:
    """Offense terminal reward plus catch and touchdown bonuses."""

    play: RewardConfig = field(default_factory=RewardConfig)
    catch_bonus: float = 0.3
    touchdown_bonus: float = 7.0


def receiver_result_reward(
    outcome: PlayOutcome,
    yards: float,
    caught: bool,
    config: ReceiverRewardConfig,
) -> float:
    """Reward for one finished play from the receiver's perspective."""
    reward = terminal_reward(outcome, yards, config.play)
    if caught:
        reward += config.catch_bonus
    if outcome is PlayOutcome.TOUCHDOWN:
        reward += config.touchdown_bonus
    return float(reward)


@dataclass(frozen=True, slots=True)
class DefenderRewardConfig:
    """Zero-sum terminal reward plus policy-invariant coverage shaping."""

    terminal: DriveRewardConfig = field(default_factory=DriveRewardConfig)
    shaping: bool = True
    shaping_gamma: float = 0.99
    shaping_scale: float = 10.0


def defender_potential(distance: float, scale: float = 10.0) -> float:
    """Potential: closer to the target is better (bounded to ``[-2, 0]``)."""
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return -min(max(float(distance), 0.0), 2.0 * scale) / scale
```

- [ ] **Step 4: Run tests and full suite**

Run: `python -m pytest tests/test_rewards.py -q` → PASS
Run: `python -m pytest -q` → PASS
Run: `ruff check .` → clean

---

### Task 5: Context-aware policy adapters

**Files:**
- Modify: `src/qb_sim/training/policy.py`
- Test: `tests/test_policy_viewer.py`

**Interfaces:**
- Produces: `ReceiverPolicy(model, config, player_id, goal_line=100.0)` callable `(state, route) -> Move | None`; `DefenderPolicy(model, config, player_id, goal_line=100.0)` callable `(state, target_id, drive_state) -> Move | None`; `load_receiver_policy(path, scenario_name, player_id)`; `load_defender_policy(path, scenario_name, player_id)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_policy_viewer.py`:

```python
from qb_sim.training.policy import DefenderPolicy, ReceiverPolicy


class _RecordingModel:
    def __init__(self, action: list[float]) -> None:
        self._action = np.array(action, dtype=np.float32)
        self.observation_shape: tuple[int, ...] | None = None

    def predict(self, observation, deterministic: bool = True):
        self.observation_shape = tuple(np.asarray(observation).shape)
        return self._action, None


def test_receiver_policy_builds_the_28_float_observation() -> None:
    play = Play()
    model = _RecordingModel([1.0, 0.0])
    policy = ReceiverPolicy(model, play.config, "wr1")

    action = policy(play.state, RouteName.POST)

    assert model.observation_shape == (28,)
    assert action is not None
    assert action.direction.tolist() == pytest.approx([1.0, 0.0])


def test_defender_policy_builds_the_24_float_observation() -> None:
    play = Play()
    model = _RecordingModel([0.0, 1.0])
    policy = DefenderPolicy(model, play.config, "cb1")

    action = policy(play.state, "wr1", DriveState(los=40.0))

    assert model.observation_shape == (24,)
    assert action is not None
    assert action.direction.tolist() == pytest.approx([0.0, 1.0])
```

Add `import pytest` to that file if missing.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_policy_viewer.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the adapters**

In `src/qb_sim/training/policy.py` add (and import `receiver_context_observation`/`defender_context_observation`):

```python
def _move_from_action(action: Any) -> Move | None:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    direction = np.array(values[:2], dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        return None
    if length > 1.0:
        direction = direction / length
    return Move(direction=direction)


class ReceiverPolicy:
    """Receiver callable ``(state, route) -> Move | None`` backed by a PPO model."""

    def __init__(
        self,
        model: Any,
        config: SimulationConfig,
        player_id: str,
        goal_line: float = 100.0,
    ) -> None:
        self._model = model
        self._config = config
        self._player_id = player_id
        self._goal_line = float(goal_line)

    def __call__(self, state: PlayState, route: RouteName | None) -> Move | None:
        observation = receiver_context_observation(
            state,
            self._player_id,
            self._config,
            route,
            self._goal_line,
        )
        action, _ = self._model.predict(observation, deterministic=True)
        return _move_from_action(action)


class DefenderPolicy:
    """Defender callable ``(state, target_id, drive_state) -> Move | None``."""

    def __init__(
        self,
        model: Any,
        config: SimulationConfig,
        player_id: str,
        goal_line: float = 100.0,
    ) -> None:
        self._model = model
        self._config = config
        self._player_id = player_id
        self._goal_line = float(goal_line)

    def __call__(
        self,
        state: PlayState,
        target_id: str | None,
        drive_state: DriveState,
    ) -> Move | None:
        observation = defender_context_observation(
            state,
            self._player_id,
            target_id,
            self._config,
            drive_state,
            self._goal_line,
        )
        action, _ = self._model.predict(observation, deterministic=True)
        return _move_from_action(action)


def load_receiver_policy(
    path: str | Path,
    scenario_name: str = "stage_b",
    player_id: str = "wr1",
) -> ReceiverPolicy:
    """Load a saved PPO receiver model and adapt it to the engine."""
    from stable_baselines3 import PPO

    scenario = get_scenario(scenario_name)
    model = PPO.load(str(path), device="auto")
    return ReceiverPolicy(model, scenario.config, player_id)


def load_defender_policy(
    path: str | Path,
    scenario_name: str = "stage_b",
    player_id: str = "cb1",
) -> DefenderPolicy:
    """Load a saved PPO defender model and adapt it to the engine."""
    from stable_baselines3 import PPO

    scenario = get_scenario(scenario_name)
    model = PPO.load(str(path), device="auto")
    return DefenderPolicy(model, scenario.config, player_id)
```

- [ ] **Step 4: Run tests and full suite**

Run: `python -m pytest tests/test_policy_viewer.py -q` → PASS
Run: `python -m pytest -q` → PASS
Run: `ruff check .` → clean

---

### Task 6: `RolePlayEnv` (WR)

**Files:**
- Create: `src/qb_sim/training/role_env.py`
- Test: `tests/test_role_env.py`

**Interfaces:**
- Consumes: `STAGE_B_SCENARIO`, `sample_routes`, `build_rule_agents`, `receiver_context_observation`, `receiver_result_reward`, policy adapters.
- Produces: `defender_target(play, defender_id)`, `decide_player_action(play, routes, player_id, policy, drive_state=None)`, `RolePlayEnv(scenario=None, learner_id="wr1", opponents=None, reward_config=None, goal_line=100.0, max_steps=240)` with public `observation()`, `rule_agent(player_id)`, `play`, `routes`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_role_env.py`:

```python
from dataclasses import replace

import numpy as np
from gymnasium.utils.env_checker import check_env

from qb_sim.engine.entities import BallStatus
from qb_sim.training.role_context_observations import RECEIVER_CONTEXT_OBSERVATION_SIZE
from qb_sim.training.role_env import RolePlayEnv


def _noop() -> np.ndarray:
    return np.zeros(2, dtype=np.float32)


def test_role_play_spaces_and_registration_contract() -> None:
    env = RolePlayEnv()

    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (RECEIVER_CONTEXT_OBSERVATION_SIZE,)
    assert env.observation_space.contains(env.reset(seed=0)[0])


def test_role_play_passes_the_gymnasium_checker() -> None:
    env = RolePlayEnv()

    check_env(env)

    env.close()


def test_role_play_reset_is_deterministic_for_a_seed() -> None:
    first, _ = RolePlayEnv().reset(seed=7)
    second, _ = RolePlayEnv().reset(seed=7)

    np.testing.assert_allclose(first, second)


def test_role_play_scores_the_catch_bonus_for_the_learner() -> None:
    env = RolePlayEnv(learner_id="wr1")
    env.reset(seed=0)
    state = env.play.state
    state.carrier_id = "wr1"
    state.ball.status = BallStatus.CARRIED
    state.carry_started_at = 0.0
    total = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        _, reward, terminated, truncated, _ = env.step(_noop())
        total += reward

    assert total >= 1.2  # completion (+1.0) plus the +0.3 catch bonus


def test_role_play_rule_agent_is_exposed_for_demonstrations() -> None:
    env = RolePlayEnv(learner_id="wr2")

    agent = env.rule_agent("wr2")

    assert agent.player_id == "wr2"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_role_env.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `RolePlayEnv` and helpers**

Create `src/qb_sim/training/role_env.py`:

```python
"""Single-learner role environments for Stage B (WR play level, CB/rusher drives)."""

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from qb_sim.agents.base import BaseAgent
from qb_sim.agents.loop import build_rule_agents
from qb_sim.engine.actions import Move
from qb_sim.engine.entities import PlayOutcome
from qb_sim.rules.drive import Drive, DriveConfig, DriveState
from qb_sim.rules.play import Play
from qb_sim.rules.routes import sample_routes
from qb_sim.rules.scenarios import STAGE_B_SCENARIO, Scenario
from qb_sim.training.rewards import (
    DriveRewardConfig,
    ReceiverRewardConfig,
    defender_drive_reward,
    receiver_result_reward,
)
from qb_sim.training.role_context_observations import (
    DEFENDER_CONTEXT_OBSERVATION_SIZE,
    RECEIVER_CONTEXT_OBSERVATION_SIZE,
    defender_context_observation,
    receiver_context_observation,
)

MOVE_ACTION_SIZE = 2
DISTANCE_TO_TARGET = 3.0


def defender_target(play: Play, defender_id: str) -> str | None:
    """Coverage target for a defender, shared by the role environments."""
    state = play.state
    if state.carrier_id is not None:
        return state.carrier_id
    if defender_id in play.rusher_ids:
        return "qb"
    coverage = [
        player_id for player_id in play.defender_ids if player_id not in set(play.rusher_ids)
    ]
    index = coverage.index(defender_id)
    return play.receiver_ids[index % len(play.receiver_ids)]


def decide_player_action(
    play: Play,
    routes: tuple,
    player_id: str,
    policy: Any,
    drive_state: DriveState | None = None,
) -> Move | None:
    """Call a rule agent or a context-aware policy with the right signature."""
    if isinstance(policy, BaseAgent):
        return policy.decide(play.state)
    if player_id == "qb":
        if drive_state is None:
            return policy(play.state)
        return policy(play.state, routes, drive_state)
    if player_id in play.receiver_ids:
        index = play.receiver_ids.index(player_id)
        return policy(play.state, routes[index])
    return policy(play.state, defender_target(play, player_id), drive_state)


def _decode_move(action: np.ndarray) -> Move | None:
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.shape != (MOVE_ACTION_SIZE,):
        raise ValueError(f"action must have {MOVE_ACTION_SIZE} values")
    direction = np.array(values, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        return None
    if length > 1.0:
        direction = direction / length
    return Move(direction=direction)


class RolePlayEnv(gym.Env[np.ndarray, np.ndarray]):
    """Train one receiver inside a single play with random routes."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: Scenario | None = None,
        learner_id: str = "wr1",
        opponents: Mapping[str, Any] | None = None,
        reward_config: ReceiverRewardConfig | None = None,
        goal_line: float = 100.0,
        max_steps: int = 240,
    ) -> None:
        super().__init__()
        self.scenario = scenario if scenario is not None else STAGE_B_SCENARIO
        self.learner_id = learner_id
        self.opponents = dict(opponents or {})
        self.reward_config = reward_config if reward_config is not None else ReceiverRewardConfig()
        self.goal_line = float(goal_line)
        self.max_steps = int(max_steps)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(MOVE_ACTION_SIZE,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(RECEIVER_CONTEXT_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.play = Play(self.scenario.config, goal_line=self.goal_line)
        self.routes: tuple = ()
        self._agents: dict[str, BaseAgent] = {}
        self._steps = 0
        self._episode_seed = self.scenario.seed
        self._separation_at_throw: float | None = None
        self._separation_at_catch: float | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._episode_seed = int(seed)
        else:
            self._episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        rng = np.random.default_rng(self._episode_seed)
        self.routes = sample_routes(rng, len(self.scenario.config.receiver_starts))
        self.play = Play(self.scenario.config, seed=self._episode_seed, goal_line=self.goal_line)
        self._agents = build_rule_agents(
            self.play,
            self.routes,
            self.scenario.config,
            include_quarterback=False,
        )
        self._steps = 0
        self._separation_at_throw = None
        self._separation_at_catch = None
        return self.observation(), self._info()

    def rule_agent(self, player_id: str) -> BaseAgent:
        """Rule controller for a player, used to collect demonstrations."""
        return self._agents[player_id]

    def observation(self) -> np.ndarray:
        """Current learner observation."""
        route = self.routes[self.play.receiver_ids.index(self.learner_id)]
        return receiver_context_observation(
            self.play.state,
            self.learner_id,
            self.scenario.config,
            route,
            self.goal_line,
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        engine_actions: dict[str, Move] = {}
        for player_id, agent in self._agents.items():
            if player_id == self.learner_id:
                continue
            if player_id in self.opponents:
                decided = decide_player_action(self.play, self.routes, player_id, self.opponents[player_id])
            else:
                decided = agent.decide(self.play.state)
            if decided is not None:
                engine_actions[player_id] = decided
        learner_action = _decode_move(action)
        if learner_action is not None:
            engine_actions[self.learner_id] = learner_action

        state = self.play.step(engine_actions)
        self._steps += 1
        self._track_diagnostics(state)
        terminated = state.outcome is not PlayOutcome.IN_PROGRESS
        truncated = not terminated and self._steps >= self.max_steps
        reward = 0.0
        if terminated:
            yards = 0.0
            if state.carrier_id is not None:
                yards = float(
                    state.player(state.carrier_id).position[0]
                    - self.scenario.config.quarterback_start[0]
                )
            caught = state.carrier_id == self.learner_id
            reward = receiver_result_reward(state.outcome, yards, caught, self.reward_config)
        return self.observation(), float(reward), terminated, truncated, self._info()

    def render(self) -> None:
        return None

    def _track_diagnostics(self, state) -> None:
        if self._separation_at_throw is None and state.throw_time is not None:
            self._separation_at_throw = self._learner_separation(state)
        if self._separation_at_catch is None and state.carrier_id == self.learner_id:
            self._separation_at_catch = self._learner_separation(state)

    def _learner_separation(self, state) -> float:
        learner = state.player(self.learner_id)
        defenders = [
            state.player(player_id).position
            for player_id in self.play.defender_ids
        ]
        if not defenders:
            return 0.0
        return float(
            min(np.linalg.norm(learner.position - defender) for defender in defenders)
        )

    def _info(self) -> dict[str, Any]:
        state = self.play.state
        return {
            "episode_seed": self._episode_seed,
            "outcome": state.outcome.value,
            "yards": (
                float(state.player(state.carrier_id).position[0] - self.scenario.config.quarterback_start[0])
                if state.carrier_id is not None
                else 0.0
            ),
            "caught": state.carrier_id == self.learner_id,
            "separation_at_throw": self._separation_at_throw,
            "separation_at_catch": self._separation_at_catch,
            "throw_time": state.throw_time,
            "elapsed_seconds": state.elapsed_seconds,
        }
```

Note: `RoleDriveEnv` is Task 7 and will be added to this same file.

- [ ] **Step 4: Run tests and full suite**

Run: `python -m pytest tests/test_role_env.py -q` → PASS
Run: `python -m pytest -q` → PASS
Run: `ruff check .` → clean

---

### Task 7: `RoleDriveEnv` (CB / rusher)

**Files:**
- Modify: `src/qb_sim/training/role_env.py`
- Test: `tests/test_role_env.py`

**Interfaces:**
- Produces: `RoleDriveEnv(scenario=None, learner_id="cb1", opponents=None, reward_config=None, drive_config=None, goal_line=100.0, max_steps=2400)` with public `observation()`, `rule_agent`, `drive`, `play`, `play_history`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_role_env.py`:

```python
from qb_sim.rules.drive import DriveConfig
from qb_sim.training.role_env import RoleDriveEnv


def _sack_scenario():
    from qb_sim.rules.scenarios import Scenario
    from qb_sim.rules.routes import RouteName
    from dataclasses import replace
    from qb_sim.config import SimulationConfig

    zero = replace(
        SimulationConfig(),
        throw_base_sigma_degrees=0.0,
        throw_move_penalty=0.0,
        throw_pressure_penalty=0.0,
        throw_distance_penalty=0.0,
        throw_power_sigma=0.0,
        defender_starts=((20.5, 26.65), (0.0, 0.0)),
    )
    return Scenario("test-sack", zero, (RouteName.POST, RouteName.SLANT))


def test_role_drive_spaces_and_checker() -> None:
    env = RoleDriveEnv(_sack_scenario(), learner_id="cb1", drive_config=DriveConfig(los_range=(20.0, 20.0)))
    observation, info = env.reset(seed=0)

    assert env.observation_space.shape == (24,)
    assert env.observation_space.contains(observation)
    assert info["down"] == 1

    check_env(env)


def test_role_drive_defender_reward_is_zero_sum() -> None:
    env = RoleDriveEnv(_sack_scenario(), learner_id="cb1", drive_config=DriveConfig(los_range=(20.0, 20.0)))
    env.reset(seed=0)
    total = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        _, reward, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        total += reward
    assert total > 0.0  # three sacks are worth +1 each for the defense


def test_role_drive_is_deterministic_for_a_seed() -> None:
    config = DriveConfig(los_range=(20.0, 20.0))
    first = RoleDriveEnv(_sack_scenario(), learner_id="ru1", drive_config=config)
    second = RoleDriveEnv(_sack_scenario(), learner_id="ru1", drive_config=config)
    first_observation, _ = first.reset(seed=3)
    second_observation, _ = second.reset(seed=3)

    np.testing.assert_allclose(first_observation, second_observation)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_role_env.py -q -k role_drive`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `RoleDriveEnv`**

Append to `src/qb_sim/training/role_env.py`:

```python
class RoleDriveEnv(gym.Env[np.ndarray, np.ndarray]):
    """Train one defender inside drives against frozen teammates/opponents."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: Scenario | None = None,
        learner_id: str = "cb1",
        opponents: Mapping[str, Any] | None = None,
        reward_config: DriveRewardConfig | None = None,
        drive_config: DriveConfig | None = None,
        goal_line: float = 100.0,
        max_steps: int = 2400,
    ) -> None:
        super().__init__()
        self.scenario = scenario if scenario is not None else STAGE_B_SCENARIO
        self.learner_id = learner_id
        self.opponents = dict(opponents or {})
        self.reward_config = reward_config if reward_config is not None else DriveRewardConfig()
        self.drive_config = drive_config if drive_config is not None else DriveConfig()
        self.goal_line = float(goal_line)
        self.max_steps = int(max_steps)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(MOVE_ACTION_SIZE,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(DEFENDER_CONTEXT_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.drive = Drive(self.scenario.config, self.drive_config)
        self.play = self.drive.start_play()
        self.play_history: list[dict[str, Any]] = []
        self._agents: dict[str, BaseAgent] = {}
        self._steps = 0
        self._episode_seed = self.scenario.seed
        self._pressure_at_throw: float | None = None
        self._arrival_time: float | None = None
        self._build_agents()

    def _build_agents(self) -> None:
        self._agents = build_rule_agents(
            self.play,
            self.drive.routes,
            self.scenario.config,
            include_quarterback=False,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._episode_seed = int(seed)
        else:
            self._episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.drive = Drive(self.scenario.config, self.drive_config, seed=self._episode_seed)
        self.play_history = []
        self._steps = 0
        self._start_play()
        return self.observation(), self._info()

    def _start_play(self) -> None:
        self.play = self.drive.start_play()
        self._build_agents()
        self._pressure_at_throw = None
        self._arrival_time = None

    def rule_agent(self, player_id: str) -> BaseAgent:
        return self._agents[player_id]

    def observation(self) -> np.ndarray:
        return defender_context_observation(
            self.play.state,
            self.learner_id,
            defender_target(self.play, self.learner_id),
            self.scenario.config,
            self.drive.state,
            self.goal_line,
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        engine_actions: dict[str, Move] = {}
        for player_id, agent in self._agents.items():
            if player_id == self.learner_id:
                continue
            if player_id in self.opponents:
                decided = decide_player_action(
                    self.play,
                    self.drive.routes,
                    player_id,
                    self.opponents[player_id],
                    self.drive.state,
                )
            else:
                decided = agent.decide(self.play.state)
            if decided is not None:
                engine_actions[player_id] = decided
        learner_action = _decode_move(action)
        if learner_action is not None:
            engine_actions[self.learner_id] = learner_action

        state = self.play.step(engine_actions)
        self._steps += 1
        self._track_diagnostics(state)
        reward = 0.0
        terminated = False
        truncated = False
        if state.outcome is not PlayOutcome.IN_PROGRESS:
            down_before = self.drive.state.down
            distance_before = self.drive.state.distance
            los_before = self.drive.state.los
            result = self.drive.advance(self.play)
            reward = defender_drive_reward(result, self.reward_config)
            self.play_history.append(
                {
                    "down": down_before,
                    "distance": distance_before,
                    "los_before": los_before,
                    "outcome": result.outcome.value,
                    "spot": result.spot,
                    "yards": result.yards,
                    "first_down": result.first_down,
                    "touchdown": result.touchdown,
                    "throw_distance": result.throw_distance,
                    "sack": result.outcome is PlayOutcome.SACK,
                    "coverage_sack": result.coverage_sack,
                    "pressure_at_throw": self._pressure_at_throw,
                    "arrival_time": self._arrival_time,
                }
            )
            terminated = self.drive.state.terminated
            if not terminated:
                self._start_play()
        if not terminated and self._steps >= self.max_steps:
            truncated = True
        return self.observation(), float(reward), terminated, truncated, self._info()

    def render(self) -> None:
        return None

    def _track_diagnostics(self, state) -> None:
        if self._pressure_at_throw is None and state.throw_time is not None:
            self._pressure_at_throw = float(state.pressure)
        if self._arrival_time is None and self.learner_id in self.play.rusher_ids:
            quarterback = state.player("qb")
            rusher = state.player(self.learner_id)
            if float(np.linalg.norm(rusher.position - quarterback.position)) <= self.scenario.config.tackle_radius:
                self._arrival_time = float(state.elapsed_seconds)

    def _info(self) -> dict[str, Any]:
        return {
            "episode_seed": self._episode_seed,
            "down": self.drive.state.down,
            "distance": self.drive.state.distance,
            "los": self.drive.state.los,
            "outcome": self.play.state.outcome.value,
            "drive_end_reason": self.drive.state.end_reason,
            "steps": self._steps,
            "plays": list(self.play_history),
        }
```

- [ ] **Step 4: Run tests and full suite**

Run: `python -m pytest tests/test_role_env.py -q` → PASS
Run: `python -m pytest -q` → PASS
Run: `ruff check .` → clean

---

### Task 8: BC demonstrations for roles

**Files:**
- Modify: `src/qb_sim/training/imitation.py`
- Test: `tests/test_imitation.py`

**Interfaces:**
- Produces: `collect_role_play_demonstrations(scenario=None, learner_id="wr1", episodes=200, seed_start=0, opponents=None, goal_line=100.0)` and `collect_role_drive_demonstrations(scenario=None, learner_id="cb1", episodes=200, seed_start=0, opponents=None, drive_config=None)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_imitation.py`:

```python
from qb_sim.training.imitation import (  # noqa: E402
    collect_role_drive_demonstrations,
    collect_role_play_demonstrations,
)
from qb_sim.training.role_context_observations import (  # noqa: E402
    DEFENDER_CONTEXT_OBSERVATION_SIZE,
    RECEIVER_CONTEXT_OBSERVATION_SIZE,
)


def test_role_play_demonstrations_have_expected_shapes() -> None:
    dataset = collect_role_play_demonstrations(get_scenario("stage_b"), learner_id="wr1", episodes=1)

    assert dataset.observations.shape[1] == RECEIVER_CONTEXT_OBSERVATION_SIZE
    assert dataset.actions.shape[1] == 2
    assert len(dataset.observations) == len(dataset.actions)


def test_role_drive_demonstrations_have_expected_shapes() -> None:
    dataset = collect_role_drive_demonstrations(get_scenario("stage_b"), learner_id="cb1", episodes=1)

    assert dataset.observations.shape[1] == DEFENDER_CONTEXT_OBSERVATION_SIZE
    assert dataset.actions.shape[1] == 2


def test_mover_behavior_cloning_loads_weights_into_a_two_action_policy() -> None:
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    from qb_sim.training.role_env import RolePlayEnv

    scenario = get_scenario("stage_b")
    dataset = collect_role_play_demonstrations(scenario, learner_id="wr1", episodes=1)
    network, _ = train_behavior_cloning(dataset, epochs=2, seed=0, action_size=2)
    model = PPO(
        "MlpPolicy",
        RolePlayEnv(scenario),
        n_steps=32,
        batch_size=16,
        seed=0,
        device="cpu",
    )

    warm_start_policy(model, dataset, epochs=2, seed=0, action_size=2)

    observations = torch.as_tensor(dataset.observations[:16])
    with torch.no_grad():
        expected = network(observations)
        loaded = model.policy.action_net(model.policy.mlp_extractor.policy_net(observations))
    assert torch.allclose(expected, loaded, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_imitation.py -q -k role`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the collectors**

Append to `src/qb_sim/training/imitation.py` (import `RoleDriveEnv`, `RolePlayEnv`, `DriveConfig`):

```python
def _encoded_move(action) -> np.ndarray:
    if action is None:
        return np.zeros(2, dtype=np.float32)
    return np.array([action.direction[0], action.direction[1]], dtype=np.float32)


def collect_role_play_demonstrations(
    scenario: Scenario | None = None,
    learner_id: str = "wr1",
    episodes: int = 200,
    seed_start: int = 0,
    opponents: Mapping[str, object] | None = None,
    goal_line: float = 100.0,
) -> DemonstrationSet:
    """Run the rule receiver and record ``(28-observation, 2-action)`` pairs."""
    chosen = scenario if scenario is not None else get_scenario("stage_b")
    env = RolePlayEnv(chosen, learner_id=learner_id, opponents=opponents, goal_line=goal_line)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for episode in range(episodes):
        env.reset(seed=seed_start + episode)
        terminated = truncated = False
        while not (terminated or truncated):
            agent = env.rule_agent(learner_id)
            action = agent.decide(env.play.state)
            observations.append(env.observation())
            actions.append(_encoded_move(action))
            _, _, terminated, truncated, _ = env.step(actions[-1])
    env.close()
    return DemonstrationSet(
        observations=np.stack(observations).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
    )


def collect_role_drive_demonstrations(
    scenario: Scenario | None = None,
    learner_id: str = "cb1",
    episodes: int = 200,
    seed_start: int = 0,
    opponents: Mapping[str, object] | None = None,
    drive_config: DriveConfig | None = None,
) -> DemonstrationSet:
    """Run the rule defender and record ``(24-observation, 2-action)`` pairs."""
    chosen = scenario if scenario is not None else get_scenario("stage_b")
    env = RoleDriveEnv(
        chosen,
        learner_id=learner_id,
        opponents=opponents,
        drive_config=drive_config,
    )
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for episode in range(episodes):
        env.reset(seed=seed_start + episode)
        terminated = truncated = False
        while not (terminated or truncated):
            agent = env.rule_agent(learner_id)
            action = agent.decide(env.play.state)
            observations.append(env.observation())
            actions.append(_encoded_move(action))
            _, _, terminated, truncated, _ = env.step(actions[-1])
    env.close()
    return DemonstrationSet(
        observations=np.stack(observations).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
    )
```

Add `from collections.abc import Mapping` if missing.

- [ ] **Step 4: Run tests and full suite**

Run: `python -m pytest tests/test_imitation.py -q` → PASS
Run: `python -m pytest -q` → PASS
Run: `ruff check .` → clean

---

### Task 9: Training pipeline for roles

**Files:**
- Modify: `src/qb_sim/training/ppo_config.py`, `src/qb_sim/training/train.py`
- Test: `tests/test_ppo_training.py`

**Interfaces:**
- Produces: `PPOConfig.learner: str = "qb"`, `PPOConfig.opponents: tuple[tuple[str, str], ...] = ()`; `--env role_play|role_drive`, `--learner`, `--opponent ID=PATH`; metadata keys `learner`, `opponents`, `fatigue`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ppo_training.py`:

```python
def test_role_play_training_smoke(tmp_path) -> None:
    from qb_sim.training.role_context_observations import RECEIVER_CONTEXT_OBSERVATION_SIZE

    config = PPOConfig(
        run_name="role-play-run",
        env_kind="role_play",
        scenario="stage_b",
        learner="wr1",
        total_timesteps=128,
        seed=0,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=1,
        attempt_decay_steps=1,
        imitation_episodes=1,
        imitation_epochs=2,
        output_dir=str(tmp_path),
        device="cpu",
    )

    _, final_path = train(config)

    metadata = json.loads((tmp_path / "role-play-run" / "config.json").read_text())
    assert metadata["env_kind"] == "role_play"
    assert metadata["learner"] == "wr1"
    assert metadata["observation_size"] == RECEIVER_CONTEXT_OBSERVATION_SIZE
    assert metadata["fatigue"]["carrier_fatigue_per_second"] > 0.0

    loaded = PPO.load(final_path, device="cpu")
    assert loaded.observation_space.shape == (RECEIVER_CONTEXT_OBSERVATION_SIZE,)


def test_role_drive_training_smoke(tmp_path) -> None:
    config = PPOConfig(
        run_name="role-drive-run",
        env_kind="role_drive",
        scenario="stage_b",
        learner="cb1",
        total_timesteps=128,
        seed=0,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=1,
        attempt_decay_steps=1,
        imitation_episodes=1,
        imitation_epochs=2,
        output_dir=str(tmp_path),
        device="cpu",
    )

    _, final_path = train(config)

    metadata = json.loads((tmp_path / "role-drive-run" / "config.json").read_text())
    assert metadata["observation_size"] == 24
    assert metadata["learner"] == "cb1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ppo_training.py -q -k role`
Expected: FAIL with `TypeError` for `learner`.

- [ ] **Step 3: Extend the config**

In `src/qb_sim/training/ppo_config.py`, after `env_kind`:

```python
    learner: str = "qb"
    opponents: tuple[tuple[str, str], ...] = ()
```

- [ ] **Step 4: Wire the role environments into `train.py`**

In `src/qb_sim/training/train.py`:
- Imports: `RoleDriveEnv`, `RolePlayEnv`, `RECEIVER_CONTEXT_OBSERVATION_SIZE`, `DEFENDER_CONTEXT_OBSERVATION_SIZE`, `collect_role_drive_demonstrations`, `collect_role_play_demonstrations`, `load_drive_policy`, `load_policy`, `load_receiver_policy`, `load_defender_policy`, `STAGE_B_SCENARIO`.
- Add:

```python
def build_opponents(env_kind: str, scenario_name: str, pairs) -> dict:
    """Load frozen opponent policies with the context-aware adapters."""
    from qb_sim.training.policy import (
        load_defender_policy,
        load_drive_policy,
        load_policy,
        load_receiver_policy,
    )

    opponents: dict = {}
    for player_id, path in pairs:
        if player_id == "qb":
            loader = load_drive_policy if env_kind in ("drive", "role_drive") else load_policy
            opponents[player_id] = loader(path, scenario_name)
        elif player_id in ("wr1", "wr2"):
            opponents[player_id] = load_receiver_policy(path, scenario_name, player_id)
        else:
            opponents[player_id] = load_defender_policy(path, scenario_name, player_id)
    return opponents
```

- Extend `build_env`:

```python
def build_env(
    scenario_names: tuple[str, ...],
    env_kind: str = "passing",
    learner: str = "qb",
    opponents: tuple[tuple[str, str], ...] = (),
):
    """Create a headless environment for the requested training mode."""
    if env_kind not in ("passing", "drive", "role_play", "role_drive"):
        raise ValueError(f"unknown env_kind: {env_kind!r}")
    if env_kind in ("drive", "role_play", "role_drive") and len(scenario_names) != 1:
        raise ValueError(f"{env_kind} training supports exactly one scenario")
    if env_kind == "drive":
        scenario = get_scenario(scenario_names[0])
        return QBDriveEnv(
            scenario,
            drive_config=DriveConfig(),
            opponents=build_opponents(env_kind, scenario_names[0], opponents),
        )
    if env_kind == "role_play":
        scenario = get_scenario(scenario_names[0])
        return RolePlayEnv(
            scenario,
            learner_id=learner,
            opponents=build_opponents(env_kind, scenario_names[0], opponents),
        )
    if env_kind == "role_drive":
        scenario = get_scenario(scenario_names[0])
        return RoleDriveEnv(
            scenario,
            learner_id=learner,
            opponents=build_opponents(env_kind, scenario_names[0], opponents),
        )
    if len(scenario_names) == 1:
        return QBPassingEnv(get_scenario(scenario_names[0]))
    return MultiScenarioEnv(scenario_names)
```

- Update the two `build_env(scenarios, config.env_kind)` calls to
  `build_env(scenarios, config.env_kind, config.learner, config.opponents)`.
- Imitation branch: add

```python
        elif config.env_kind == "role_play":
            datasets = [
                collect_role_play_demonstrations(
                    get_scenario(name),
                    learner_id=config.learner,
                    episodes=config.imitation_episodes,
                    seed_start=config.seed,
                    opponents=build_opponents(config.env_kind, name, config.opponents),
                )
                for name in scenarios
            ]
        elif config.env_kind == "role_drive":
            datasets = [
                collect_role_drive_demonstrations(
                    get_scenario(name),
                    learner_id=config.learner,
                    episodes=config.imitation_episodes,
                    seed_start=config.seed,
                    opponents=build_opponents(config.env_kind, name, config.opponents),
                )
                for name in scenarios
            ]
```

- Metadata: add

```python
    metadata["learner"] = config.learner
    metadata["opponents"] = [list(pair) for pair in config.opponents]
    if config.env_kind in ("role_play", "role_drive", "drive"):
        stage_b = get_scenario(scenarios[0]).config
        metadata["fatigue"] = {
            "carrier_fatigue_per_second": stage_b.carrier_fatigue_per_second,
            "carrier_min_speed_factor": stage_b.carrier_min_speed_factor,
        }
```

- `observation_size` selection:

```python
    from qb_sim.training.role_context_observations import (
        DEFENDER_CONTEXT_OBSERVATION_SIZE,
        RECEIVER_CONTEXT_OBSERVATION_SIZE,
    )

    if config.env_kind == "drive":
        metadata["observation_size"] = DRIVE_OBSERVATION_SIZE
    elif config.env_kind == "role_play":
        metadata["observation_size"] = RECEIVER_CONTEXT_OBSERVATION_SIZE
    elif config.env_kind == "role_drive":
        metadata["observation_size"] = DEFENDER_CONTEXT_OBSERVATION_SIZE
    else:
        metadata["observation_size"] = OBSERVATION_SIZE
```
- CLI: `--env` choices extended; `--learner` default `defaults.learner`; `--opponent` action append; parse pairs with `entry.split("=", 1)`; pass `learner=` and `opponents=tuple(pairs)` into `PPOConfig`.

- [ ] **Step 5: Run tests and full suite**

Run: `python -m pytest tests/test_ppo_training.py -q` → PASS (may take ~1–2 min)
Run: `python -m pytest -q` → PASS
Run: `ruff check .` → clean

---

### Task 10: Role evaluation and composition script

**Files:**
- Create: `src/qb_sim/training/role_evaluation.py`
- Create: `scripts/evaluate_stage_b.py`
- Test: `tests/test_role_evaluation.py`

**Interfaces:**
- Produces: `summarize_role_play(results)`, `summarize_role_drive(results)`, `summarize_composition(results)`, `evaluate_role_play`, `evaluate_role_drive`, `evaluate_composition` (all returning JSON-friendly dicts with Wilson intervals), and `run_composition_episode(env, qb_policy, seed)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_role_evaluation.py`:

```python
import pytest

from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.evaluation import RuleQuarterbackPolicy
from qb_sim.training.role_evaluation import (
    evaluate_composition,
    summarize_composition,
)


def test_summarize_composition_computes_team_metrics() -> None:
    results = [
        {
            "outcome": "touchdown",
            "points": 7.0,
            "first_downs": 2,
            "plays": [
                {"down": 3, "distance": 12.0, "first_down": True, "touchdown": False,
                 "throw_distance": 18.0, "outcome": "completed", "coverage_sack": False},
            ],
            "separation_at_throw": 3.0,
            "separation_at_catch": 1.0,
            "pressure_at_throw": 0.4,
            "arrival_time": 1.6,
        },
        {
            "outcome": "turnover_on_downs",
            "points": 0.0,
            "first_downs": 0,
            "plays": [
                {"down": 3, "distance": 5.0, "first_down": False, "touchdown": False,
                 "throw_distance": 9.0, "outcome": "incomplete", "coverage_sack": False},
            ],
            "separation_at_throw": 2.0,
            "separation_at_catch": None,
            "pressure_at_throw": None,
            "arrival_time": None,
        },
    ]

    summary = summarize_composition(results)

    assert summary["episodes"] == 2.0
    assert summary["touchdown_pct"] == pytest.approx(50.0)
    assert summary["points_per_drive"] == pytest.approx(3.5)
    assert summary["third_down_conversion_pct"] == pytest.approx(50.0)
    assert summary["mean_separation_at_throw"] == pytest.approx(2.5)


def test_evaluate_composition_with_the_rule_quarterback_is_reproducible() -> None:
    scenario = get_scenario("stage_b")
    policy = RuleQuarterbackPolicy(scenario.config)

    first = evaluate_composition(policy, {}, "stage_b", range(2))
    second = evaluate_composition(policy, {}, "stage_b", range(2))

    assert first == second


def test_qb_drive_env_calls_context_aware_opponents() -> None:
    from qb_sim.training.drive_env import QBDriveEnv

    calls: list[object] = []

    def opponent(state, route):
        calls.append(route)
        return None

    env = QBDriveEnv(get_scenario("stage_b"), opponents={"wr1": opponent})
    env.reset(seed=0)
    env.step(np.zeros(6, dtype=np.float32))

    assert calls
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_role_evaluation.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `role_evaluation.py`**

Create `src/qb_sim/training/role_evaluation.py`:

```python
"""Role-level and composition metrics for Stage B over fixed seed sets."""

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.drive_env import QBDriveEnv
from qb_sim.training.drive_evaluation import wilson_interval
from qb_sim.training.role_env import RoleDriveEnv, RolePlayEnv

QBPolicy = Callable[[QBDriveEnv, np.ndarray], np.ndarray]


def _pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def _ci(count: int, total: int) -> list[float]:
    low, high = wilson_interval(count, total)
    return [100.0 * low, 100.0 * high]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def run_role_play_episode(env: RolePlayEnv, action_fn, seed: int) -> dict[str, Any]:
    """Run one seeded play with a learner policy and collect receiver metrics."""
    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        action = np.asarray(action_fn(env, observation), dtype=np.float32)
        observation, _, terminated, truncated, info = env.step(action)
    return dict(info)


def evaluate_role_play(action_fn, scenario_name: str, seeds: Iterable[int], **kwargs) -> dict:
    env = RolePlayEnv(get_scenario(scenario_name), **kwargs)
    results = [run_role_play_episode(env, action_fn, seed) for seed in seeds]
    env.close()
    return summarize_role_play(results)


def summarize_role_play(results: list[dict[str, Any]]) -> dict:
    total = len(results)
    if total == 0:
        raise ValueError("at least one play is required")
    catches = sum(bool(result["caught"]) for result in results)
    touchdowns = sum(result["outcome"] == "touchdown" for result in results)
    separations = [
        float(result["separation_at_throw"])
        for result in results
        if result["separation_at_throw"] is not None
    ]
    return {
        "episodes": float(total),
        "catch_pct": _pct(catches, total),
        "catch_ci": _ci(catches, total),
        "touchdown_pct": _pct(touchdowns, total),
        "touchdown_ci": _ci(touchdowns, total),
        "yards_per_play": _mean([float(result["yards"]) for result in results]),
        "mean_separation_at_throw": _mean(separations),
    }


def run_role_drive_episode(env: RoleDriveEnv, action_fn, seed: int) -> dict[str, Any]:
    """Run one seeded drive with a learner policy and collect defender metrics."""
    env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        observation = env.observation()
        action = np.asarray(action_fn(env, observation), dtype=np.float32)
        _, _, terminated, truncated, _ = env.step(action)
    state = env.drive.state
    return {
        "outcome": state.end_reason or "truncated",
        "points": state.points,
        "plays": list(env.play_history),
    }


def evaluate_role_drive(action_fn, scenario_name: str, seeds: Iterable[int], **kwargs) -> dict:
    env = RoleDriveEnv(get_scenario(scenario_name), **kwargs)
    results = [run_role_drive_episode(env, action_fn, seed) for seed in seeds]
    env.close()
    return summarize_role_drive(results)


def summarize_role_drive(results: list[dict[str, Any]]) -> dict:
    total = len(results)
    if total == 0:
        raise ValueError("at least one drive is required")
    plays = [play for result in results for play in result["plays"]]
    sacks = sum(play["outcome"] == "sack" for play in plays)
    pressures = [
        float(play["pressure_at_throw"])
        for play in plays
        if play["pressure_at_throw"] is not None
    ]
    arrivals = [float(play["arrival_time"]) for play in plays if play["arrival_time"] is not None]
    separations = [
        float(play["separation_allowed_throw"])
        for play in plays
        if play["separation_allowed_throw"] is not None
    ]
    touchdowns = sum(result["outcome"] == "touchdown" for result in results)
    return {
        "episodes": float(total),
        "touchdown_pct_allowed": _pct(touchdowns, total),
        "sack_pct": _pct(sacks, len(plays)),
        "sack_ci": _ci(sacks, len(plays)),
        "mean_pressure_at_throw": _mean(pressures),
        "mean_arrival_time": _mean(arrivals),
        "mean_separation_allowed_throw": _mean(separations),
        "plays_per_drive": len(plays) / total,
    }


def run_composition_episode(env: QBDriveEnv, qb_policy: QBPolicy, seed: int) -> dict[str, Any]:
    """Run one seeded drive with the QB policy and frozen role opponents."""
    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    separation_throw: float | None = None
    separation_catch: float | None = None
    pressure_throw: float | None = None
    arrival: float | None = None
    while not (terminated or truncated):
        pre_state = env.play.state
        action = np.asarray(qb_policy(env, observation), dtype=np.float32)
        observation, _, terminated, truncated, _ = env.step(action)
        current = env.play.state
        if separation_throw is None and current.throw_time is not None:
            separation_throw = _target_separation(env)
            pressure_throw = float(pre_state.pressure)
        if separation_catch is None and current.carrier_id is not None:
            separation_catch = _target_separation(env)
        if arrival is None and env.play.rusher_ids:
            rusher = current.player(env.play.rusher_ids[0])
            quarterback = current.player("qb")
            if float(np.linalg.norm(rusher.position - quarterback.position)) <= env.play.config.tackle_radius:
                arrival = float(current.elapsed_seconds)
    state = env.drive.state
    return {
        "outcome": state.end_reason or "truncated",
        "points": state.points,
        "first_downs": state.first_downs,
        "plays": list(env.play_history),
        "separation_at_throw": separation_throw,
        "separation_at_catch": separation_catch,
        "pressure_at_throw": pressure_throw,
        "arrival_time": arrival,
    }


def _target_separation(env: QBDriveEnv) -> float | None:
    state = env.play.state
    if state.carrier_id is not None:
        receiver_id = state.carrier_id
    else:
        target = env.play.state.ball.target
        if target is None:
            return None
        receiver_id = min(
            env.play.receiver_ids,
            key=lambda player_id: float(np.linalg.norm(state.player(player_id).position - target)),
        )
    receiver = state.player(receiver_id)
    defenders = [state.player(player_id).position for player_id in env.play.defender_ids]
    if not defenders:
        return None
    return float(min(np.linalg.norm(receiver.position - defender) for defender in defenders))


def evaluate_composition(
    qb_policy: QBPolicy,
    opponents: dict,
    scenario_name: str,
    seeds: Iterable[int],
) -> dict:
    env = QBDriveEnv(get_scenario(scenario_name), opponents=opponents)
    results = [run_composition_episode(env, qb_policy, seed) for seed in seeds]
    env.close()
    return summarize_composition(results)


def summarize_composition(results: list[dict[str, Any]]) -> dict:
    total = len(results)
    if total == 0:
        raise ValueError("at least one drive is required")
    touchdowns = sum(result["outcome"] == "touchdown" for result in results)
    interceptions = sum(result["outcome"] == "interception" for result in results)
    turnovers = sum(result["outcome"] == "turnover_on_downs" for result in results)
    plays = [play for result in results for play in result["plays"]]
    third_downs = [play for play in plays if play["down"] == 3]
    conversions = sum(bool(play["first_down"] or play["touchdown"]) for play in third_downs)
    sacks = sum(play["outcome"] == "sack" for play in plays)
    separations = [
        float(result["separation_at_throw"])
        for result in results
        if result["separation_at_throw"] is not None
    ]
    pressures = [
        float(result["pressure_at_throw"])
        for result in results
        if result["pressure_at_throw"] is not None
    ]
    arrivals = [float(result["arrival_time"]) for result in results if result["arrival_time"] is not None]
    return {
        "episodes": float(total),
        "touchdown_pct": _pct(touchdowns, total),
        "touchdown_ci": _ci(touchdowns, total),
        "points_per_drive": _mean([float(result["points"]) for result in results]),
        "first_downs_per_drive": _mean([float(result["first_downs"]) for result in results]),
        "plays_per_drive": len(plays) / total,
        "third_down_conversion_pct": _pct(conversions, len(third_downs)),
        "third_down_conversion_ci": _ci(conversions, len(third_downs)),
        "interception_pct": _pct(interceptions, total),
        "interception_ci": _ci(interceptions, total),
        "turnover_on_downs_pct": _pct(turnovers, total),
        "sack_pct": _pct(sacks, len(plays)),
        "mean_separation_at_throw": _mean(separations),
        "mean_pressure_at_throw": _mean(pressures),
        "mean_arrival_time": _mean(arrivals),
    }
```

Note: `QBDriveEnv.opponents` must accept context-aware callables — Task 9 changes `build_opponents`, but the env itself must route them. Add to Task 9 (or this task) the `decide_player_action` usage in `QBDriveEnv.step`/`_build_agents`: when an opponent is not a `BaseAgent`, call `decide_player_action(play, drive.routes, player_id, opponent, drive.state)`. Implement that in this task:

In `src/qb_sim/training/drive_env.py`, replace the agent loop:

```python
        for player_id, agent in self._agents.items():
            if player_id in self.opponents:
                agent_action = decide_player_action(
                    self.play,
                    self.drive.routes,
                    player_id,
                    self.opponents[player_id],
                    self.drive.state,
                )
            else:
                agent_action = agent.decide(self.play.state)
            if agent_action is not None:
                engine_actions[player_id] = agent_action
```

with `from qb_sim.training.role_env import decide_player_action`. Keep the `_build_agents` override `self._agents.update(self.opponents)` (rules replaced by opponents; the loop decides by membership).

- [ ] **Step 4: Create the script**

Create `scripts/evaluate_stage_b.py`:

```python
"""Evaluate Stage B roles and the frozen composition on fixed seeds.

Examples:

    python scripts/evaluate_stage_b.py wr --episodes 100 \
        --ppo wr1=runs/ppo_wr_v1/final_model.zip \
        --opponent qb=runs/ppo_rushed_v10/final_model.zip
    python scripts/evaluate_stage_b.py composition --episodes 100 \
        --ppo qb=runs/ppo_drive_qb_v12/final_model.zip \
        --opponent wr1=runs/ppo_wr_v1/final_model.zip \
        --opponent cb1=runs/ppo_cb_v1/final_model.zip
"""

import argparse
import json
from pathlib import Path

import numpy as np

from qb_sim.rules.scenarios import get_scenario
from qb_sim.training.evaluation import PPOActionPolicy, RuleQuarterbackPolicy
from qb_sim.training.role_evaluation import (
    evaluate_composition,
    evaluate_role_drive,
    evaluate_role_play,
)
from qb_sim.training.train import build_opponents

LEARNERS = {"wr": "wr1", "cb": "cb1", "ru": "ru1"}
ENV_KINDS = {"composition": "drive", "wr": "role_play", "cb": "role_drive", "ru": "role_drive"}


def _pairs(entries: list[str]) -> tuple[tuple[str, str], ...]:
    return tuple(entry.split("=", 1) for entry in entries)


def _rule_role_action(env, observation) -> np.ndarray:
    action = env.rule_agent(env.learner_id).decide(env.play.state)
    if action is None:
        return np.zeros(2, dtype=np.float32)
    return np.array([action.direction[0], action.direction[1]], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage B evaluation")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("composition", "wr", "cb", "ru"):
        sub = subparsers.add_parser(mode)
        sub.add_argument("--ppo", action="append", default=[], metavar="LABEL=PATH")
        sub.add_argument("--opponent", action="append", default=[], metavar="ID=PATH")
        sub.add_argument("--episodes", type=int, default=100)
        sub.add_argument("--seed-start", type=int, default=10_000)
        sub.add_argument("--scenario", default="stage_b")
        sub.add_argument("--output", default=None)
    args = parser.parse_args()

    seeds = range(args.seed_start, args.seed_start + args.episodes)
    scenario = get_scenario(args.scenario)
    env_kind = ENV_KINDS[args.mode]
    opponents = build_opponents(env_kind, args.scenario, _pairs(args.opponent))
    results: dict[str, dict] = {}

    if args.mode == "composition":
        results["rules"] = evaluate_composition(
            RuleQuarterbackPolicy(scenario.config), opponents, args.scenario, seeds
        )
        for label, path in _pairs(args.ppo):
            results[label] = evaluate_composition(
                PPOActionPolicy(path), opponents, args.scenario, seeds
            )
    else:
        learner = LEARNERS[args.mode]
        env_kwargs = {"learner_id": learner, "opponents": opponents}
        evaluate = evaluate_role_play if args.mode == "wr" else evaluate_role_drive
        results["rules"] = evaluate(_rule_role_action, args.scenario, seeds, **env_kwargs)
        for label, path in _pairs(args.ppo):
            results[label] = evaluate(PPOActionPolicy(path), args.scenario, seeds, **env_kwargs)

    for name, summary in results.items():
        print(f"\n{args.mode} — {name}")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    output = Path(args.output or f"runs/stage_b_evaluation_{args.mode}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests and full suite**

Run: `python -m pytest tests/test_role_evaluation.py -q` → PASS
Run: `python -m pytest -q` → PASS
Run: `ruff check .` → clean

---

### Task 11: WR pilot, gate, and long run (`ppo_wr_v1`)

**Files:** none (experiment); outputs in `runs/` (gitignored).

- [ ] **Step 1: Pilot (50k, BC warm start)**

Note (2026-10-02 ruling): the WR trains against the **frozen drive QB v11**,
which is route-aware. v10 cannot play random routes (3% catches) and the rule
QB's random-route baseline is only 14%; `RolePlayEnv` passes a default drive
context to the QB opponent so `DrivePolicyQuarterback` (57-float obs) works.

```bash
python -m qb_sim.training.train --env role_play --scenario stage_b --learner wr1 \
  --opponent qb=runs/ppo_drive_qb_v11/final_model.zip \
  --timesteps 50000 --seed 0 --run-name ppo_wr_v1_pilot \
  --imitation-episodes 300 --imitation-epochs 800 --imitation-learning-rate 1e-4
```

- [ ] **Step 2: Evaluate the pilot and the rule-WR baseline**

```bash
python scripts/evaluate_stage_b.py wr --episodes 100 --seed-start 10000 \
  --ppo wr1=runs/ppo_wr_v1_pilot/final_model.zip --output runs/stage_b_wr_pilot.json
```

- [ ] **Step 3: Apply the gate**

Pass if: `catch_pct >= rules + 5` and (`touchdown_pct >= rules` or `yards_per_play >= rules` or `mean_separation_at_throw >= rules`). If it fails, adjust only `ReceiverRewardConfig.catch_bonus` or the fatigue values, document, and re-run the pilot once per adjustment.

- [ ] **Step 4: Long run (250k)**

```bash
python -m qb_sim.training.train --env role_play --scenario stage_b --learner wr1 \
  --opponent qb=runs/ppo_drive_qb_v11/final_model.zip \
  --timesteps 250000 --seed 0 --run-name ppo_wr_v1 \
  --imitation-episodes 300 --imitation-epochs 800 --imitation-learning-rate 1e-4
```

- [ ] **Step 5: Final WR evaluation and report**

```bash
python scripts/evaluate_stage_b.py wr --episodes 100 --seed-start 10000 \
  --ppo wr1=runs/ppo_wr_v1/final_model.zip --output runs/stage_b_wr_v1.json
```

Report the table (rules vs WR) with CIs before moving to Task 12.

---

### Task 12: CB pilot, gate, and long run (`ppo_cb_v1`)

**Files:** none (experiment).

- [ ] **Step 1: Pilot (50k) with frozen QB v11 + WR v1**

```bash
python -m qb_sim.training.train --env role_drive --scenario stage_b --learner cb1 \
  --opponent qb=runs/ppo_drive_qb_v11/final_model.zip \
  --opponent wr1=runs/ppo_wr_v1/final_model.zip \
  --opponent wr2=runs/ppo_wr_v1/final_model.zip \
  --timesteps 50000 --seed 0 --run-name ppo_cb_v1_pilot \
  --imitation-episodes 300 --imitation-epochs 800 --imitation-learning-rate 1e-4
```

- [ ] **Step 2: Evaluate the pilot vs rules**

```bash
python scripts/evaluate_stage_b.py cb --episodes 100 --seed-start 10000 \
  --ppo cb1=runs/ppo_cb_v1_pilot/final_model.zip --output runs/stage_b_cb_pilot.json
```

- [ ] **Step 3: Gate**

Pass if the completion percentage allowed is ≤ rules − 5 points and the mean separation allowed is ≤ rules, without lowering forced interceptions below the baseline. Adjust only the defender reward/fatigue if needed, document, and re-run once per adjustment.

- [ ] **Step 4: Long run (250k) → `ppo_cb_v1` and final evaluation** (same commands with `--run-name ppo_cb_v1`, output `runs/stage_b_cb_v1.json`).

---

### Task 13: Rusher pilot, gate, and long run (`ppo_ru_v1`)

**Files:** none (experiment).

- [ ] **Step 1: Pilot (50k) with frozen QB v11 + WR v1 + CB v1**

```bash
python -m qb_sim.training.train --env role_drive --scenario stage_b --learner ru1 \
  --opponent qb=runs/ppo_drive_qb_v11/final_model.zip \
  --opponent wr1=runs/ppo_wr_v1/final_model.zip \
  --opponent wr2=runs/ppo_wr_v1/final_model.zip \
  --opponent cb1=runs/ppo_cb_v1/final_model.zip \
  --opponent cb2=runs/ppo_cb_v1/final_model.zip \
  --timesteps 50000 --seed 0 --run-name ppo_ru_v1_pilot \
  --imitation-episodes 300 --imitation-epochs 800 --imitation-learning-rate 1e-4
```

- [ ] **Step 2: Evaluate and gate**

```bash
python scripts/evaluate_stage_b.py ru --episodes 100 --seed-start 10000 \
  --ppo ru1=runs/ppo_ru_v1_pilot/final_model.zip --output runs/stage_b_ru_pilot.json
```

Pass if `sack_pct >= rules + 3` or `mean_pressure_at_throw >= rules`; adjust only if needed and document.

- [ ] **Step 3: Long run (250k) → `ppo_ru_v1` and final evaluation** (`runs/stage_b_ru_v1.json`).

---

### Task 14: Composition, QB v12, and docs

**Files:**
- Modify: `AGENTS.md` (§11)

- [ ] **Step 1: Composition evaluation (all trained roles + QB v11)**

```bash
python scripts/evaluate_stage_b.py composition \
  --ppo qb=runs/ppo_drive_qb_v11/final_model.zip \
  --opponent wr1=runs/ppo_wr_v1/final_model.zip \
  --opponent wr2=runs/ppo_wr_v1/final_model.zip \
  --opponent cb1=runs/ppo_cb_v1/final_model.zip \
  --opponent cb2=runs/ppo_cb_v1/final_model.zip \
  --opponent ru1=runs/ppo_ru_v1/final_model.zip \
  --episodes 100 --seed-start 10000 --output runs/stage_b_composition_v11.json
```

- [ ] **Step 2: Re-fine-tune the drive QB against the trained roles → `ppo_drive_qb_v12`**

```bash
python -m qb_sim.training.train --env drive --scenario stage_b --timesteps 250000 --seed 0 \
  --run-name ppo_drive_qb_v12 --resume runs/ppo_drive_qb_v11/final_model.zip \
  --opponent wr1=runs/ppo_wr_v1/final_model.zip \
  --opponent wr2=runs/ppo_wr_v1/final_model.zip \
  --opponent cb1=runs/ppo_cb_v1/final_model.zip \
  --opponent cb2=runs/ppo_cb_v1/final_model.zip \
  --opponent ru1=runs/ppo_ru_v1/final_model.zip \
  --imitation-episodes 0
```

Note: `QBDriveEnv` loads opponents through `build_opponents`, so `train.py` must pass them for `env_kind="drive"` too (Task 9 covers this).

- [ ] **Step 3: Final composition with v12 and comparison**

```bash
python scripts/evaluate_stage_b.py composition \
  --ppo qb=runs/ppo_drive_qb_v12/final_model.zip \
  --opponent wr1=runs/ppo_wr_v1/final_model.zip \
  --opponent wr2=runs/ppo_wr_v1/final_model.zip \
  --opponent cb1=runs/ppo_cb_v1/final_model.zip \
  --opponent cb2=runs/ppo_cb_v1/final_model.zip \
  --opponent ru1=runs/ppo_ru_v1/final_model.zip \
  --episodes 100 --seed-start 10000 --output runs/stage_b_composition_v12.json
```

Report v11 vs v12 vs rules with CIs.

- [ ] **Step 4: Update `AGENTS.md` §11** with Stage B status: carrier/fatigue mechanics, role observations 28/24, trained runs `ppo_wr_v1`/`ppo_cb_v1`/`ppo_ru_v1`, composition numbers, `ppo_drive_qb_v12`, and the remaining M7/M8 work.

- [ ] **Step 5: Final verification**

Run: `python -m pytest -q` → PASS
Run: `ruff check .` → clean
