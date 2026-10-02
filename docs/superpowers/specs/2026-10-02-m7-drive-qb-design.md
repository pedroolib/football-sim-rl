# M7 — Drive System, Route Randomization, and QB v11: Design

**Goal:** Train a quarterback that makes strategic decisions inside a full drive
(3 downs, first downs, touchdown) with randomly sampled receiver routes, and
observe down/distance/route context in its observation.

**Scope of this spec:** routes + drive rules + drive observation + `QBDriveEnv` +
QB v11 training and evaluation. Per-role training (WR/CB/rusher) and composition
come afterwards in their own cycle.

**Status:** sections A (rules), B (observation/reward/env), and C (training/
versioning/tests) approved by the user on 2026-10-02.

## 1. Context

- Field is 120 yards long; the offense attacks `+x`. Own goal is `x=0`; the
  opponent goal line is fixed at `x=100` (the 20 yards beyond it act as the end
  zone; arcade simplification, no own-goal/safety logic).
- `Play` always starts at the scenario's fixed positions (QB at `x=20`) and has
  no concept of line of scrimmage (LOS), first downs, or touchdowns.
- Routes are fixed per scenario (`go`/`slant`/`post`).
- The QB observation is 36 floats without down, distance, or route context.
- `runs/ppo_rushed_v10` stays frozen: it is the play-level baseline and an
  opponent for Stage B. The drive QB is a new model (`v11`) with a new layout.

## 2. Approved decisions

1. QB trains inside drives (episodio = drive); WR/CB/rusher keep play-level
   training later. Order: drive + QB v11 first, then skill positions.
2. Drive rules: minimum set, **3 downs**, 10 yards for a first down, TD at the
   goal line, interception or turnover on downs ends the drive. No punts, no
   field goals, no clock, no safeties.
3. Initial LOS random in `[25, 50]` per drive, sampled with the drive seed.
4. QB observation includes a per-receiver route one-hot (a real QB knows the
   called routes). New layout is incompatible with v10 by design.
5. Reward: per-play rewards (reused) + first-down bonus + touchdown + turnover
   penalty; `gamma` propagates future drive value.
6. Routes: expand to 8 and sample per play per receiver.
7. Time limit is a safety cap. A drive maps "expired with the ball in hand"
   (never threw) to a coverage sack; expiry with the ball in flight counts as
   an incompletion. The engine keeps emitting `TIME_LIMIT` so M5 comparisons
   with v10 stay valid.
8. Drive lives in `rules` (`rules/drive.py`); the Gym wrapper lives in
   `training/drive_env.py`.

## 3. Routes

### 3.1 New route names

`RouteName` gains `OUT`, `CURL`, `FLAT`, `WHEEL`, `CORNER`. Fixed observation
order (one-hot): `ROUTE_OBSERVATION_ORDER = (GO, SLANT, POST, OUT, CURL, FLAT, WHEEL, CORNER)`.
`DEFAULT_ROUTE_POOL` is that same tuple.

### 3.2 New `SimulationConfig` parameters

| Field | Default | Used by |
|---|---|---|
| `out_break_yards` | `5.0` | OUT vertical stem |
| `out_angle_degrees` | `75.0` | OUT cut |
| `curl_depth_yards` | `10.0` | CURL stem |
| `curl_back_yards` | `3.0` | CURL return |
| `flat_length_yards` | `6.0` | FLAT lateral length |
| `wheel_break_yards` | `4.0` | WHEEL lateral break |
| `wheel_length_yards` | `20.0` | WHEEL vertical leg |
| `corner_break_yards` | `12.0` | CORNER stem |
| `corner_angle_degrees` | `45.0` | CORNER cut |

Validation: lengths positive; angles in `[0, 90]`.

### 3.3 Waypoints

Offense runs `+x`; `outward_sign = -inward_sign` (toward the nearest sideline).
All waypoints are clamped to the field as today.

- `GO`: `(start + (go_length_yards, 0))` (unchanged).
- `SLANT` / `POST`: unchanged.
- `OUT`: break at `+out_break_yards`, then outward diagonal of
  `cut_length_yards` at `out_angle_degrees`.
- `CURL`: waypoint at `+curl_depth_yards`, then `(-curl_back_yards, 0)`.
- `FLAT`: single waypoint `start + (0, outward_sign * flat_length_yards)`.
- `WHEEL`: waypoint `start + (0, outward_sign * wheel_break_yards)`, then
  `+ (wheel_length_yards, 0)` from there.
- `CORNER`: break at `+corner_break_yards`, then outward diagonal of
  `cut_length_yards` at `corner_angle_degrees`.

### 3.4 Sampling

`sample_routes(rng, count, pool=DEFAULT_ROUTE_POOL) -> tuple[RouteName, ...]`
draws uniform indices with `rng.integers(0, len(pool), size=count)`.
Receiver `i` (roster order) gets `routes[i]`. The same route may repeat across
receivers. Fixed-route scenarios (`Scenario.receiver_routes`) remain for the
play-level env and M5.

## 4. Play changes

### 4.1 Line of scrimmage

`Play(config=None, seed=0, line_of_scrimmage: float | None = None)`.

- Default `None` keeps today's behavior exactly (LOS = `config.quarterback_start[0]`).
- `reset()` shifts every starting position by `dx = line_of_scrimmage - config.quarterback_start[0]`
  (formation preserved). Shifted starts must stay inside the field; otherwise
  `ValueError` at construction time.
- `Play.line_of_scrimmage` property returns the effective LOS.

### 4.2 Touchdown

- `Play` gains `goal_line: float | None = None`. Touchdown detection is only
  active when the drive passes its `DriveConfig.goal_line` (default `100.0`).
  With the default `None`, `Play` keeps the exact M5 behavior so existing
  scenarios and tests (which intentionally use `x > 100`) are unaffected.
- New `PlayOutcome.TOUCHDOWN`.
- A TD ends the play immediately when:
  - the QB crosses `x >= goal_line` while holding the ball (scramble), checked
    before the sack check; or
  - the carrier crosses `x >= goal_line`, checked before the tackle check; or
  - a completed catch happens at a contact point `x >= goal_line` (catch in the
    end zone, no chance to be tackled).
- Ordering: touchdown wins over a simultaneous tackle/sack.
- Existing scenarios cannot reach `x=100` in 6 s, so M5 behavior is unchanged.

### 4.3 Spots (used by the drive)

| Play result | Spot |
|---|---|
| `COMPLETED` | carrier `x` at the end (clamped to `[0, goal_line]`) |
| `SACK` | QB `x` (can be behind the LOS) |
| `INCOMPLETE`, `OUT_OF_BOUNDS` | LOS |
| expired with ball in flight | LOS (incompletion) |
| expired with ball in hand (`throw_time is None`) | QB `x` (coverage sack) |
| `INTERCEPTION`, `TOUCHDOWN` | drive ends; spot irrelevant |

The engine still emits `TIME_LIMIT`; the coverage-sack mapping lives in the
drive layer and is detectable because `PlayState.throw_time` stays `None` when
the QB never threw.

## 5. Drive rules (`rules/drive.py`)

```python
@dataclass(frozen=True, slots=True)
class DriveConfig:
    downs: int = 3
    first_down_yards: float = 10.0
    los_range: tuple[float, float] = (25.0, 50.0)
    max_plays: int = 20
    touchdown_points: float = 7.0  # arcade: TD + extra point
    goal_line: float = 100.0
```

`DriveState`: `down` (1-based), `distance`, `los`, `play_number`, `first_downs`,
`points`, `terminated`, `end_reason`.

`Drive(config, drive_config=None, seed=0)`:

- `reset(seed=None)`: samples the initial LOS uniformly in `los_range` and
  `distance = min(first_down_yards, goal_line - los)`; resets state and RNG.
- `routes` property: routes for the current play.
- `start_play() -> Play`: increments `play_number`, samples routes, and returns
  `Play(config, seed=<draw from drive RNG>, line_of_scrimmage=state.los)`.
- `advance(play) -> PlayResult` applies the table below and mutates the state.

`PlayResult` (frozen dataclass): `outcome`, `spot`, `yards`, `first_down`,
`touchdown`, `drive_over`, `end_reason`, `points`, `throw_distance`.

Transitions:

1. `TOUCHDOWN` → `points += touchdown_points`; `terminated`, `end_reason="touchdown"`.
2. `INTERCEPTION` → `terminated`, `end_reason="interception"` (touchback ignored).
3. Conversion when `spot - los >= distance`: first down → `first_downs += 1`,
   `down = 1`, `los = spot`, `distance = min(first_down_yards, goal_line - los)`.
4. Otherwise move the spot (`los = spot`) and `down += 1`; if `down > downs` →
   `terminated`, `end_reason="turnover_on_downs"`. Otherwise
   `distance = (old_los + old_distance) - new_los`.
5. `max_plays` reached → `terminated`, `end_reason="max_plays"` (safety).

`Drive` has no Gym/PyTorch/agents imports. A rule-driven full-drive helper for
evaluation lives in `training/drive_evaluation.py` (it needs the rule
controllers).

## 6. Observation v11 (`training/drive_observations.py`)

`DRIVE_OBSERVATION_SIZE = 57`: the existing 36 floats at indices 0–35 plus:

- 36–51: receiver-slot route one-hot in roster order (2 slots × 8 routes in
  `ROUTE_OBSERVATION_ORDER`); zeros for empty slots or missing routes.
- 52–54: down one-hot for down 1/2/3.
- 55: `min(distance, 40) / 40`.
- 56: `(goal_line - los) / goal_line`.

`drive_observation_from_state(state, routes, drive_state, config, goal_line=100.0)`
builds it by extending `observation_from_state`, then clipping to `[-1, 1]`.
`OBSERVATION_SIZE` (36) stays for the passing env.

## 7. Rewards (`training/rewards.py`)

```python
@dataclass(frozen=True, slots=True)
class DriveRewardConfig:
    play: RewardConfig = field(default_factory=RewardConfig)
    first_down: float = 0.3
    turnover_on_downs: float = -0.5
```

Per play: `terminal_reward(outcome, yards, play_reward)` with `yards = spot - los`
(completion +1.0 + 0.02/yard, interception −1.0, incompletion/out-of-bounds
−0.2, sack −1.0, coverage sack mapped to SACK). Plus `first_down` when the play
converts (not on a TD), plus `result.points` on a touchdown, plus
`turnover_on_downs` when the drive ends without converting. The env adds the
per-tick `time_penalty` and keeps optional attempt shaping with
`set_attempt_scale` for pipeline compatibility.

Strategy is expected to emerge from: the turnover penalty, the first-down bonus,
and the discounted TD value; the 3rd-and-long behavior is validated in
evaluation (`third_and_long_*` metrics) before any long run.

## 8. `QBDriveEnv` (`training/drive_env.py`, registered `QBDrive-v0`)

```python
QBDriveEnv(
    scenario=RUSHED_SCENARIO,
    drive_config=None,          # DriveConfig()
    reward_config=None,         # DriveRewardConfig()
    max_episode_steps=2400,     # 20 plays x 120 ticks safety truncation
    opponents=None,             # optional player_id -> BaseAgent overrides
)
```

- Spaces: `Box(6)` action, `Box(57)` observation.
- `reset(seed)`: new `Drive`, sample LOS, `start_play()`, build rule agents
  (`build_rule_agents` with `drive.routes`), return the observation.
- `step(action)`: decode the QB action (same 6-value scheme), gather rule-agent
  actions (with `opponents` overrides), `play.step()`, add per-tick time
  penalty. When the play ends: `drive.advance(play)`, add the play reward, and
  if the drive continues, start the next play at the new LOS with freshly
  sampled routes (same episode). Terminated on TD/interception/turnover;
  truncated at `max_episode_steps`.
- `info`: `episode_seed`, `scenario`, `down`, `distance`, `los`, `play_number`,
  `points`, `first_downs`, `outcome`, `drive_end_reason`, `throw_time`, and
  `plays` (history of per-play dicts: down, distance, los_before, outcome, spot,
  yards, first_down, touchdown, throw_distance, sack, coverage_sack).
- Deterministic for a fixed seed and action sequence.

## 9. Training pipeline

- `PPOConfig` gains `env_kind: str = "passing"`; `train.build_env` returns
  `QBDriveEnv(get_scenario(scenario))` for `"drive"`. New CLI flag `--env`.
  Recommended drive scenario: `rushed` (pass rusher on the field).
- `training/imitation.py`:
  - `BehaviorCloningNetwork(input_size, action_size=6)`.
  - `collect_drive_demonstrations(scenario, episodes, seed_start, drive_config=None)`
    runs the rule QB inside `QBDriveEnv` and records `(57-obs, 6-action)` pairs.
  - Warm start into PPO as today; a test asserts weights actually load
    (BC net output equals the PPO policy's deterministic action on sample
    observations).
- Pilot command:
  `python -m qb_sim.training.train --env drive --scenario rushed --timesteps 50000 --run-name ppo_drive_qb_v11_pilot`
- Gate: pilot beats the rule QB on 3rd-down conversion and shows deeper throws
  on 3rd and long. If not, adjust rewards (documented) and re-run the pilot
  before any long run.
- Long run: `--timesteps 250000 --run-name ppo_drive_qb_v11`.
- `config.json` metadata: `env_kind`, drive config, route pool order, scenario,
  imitation samples/loss.

## 10. Evaluation (`training/drive_evaluation.py`, `scripts/evaluate_drive.py`)

- Fixed evaluation seeds, separate from training (default `10000+`, 100 drives).
- Policies: `rules` (heuristic QB via the drive env) and any PPO model whose
  observation size is 57 (`supports_drive_layout`).
- Metrics: `touchdown_pct`, `points_per_drive`, `first_downs_per_drive`,
  `plays_per_drive`, `third_down_conversion_pct`,
  `third_and_long_conversion_pct` (distance ≥ 8),
  `third_and_long_throw_depth_yards`, `checkdown_pct` on 3rd and long
  (target distance ≤ 10), `interception_pct`, `sack_pct`, `coverage_sack_pct`,
  `incomplete_pct`.
- Wilson 95% intervals for the proportion metrics (no SciPy dependency).
- Output JSON to `runs/drive_evaluation.json`; the rule QB is always the
  baseline on the same seeds. Never pool drive metrics with play-level metrics
  from v10 evaluations.

## 11. Versioning and compatibility

- `runs/ppo_rushed_v10` is untouched; it keeps serving play-level evaluation
  and Stage B opponents.
- v11 is a fresh model with the 57-float layout; `PPOActionPolicy.supports_layout`
  (36) naturally rejects it and the drive evaluator rejects non-57 models.
- Play-level env, multiagent env, UI, and human mode keep working unchanged:
  `Play(goal_line=None)` disables touchdown detection, so no existing
  trajectory changes. The drive enables it explicitly with `goal_line=100.0`.

## 12. Testing

TDD for every piece; the suite currently has 229 passing tests.

1. `tests/test_routes.py`: waypoints for `out/curl/flat/wheel/corner`; clamping;
   `sample_routes` determinism and uniform coverage.
2. `tests/test_play.py`: LOS shift preserves formation and validates bounds;
   scramble TD; run-after-catch TD; catch in the end zone; touchdown priority
   over tackle; default behavior unchanged.
3. `tests/test_drive.py`: 3-down transitions; conversion → 1st and 10 /
   1st and goal; distance recompute on non-conversion; turnover on downs;
   interception and touchdown end the drive; coverage-sack mapping; points;
   determinism for a fixed seed.
4. `tests/test_drive_observations.py`: size 57; range; route/down one-hots;
   empty slots zeroed; distance and yards-to-goal normalization.
5. `tests/test_drive_env.py`: Gymnasium `check_env`; first-down/TD/turnover
   rewards; termination/truncation; play history; same-seed determinism.
6. `tests/test_imitation.py`: drive BC training and verified weight loading.
7. `tests/test_ppo_training.py`: short PPO smoke test with `env_kind="drive"`.
8. Reproducibility: same seed + same action sequence reproduces a full drive.

## 13. Execution order

1. Routes + sampling (+ tests).
2. `Play`: LOS, touchdown, spots (+ tests).
3. `Drive` (+ tests).
4. Observation v11 + `QBDriveEnv` (+ tests).
5. BC + pilot (50k) + gate report versus rules.
6. Long run (250k) + `evaluate_drive.py` + results report.

Stage B (WR/CB/rusher play-level training; WR observation will need the route
one-hot to execute random routes; composition and QB re-fine-tune) follows in
its own cycle, reusing this drive infrastructure.

## 14. Out of scope

Clock/quarters, punts, field goals, extra-point decisions, safeties, human
drive mode, multi-scenario drive training, MAPPO, and route one-hot for WR/CB
observations (Stage B).
