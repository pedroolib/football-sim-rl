# Stage B2 — Config Adaptation and Training Throughput: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add capability features (speeds, arm range, accuracy) with asymmetric information, randomize configs with a curriculum, train generalized role/QB policies faster with parallel envs, and validate adaptation on a config matrix.

**Architecture:** `train.py` gains a picklable env factory and `SubprocVecEnv` support; observations gain capability blocks (QB 61, WR 35, defender 27) plus `DriveState.last_throw_distance`; rule teachers become range-aware; `training/randomization.py` samples configs with a two-phase curriculum that the role/drive envs apply per episode; generalized models are trained from scratch and validated on a matrix.

**Tech Stack:** Python 3.11+ (env 3.14), NumPy, Gymnasium, Stable-Baselines3 + PyTorch, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-10-02-stage-b2-config-adaptation-design.md` (approved 2026-10-02).

## Global Constraints

- Stage B specialists (`runs/ppo_wr_v1`, `runs/ppo_cb_v1`, `runs/ppo_ru_v1`, `runs/ppo_drive_qb_v12`) are never modified; they are regression references.
- `engine`/`rules` never import training/gym/torch/pygame; defaults keep M5/M6/M7 trajectories unchanged unless randomization is explicitly enabled.
- Observations are clipped to `[-1, 1]`, dtype float32, fixed sizes (QB 61, WR 35, defenders 27); M6 play-level layouts stay untouched.
- Determinism for a fixed seed and action sequence is mandatory; sampled configs derive from the episode seed.
- Commit each task with a focused message once `python -m pytest -q` and `ruff check .` pass; never commit `runs/` artifacts.

---

### Task 1: Parallel environments and env_method callbacks

**Files:**
- Modify: `src/qb_sim/training/train.py`
- Test: `tests/test_ppo_training.py`

**Interfaces:**
- Produces: `class _EnvSpec` (picklable factory) and `build_vec_env(config, scenarios)`; `AttemptAnnealingCallback` uses `env_method`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ppo_training.py`:

```python
def test_training_with_two_subprocess_envs(tmp_path) -> None:
    config = PPOConfig(
        run_name="subproc-run",
        env_kind="role_play",
        scenario="stage_b",
        learner="wr1",
        total_timesteps=128,
        seed=0,
        n_envs=2,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        checkpoint_freq=1000,
        eval_freq=1000,
        eval_episodes=1,
        attempt_decay_steps=1,
        imitation_episodes=0,
        output_dir=str(tmp_path),
        device="cpu",
    )

    _, _ = train(config)

    metadata = json.loads((tmp_path / "subproc-run" / "config.json").read_text())
    assert metadata["n_envs"] == 2
```

The training call itself proves the annealing callback works on `SubprocVecEnv`.

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_ppo_training.py -q -k subproc`
Expected: FAIL because `n_envs` is ignored (num_envs stays 1) or metadata lacks it.

- [ ] **Step 3: Implement the factory and vec env**

In `src/qb_sim/training/train.py` add a module-level picklable factory and replace the
`DummyVecEnv([lambda: Monitor(build_env(...))])` calls:

```python
class _EnvSpec:
    """Picklable env factory so SubprocVecEnv workers can rebuild the env."""

    def __init__(self, scenarios, env_kind, learner, opponents):
        self.scenarios = tuple(scenarios)
        self.env_kind = env_kind
        self.learner = learner
        self.opponents = tuple(opponents)

    def __call__(self):
        return Monitor(build_env(self.scenarios, self.env_kind, self.learner, self.opponents))


def build_vec_env(config: PPOConfig, scenarios: tuple[str, ...]):
    """Single DummyVecEnv or a forked SubprocVecEnv depending on ``n_envs``."""
    factories = [
        _EnvSpec(scenarios, config.env_kind, config.learner, config.opponents)
        for _ in range(max(config.n_envs, 1))
    ]
    if config.n_envs <= 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories, start_method="fork")
```

- Use `build_vec_env(config, scenarios)` for the training env; keep a single `DummyVecEnv`
  for the eval env (callbacks compare on one env).

- [ ] **Step 4: Make the annealing callback vec-env agnostic**

```python
    def _on_step(self) -> bool:
        scale = max(0.0, 1.0 - self.num_timesteps / self._decay_steps)
        self.model.get_env().env_method("set_attempt_scale", scale)
        return True
```

- [ ] **Step 5: Metadata and CLI**

- `metadata["n_envs"] = config.n_envs` in the metadata block.
- `--n-envs` already exists; no change needed.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_ppo_training.py -q` → PASS
Run: `python -m pytest -q` → PASS
Run: `ruff check .` → clean

- [ ] **Step 7: Benchmark (record in report)**

```bash
python -m qb_sim.training.train --env drive --scenario stage_b --timesteps 50000 --seed 0 \
  --run-name bench_n8 --n-envs 8 --imitation-episodes 0
```
Expected: fps at least 4x the single-env baseline (~1.6k), recorded in the report.

---

### Task 2: `DriveState.last_throw_distance`

**Files:**
- Modify: `src/qb_sim/rules/drive.py`
- Test: `tests/test_drive.py`

**Interfaces:**
- Produces: `DriveState.last_throw_distance: float | None = None`; updated in `Drive.advance` and reset in `Drive.reset`.

- [ ] **Step 1: Failing tests**

```python
def test_drive_stores_the_last_throw_distance() -> None:
    drive = _drive(los_range=(40.0, 40.0))
    play = drive.start_play()
    _finish_play(play, PlayOutcome.INCOMPLETE, throw_time=0.5)
    play.state.ball.flight_distance = 22.5

    drive.advance(play)

    assert drive.state.last_throw_distance == pytest.approx(22.5)


def test_last_throw_distance_persists_across_plays_without_a_throw() -> None:
    drive = _drive(los_range=(40.0, 40.0))
    play = drive.start_play()
    _finish_play(play, PlayOutcome.INCOMPLETE, throw_time=0.5)
    play.state.ball.flight_distance = 18.0
    drive.advance(play)
    play = drive.start_play()
    _finish_play(play, PlayOutcome.SACK, qb_x=36.0)

    drive.advance(play)

    assert drive.state.last_throw_distance == pytest.approx(18.0)


def test_last_throw_distance_resets_with_the_drive() -> None:
    drive = _drive(los_range=(40.0, 40.0))
    play = drive.start_play()
    _finish_play(play, PlayOutcome.INCOMPLETE, throw_time=0.5)
    play.state.ball.flight_distance = 18.0
    drive.advance(play)

    drive.reset(seed=7)

    assert drive.state.last_throw_distance is None
```

- [ ] **Step 2: Implement**

- Add the field to `DriveState`.
- `reset`: `last_throw_distance=None` (dataclass construction).
- `advance`: after computing `throw_distance`, if not None set `self.state.last_throw_distance = throw_distance`.

- [ ] **Step 3: Run tests** (`tests/test_drive.py`, full suite, ruff).

---

### Task 3: Capability features in observations

**Files:**
- Modify: `src/qb_sim/training/drive_observations.py` (QB 57 -> 61)
- Modify: `src/qb_sim/training/role_context_observations.py` (WR 31 -> 35, defenders 24 -> 27)
- Test: `tests/test_drive_observations.py`, `tests/test_role_context_observations.py`, and update size assertions in `tests/test_policy_viewer.py`, `tests/test_role_env.py`, `tests/test_imitation.py`, `tests/test_ppo_training.py` (grep for 57/31/28/24).

**Interfaces:**
- Produces: `DRIVE_OBSERVATION_SIZE = 61`, `RECEIVER_CONTEXT_OBSERVATION_SIZE = 35`, `DEFENDER_CONTEXT_OBSERVATION_SIZE = 27`.

- [ ] **Step 1: Failing tests**

Add to `tests/test_drive_observations.py`:

```python
def test_qb_observation_encodes_own_capabilities_and_advantage() -> None:
    play = Play()
    state = _observation()  # existing helper building a 61-float observation

    assert state[57] == pytest.approx(play.config.quarterback_speed / 10.0)
    assert state[58] == pytest.approx(play.config.throw_max_range / 50.0)
    assert state[59] == pytest.approx(play.config.throw_base_sigma_degrees / 5.0)
    expected = max(0.0, play.config.receiver_speed - play.config.defender_speed) / 2.0
    assert state[60] == pytest.approx(expected)
```

Add to `tests/test_role_context_observations.py`:

```python
def test_receiver_context_encodes_capabilities() -> None:
    play = Play()
    observation = receiver_context_observation(play.state, "wr1", play.config, RouteName.POST)

    assert observation[31] == pytest.approx(play.config.receiver_speed / 10.0)
    assert observation[32] == pytest.approx(play.config.defender_speed / 10.0)
    assert observation[33] == pytest.approx(play.config.throw_max_range / 50.0)
    assert observation[34] == pytest.approx(play.config.quarterback_speed / 10.0)


def test_defender_context_encodes_capabilities_and_last_throw() -> None:
    play = Play()
    drive_state = DriveState(los=40.0)
    drive_state.last_throw_distance = 20.0
    observation = defender_context_observation(
        play.state, "cb1", "wr1", play.config, drive_state
    )

    assert observation[24] == pytest.approx(play.config.defender_speed / 10.0)
    assert observation[25] == pytest.approx(play.config.receiver_speed / 10.0)
    assert observation[26] == pytest.approx(20.0 / 50.0)
```

- [ ] **Step 2: Implement**

- `drive_observations.py`: append the four features, using per-player `max_speed` for the
  advantage: for each present receiver, nearest defender distance and speeds; take
  `max(0, recv.max_speed - nearest.max_speed)` over receivers, `/2`.
- `role_context_observations.py`: append the WR block (own `max_speed`, nearest defender
  `max_speed`, QB range, QB speed) after index 30; append the defender block (own
  `max_speed`, target `max_speed` or 0, `last_throw_distance or 0` / 50) after index 23.
- Update every size constant and test assertion; grep `OBSERVATION_SIZE`/31/24 literals.

- [ ] **Step 3: Run the affected suites, then the full suite and ruff.**

---

### Task 4: Range-aware rule teachers

**Files:**
- Modify: `src/qb_sim/rules/routes.py`, `src/qb_sim/agents/loop.py`, `src/qb_sim/agents/rule_controllers.py`
- Test: `tests/test_routes.py`, `tests/test_rule_controllers.py`, `tests/test_agents.py`

**Interfaces:**
- `build_route(name, start, field, config, max_depth_yards: float | None = None)`.
- `build_rule_agents` caps receiver routes at `max(5.0, 0.85 * config.throw_max_range)`.
- `HeuristicQuarterback` only targets receivers within `0.95 * throw_max_range`.

- [ ] **Step 1: Failing tests**

```python
def test_go_route_is_capped_by_max_depth() -> None:
    config = SimulationConfig()
    route = build_route(RouteName.GO, vec2(20.0, 43.0), config.field, config, max_depth_yards=10.0)

    assert route.waypoints[0][0] == pytest.approx(30.0)


def test_heuristic_quarterback_ignores_receivers_beyond_range() -> None:
    config = replace(SimulationConfig(), throw_max_range=30.0)
    play = Play(config)
    for defender_id in ("cb1", "cb2"):
        play.state.player(defender_id).position = vec2(119.0, 0.0)
    play.state.player("wr1").position = vec2(60.0, 43.0)
    play.state.player("wr2").position = vec2(60.0, 10.0)
    play.state.elapsed_seconds = 1.0
    controller = HeuristicQuarterback("qb", config)

    action = controller.decide(play.state)

    assert not isinstance(action, Throw)
```

- [ ] **Step 2: Implement** the cap (clamp waypoint x), the `build_rule_agents` cap, and the
  QB in-range filter (skip receivers whose lead distance exceeds the limit; keep existing
  scramble/wait behavior).
- [ ] **Step 3: Update any affected existing tests and note them in the report.**

---

### Task 5: Randomization and curriculum

**Files:**
- Modify: `src/qb_sim/training/randomization.py`
- Modify: `src/qb_sim/training/role_env.py`, `src/qb_sim/training/drive_env.py`
- Test: `tests/test_randomization.py`, `tests/test_role_env.py`, `tests/test_drive_env.py`

**Interfaces:**
- `AdaptationRanges` dataclass with the six ranges from the spec.
- `sample_adaptation_config(base, rng, progress) -> SimulationConfig`.
- Envs gain `randomization=None` and `set_curriculum_progress(progress)`; `active_config` property; resets sample deterministically and apply per-player speeds/arm.

- [ ] **Step 1: Failing tests**

```python
def test_sample_adaptation_config_interpolates_with_progress() -> None:
    base = SimulationConfig()
    narrow = sample_adaptation_config(base, np.random.default_rng(0), 0.0)
    wide = sample_adaptation_config(base, np.random.default_rng(0), 1.0)

    # Narrow phase is each full-range center +/- 0.5 (receiver center 7.25).
    assert 6.75 <= narrow.receiver_speed <= 7.75
    assert 39.0 <= narrow.throw_max_range <= 51.0
    assert 6.0 <= wide.receiver_speed <= 8.5
    assert 30.0 <= wide.throw_max_range <= 55.0


def test_randomized_env_applies_sampled_speeds() -> None:
    env = RolePlayEnv(randomization=AdaptationRanges())
    env.reset(seed=5)

    assert env.active_config.receiver_speed == pytest.approx(
        env.play.config.receiver_speed
    )
    assert env.play.state.player("wr1").max_speed == pytest.approx(
        env.active_config.receiver_speed
    )
```

- [ ] **Step 2: Implement** the sampler and the env integration (`_active_config`, sampling
  on reset from the episode RNG, rebuilding `Play`/`Drive` and rule agents with it,
  `set_curriculum_progress`). Apply to `RolePlayEnv`, `RoleDriveEnv`, and `QBDriveEnv`.

- [ ] **Step 3: Run tests, full suite, ruff.**

---

### Task 6: CLI/curriculum wiring and metadata

**Files:**
- Modify: `src/qb_sim/training/ppo_config.py`, `src/qb_sim/training/train.py`
- Test: `tests/test_ppo_training.py`

**Interfaces:**
- `PPOConfig.randomize_config: bool = False`; `--randomize-config` CLI flag.
- `build_env(..., randomization=None)` passes `AdaptationRanges()` to the role/drive envs.
- `AdaptationCurriculumCallback` (narrow until 40% of steps, linear to full by 80%) added
  when randomization is on; metadata records the schedule and ranges.

- [ ] **Step 1: Smoke test**: role_play + randomization + `n_envs=2`, 128 steps; metadata
  contains `randomize_config=True` and the ranges.
- [ ] **Step 2: Implement** the flag, env wiring, callback, metadata.
- [ ] **Step 3: Run tests, full suite, ruff.**

---

### Task 7: Generalized WR (`ppo_wr_g1`)

- [ ] **Step 1: Pilot (50k, BC + PPO, 8 envs, randomization)**

```bash
python -m qb_sim.training.train --env role_play --scenario stage_b --learner wr1 \
  --timesteps 50000 --seed 0 --run-name ppo_wr_g1_pilot --n-envs 8 --randomize-config \
  --imitation-episodes 300 --imitation-epochs 800 --imitation-learning-rate 1e-4
```

- [ ] **Step 2: Evaluate on the matrix** (Task 10 script lands later; for the pilot use the
  existing `evaluate_stage_b.py wr` on `stage_b` without `--opponent`, since the rule QB is
  the teammate). Gate: at the Stage B config, catch within 5 points of the recorded Stage B
  specialist (`ppo_wr_v1` = 28%) and monotone catch/TD with the speed gap in a 3-cell sanity
  sweep (Task 10).
- [ ] **Step 3: Long run (250k) -> `runs/ppo_wr_g1`** and final evaluation.

---

### Task 8: Generalized CB (`ppo_cb_g1`)

- [ ] **Step 1: Pilot (50k)** with `--env role_drive --learner cb1 --n-envs 8 --randomize-config`.
  Opponents: rule QB (no `--opponent qb`) plus generalized WR when available (layout-compatible;
  the 31-obs `ppo_wr_v1` cannot run against the 35-obs builder).
- [ ] **Step 2: Gate**: at `stage_b`, INT forced not lower and separation/completion within
  5 points of `ppo_cb_v1`/rules; monotone coverage vs gap sanity.
- [ ] **Step 3: Long run (250k) -> `runs/ppo_cb_g1`.**

---

### Task 9: Generalized rusher (`ppo_ru_g1`)

- [ ] **Step 1: Pilot (50k)** with `--env role_drive --learner ru1 --n-envs 8 --randomize-config`.
  Opponents: rule QB plus generalized WR/CB when available.
- [ ] **Step 2: Gate**: pressure/arrival within 5% of `ppo_ru_v1`/rules at `stage_b`.
- [ ] **Step 3: Long run (250k) -> `runs/ppo_ru_g1`.**

---

### Task 10: Generalized QB (`ppo_qb_g1`) and validation matrix

**Files:**
- Create: `scripts/evaluate_adaptation.py` (config-matrix evaluation for role/composition
  policies; reuses `role_evaluation`/`drive_evaluation` with explicit `SimulationConfig`
  overrides).
- Modify: `AGENTS.md` (§11).

- [ ] **Step 1: Train the generalized QB**

```bash
python -m qb_sim.training.train --env drive --scenario stage_b --timesteps 250000 --seed 0 \
  --run-name ppo_qb_g1 --n-envs 8 --randomize-config \
  --opponent wr1=runs/ppo_wr_g1/final_model.zip \
  --opponent wr2=runs/ppo_wr_g1/final_model.zip \
  --opponent cb1=runs/ppo_cb_g1/final_model.zip \
  --opponent cb2=runs/ppo_cb_g1/final_model.zip \
  --opponent ru1=runs/ppo_ru_g1/final_model.zip \
  --imitation-episodes 300 --imitation-epochs 800 --imitation-learning-rate 1e-4
```

- [ ] **Step 2: Build `evaluate_adaptation.py`** with `--policies` (role model paths),
  `--gap`, `--range`, `--sigma`, `--episodes`, `--seed-start`; it constructs configs with
  `replace(STAGE_B_CONFIG, receiver_speed=..., defender_speed=..., quarterback_speed=...,
  throw_max_range=..., throw_base_sigma_degrees=...)` and evaluates the composition
  (`ppo_qb_g1` + generalized roles) plus the rule baseline per cell, reporting TD%,
  completion%, mean target depth, and scramble rate.
- [ ] **Step 3: Run the matrix**: gap `{0, 0.5, 1.0}` x range `{35, 45, 55}` (sigma 1.0) and
  sigma `{1, 3}` at gap 0.5/range 45; 100 episodes per key cell. Checks: monotone TD in gap
  and range; WR depth bounded by range at 35; archetypes by region; regression at the
  Stage B config within 5 points of `ppo_drive_qb_v12`.
- [ ] **Step 4: Update `AGENTS.md` §11** with the B2 status, runs, matrix results, and the
  remaining MARL/coordination work.

---

## Self-review notes

- Task 3 changes every observation size; the plan instructs grepping literals and running
  the full suite so no stale assertion survives.
- Envs sample configs from the episode RNG, preserving determinism; `active_config` exposes
  what was sampled for tests and evaluation.
- Specialists stay untouched and are used as regression references; generalization gaps are
  documented rather than hidden.
- MARL (IPPO with snapshots) starts only after B2 validation, in its own spec/plan cycle.
