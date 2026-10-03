# Stage B2 — Config Adaptation (Speed, Arm, Accuracy) and Training Throughput: Design

**Goal:** policies that condition on their own and the matchup's capabilities
(speeds, throw range, accuracy) so behavior adapts: a fast receiver against a
slow corner with a strong-armed QB converts deep, a short-armed QB makes the
receiver settle at reachable depth, and the QB develops archetypes (pocket vs
scrambler). Training runs ~5x faster with parallel environments.

**Scope of this spec:** throughput infrastructure, capability features with
asymmetric information, `DriveState.last_throw_distance`, range-aware rule
teachers, config randomization with curriculum, new generalized models, and the
validation matrix. MARL/coordination comes afterwards on top of these policies.

**Status:** approved by the user on 2026-10-02.

## 1. Context

- Stage B trained config-blind specialists: `runs/ppo_wr_v1`,
  `runs/ppo_cb_v1`, `runs/ppo_ru_v1`, `runs/ppo_drive_qb_v12`/`v13`.
- Velocities are normalized by each player's `max_speed`, so absolute speeds
  (and therefore the matchup) are invisible to the policies; the QB's arm and
  accuracy are not observable at all.
- Training ignores `PPOConfig.n_envs` (always one `DummyVecEnv`) and uses CUDA
  for a tiny MLP: measured ~1.6k fps, GPU mostly idle.

## 2. Approved decisions

1. New generalized models trained from scratch; the Stage B specialists stay as
   references for per-config regression.
2. Asymmetric information: the offense knows team capabilities (QB and WR train
   together); defenders see only in-game kinematics plus the last observed
   throw distance of the drive. Defenders never observe the QB's range or
   accuracy.
3. Randomize six scalars per episode/drive (WR/CB/QB/rusher speeds, QB max
   range, QB base sigma) uniformly with a two-phase curriculum.
4. Rule teachers become range-aware so behavior cloning is correct for short
   arms; coverage teachers are unchanged (consistent with the asymmetry).
5. Throughput: honor `n_envs` with `SubprocVecEnv` and drive callback env
   mutations through `env_method`.
6. Validation on a config matrix (speed gap x arm range x accuracy) plus
   archetype and regression checks.

## 3. Throughput

- `train.py` builds `DummyVecEnv` when `n_envs == 1` and
  `SubprocVecEnv(start_method="fork")` otherwise, using a module-level,
  picklable `make_env(...)` factory with `Monitor` inside each worker.
- `AttemptAnnealingCallback` calls
  `self.model.get_env().env_method("set_attempt_scale", scale)` (works for both
  vec-env types).
- New `AdaptationCurriculumCallback` calls
  `self.model.get_env().env_method("set_curriculum_progress", progress)`.
- Reference training commands use `--n-envs 8`. Benchmark target after the
  change: at least 4x the single-env fps on the same machine.

## 4. Observations (capability features)

All blocks are appended after the existing layouts, clipped to `[-1, 1]`,
float32. Per-player speeds come from `PlayerState.max_speed` (set by
`Play.reset` from the active config), so randomized configs flow through.

- **QB drive observation, 57 -> 61** (`DRIVE_OBSERVATION_SIZE`):
  - 57: `quarterback_speed / 10`
  - 58: `throw_max_range / 50`
  - 59: `throw_base_sigma_degrees / 5`
  - 60: best receiver speed advantage:
    `max(0, max over present receivers of (receiver.max_speed - nearest defender.max_speed)) / 2`
- **WR observation, 31 -> 35** (`RECEIVER_CONTEXT_OBSERVATION_SIZE`):
  - 31: own `max_speed / 10`
  - 32: nearest defender `max_speed / 10`
  - 33: QB `throw_max_range / 50`
  - 34: QB `quarterback_speed / 10`
- **CB/rusher observation, 24 -> 27** (`DEFENDER_CONTEXT_OBSERVATION_SIZE`):
  - 24: own `max_speed / 10`
  - 25: target `max_speed / 10`
  - 26: `last_throw_distance / 50` (0.0 when none), where
    `last_throw_distance` is the most recent throw distance observed in the
    current drive.

`DriveState` gains `last_throw_distance: float | None = None`, reset per drive
and updated in `Drive.advance` when `PlayResult.throw_distance` is not None
(the value persists across plays without throws). Play-level M6 layouts stay
untouched.

## 5. Range-aware rule teachers

- `build_route` gains `max_depth_yards: float | None = None`, clamping every
  waypoint's x to `start_x + max_depth_yards`.
- `build_rule_agents` computes `max_depth_yards = max(5.0, 0.85 * throw_max_range)`
  for receiver routes (measured from the receiver's start), so demonstrations
  and the rule baseline never run far beyond the arm.
- `HeuristicQuarterback` only considers receivers whose projected throw distance
  is `<= 0.95 * throw_max_range`; when none qualifies it keeps waiting or
  scrambling. This changes the rule baseline and BC teacher for short arms and
  must be reflected in existing tests' expectations where affected.
- `ManCoverageController` and `PassRushController` are unchanged (defenders do
  not get arm information).

## 6. Randomization and curriculum

`training/randomization.py` gains:

```python
@dataclass(frozen=True, slots=True)
class AdaptationRanges:
    receiver_speed: tuple[float, float] = (6.0, 8.5)
    defender_speed: tuple[float, float] = (5.5, 7.5)
    quarterback_speed: tuple[float, float] = (5.0, 8.0)
    rusher_speed: tuple[float, float] = (5.0, 8.0)
    throw_max_range: tuple[float, float] = (30.0, 55.0)
    throw_base_sigma_degrees: tuple[float, float] = (0.5, 4.0)
```

- `sample_adaptation_config(base, rng, progress)` interpolates each range
  linearly from a narrow phase-1 range (speeds: center +/- 0.5; range:
  `(40, 50)`; sigma: `(1.0, 2.0)`) to the full range as `progress` goes
  `0 -> 1`, and returns `replace(base, ...)` keeping the roster and fatigue
  settings.
- `RolePlayEnv`, `RoleDriveEnv`, and `QBDriveEnv` accept
  `randomization: AdaptationRanges | None = None` and expose
  `set_curriculum_progress(progress)`; on reset they sample a config with the
  episode RNG and rebuild `Play`/`Drive` with it. Sampling is deterministic per
  seed.
- `AdaptationCurriculumCallback(decay_start_fraction=0.4, full_fraction=0.8)`
  maps training progress to `[0, 1]` (narrow until 40%, linear to full by 80%,
  full afterwards).

## 7. Training

New runs (from scratch, BC + PPO, `--n-envs 8`, randomization enabled):

- `runs/ppo_wr_g1`: WR, play env with randomization, opponents = rule QB +
  generalized defenders when available (specialists otherwise).
- `runs/ppo_cb_g1`: CB shared policy, drives with randomization, opponents =
  generalized QB/WR/rusher (specialists while a role is missing).
- `runs/ppo_ru_g1`: rusher, drives with randomization.
- `runs/ppo_qb_g1`: drive QB, `QBDriveEnv` with randomization, opponents = the
  generalized roles; BC from the range-aware rule QB.

Order: WR -> CB -> rusher -> QB (same freeze discipline as Stage B). Config
metadata records the randomization ranges, curriculum schedule, and frozen
opponent paths.

## 8. Validation

Config matrix: speed gap `{0.0, 0.5, 1.0}`, QB range `{35, 45, 55}`, sigma
`{1.0, 3.0}`, fixed seeds (10000+), 100 episodes per cell for the key cells.

Checks:

1. Monotonicity: TD% and completion% increase with the speed gap at fixed arm,
   and with the arm range at fixed gap.
2. Short arm: with range 35, the WR's mean target depth stays within
   `range - margin` and the offense does not collapse to zero conversions.
3. Archetypes: fast QBs scramble more and throw later; accurate QBs complete
   more; a "pocket" region (slow, strong-ish arm) shows the lowest scramble
   rate. Reported by config region, not by label.
4. Regression: at the Stage B config (`stage_b`), generalized models stay
   within ~5 points of the specialists (v12/WR v1/CB v1/RU v1); larger gaps are
   documented.
5. No config collapses completely (e.g., no cell with 0 completions and 0
   conversions while the rule baseline performs).

## 9. Testing

1. Layout tests for 61/35/27, capability values, empty slots, clipping.
2. `DriveState.last_throw_distance` update/persistence/reset.
3. Range-aware route cap and QB in-range filter (unit tests).
4. `sample_adaptation_config`: phase interpolation, bounds, determinism.
5. Env resets actually apply the sampled speeds/arm (per-player `max_speed` and
   `config.throw_max_range`).
6. Throughput smoke: `n_envs=2` `SubprocVecEnv` training completes for a role
   env; callbacks work through `env_method`.
7. BC smoke per generalized role and PPO smoke per new run.

## 10. Risks and out of scope

- **Generalization cost:** more variance can slow learning; the curriculum and
  parallel envs mitigate it. Specialists may stay slightly better at the
  matched config (measured, bounded target).
- **Hidden arm for defenders:** coverage optimizes against the QB distribution;
  `last_throw_distance` provides in-drive adaptation without recurrence.
- **Rule baseline shift:** range-aware teachers change rule benchmarks again;
  numbers are labeled per phase.
- **Out of scope:** MARL/coordination (next cycle), O-line/D-line and roster
  expansion, full-game clock/possessions, UI polish.
