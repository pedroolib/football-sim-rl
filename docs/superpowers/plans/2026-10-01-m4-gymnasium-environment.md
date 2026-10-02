# M4 — Gymnasium Environment: Implementation Plan

> **For agentic workers:** executed inline with TDD. Steps use checkbox syntax for tracking.

**Goal:** Train the quarterback with a Gymnasium-compatible environment while teammates and defenders use the M3 rule controllers.

**Architecture:** `training` depends on `agents`/`rules`; the environment drives the same `Play.step(actions)` API. Observations are normalized role-local vectors and the action is the hybrid `(move, aim, power, trigger)` vector that maps onto `Move`/`Throw`, so humans, rule controllers, and learned policies share the engine.

**Tech Stack:** Gymnasium 1.3, NumPy, pytest, Ruff. Headless only; no Pygame, no SB3 yet (M5).

**Spec:** [`AGENTS.md`](../../../AGENTS.md) §7 (M4) and §8.

## Global Constraints

- Simulation never imports the training framework.
- The environment handles terminations, truncations, and rewards per Gymnasium.
- Same seed + same action sequence reproduce the episode.
- Reward coefficients are configurable and documented.

---

### Task 1: Reward configuration

**Files:** `training/rewards.py`, `tests/test_rewards.py`.

- [x] `RewardConfig` with completion, interception, incompletion, out-of-bounds, time-limit, yard, and time-penalty coefficients.
- [x] `terminal_reward(outcome, yards, config)` pure function with literal mapping tests.

### Task 2: `QBPassingEnv`

**Files:** `training/env.py`, `training/__init__.py`, `tests/test_env.py`.

- [x] Spaces: `Box(-1, 1, (27,), float32)` observation and `Box(-1, 1, (6,), float32)` action.
- [x] Observation: QB position/velocity, ball status one-hot + height, relative receiver/defender position and velocity, pressure, remaining time.
- [x] Action: `[move_x, move_y, aim_x, aim_y, power_raw, throw_trigger]` mapping to `Move`/`Throw`.
- [x] `reset(seed=...)` reproducible; `step` returns `(obs, reward, terminated, truncated, info)`; terminal outcomes terminate and `max_steps` truncates.
- [x] Registered as `QBPassing-v0`; validated with `gymnasium.utils.env_checker.check_env`.
- [x] Tests: spaces, determinism, API shape, checker, scripted completion (positive reward), interception (negative reward), stalling penalty, truncation, bounded random observations, headless import.
