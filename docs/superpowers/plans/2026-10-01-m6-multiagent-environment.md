# M6 — Multi-Agent Environment: Implementation Plan

> **For agentic workers:** executed inline with TDD. Steps use checkbox syntax for tracking.

**Goal:** A PettingZoo `ParallelEnv` where every player decides for itself, with per-role observations and actions, keeping rule controllers and human mode compatible.

**Architecture:** `training.multiagent.QBPassingMultiAgentEnv` wraps the mechanical `Play`; one agent per roster player. Rule controllers from M3 act as evaluation policies through `run_rule_episode`. No MARL training yet (M7).

**Tech Stack:** PettingZoo 1.27, Gymnasium spaces, NumPy, pytest, Ruff.

**Spec:** [`AGENTS.md`](../../../AGENTS.md) §7 (M6).

## Global Constraints

- Human mode and the single-agent environment stay unchanged.
- Rewards favor the collective result while keeping per-agent values analyzable.
- Importing the core never loads PettingZoo.

---

### Task 1: Per-role observations

**Files:** `training/role_observations.py`, `tests/test_multiagent.py`.

- [x] 19-float role-local layout: self, QB relative, ball status/height, target relative + presence, remaining time.
- [x] Receivers target the nearest defender; defenders target the carrier, the QB (rushers), or their assigned receiver.

### Task 2: PettingZoo ParallelEnv

**Files:** `training/multiagent.py`.

- [x] `possible_agents` from the roster (`qb`, receivers, corners, rushers); one agent per player.
- [x] Spaces: QB `Box(6)`, movers `Box(2)`; observations 36 (QB) / 19 (rest).
- [x] `reset(seed)` reproducible; `step(actions)` decodes by role and returns the five Parallel API dictionaries.
- [x] Zero-sum terminal rewards: offense gets the single-agent terminal reward, defense its negative; per-tick time penalty on offense.
- [x] `run_rule_episode(env, seed)` runs the M3 rule controllers as every player's policy.

### Task 3: Tests and compatibility

**Files:** `tests/test_multiagent.py`, `pyproject.toml`.

- [x] `pettingzoo.test.parallel_api_test` passes.
- [x] Roster agents for `rushed` (6) and `default` (5); per-role spaces; deterministic rule episodes; zero-sum rewards; mover action validation.
- [x] Core import does not load PettingZoo; extra `multi = ["pettingzoo>=1.24"]`.
