# M5 — PPO Training and Evaluation: Implementation Plan

> **For agentic workers:** executed inline with TDD. Steps use checkbox syntax for tracking.

**Goal:** Train the quarterback with Stable-Baselines3 PPO, evaluate it against rule and random baselines on fixed seeds, and visualize its plays.

**Architecture:** `training` depends on `agents`/`rules`; PPO consumes `QBPassing-v0`. Precise aiming is hard to explore from scratch, so the pipeline warm-starts the actor by behavior cloning the rule QB and then fine-tunes with conservative PPO. A pass rusher with sacks creates pressure so waiting is risky.

**Tech Stack:** Gymnasium, Stable-Baselines3 2.9, PyTorch (CUDA), NumPy, pytest, Ruff, Pygame (optional dashboard).

**Spec:** [`AGENTS.md`](../../../AGENTS.md) §7 (M5) and §8.

## Global Constraints

- Importing `qb_sim` never loads PyTorch or Stable-Baselines3.
- Evaluation uses seeds disjoint from training (10000+) and reports the episode count.
- No long or stochastic assertions in automated tests.
- Do not advance to M6.

---

### Task 1: Pure training helpers

**Files:** `training/actions.py`, `training/observations.py`, `tests/test_training_actions.py`, `tests/test_training_observations.py`.

- [x] Extract `decode_action`/`encode_engine_action` and `observation_from_state`; env reuses them.
- [x] Fix trigger semantics: a positive trigger always throws, defaulting aim to `(1, 0)`.

### Task 2: Rewards and the no-stall / no-spam balance

**Files:** `training/rewards.py`, `training/env.py`, `tests/test_rewards.py`, `tests/test_env.py`.

- [x] Time-limit penalty worse than an incompletion; per-tick time cost.
- [x] Throw-attempt shaping with a minimum time gate (`attempt_min_seconds`).
- [x] `attempt_scale` so training can anneal the shaping to zero.

### Task 3: Pass rusher and sacks

**Files:** `engine/entities.py`, `rules/play.py`, `agents/rule_controllers.py`, `rules/scenarios.py`, `agents/loop.py`, `tests/test_play.py`, `tests/test_rule_controllers.py`, `tests/test_scenarios.py`.

- [x] `PlayOutcome.SACK` when a defender touches the QB while he holds the ball.
- [x] `PassRushController` chases the QB and switches to the carrier.
- [x] `rushed` scenario with `cb2` blitzing; `rusher_ids` flows through loop, env, session, and dashboard.

### Task 4: Behavior-cloning warm start

**Files:** `training/imitation.py`, `tests/test_imitation.py`.

- [x] Collect rule-QB demonstrations (`stand` encoded with trigger `-1`, throw samples upweighted).
- [x] MLP matching the SB3 actor (`pi_layers`), targets `atanh(action)`, weights loaded into the PPO policy.
- [x] BC-only policy reaches ~85% completions on `rushed`.

### Task 5: Conservative PPO fine-tuning

**Files:** `training/ppo_config.py`, `training/train.py`, `tests/test_ppo_training.py`.

- [x] Defaults: `lr=2e-5`, `n_epochs=4`, `clip_range=0.05`, `target_kl=0.01`, `ent_coef=0`, `log_std_init=-2`, separate `pi`/`vf` [128, 128].
- [x] CLI (`--timesteps`, `--seed`, `--scenario`, `--resume`, `--imitation-*`, `--attempt-decay-steps`), checkpoints, `config.json`, CSV curves.

### Task 6: Evaluation, analysis, dashboard

**Files:** `training/evaluation.py`, `scripts/evaluate_ppo.py`, `scripts/analyze_ppo.py`, `training/monitoring.py`, `ui/dashboard.py`, `scripts/train_dashboard.py`, `tests/test_evaluation.py`, `tests/test_dashboard.py`, `tests/test_monitoring.py`.

- [x] Metrics: completions/interceptions/incompletions/sacks, yards per attempt, mean reward, duration, throw and move frequency, pressure and QB speed at throw.
- [x] Fixed-seed comparison of random, rules, and PPO checkpoints across scenarios.
- [x] Six-panel live dashboard with learning curve, checkpoint hot-reload, and optional training subprocess for recording.

### Results (seed 0, 200k steps, `rushed`)

- `rushed` (trained): PPO 100% completions vs rules 81%; reward 1.29 vs 1.02.
- `default` (unseen): PPO 86% vs rules 88%.
- `pressured` / `deep` (unseen): 0%; multi-scenario training is the next experiment.
