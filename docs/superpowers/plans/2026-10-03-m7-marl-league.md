# M7-MARL — League Training with Snapshots: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax.

**Goal:** Implement and run the league orchestrator that co-trains QB/WR/CB/RU against frozen snapshots and reports per-round coordination metrics.

**Architecture:** `training/league.py` owns snapshot pools, rotation, opponent specs, the in-process round trainer, and the per-round composition evaluation; `scripts/marl_league.py` is the CLI orchestrator that persists `league_state.json` and the league table. Training reuses `train.py` and evaluation reuses `role_evaluation`.

**Tech Stack:** Python 3.11+ (env 3.14), NumPy, SB3 + PyTorch, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-10-03-m7-marl-league-design.md` (approved 2026-10-03).

## Global Constraints

- League rounds are stationary single-agent PPO runs: opponents are always frozen files (`--opponent` paths), never live policies.
- The B2 generalized models are the starting population; the Stage B specialists and `stage_b` rule numbers are recorded references.
- Determinism: pool sampling and round order derive from the league seed.
- Every task ends with `python -m pytest -q` and `ruff check .` green; commit per task; never commit `runs/`.

---

### Task 1: Snapshot pools, config, opponent specs, improvement rule

**Files:** Create `src/qb_sim/training/league.py`; Test `tests/test_league.py`.

**Interfaces:**
- `SnapshotPool(root: Path, pattern: str = "*.zip", pool_size: int = 3)` with `files() -> list[Path]`, `latest() -> Path | None`, `sample(rng) -> Path | None`, `add(path)`.
- `LeagueConfig` dataclass (fields from spec §3).
- `ROLE_OPPONENTS: dict[str, tuple[str, ...]]` mapping the trained role to the opponent player ids: qb -> (wr1, wr2, cb1, cb2, ru1); wr -> (qb, cb1, cb2, ru1); cb -> (qb, wr1, wr2, ru1); ru -> (qb, wr1, wr2, cb1, cb2).
- `opponent_spec(role, pools: dict[str, SnapshotPool], rng) -> tuple[tuple[str, str], ...]` where `pools` maps player id -> pool; raises `ValueError` when a required pool has no files.
- `improved(points: float, previous: float | None, tolerance: float = 0.05) -> bool`.

- [ ] **Step 1: Failing tests** covering discovery/sorting, `latest`, deterministic `sample` (same seed -> same file), trimming to `pool_size`, empty pool error in `opponent_spec`, correct pairs per role, and `improved` tolerance.
- [ ] **Step 2: Implement** the module exactly as specified.
- [ ] **Step 3: Run tests, full suite, ruff; commit `feat: league snapshot pools and opponent specs`.**

---

### Task 2: In-process round trainer

**Files:** Modify `src/qb_sim/training/league.py`; Test `tests/test_league.py`.

**Interfaces:**
- `train_round(role: str, config: LeagueConfig, opponents: tuple[tuple[str, str], ...], round_index: int) -> Path` that builds a `PPOConfig` for the role:
  - `qb`: `env_kind="drive"`, `learner="qb"`, scenario `stage_b`, `n_envs`, `n_steps`, `randomize_config=False`.
  - `wr`: `env_kind="role_play"`, `learner="wr1"`.
  - `cb`: `env_kind="role_drive"`, `learner="cb1"`; `ru`: `learner="ru1"`.
  - `run_name=f"league/r{round_index}_{role}"`, `total_timesteps=config.steps`, `opponents=opponents`, `output_dir=config.output`, BC settings `imitation_episodes=300`, `imitation_epochs=800`, `imitation_learning_rate=1e-4`, `seed=config.seed + round_index`, `device="auto"`.
  - Calls `qb_sim.training.train.train(ppo_config)` and returns the `final_model.zip` path; creates `output/league/` as needed.
- [ ] **Step 1: Smoke test** with `steps=256`, `n_envs=1`, `n_steps=64`, `eval_episodes=1`, opponents `(("qb","rules"),)` for a `wr` round in `tmp_path`; assert the returned path exists and `config.json` records the opponents.
- [ ] **Step 2: Implement, run tests/suite/ruff; commit `feat: league round trainer`.**

---

### Task 3: Per-round composition evaluation

**Files:** Modify `src/qb_sim/training/league.py`; Test `tests/test_league.py`.

**Interfaces:**
- `load_role_policies(state: dict[str, str], scenario_name: str) -> dict[str, object]` building `DrivePolicyQuarterback`/`ReceiverPolicy`/`DefenderPolicy` from the current model paths (qb, wr1/wr2 from the same WR path, cb1/cb2, ru1).
- `evaluate_league(state: dict[str, str], config: LeagueConfig) -> dict` running `run_composition_episode` over `range(config.eval_seed_start, +config.eval_episodes)` with the learned composition and the rule baseline (`RuleQuarterbackPolicy`) on a `QBDriveEnv(stage_b)`, returning `{"learned": summary, "rules": summary}` from `summarize_composition`.
- `league_row(round_index, role, opponents, learned, rules, previous_points) -> dict` with the metrics and `improved` flag.

- [ ] **Step 1: Test** with a tiny evaluation (`eval_episodes=2`) using the current B2 models if present, else rule policies as stand-ins; assert keys and the improvement flag.
- [ ] **Step 2: Implement, run tests/suite/ruff; commit `feat: league round evaluation`.**

---

### Task 4: Orchestrator script

**Files:** Create `scripts/marl_league.py`; Test `tests/test_league.py`.

**Interfaces:** CLI from spec §4. Flow per round: load/create `league_state.json` (role -> model path, starting from `--base` defaults), sample opponents, `train_round`, `evaluate_league`, append the row to `league_table.json`/`league_table.csv`, update the state, print the table, stop on patience.

- [ ] **Step 1: Smoke test** invoking `main([...])` with `--rounds 1 --steps 256 --n-envs 1 --n-steps 64 --eval-episodes 1 --output tmp_path` and a `--base` JSON pointing to rule agents (`"rules"` paths) or the B2 models; assert the state and table files exist with one row.
- [ ] **Step 2: Implement, run tests/suite/ruff; commit `feat: marl league orchestrator`.**

---

### Task 5: Run the league (4 rounds) and report

- [ ] **Step 1:** Ensure the B2 base models exist (`runs/wr/adaptation_g1`, `runs/cb/adaptation_g1`, `runs/ru/adaptation_g1`, `runs/qb/drive/adaptation_g1`); write `runs/league/base.json` with those paths.
- [ ] **Step 2:** Run `python scripts/marl_league.py --rounds 4 --steps 150000 --n-envs 8 --n-steps 128 --eval-episodes 100 --output runs/league --base runs/league/base.json` (allow ~20-30 min).
- [ ] **Step 3:** Report the league table (per-round learned vs rules: TD%, points/drive, completions, INT, sacks, 3rd down) and the improvement flags; identify which roles improved and whether the QB matchup behavior changed.

---

### Task 6: Final composition, optional extra rounds, docs

- [ ] **Step 1:** Run the final composition (learned vs rules vs recorded Stage B numbers) with `scripts/evaluate_adaptation.py` on `stage_b`; save `runs/league/final_composition.json`.
- [ ] **Step 2:** If the table improved in the last round, run up to 4 extra rounds; otherwise document the plateau.
- [ ] **Step 3:** Update `AGENTS.md` §11 with the league design, round table, coordination findings, and the remaining work (QB matchup exploitation, hybrid simultaneous fine-tune, M8).

## Self-review notes

- Rounds are stationary: `train_round` only ever passes frozen files to `--opponent`; the orchestrator samples pools per round deterministically.
- The orchestrator reuses the tested `train()` and `role_evaluation` paths, so the new code surface is pools/specs/table only.
- Evaluation uses the existing composition runner with Wilson CIs; no new metric definitions.
