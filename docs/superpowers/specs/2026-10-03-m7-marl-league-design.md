# M7-MARL — League Training with Snapshots: Design

**Goal:** Co-train the quarterback, receivers, corners, and rusher through league
rounds where each role trains against frozen snapshots of the others, then
measure coordination against the rule baseline and the Stage B composition.

**Scope:** a league orchestrator (round rotation, snapshot pools, per-round
composition evaluation, league table), starting from the B2 generalized models,
plus the league experiments and the final report. Simultaneous IPPO, centralized
critics (MAPPO), and roster expansion are out of scope.

**Status:** approved by the user on 2026-10-03 ("hazlo").

## 1. Context

- B2 produced generalized policies at parity with rules in the base config:
  `runs/wr/adaptation_g1`, `runs/cb/adaptation_g1`, `runs/ru/adaptation_g1`,
  and a drive QB (`runs/qb/drive/adaptation_g1`) that still fails to exploit
  speed matchups (open problem).
- The single-agent pipeline (`train.py` with `--opponent ID=PATH` and
  `--opponent ID=rules`) and the composition evaluation
  (`scripts/evaluate_adaptation.py`, `role_evaluation.py`) already exist.
- Naive simultaneous independent PPO is unstable with sparse terminal rewards;
  league training with frozen snapshots keeps every round a stationary
  single-agent PPO problem and is the practical recipe for multiagent games.

## 2. Approved approach

League rounds with snapshot opponents (a form of independent PPO with
population-based opponent sampling):

1. Roles and rotation: `qb -> wr -> cb -> ru`, repeated for `rounds` rounds.
2. Each round trains one role for `steps` timesteps against frozen snapshots of
   the other roles, using the existing `train.py` entry point and BC warm start.
3. Snapshot pools keep the last `pool_size` policies per role (plus the B2 base
   model); each round samples one snapshot per opponent role from its pool
   (latest by default; older snapshots add diversity).
4. After each round, evaluate the full learned composition on fixed seeds in
   drives (`stage_b`) against the rule baseline, append a league table row, and
   save the new snapshot.
5. Stop after the configured rounds or after `patience` rounds without
   improvement; report per-round curves and the final composition.
6. If the league plateaus, an optional hybrid phase fine-tunes all roles
   simultaneously (documented, not part of the first pass).

## 3. League components (`training/league.py`)

- `SnapshotPool(root, pattern)`: discovers checkpoint/model files, exposes
  `latest()` and `sample(rng)`; deterministic given the RNG. Pools start with
  the B2 base model so early rounds always have an opponent.
- `LeagueConfig`: `rounds=4`, `steps=150_000`, `rotation=("qb","wr","cb","ru")`,
  `pool_size=3`, `patience=2`, `scenario="stage_b"`, `seed=0`, `n_envs=8`,
  `n_steps=128`, `eval_episodes=100`, `eval_seed_start=10_000`, `output="runs/league"`.
- `opponent_spec(role, league)`: builds the `(player_id, path)` pairs for
  `train.py --opponent` for the role being trained (QB trains against
  wr1/wr2/cb1/cb2/ru1; WR against qb/cb/ru; CB against qb/wr/ru; RU against
  qb/wr/cb), sampling each opponent role's pool once per round.
- `LeagueState`: current model path per role, round number, table rows; JSON
  serialization under `output/league_state.json`.
- `train_round(role, config)`: invokes the training pipeline in-process
  (`qb_sim.training.train.train`) with the appropriate `PPOConfig`
  (`env_kind`, `learner`, `opponents`, `n_envs`, `n_steps`, BC settings) and
  returns the final model path; the run name is
  `league/r<round>_<role>` (e.g., `runs/league/league/r1_wr` with the default output).
- `evaluate_league(state, config)`: builds the learned composition opponents
  from `state` and runs `run_composition_episode` over fixed seeds for the
  learned policies and the rule baseline; returns the summary rows.
- `improved(rows, previous)`: improvement rule on `points_per_drive` with a
  tolerance (e.g., `+0.05`).

## 4. Orchestrator (`scripts/marl_league.py`)

CLI: `--rounds`, `--steps`, `--rotation`, `--scenario`, `--seed`, `--n-envs`,
`--n-steps`, `--eval-episodes`, `--seed-start`, `--output`, `--base` (JSON with
initial role paths, defaulting to the B2 models), `--resume` (league state).

Flow per round: sample opponents, train, evaluate, append the league table row
(JSON + CSV under `output/`), update the state, print a compact table. Stop
early on patience. The script is deterministic given the seed.

## 5. Rewards and information

Unchanged from B2: role rewards (WR terminal + catch/TD bonuses + carry
progress; defenders zero-sum terminal + potential shaping), asymmetric
capability features, config randomization available but disabled by default in
league rounds for stability (enabled as a documented option).

## 6. Evaluation and metrics

Per round, on `stage_b` drives with seeds `10000+`:
- Team: TD%, points/drive, completion%, interceptions, sacks, 3rd-down
  conversion, plays/drive (Wilson CIs where proportions).
- Coordination: QB-to-WR connection rate (completions per throw), mean
  separation at throw/catch, rusher pressure/arrival, coverage INTs.
- Baselines: rule team on the same seeds; the Stage B composition numbers as
  recorded references.
- League table columns: round, trained role, opponent snapshot ids, team
  metrics, and the improvement flag.

## 7. Testing

1. `SnapshotPool`: discovery, `latest`, deterministic `sample`, empty-pool
   error, `pool_size` trimming.
2. `opponent_spec`: correct pairs per trained role and pool sampling.
3. `improved`: tolerance behavior.
4. Orchestrator smoke: one round with `steps=256`, `n_envs=1`, tiny eval,
   using temporary run/output dirs; asserts state/table files and that the
   trained model path exists.
5. Determinism: same seed -> same sampled opponents and same table order.

## 8. Risks and out of scope

- **Plateau/cycling:** frozen-opponent rounds can overfit the current opponent;
  pools with older snapshots mitigate it; patience stops the loop and reports.
- **QB matchup exploitation** remains open from B2; league rounds may improve it
  indirectly, otherwise it stays a documented research item.
- **Out of scope:** simultaneous IPPO/MAPPO, centralized critics, roster and
  O-line/D-line expansion (M8), full-game clock/possessions.
