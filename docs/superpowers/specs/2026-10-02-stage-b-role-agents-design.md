# Stage B — Role Agents (WR, CB, Rusher) and QB v12: Design

**Goal:** Turn the receivers, cornerbacks, and rusher from rule controllers into
trained policies, compose them with the drive quarterback, and re-fine-tune the
QB against them, while adding carrier-run and fatigue mechanics that make long
catches situational instead of automatic touchdowns.

**Scope of this spec:** carrier/fatigue mechanics, per-role observations,
`RolePlayEnv`/`RoleDriveEnv`, per-role rewards, BC + PPO training pipeline,
per-role gates, drive composition evaluation, and `ppo_drive_qb_v12`.
This is the Stage B cycle announced in the M7 drive spec §13.

**Status:** approved by the user on 2026-10-02.

## 1. Context after M7

- `runs/ppo_rushed_v10` (play-level QB, 36-float obs) and `runs/ppo_drive_qb_v11`
  (drive QB, 57-float obs) are frozen baselines.
- `rules/drive.py` provides 3-down drives with random LOS/routes and TD/INT/
  turnover endings; `training/drive_env.py` exposes `QBDriveEnv`.
- Receivers, corners, and rushers are agents (same `BaseAgent` interface) but
  their policies are rule controllers: `ReceiverRouteController`,
  `ManCoverageController`, `PassRushController`.
- `training/role_observations.py` keeps the 19-float mover layout used by the
  M6 multiagent environment; that environment stays untouched in Stage B.
- `train_behavior_cloning` already trains on raw action targets (BCE trigger for
  the 6-value QB action; plain MSE for 2-value movers), matching SB3 2.9's
  unsquashed Gaussian policies.

## 2. Approved decisions

1. Order: WR → CB → rusher, freezing previous policies; then re-fine-tune the
   drive QB against the trained roles → `ppo_drive_qb_v12`. v10 and v11 stay
   untouched.
2. Training environments are mixed: WR at play level (fast single plays, random
   routes), CB and rusher inside drives (coverage/pressure depend on down
   context).
3. Defenders observe drive context (down, distance, yards to goal) so they can
   play cushion on 3rd and long.
4. Receiver reward: offense terminal reward + catch bonus (+0.3 when the learner
   catches) + yards. Separation is measured, never rewarded.
5. Defender reward: zero-sum terminal (negated drive play reward, first-down
   bonus, points, and turnover penalty) plus potential-based shaping
   `γ·Φ(s') − Φ(s)` with `Φ = −min(distance(learner, target), 20)/10` and
   terminal `Φ = 0` (user-approved 2026-10-02 after pure zero-sum made PPO
   drift from 1.74 to 2.65 yd of separation allowed). Sacks/interceptions are
   worth +1 through the negation; pressure and arrival are measured only.
6. Composition and final evaluation happen in drives with frozen policies;
   `QBPassingMultiAgentEnv` (M6) stays as-is until MARL/M8.
7. The receiver runs to the end zone after the catch instead of stopping when
   the route ends.
8. Fatigue applies only to the ball carrier after the catch, so a long pass is
   not automatically a touchdown; a receiver with a large speed edge still
   scores.

## 3. New engine mechanics

### 3.1 Carrier runs to the end zone

`ReceiverRouteController.act` changes behavior when its player is the carrier
(`state.carrier_id == player_id`): it stops following waypoints and steers
toward the opponent goal line (+x), with a small lateral component away from the
nearest defender and a soft boundary correction so it does not run out of
bounds. This behavior is used by the rule baseline, by BC demonstrations, and by
rule-controlled receivers inside drives.

### 3.2 Carrier fatigue

`SimulationConfig` gains:

| Field | Default | Meaning |
|---|---|---|
| `carrier_fatigue_per_second` | `0.0` | Linear speed-factor decay per second of carrying |
| `carrier_min_speed_factor` | `0.55` | Floor of the fatigue factor |

`PlayState` gains `carry_started_at: float | None = None`, assigned in
`Play.step` right after the elapsed-time increment on the tick the catch is
resolved (single guarded write; the value is the catch instant). While a player
is the carrier, `Play._apply_action` multiplies its speed by
`speed_factor = max(carrier_min_speed_factor, 1 - carrier_fatigue_per_second * (elapsed_seconds - carry_started_at))`.
Fatigue does not apply during route running or to defenders. The default `0.0`
keeps every existing M5/M7 trajectory unchanged; Stage B scenarios and
environments enable it (initial calibration target: `0.12`/s, floor `0.55`,
validated by the feasibility script before training).

### 3.3 Calibration script

`scripts/carrier_feasibility.py` runs rule-QB drives with fatigue on and reports
touchdown rate, yards after catch, and time-to-tackle grouped by the
receiver–corner speed gap, so the fatigue values are chosen from data before any
training run (same methodology as the M7 Stage A feasibility report).

## 4. Per-role observations

New module `training/role_context_observations.py`; the existing 19/36 layouts
used by M6 stay untouched.

- `RECEIVER_CONTEXT_OBSERVATION_SIZE = 31`: the 19-float role observation plus
  the route one-hot (8, `ROUTE_OBSERVATION_ORDER`), yards to the goal line
  `/100`, and route progress (3): relative position of the next route waypoint
  (`dx/field_length`, `dy/field_width`) and `min(distance_to_waypoint, 10)/10`.
  The next waypoint is derived from the called route and the receiver's start
  position; it makes the rule policy near-Markov so behavior cloning can
  reproduce route running (measured: 22% vs 17% baseline catches).
  Builder: `receiver_context_observation(state, player_id, config, route, goal_line=100.0)`.
- `DEFENDER_CONTEXT_OBSERVATION_SIZE = 24`: the 19-float role observation plus
  down one-hot (3), `min(distance, 40)/40`, and yards to the goal line `/100`.
  Builder: `defender_context_observation(state, player_id, target_id, config, drive_state, goal_line=100.0)`.

Both clip to `[-1, 1]`, dtype float32, fixed slots with zeroed empty entries.

## 5. Environments

New module `training/role_env.py` (registered as Gymnasium environments).

### 5.1 `RolePlayEnv` (WR)

- `RolePlayEnv(scenario=RUSHED_SCENARIO, learner_id="wr1", opponents=None, receiver_reward=None, goal_line=100.0, max_steps=240)`.
- Episode = one play. Random routes sampled per episode from
  `DEFAULT_ROUTE_POOL` with the environment RNG; `Play(config, seed, goal_line=100.0)`.
- Stage B defines the fatigue-enabled configuration explicitly in
  `rules/scenarios.py`:
  `STAGE_B_CONFIG = replace(SimulationConfig(), carrier_fatigue_per_second=<calibrated>, carrier_min_speed_factor=0.55)`
  and a `STAGE_B_SCENARIO` (rushed roster) built from it. The fatigue values come
  from the §3.3 calibration report; `RolePlayEnv`, `RoleDriveEnv`, composition,
  and QB v12 all use `STAGE_B_CONFIG`.
- Learner observation: 28 floats; action: `Box(2)` movement.
- Teammates: QB policy (`opponents["qb"]`, e.g. v10), the other receiver and all
  defenders by rule controllers (overridable via `opponents`).
- Shared policy across `wr1`/`wr2`: train with a `DummyVecEnv` of two envs
  (learner `wr1` and learner `wr2`).
- `info`: seed, outcome, yards, caught (carrier == learner), separation at throw
  and at catch, throw distance, time to tackle.

### 5.2 `RoleDriveEnv` (CB / rusher)

- `RoleDriveEnv(scenario=RUSHED_SCENARIO, learner_id="cb1", opponents=None, defender_reward=None, drive_config=None, max_steps=2400)`.
- Episode = one drive; learners `cb1`, `cb2`, or `ru1`; learner observation 24
  floats; action `Box(2)`.
- Frozen teammates/opponents: QB v11 (`opponents["qb"]`), WR v1
  (`opponents["wr1"]`, `opponents["wr2"]`), remaining defenders by rule
  controllers. `opponents` maps `player_id` to a context-aware callable; the
  environment passes each opponent the context its layout requires (`(state)` for
  rules, `(state, routes, drive_state)` for the QB, `(state, route)` for a
  receiver, `(state, drive_state)` for a defender).
- Shared CB policy across `cb1`/`cb2`: two envs in a `DummyVecEnv`; rusher is
  its own policy.
- `info`: drive down/distance/LOS, play history (same keys as `QBDriveEnv`),
  defender-specific diagnostics (separation allowed at throw/catch, pressure,
  rusher arrival time, sack).

### 5.3 Policy adapters (`training/policy.py`)

- `ReceiverPolicy(model, config, player_id, goal_line=100.0)`: callable
  `(state, route) -> Move | None`, builds the 28-float observation.
- `DefenderPolicy(model, config, player_id, goal_line=100.0)`: callable
  `(state, target_id, drive_state) -> Move | None`, builds the 24-float
  observation.
- `load_receiver_policy(path, scenario_name, player_id)` and
  `load_defender_policy(path, scenario_name, player_id)` load PPO models lazily
  (no torch at import time).

## 6. Rewards

`training/rewards.py` gains:

```python
@dataclass(frozen=True, slots=True)
class ReceiverRewardConfig:
    play: RewardConfig = field(default_factory=RewardConfig)
    catch_bonus: float = 0.3
    touchdown_bonus: float = 7.0
```

- WR: `terminal_reward(outcome, yards, config.play)` + `catch_bonus` when the
  learner caught the ball + `touchdown_bonus` when the play is a touchdown
  (`yards = carrier_x - LOS`).
- CB/rusher: `defender_drive_reward(result, config) = -drive_result_reward(result, config)`
  with `DriveRewardConfig`, evaluated at play end; per-tick reward is 0.

## 7. Training pipeline and versioning

- `train.py` gains `--env role_play|role_drive`, `--learner <player_id>`, and
  repeatable `--opponent <player_id>=<path>`; `config.json` records the learner,
  the frozen opponents (id + path), the fatigue config, and the goal line.
- BC warm start per role uses the same opponents as training:
  - WR: `ReceiverRouteController` demonstrations in `RolePlayEnv` with the
    **frozen drive QB v11** as teammate (route-aware; v10 has no route
    information and the rule QB completes too few random-route passes), with a
    default drive context (`1st & 10` at the play's LOS) supplied by
    `RolePlayEnv` to the QB opponent. 28-float observations.
  - CB: `ManCoverageController` demonstrations in `RoleDriveEnv` with frozen
    QB v11 and WR v1, 24-float observations.
  - Rusher: `PassRushController` demonstrations in `RoleDriveEnv` with frozen
    QB v11, WR v1, and CB v1.
- Versioned runs: `runs/ppo_wr_v1`, `runs/ppo_cb_v1`, `runs/ppo_ru_v1`, and
  `runs/ppo_drive_qb_v12` (resumed from v11, imitation off).
- Training and evaluation seeds are separate; evaluations use 100 fixed episodes
  (seeds 10000+) with the rule baseline on the same seeds.

## 8. Gates

Pilot (50k steps with BC) must pass before the long run (250k) for each role:

- **WR**: catch rate ≥ rule baseline + 5 points, and (TD/YAC ≥ baseline or mean
  separation ≥ baseline).
- **CB**: completion percentage allowed ≤ rule baseline − 5 points; forced
  interceptions ≥ baseline (not lower); mean separation allowed ≤ baseline.
- **Rusher**: sack rate ≥ rule baseline + 3 points, or pressure at throw ≥
  baseline.
- If a gate fails: adjust only that role's reward or the fatigue parameters,
  document the change and the numbers, and re-run the pilot once per adjustment.

## 9. Composition and QB v12

- `scripts/evaluate_stage_b.py`: fixed-seed drives with frozen policies
  (QB v11/v12, WR v1, CB v1, RU v1) versus the all-rules baseline; Wilson 95%
  intervals for proportion metrics.
- Team metrics: TD%, points/drive, first downs/drive, 3rd-down conversion,
  interceptions, sacks, plays/drive.
- Role metrics: WR catch rate/TD per catch/YAC/separation at throw and catch;
  CB completion allowed/separation allowed/interceptions; rusher sack rate/
  pressure at throw/arrival time. Separation is the distance from the targeted
  receiver to the nearest defender at the throw and at the catch; YAC is the
  distance from the catch spot to the final spot.
- `QBDriveEnv.opponents` accepts context-aware callables (same contract as
  §5.2), so the composition and the QB v12 opponents plug in directly.
- QB v12: `train.py --env drive --resume runs/ppo_drive_qb_v11/final_model.zip`
  with the trained roles as opponents (fatigue on), 250k steps, no imitation;
  compare v11 vs v12 vs rules on the same seeds.

## 10. Testing

1. Fatigue: only the carrier after the catch loses speed; defenders and route
   running are unaffected; `0.0` default reproduces M5 trajectories.
2. Carrier run: the rule receiver steers toward the end zone after catching and
   does not stop at the route end.
3. Observations: sizes 28/24, ranges, route/down one-hots, empty slots zeroed,
   goal-line context.
4. `RolePlayEnv`/`RoleDriveEnv`: Gymnasium `check_env`, per-role reward
   attribution, determinism for fixed seed and actions, termination/truncation,
   frozen-opponent overrides.
5. BC per role: demonstration shapes and verified weight loading.
6. PPO smoke tests per role; `config.json` metadata.
7. Evaluation: Wilson intervals and metric definitions on synthetic fixtures;
   composition smoke test.
8. Import purity: `engine`/`rules` never import training/gym/torch/pygame.

## 11. Risks and out of scope

- **Benchmark shift:** carrier-run and fatigue change the dynamics for the
  frozen v10/v11 evaluations (same models, new game). Previous numbers are
  documented as pre-Stage-B; comparisons against v10/v11 use their original
  config (fatigue off).
- **Mixed distributions:** WR trains at play level and CB/rusher in drives;
  composition measures the mismatch and QB v12 absorbs it.
- **Out of scope:** O-line/D-line and roster expansion (M8), full-game clock and
  possessions (M8), MARL/MAPPO and multiagent layout updates (M7 rest/M8), UI
  polish (M8).
