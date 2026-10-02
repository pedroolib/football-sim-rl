# QB Lab — Agent Instructions

## 1. Project Overview

Build a 2D American football quarterback simulation designed for Reinforcement Learning research.

The objective is to train an autonomous AI quarterback that learns to:
- Identify open receivers.
- Anticipate receiver movement.
- Select when to throw.
- Predict where a receiver will be when the ball arrives.
- Optimize passing accuracy and yardage.
- Avoid interceptions, sacks, and poor decisions.
- Eventually learn to read increasingly complex defensive formations.

This project is inspired by the simple passing mechanics of Retro Bowl, but it is an original simulation with its own physics, rules, visuals, and training environment.

**The primary objective is learning quarterback decision-making, not recreating an existing commercial game.**

## 2. Development Philosophy

Follow these principles throughout the project:

- Build incrementally. Never implement the entire roadmap in one step.
- Prioritize a correct, testable simulation over visual polish.
- Separate simulation, RL environment, training, and rendering.
- Keep the environment fast enough to support thousands of training episodes.
- Use simple mechanics initially and introduce complexity only when justified by experiments.
- Prefer minimal dependencies and avoid premature abstractions.
- Explain significant technical decisions in clear, accessible language.
- Communicate with the user in Spanish. Use English for code identifiers and standard technical terminology.
- Do not silently make major architectural changes.
- If an important requirement is ambiguous, ask the user before implementing it.
- After each milestone, explain what was built, which files changed, which tests passed, and what should be done next.

## 3. Technology Stack

Primary stack:

- Python 3.11+
- NumPy for numerical calculations.
- Gymnasium for the RL environment API.
- PyTorch for neural network experimentation.
- Stable-Baselines3 for PPO and baseline RL algorithms.
- Pygame for optional visualization.
- Pytest for automated testing.
- Ruff for formatting and linting.

Use a Python-first architecture. Do not introduce Unity, computer vision, external LLMs, or distributed training unless explicitly requested or justified by a later experiment.

The simulator must work in headless mode without rendering.

## 4. Architecture

Keep the following responsibilities separate:

### Simulation Core

Responsible for:
- Field geometry.
- Player positions and velocities.
- Receiver route execution.
- Defensive movement.
- Ball movement and flight.
- Catch, interception, and out-of-bounds detection.
- Play outcomes.
- Simulation time and fixed time steps.

The simulation core must not depend on Gymnasium or a rendering library.

### RL Environment

Implement a Gymnasium-compatible environment, initially named `QBPassingEnv`.

It is responsible for:
- Initializing plays.
- Producing observations.
- Validating and applying agent actions.
- Advancing the simulation.
- Calculating rewards.
- Determining episode termination.
- Returning diagnostic information.

Implement the standard API:

- `reset(seed=None, options=None)`
- `step(action)`
- `render()`
- `close()`

Follow Gymnasium's current return conventions, including separate `terminated` and `truncated` values.

### Agent and Training

Responsible for:
- PPO configuration.
- Training loops.
- Checkpoint management.
- Evaluation on separate random seeds.
- Metrics and experiment logging.

Training code must not contain football physics or duplicate environment logic.

### Rendering

Use Pygame as a visualization layer.

Render:
- Field and yard markings.
- Quarterback.
- Receivers.
- Defenders.
- Routes.
- Football trajectory.
- Play status and outcome.

Rendering must read simulation state without changing its behavior.

## 5. Initial Simulation Scope — Version 0.1

Start with the smallest useful passing environment.

Entities:
- One quarterback.
- One wide receiver.
- One cornerback.
- One football.

Mechanics:
- Fixed simulation time step.
- A small set of scripted receiver routes.
- Simple player movement and collision detection.
- A deterministic, configurable passing model.
- Receiver and defender movement during a play.
- Catch, incompletion, and interception outcomes.
- A maximum play duration to prevent infinite episodes.

Use a simple 2D field coordinate system with normalized positions where appropriate.

Start with basic kinematics. Do not attempt realistic aerodynamic or biomechanical simulation in the first version.

The simulation must be deterministic when initialized with the same seed and configuration.

## 6. Observation Space

Initially, give the agent structured, ground-truth observations.

Include normalized values for:
- Quarterback position.
- Receiver position and velocity.
- Defender position and velocity.
- Relative distances between relevant players.
- Time since snap.
- Time remaining before the play ends.
- Ball status.
- Relevant field boundaries.

Use a consistent coordinate convention and document it.

Include only information available to the agent in the chosen experiment. Do not reveal future player positions or hidden random variables.

Keep the initial observation space small and understandable.

Later experiments may introduce multiple receivers, multiple defenders, incomplete information, or visual observations.

## 7. Action Space

Start with a discrete action space compatible with Stable-Baselines3 PPO.

The initial actions should include:
- Wait.
- Throw with a short lead.
- Throw with a medium lead.
- Throw with a long lead.

For each throw, calculate a target based on the receiver's current position, velocity, and a configurable lead parameter.

Keep action-to-throw mapping explicit and documented.

The first version should focus on learning the timing and basic placement of a pass rather than solving arbitrary continuous aiming.

After establishing a working baseline, investigate continuous or hybrid action spaces for more precise control over:
- Throw target.
- Throw power.
- Release timing.

Do not introduce custom policy distributions before a working discrete PPO baseline exists.

## 8. Reward Design

Implement an initial, configurable reward function.

Consider:
- Positive reward for successful completions.
- Reward proportional to useful yardage.
- Additional reward for reaching a first down.
- A significant touchdown bonus.
- Penalties for interceptions, sacks, and incompletions.
- Optional small time or pressure penalties to discourage indefinite waiting.

Avoid rewarding behavior that can be exploited without advancing the play.

Separate terminal rewards from optional intermediate reward shaping.

Document every reward coefficient and its intended effect.

Use `info` to expose detailed outcomes, even when they do not directly contribute to reward.

Reward design must be evaluated experimentally rather than assumed to be correct.

## 9. Baselines

Before training PPO, implement and evaluate:

1. Random Agent: selects valid actions randomly.
2. Simple Heuristic Agent: uses receiver separation and a basic throwing rule.

These baselines are essential for determining whether the learned policy actually improves.

Keep evaluation scenarios and random seeds consistent when comparing agents.

## 10. PPO Training

Use Stable-Baselines3 PPO for the first RL experiment.

Start with a small multilayer perceptron policy.

Expose important hyperparameters in a configuration file, including:
- Learning rate.
- Discount factor.
- Rollout length.
- Batch size.
- Number of training epochs.
- Entropy coefficient.
- Clip range.
- Total training steps.
- Random seed.

Do not prematurely optimize hyperparameters.

Provide scripts for:
- Starting training.
- Saving checkpoints.
- Resuming training.
- Evaluating a saved policy.
- Running deterministic demonstration episodes.

Keep training independent of rendering.

Use TensorBoard or another lightweight logging solution when useful.

## 11. Evaluation

Track at least these metrics:

- Average episode reward.
- Completion percentage.
- Interception percentage.
- Incompletion percentage.
- Average yards per attempt.
- Average yards per completion.
- Touchdown percentage, when applicable.
- Average time before throwing.
- Episode length.

Evaluate policies on seeds not used during training.

Compare PPO against both baselines.

Do not claim the agent has learned a skill merely because training reward increased. Verify that relevant gameplay metrics improve on unseen scenarios.

Save evaluation results in a reproducible format.

## 12. Testing and Code Quality

Every milestone must include appropriate tests.

At minimum, test:
- Environment reset behavior.
- Observation shape, type, and valid ranges.
- Action validation.
- Simulation boundaries.
- Ball flight and catch detection.
- Interception detection.
- Episode termination.
- Seed reproducibility.
- Reward calculations.
- Gymnasium environment compatibility.

Use type hints and clear function names.

Keep functions focused and avoid unnecessary complexity.

Run tests and linting before declaring a milestone complete.

Do not hide failing tests or silently weaken assertions to make a build pass.

## 13. Development Roadmap

Work through these milestones in order.

### Milestone 0 — Repository Setup
- Inspect the existing repository.
- Identify its current structure and tooling.
- Propose a minimal directory structure.
- Identify dependencies and setup requirements.
- Present a short implementation plan.
- Wait for user approval before making substantial code changes.

### Milestone 1 — Simulation Core
- Implement field geometry and entities.
- Implement player movement.
- Implement basic receiver routes.
- Implement ball movement and collision detection.
- Create deterministic simulation tests.
- Do not implement RL yet.

### Milestone 2 — Gymnasium Environment
- Implement `QBPassingEnv`.
- Define observation and discrete action spaces.
- Implement rewards and episode termination.
- Validate with Gymnasium's environment checker.
- Add environment tests.

### Milestone 3 — Baseline Agents
- Implement random and heuristic agents.
- Run evaluation episodes.
- Record baseline metrics.
- Verify that the environment produces meaningful gameplay outcomes.

### Milestone 4 — PPO Training
- Integrate Stable-Baselines3 PPO.
- Create configurable training and evaluation scripts.
- Save and reload checkpoints.
- Compare results against baselines.

### Milestone 5 — Visualization
- Add Pygame rendering.
- Display trained agent demonstrations.
- Visualize routes, decisions, and ball trajectories.
- Ensure visualization does not alter training dynamics.

### Milestone 6 — Increasing Difficulty
Only after the initial environment works:
- Add a second receiver and defender.
- Introduce additional receiver routes.
- Randomize player speeds and initial positions.
- Add more defensive behaviors.
- Experiment with curriculum learning and domain randomization.

### Milestone 7 — Advanced Experiments
Explore independently measurable experiments:
- Continuous-action PPO or SAC.
- Behavior cloning from human demonstrations.
- Partial observability and recurrent policies.
- More realistic passing physics.
- Defensive agents and self-play.
- Hierarchical policies.
- Visual observations and CNN-based policies.

Do not combine every advanced technique into one experiment. Establish clear comparisons.

## 14. Git and Experiment Management

Use Git with focused commits.

Keep configurations and experiment results reproducible.

Do not commit:
- Virtual environments.
- Large temporary datasets.
- Unnecessary model checkpoints.
- Generated cache files.

Provide a `.gitignore` and document how to reproduce important experiments.

Use clear experiment names and preserve the configuration associated with each checkpoint.

## 15. Communication and Workflow

At the beginning of each new development session:
1. Read this file.
2. Inspect the current repository and relevant existing code.
3. Identify the current milestone and project status.
4. Summarize what is already implemented and what remains.

Before implementing a new milestone:
1. Explain the proposed approach.
2. Identify the files to create or modify.
3. Mention important technical decisions and trade-offs.
4. Ask for approval if the work involves a significant architectural decision.

For each completed milestone:
1. Summarize the implementation.
2. List changed files.
3. Report tests and their actual results.
4. Explain relevant RL concepts introduced.
5. Identify the next logical task without implementing it automatically.

**Always prioritize a small, working, measurable system over a large unfinished implementation.**
