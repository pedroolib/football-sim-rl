"""Behavior-cloning warm start from the rule-based quarterback.

The rule QB provides high-quality demonstrations (lead throws to the openest
receiver). Imitating them gives PPO a starting point from which exploration of
precise aiming is feasible.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from qb_sim.agents.loop import build_rule_agents
from qb_sim.agents.rule_controllers import HeuristicQuarterback
from qb_sim.engine.entities import PlayOutcome
from qb_sim.rules.drive import DriveConfig
from qb_sim.rules.play import Play
from qb_sim.rules.scenarios import Scenario, get_scenario
from qb_sim.training.actions import ACTION_SIZE, encode_engine_action
from qb_sim.training.drive_env import QBDriveEnv
from qb_sim.training.drive_observations import drive_observation_from_state
from qb_sim.training.observations import observation_from_state
from qb_sim.training.role_env import RoleDriveEnv, RolePlayEnv

_STAND_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class DemonstrationSet:
    """Recorded observations and the rule quarterback's actions."""

    observations: np.ndarray
    actions: np.ndarray


def collect_demonstrations(
    scenario: Scenario | None = None,
    episodes: int = 200,
    seed_start: int = 0,
) -> DemonstrationSet:
    """Run the rule QB and record ``(observation, action)`` pairs every tick."""
    chosen = scenario if scenario is not None else get_scenario("default")
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for episode in range(episodes):
        play = Play(chosen.config, seed=seed_start + episode)
        agents = build_rule_agents(
            play,
            chosen.receiver_routes,
            chosen.config,
            include_quarterback=True,
        )
        state = play.state
        while state.outcome is PlayOutcome.IN_PROGRESS:
            engine_actions = {}
            for player_id, agent in agents.items():
                action = agent.decide(state)
                if action is not None:
                    engine_actions[player_id] = action
            observations.append(observation_from_state(state, chosen.config))
            quarterback_action = engine_actions.get("qb")
            actions.append(
                encode_engine_action(quarterback_action)
                if quarterback_action is not None
                else _STAND_ACTION.copy()
            )
            state = play.step(engine_actions)
    return DemonstrationSet(
        observations=np.stack(observations).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
    )


def concatenate_demonstrations(datasets: Sequence[DemonstrationSet]) -> DemonstrationSet:
    """Join demonstration sets from several scenarios into one."""
    if not datasets:
        raise ValueError("at least one demonstration set is required")
    return DemonstrationSet(
        observations=np.concatenate([dataset.observations for dataset in datasets]),
        actions=np.concatenate([dataset.actions for dataset in datasets]),
    )


def collect_drive_demonstrations(
    scenario: Scenario | None = None,
    episodes: int = 200,
    seed_start: int = 0,
    drive_config: DriveConfig | None = None,
) -> DemonstrationSet:
    """Run the rule QB inside a drive and record ``(observation, action)`` pairs."""
    chosen = scenario if scenario is not None else get_scenario("rushed")
    env = QBDriveEnv(chosen, drive_config=drive_config)
    quarterback = HeuristicQuarterback("qb", chosen.config)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for episode in range(episodes):
        _observation, _ = env.reset(seed=seed_start + episode)
        terminated = truncated = False
        while not (terminated or truncated):
            engine_action = quarterback.decide(env.play.state)
            action = (
                encode_engine_action(engine_action)
                if engine_action is not None
                else _STAND_ACTION.copy()
            )
            observations.append(
                drive_observation_from_state(
                    state=env.play.state,
                    routes=env.drive.routes,
                    drive_state=env.drive.state,
                    config=chosen.config,
                    goal_line=env.drive_config.goal_line,
                )
            )
            actions.append(action)
            _observation, _, terminated, truncated, _ = env.step(action)
    env.close()
    return DemonstrationSet(
        observations=np.stack(observations).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
    )


class BehaviorCloningNetwork(nn.Module):
    """MLP matching the structure of Stable-Baselines3's policy network."""

    def __init__(
        self,
        input_size: int,
        hidden: tuple[int, ...] = (64, 64),
        action_size: int = ACTION_SIZE,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        size = input_size
        for hidden_size in hidden:
            layers.extend((nn.Linear(size, hidden_size), nn.Tanh()))
            size = hidden_size
        self.policy_net = nn.Sequential(*layers)
        self.action_net = nn.Linear(size, action_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.action_net(self.policy_net(observations))


def train_behavior_cloning(
    dataset: DemonstrationSet,
    hidden: tuple[int, ...] = (64, 64),
    epochs: int = 60,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
    throw_weight: float = 25.0,
    action_size: int = ACTION_SIZE,
) -> tuple[BehaviorCloningNetwork, float]:
    """Train a network to imitate the demonstrations; return it and the loss.

    Stable-Baselines3's ``MlpPolicy`` outputs raw Gaussian means (no ``tanh``
    squashing), and the environment decodes those raw values, so the targets
    are the recorded actions themselves. The trigger is trained with binary
    cross-entropy and the rare throw samples upweight the aim and power
    components that decide the pass.
    """
    torch.manual_seed(seed)
    network = BehaviorCloningNetwork(dataset.observations.shape[1], hidden, action_size)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    observations = torch.as_tensor(dataset.observations)
    targets = torch.as_tensor(dataset.actions)
    generator = torch.Generator().manual_seed(seed)

    def batch_loss(predictions: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        if action_size != ACTION_SIZE:
            return nn.functional.mse_loss(predictions, targets[index])
        is_throw = targets[index, 5] > 0.0
        trigger_loss = nn.functional.binary_cross_entropy_with_logits(
            predictions[:, 5],
            is_throw.float(),
        )
        move_mse = nn.functional.mse_loss(
            predictions[:, :5],
            targets[index, :5],
            reduction="none",
        ).mean(dim=1)
        throw_mse = nn.functional.mse_loss(
            predictions[:, 2:5],
            targets[index, 2:5],
            reduction="none",
        ).mean(dim=1)
        action_loss = torch.where(is_throw, throw_weight * throw_mse, move_mse).mean()
        return trigger_loss + action_loss

    with torch.no_grad():
        final_loss = float(batch_loss(network(observations), torch.arange(len(observations))))
    for _ in range(epochs):
        permutation = torch.randperm(len(observations), generator=generator)
        for start in range(0, len(observations), batch_size):
            index = permutation[start : start + batch_size]
            optimizer.zero_grad()
            loss = batch_loss(network(observations[index]), index)
            loss.backward()
            optimizer.step()
        final_loss = float(loss.detach())
    return network, final_loss


def warm_start_policy(model: Any, dataset: DemonstrationSet, **kwargs) -> float:
    """Copy imitated weights into a Stable-Baselines3 PPO policy."""
    network, loss = train_behavior_cloning(dataset, **kwargs)
    network.to(model.device)
    model.policy.mlp_extractor.policy_net.load_state_dict(network.policy_net.state_dict())
    model.policy.action_net.load_state_dict(network.action_net.state_dict())
    return loss


def _encoded_move(action) -> np.ndarray:
    if action is None:
        return np.zeros(2, dtype=np.float32)
    return np.array([action.direction[0], action.direction[1]], dtype=np.float32)


def collect_role_play_demonstrations(
    scenario: Scenario | None = None,
    learner_id: str = "wr1",
    episodes: int = 200,
    seed_start: int = 0,
    opponents: Mapping[str, object] | None = None,
    goal_line: float = 100.0,
) -> DemonstrationSet:
    """Run the rule receiver and record ``(31-observation, 2-action)`` pairs."""
    chosen = scenario if scenario is not None else get_scenario("stage_b")
    env = RolePlayEnv(chosen, learner_id=learner_id, opponents=opponents, goal_line=goal_line)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for episode in range(episodes):
        env.reset(seed=seed_start + episode)
        terminated = truncated = False
        while not (terminated or truncated):
            agent = env.rule_agent(learner_id)
            action = agent.decide(env.play.state)
            observations.append(env.observation())
            actions.append(_encoded_move(action))
            _, _, terminated, truncated, _ = env.step(actions[-1])
    env.close()
    return DemonstrationSet(
        observations=np.stack(observations).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
    )


def collect_role_drive_demonstrations(
    scenario: Scenario | None = None,
    learner_id: str = "cb1",
    episodes: int = 200,
    seed_start: int = 0,
    opponents: Mapping[str, object] | None = None,
    drive_config: DriveConfig | None = None,
) -> DemonstrationSet:
    """Run the rule defender and record ``(27-observation, 2-action)`` pairs."""
    chosen = scenario if scenario is not None else get_scenario("stage_b")
    env = RoleDriveEnv(
        chosen,
        learner_id=learner_id,
        opponents=opponents,
        drive_config=drive_config,
    )
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for episode in range(episodes):
        env.reset(seed=seed_start + episode)
        terminated = truncated = False
        while not (terminated or truncated):
            agent = env.rule_agent(learner_id)
            action = agent.decide(env.play.state)
            observations.append(env.observation())
            actions.append(_encoded_move(action))
            _, _, terminated, truncated, _ = env.step(actions[-1])
    env.close()
    return DemonstrationSet(
        observations=np.stack(observations).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
    )
