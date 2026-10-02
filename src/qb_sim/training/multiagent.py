"""PettingZoo ParallelEnv where every player is an autonomous agent (M6)."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from qb_sim.agents.loop import build_rule_agents
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import PlayOutcome, PlayState
from qb_sim.rules.play import Play
from qb_sim.rules.scenarios import DEFAULT_SCENARIO, Scenario
from qb_sim.training.actions import ACTION_SIZE, decode_action, encode_engine_action
from qb_sim.training.observations import OBSERVATION_SIZE, observation_from_state
from qb_sim.training.rewards import RewardConfig, terminal_reward
from qb_sim.training.role_observations import (
    ROLE_OBSERVATION_SIZE,
    defender_observation,
    receiver_observation,
)

MOVE_ACTION_SIZE = 2


class QBPassingMultiAgentEnv(ParallelEnv):
    """Every player (QB, receivers, corners, rushers) is an agent."""

    metadata = {"name": "qb_passing_multi_v0", "render_modes": []}

    def __init__(
        self,
        scenario: Scenario | None = None,
        reward_config: RewardConfig | None = None,
        max_steps: int = 240,
    ) -> None:
        super().__init__()
        self.scenario = scenario if scenario is not None else DEFAULT_SCENARIO
        self.reward_config = reward_config if reward_config is not None else RewardConfig()
        self.max_steps = int(max_steps)
        self.play = Play(self.scenario.config)
        self.possible_agents = ["qb", *self.play.receiver_ids, *self.play.defender_ids]
        self.agents: list[str] = []
        self._steps = 0
        self._seed = self.scenario.seed
        self._qb_space = spaces.Box(-1.0, 1.0, shape=(ACTION_SIZE,), dtype=np.float32)
        self._mover_space = spaces.Box(-1.0, 1.0, shape=(MOVE_ACTION_SIZE,), dtype=np.float32)
        self._qb_observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self._role_observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(ROLE_OBSERVATION_SIZE,),
            dtype=np.float32,
        )

    def observation_space(self, agent: str):
        """Observation space for a role; the QB sees more than movers."""
        return self._qb_observation_space if agent == "qb" else self._role_observation_space

    def action_space(self, agent: str):
        """The QB throws and moves; every other role only moves."""
        return self._qb_space if agent == "qb" else self._mover_space

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        self._seed = self.scenario.seed if seed is None else int(seed)
        self.play = Play(self.scenario.config, seed=self._seed)
        self.agents = list(self.possible_agents)
        self._steps = 0
        observations = {agent: self._observe(agent) for agent in self.agents}
        infos = {agent: self._info() for agent in self.agents}
        return observations, infos

    def step(
        self,
        actions: Mapping[str, np.ndarray],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        engine_actions: dict[str, Move | Throw] = {}
        for agent, action in actions.items():
            decoded = self._decode(agent, action)
            if decoded is not None:
                engine_actions[agent] = decoded
        state = self.play.step(engine_actions)
        self._steps += 1
        terminated = state.outcome is not PlayOutcome.IN_PROGRESS
        truncated = not terminated and self._steps >= self.max_steps
        rewards = {agent: self._reward(agent, state, terminated) for agent in self.agents}
        observations = {agent: self._observe(agent) for agent in self.agents}
        infos = {agent: self._info() for agent in self.agents}
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        if terminated or truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        """No rendering for the multi-agent training environment."""
        return None

    def close(self) -> None:
        return None

    def _decode(self, agent: str, action: np.ndarray) -> Move | Throw | None:
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        if agent == "qb":
            return decode_action(values, self.play.state.ball.status).get("qb")
        if values.shape != (MOVE_ACTION_SIZE,):
            raise ValueError(f"{agent} action must have {MOVE_ACTION_SIZE} values")
        direction = np.array(values, dtype=np.float64)
        length = float(np.linalg.norm(direction))
        if length <= 1e-9:
            return None
        if length > 1.0:
            direction = direction / length
        return Move(direction=direction)

    def _observe(self, agent: str) -> np.ndarray:
        state = self.play.state
        if agent == "qb":
            return observation_from_state(state, self.play.config)
        if agent in self.play.receiver_ids:
            return receiver_observation(state, agent, self.play.config)
        return defender_observation(
            state,
            agent,
            self._defender_target(agent),
            self.play.config,
        )

    def _defender_target(self, defender_id: str) -> str | None:
        state = self.play.state
        if state.carrier_id is not None:
            return state.carrier_id
        if defender_id in self.play.rusher_ids:
            return "qb"
        coverage = [
            player_id
            for player_id in self.play.defender_ids
            if player_id not in set(self.play.rusher_ids)
        ]
        index = coverage.index(defender_id)
        return self.play.receiver_ids[index % len(self.play.receiver_ids)]

    def _reward(self, agent: str, state: PlayState, terminated: bool) -> float:
        is_offense = agent == "qb" or agent in self.play.receiver_ids
        reward = self.reward_config.time_penalty if is_offense else 0.0
        if terminated:
            yards = 0.0
            if state.carrier_id is not None:
                yards = float(
                    state.player(state.carrier_id).position[0]
                    - self.scenario.config.quarterback_start[0]
                )
            offense_reward = terminal_reward(state.outcome, yards, self.reward_config)
            reward += offense_reward if is_offense else -offense_reward
        return reward

    def _info(self) -> dict[str, Any]:
        state = self.play.state
        return {
            "episode_seed": self._seed,
            "scenario": self.scenario.name,
            "outcome": state.outcome.value,
            "elapsed_seconds": state.elapsed_seconds,
            "pressure": state.pressure,
            "throw_time": state.throw_time,
            "carrier_id": state.carrier_id,
            "steps": self._steps,
        }


def run_rule_episode(env: QBPassingMultiAgentEnv, seed: int) -> dict[str, Any]:
    """Run one episode with the M3 rule controllers as every player's policy."""
    env.reset(seed=seed)
    controllers = build_rule_agents(
        env.play,
        env.scenario.receiver_routes,
        env.scenario.config,
        include_quarterback=True,
    )
    first_agent = env.possible_agents[0]
    infos: dict[str, dict[str, Any]] = {}
    while env.agents:
        actions: dict[str, np.ndarray] = {}
        for agent in env.agents:
            action = controllers[agent].decide(env.play.state)
            if action is None:
                actions[agent] = np.zeros(env.action_space(agent).shape, dtype=np.float32)
            elif agent == "qb":
                actions[agent] = encode_engine_action(action)
            else:
                actions[agent] = np.array(
                    [action.direction[0], action.direction[1]],
                    dtype=np.float32,
                )
        _, _, _, _, infos = env.step(actions)
    return infos[first_agent]
