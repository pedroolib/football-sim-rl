"""Gymnasium environment for training the quarterback (M4).

The QB is the learning agent; receivers and defenders use the rule-based
controllers from M3. Observations are normalized, role-local vectors and the
hybrid action is ``[move_x, move_y, aim_x, aim_y, power_raw, throw_trigger]``
with every component in ``[-1, 1]``. A positive trigger throws toward
``(aim_x, aim_y)`` with ``power = (power_raw + 1) / 2``; otherwise the QB moves.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from qb_sim.agents.base import BaseAgent
from qb_sim.agents.loop import build_rule_agents
from qb_sim.engine.actions import Move, Throw
from qb_sim.engine.entities import BallStatus, PlayOutcome, Role
from qb_sim.engine.geometry import vec2
from qb_sim.rules.play import Play
from qb_sim.rules.scenarios import DEFAULT_SCENARIO, Scenario
from qb_sim.training.rewards import RewardConfig, terminal_reward

OBSERVATION_SIZE = 27
ACTION_SIZE = 6
_THROW_TRIGGER_THRESHOLD = 0.0
_BALL_STATUS_ORDER = (
    BallStatus.HELD,
    BallStatus.IN_FLIGHT,
    BallStatus.CARRIED,
    BallStatus.DEAD,
)


class QBPassingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Train a quarterback against rule-based teammates and defenders."""

    metadata = {"render_modes": []}

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
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_SIZE,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.play = Play(self.scenario.config)
        self._agents: dict[str, BaseAgent] = {}
        self._build_agents()
        self._steps = 0
        self._episode_seed = self.scenario.seed

    def _build_agents(self) -> None:
        self._agents = build_rule_agents(
            self.play,
            self.scenario.receiver_routes,
            self.scenario.config,
            include_quarterback=False,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new play; pass ``seed`` for a reproducible episode."""
        super().reset(seed=seed)
        if seed is not None:
            self._episode_seed = int(seed)
        else:
            self._episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.play = Play(self.scenario.config, seed=self._episode_seed)
        self._build_agents()
        self._steps = 0
        return self._observation(), self._info()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply the QB action, advance one tick, and resolve the play."""
        engine_actions: dict[str, Move | Throw] = {}
        for player_id, agent in self._agents.items():
            agent_action = agent.decide(self.play.state)
            if agent_action is not None:
                engine_actions[player_id] = agent_action
        engine_actions.update(self._decode_action(action))

        state = self.play.step(engine_actions)
        self._steps += 1
        terminated = state.outcome is not PlayOutcome.IN_PROGRESS
        truncated = not terminated and self._steps >= self.max_steps
        reward = self._step_reward(state, terminated)
        return self._observation(), reward, terminated, truncated, self._info()

    def render(self) -> None:
        """No rendering in M4; visual demos belong to M5."""
        return None

    def _decode_action(self, action: np.ndarray) -> dict[str, Move | Throw]:
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        if values.shape != (ACTION_SIZE,):
            raise ValueError(f"action must have {ACTION_SIZE} values")
        trigger = float(values[5])
        if trigger > _THROW_TRIGGER_THRESHOLD and self.play.state.ball.status is BallStatus.HELD:
            aim = vec2(float(values[2]), float(values[3]))
            if float(np.linalg.norm(aim)) > 1e-9:
                power = min(max((float(values[4]) + 1.0) / 2.0, 0.0), 1.0)
                return {"qb": Throw(direction=aim, power=power)}

        move = vec2(float(values[0]), float(values[1]))
        length = float(np.linalg.norm(move))
        if length <= 1e-9:
            return {}
        if length > 1.0:
            move = move / length
        return {"qb": Move(direction=move)}

    def _step_reward(self, state, terminated: bool) -> float:
        reward = self.reward_config.time_penalty
        if terminated:
            yards = 0.0
            if state.carrier_id is not None:
                yards = float(
                    state.player(state.carrier_id).position[0]
                    - self.scenario.config.quarterback_start[0]
                )
            reward += terminal_reward(state.outcome, yards, self.reward_config)
        return float(reward)

    def _observation(self) -> np.ndarray:
        config = self.play.config
        field = config.field
        state = self.play.state
        quarterback = state.player("qb")
        values: list[float] = [
            quarterback.position[0] / field.length_yards,
            quarterback.position[1] / field.width_yards,
            quarterback.velocity[0] / config.quarterback_speed,
            quarterback.velocity[1] / config.quarterback_speed,
        ]
        values.extend(1.0 if state.ball.status is status else 0.0 for status in _BALL_STATUS_ORDER)
        values.append(state.ball.height / config.max_arc_height)

        for role, speed in (
            (Role.RECEIVER, config.receiver_speed),
            (Role.DEFENDER, config.defender_speed),
        ):
            for player in state.players:
                if player.role is not role:
                    continue
                values.extend(
                    (
                        (player.position[0] - quarterback.position[0]) / field.length_yards,
                        (player.position[1] - quarterback.position[1]) / field.width_yards,
                        player.velocity[0] / speed,
                        player.velocity[1] / speed,
                    )
                )

        values.append(state.pressure)
        values.append(1.0 - state.elapsed_seconds / config.max_play_duration)
        return np.clip(np.array(values, dtype=np.float32), -1.0, 1.0)

    def _info(self) -> dict[str, Any]:
        state = self.play.state
        return {
            "episode_seed": self._episode_seed,
            "scenario": self.scenario.name,
            "outcome": state.outcome.value,
            "elapsed_seconds": state.elapsed_seconds,
            "pressure": state.pressure,
            "throw_time": state.throw_time,
            "carrier_id": state.carrier_id,
            "steps": self._steps,
        }
