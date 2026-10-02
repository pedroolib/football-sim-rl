"""Gymnasium environments for QB Lab."""

import gymnasium as gym

from qb_sim.training.drive_env import QBDriveEnv
from qb_sim.training.env import QBPassingEnv
from qb_sim.training.rewards import RewardConfig

__all__ = ["QBDriveEnv", "QBPassingEnv", "RewardConfig"]

try:
    gym.register(id="QBPassing-v0", entry_point="qb_sim.training.env:QBPassingEnv")
except gym.error.Error:
    pass

try:
    gym.register(id="QBDrive-v0", entry_point="qb_sim.training.drive_env:QBDriveEnv")
except gym.error.Error:
    pass
