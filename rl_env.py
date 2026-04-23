"""Simple tactical RL environment using offline synthetic observations."""

from __future__ import annotations

import numpy as np


class TacticalRLEnv:
    """
    Episodic environment backed by offline samples.
    State: obs[256]
    Action: discrete tactical decision in [0..5]
    """

    def __init__(self, observations: np.ndarray, tactical_targets: np.ndarray, max_steps: int = 120):
        if observations.ndim != 2 or observations.shape[1] != 256:
            raise ValueError("observations must be shape [N, 256]")
        if tactical_targets.ndim != 1:
            raise ValueError("tactical_targets must be shape [N]")
        if len(observations) != len(tactical_targets):
            raise ValueError("observations and tactical_targets length mismatch")

        self.observations = observations
        self.targets = tactical_targets
        self.max_steps = max_steps
        self.rng = np.random.default_rng(123)
        self.steps = 0
        self._index = 0

    def reset(self) -> np.ndarray:
        self.steps = 0
        self._index = int(self.rng.integers(0, len(self.observations)))
        return self.observations[self._index]

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self.steps += 1
        target = int(self.targets[self._index])

        # Base reward for matching tactical decision.
        reward = 1.0 if action == target else -0.5

        # Extra shaping from "humanization-like" smoothness on state features.
        obs = self.observations[self._index]
        threat = float(np.clip(obs[86], 0.0, 1.0))
        health = float(np.clip(obs[6], 0.0, 1.0))

        # Encourage retreat/heal when danger is high and health is low.
        if threat > 0.7 and health < 0.4 and action in (3, 4):
            reward += 0.25
        # Encourage push/flank in low threat scenarios.
        if threat < 0.35 and action in (0, 2):
            reward += 0.1

        done = self.steps >= self.max_steps

        # Move to next random sample to simulate diverse trajectory.
        self._index = int(self.rng.integers(0, len(self.observations)))
        next_obs = self.observations[self._index]
        info = {"target": target}
        return next_obs, reward, done, info
