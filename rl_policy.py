"""Small softmax policy (NumPy) for tactical RL training."""

from __future__ import annotations

import numpy as np


class SoftmaxPolicy:
    def __init__(self, obs_dim: int, n_actions: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.01, size=(obs_dim, n_actions)).astype(np.float32)
        self.b = np.zeros((n_actions,), dtype=np.float32)

    def logits(self, obs: np.ndarray) -> np.ndarray:
        return obs @ self.w + self.b

    def probs(self, obs: np.ndarray) -> np.ndarray:
        z = self.logits(obs)
        z = z - np.max(z)  # numerical stability
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def sample_action(self, obs: np.ndarray, rng: np.random.Generator) -> tuple[int, np.ndarray]:
        p = self.probs(obs)
        action = int(rng.choice(len(p), p=p))
        return action, p

    def reinforce_update(
        self,
        obs_batch: list[np.ndarray],
        act_batch: list[int],
        return_batch: list[float],
        learning_rate: float,
    ) -> float:
        """REINFORCE gradient update; returns mean loss."""
        grad_w = np.zeros_like(self.w)
        grad_b = np.zeros_like(self.b)
        losses = []

        for obs, action, g_t in zip(obs_batch, act_batch, return_batch):
            p = self.probs(obs)
            one_hot = np.zeros_like(p)
            one_hot[action] = 1.0

            # Gradient of -G_t * log pi(a|s)
            diff = (p - one_hot) * g_t
            grad_w += np.outer(obs, diff)
            grad_b += diff
            losses.append(-g_t * np.log(max(p[action], 1e-8)))

        n = max(len(obs_batch), 1)
        self.w -= learning_rate * (grad_w / n)
        self.b -= learning_rate * (grad_b / n)
        return float(np.mean(losses)) if losses else 0.0
