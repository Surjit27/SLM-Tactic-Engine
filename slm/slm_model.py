"""Minimal sequence-style tactical model (NumPy) for SLM path."""

from __future__ import annotations

import numpy as np


class TinySLMPolicy:
    """
    Lightweight sequence-inspired model:
    - Uses full obs plus a temporal summary from obs[96:256]
    - Outputs logits for 6 tactical classes
    """

    def __init__(self, obs_dim: int = 256, n_actions: int = 6, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.obs_w = rng.normal(0.0, 0.02, size=(obs_dim, n_actions)).astype(np.float32)
        self.hist_w = rng.normal(0.0, 0.02, size=(5, n_actions)).astype(np.float32)
        self.b = np.zeros((n_actions,), dtype=np.float32)

    @staticmethod
    def _history_features(obs: np.ndarray) -> np.ndarray:
        hist = obs[96:256].reshape(5, 32)
        return np.array(
            [
                float(hist.mean()),
                float(hist.std()),
                float(np.abs(hist).mean()),
                float(hist[-1].mean()),
                float(hist[0].mean()),
            ],
            dtype=np.float32,
        )

    def logits(self, obs: np.ndarray) -> np.ndarray:
        return obs @ self.obs_w + self._history_features(obs) @ self.hist_w + self.b

    def probs(self, obs: np.ndarray) -> np.ndarray:
        z = self.logits(obs)
        z = z - np.max(z)
        ex = np.exp(z)
        return ex / np.sum(ex)

    def predict(self, obs: np.ndarray) -> int:
        return int(np.argmax(self.probs(obs)))
