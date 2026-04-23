"""Prototype in-memory vector database for top-k retrieval."""

from __future__ import annotations

import numpy as np


class InMemoryVectorDB:
    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None
        self._tactical_labels: np.ndarray | None = None
        self._risk_scores: np.ndarray | None = None

    def build_index(self, embeddings: np.ndarray, tactical_labels: np.ndarray, risk_scores: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D [N, D]")
        if tactical_labels.ndim != 1 or risk_scores.ndim != 1:
            raise ValueError("tactical_labels and risk_scores must be 1D")
        if len(embeddings) != len(tactical_labels) or len(embeddings) != len(risk_scores):
            raise ValueError("index arrays must have same length")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        self._embeddings = (embeddings / norms).astype(np.float32)
        self._tactical_labels = tactical_labels.astype(np.int64)
        self._risk_scores = risk_scores.astype(np.float32)

    def query(self, query_embedding: np.ndarray, top_k: int = 8) -> dict[str, np.ndarray | float]:
        if self._embeddings is None or self._tactical_labels is None or self._risk_scores is None:
            raise RuntimeError("index not built")

        q = query_embedding.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        sims = self._embeddings @ q

        k = min(top_k, len(sims))
        nn_idx = np.argpartition(-sims, kth=k - 1)[:k]
        nn_sims = sims[nn_idx]
        order = np.argsort(-nn_sims)
        idx = nn_idx[order]
        sim_sorted = nn_sims[order]

        tactical_prior = np.zeros((6,), dtype=np.float32)
        for i, score in zip(idx, sim_sorted):
            label = int(self._tactical_labels[i])
            tactical_prior[label] += max(float(score), 0.0)
        if tactical_prior.sum() > 0:
            tactical_prior /= tactical_prior.sum()

        risk = float(np.mean(self._risk_scores[idx])) if len(idx) > 0 else 0.0
        rare_flag = float(np.mean(sim_sorted) < 0.45)
        return {
            "tactical_prior": tactical_prior,
            "risk_score": risk,
            "rare_case_flag": rare_flag,
            "neighbor_similarity_mean": float(np.mean(sim_sorted)) if len(sim_sorted) else 0.0,
        }
