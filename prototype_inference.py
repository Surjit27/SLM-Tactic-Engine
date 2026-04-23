"""Prototype numeric-only inference pipeline: SLM + specialists + fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from prototype_latency import LatencyRecorder
from prototype_vector_db import InMemoryVectorDB
from slm.slm_model import TinySLMPolicy


@dataclass
class PrototypeOutput:
    tactical_class: int
    tactical_probs: np.ndarray
    action: np.ndarray
    latency_breakdown_ms: dict[str, float]
    retrieval: dict[str, np.ndarray | float]


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


class SpecialistHeads:
    def __init__(self, obs_dim: int = 256, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.w_aim = rng.normal(0.0, 0.02, size=(obs_dim + 6, 4)).astype(np.float32)
        self.w_move = rng.normal(0.0, 0.02, size=(obs_dim + 6, 5)).astype(np.float32)
        self.w_weapon = rng.normal(0.0, 0.02, size=(obs_dim, 6)).astype(np.float32)
        self.w_fusion = rng.normal(0.0, 0.02, size=(4 + 5 + 6 + 6, config.ACTION_SIZE)).astype(np.float32)

    def run_aim(self, obs: np.ndarray, tactical_probs: np.ndarray) -> np.ndarray:
        return np.tanh(np.concatenate([obs, tactical_probs]) @ self.w_aim)

    def run_move(self, obs: np.ndarray, tactical_probs: np.ndarray) -> np.ndarray:
        return np.tanh(np.concatenate([obs, tactical_probs]) @ self.w_move)

    def run_weapon(self, obs: np.ndarray) -> np.ndarray:
        return _softmax(obs @ self.w_weapon)

    def run_fusion(self, tactical_probs: np.ndarray, aim: np.ndarray, move: np.ndarray, weapon: np.ndarray) -> np.ndarray:
        fused_in = np.concatenate([tactical_probs, aim, move, weapon])
        return np.tanh(fused_in @ self.w_fusion)


class PrototypeInferencePipeline:
    def __init__(self, vector_db: InMemoryVectorDB, top_k: int = 8):
        self.vector_db = vector_db
        self.top_k = top_k
        self.slm = TinySLMPolicy(obs_dim=config.OBS_SIZE, n_actions=config.TACTICAL_ACTIONS, seed=config.RANDOM_SEED)
        self.heads = SpecialistHeads(obs_dim=config.OBS_SIZE, seed=config.RANDOM_SEED + 1)

    @staticmethod
    def _embed_obs(obs: np.ndarray) -> np.ndarray:
        # Lightweight numeric embedding for vector retrieval.
        h = obs[96:256].reshape(5, 32)
        return np.array(
            [
                float(obs[:32].mean()),
                float(obs[32:64].mean()),
                float(obs[64:96].mean()),
                float(h.mean()),
                float(h.std()),
                float(np.abs(h).mean()),
                float(obs[86]),
                float(obs[6]),
            ],
            dtype=np.float32,
        )

    def predict(self, obs: np.ndarray) -> PrototypeOutput:
        if obs.shape != (config.OBS_SIZE,):
            raise ValueError(f"obs must have shape ({config.OBS_SIZE},)")

        rec = LatencyRecorder()
        stage_latency: dict[str, float] = {}

        t0 = rec.start()
        emb = self._embed_obs(obs)
        rec.stop("embed", t0)
        stage_latency["embed_ms"] = rec.samples["embed"][-1]

        t1 = rec.start()
        retrieval = self.vector_db.query(emb, top_k=self.top_k)
        rec.stop("vector_db_query", t1)
        stage_latency["vector_db_query_ms"] = rec.samples["vector_db_query"][-1]

        t2 = rec.start()
        base_probs = self.slm.probs(obs)
        rec.stop("slm_tactical", t2)
        stage_latency["slm_tactical_ms"] = rec.samples["slm_tactical"][-1]

        t3 = rec.start()
        tactical_prior = retrieval["tactical_prior"]
        if not isinstance(tactical_prior, np.ndarray):
            raise RuntimeError("invalid tactical_prior type")
        risk = float(retrieval["risk_score"])
        combined = 0.70 * base_probs + 0.25 * tactical_prior - 0.05 * risk
        tactical_probs = _softmax(combined)
        tactical_class = int(np.argmax(tactical_probs))
        rec.stop("tactical_blend", t3)
        stage_latency["tactical_blend_ms"] = rec.samples["tactical_blend"][-1]

        t4 = rec.start()
        aim = self.heads.run_aim(obs, tactical_probs)
        rec.stop("aim_head", t4)
        stage_latency["aim_head_ms"] = rec.samples["aim_head"][-1]

        t5 = rec.start()
        move = self.heads.run_move(obs, tactical_probs)
        rec.stop("movement_head", t5)
        stage_latency["movement_head_ms"] = rec.samples["movement_head"][-1]

        t6 = rec.start()
        weapon = self.heads.run_weapon(obs)
        rec.stop("weapon_head", t6)
        stage_latency["weapon_head_ms"] = rec.samples["weapon_head"][-1]

        t7 = rec.start()
        action = self.heads.run_fusion(tactical_probs, aim, move, weapon).astype(np.float32)
        rec.stop("fusion_head", t7)
        stage_latency["fusion_head_ms"] = rec.samples["fusion_head"][-1]

        stage_latency["total_ms"] = float(sum(stage_latency.values()))
        return PrototypeOutput(
            tactical_class=tactical_class,
            tactical_probs=tactical_probs.astype(np.float32),
            action=action,
            latency_breakdown_ms=stage_latency,
            retrieval=retrieval,
        )
