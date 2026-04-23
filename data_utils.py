"""Synthetic dataset creation utilities aligned to obs[256], action[32]."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

import config


@dataclass
class DatasetSplit:
    observations: np.ndarray
    actions: np.ndarray
    tactical_targets: np.ndarray


def _sample_observation(rng: np.random.Generator) -> np.ndarray:
    """Create one random-but-structured observation vector of size 256."""
    obs = rng.uniform(-1.0, 1.0, size=(config.OBS_SIZE,)).astype(np.float32)

    # Mimic bounded fields from the design doc.
    obs[0:3] = rng.uniform(0.0, 1.0, size=(3,))  # position
    obs[6:13] = rng.uniform(0.0, 1.0, size=(7,))  # health/armor/time/phase
    obs[13] = float(rng.integers(0, 2))  # in_zone
    obs[87] = float(rng.integers(0, 2))  # gunfire_nearby
    obs[91] = float(rng.integers(0, 2))  # zone_shrinking
    obs[93] = float(rng.integers(0, 2))  # has_medkit
    obs[94] = float(rng.integers(0, 2))  # has_grenade
    return obs


def _rule_based_tactical_label(obs: np.ndarray) -> int:
    """
    Produce tactical class (0..5):
    0 push, 1 hold, 2 flank, 3 retreat, 4 heal, 5 ability.
    """
    health = obs[6]
    threat = np.clip(obs[86], 0.0, 1.0)
    has_medkit = obs[93] > 0.5
    has_ability = obs[95] < 0.35  # lower cooldown -> ready soon
    in_zone = obs[13] > 0.5
    gunfire = obs[87] > 0.5

    if health < 0.25:
        return 4 if has_medkit else 3
    if threat > 0.75 and health < 0.45:
        return 3
    if gunfire and threat > 0.6:
        return 1
    if not in_zone:
        return 0
    if has_ability and threat > 0.5:
        return 5
    return 2 if threat < 0.35 else 0


def _build_action_vector(rng: np.random.Generator, tactical_label: int) -> np.ndarray:
    """Create action[32] with tactical portion encoded in indices 20..25."""
    action = rng.uniform(-1.0, 1.0, size=(config.ACTION_SIZE,)).astype(np.float32)

    # Binary-like movement/combat switches in [0,1] where useful.
    for idx in [2, 3, 4, 7, 9, 11, 12]:
        action[idx] = rng.uniform(0.0, 1.0)

    # One-hot tactical block in action tensor (indices 20..25).
    action[20:26] = 0.0
    action[20 + tactical_label] = 1.0
    return action


def generate_split(samples: int, seed: int) -> DatasetSplit:
    rng = np.random.default_rng(seed)
    observations = np.zeros((samples, config.OBS_SIZE), dtype=np.float32)
    actions = np.zeros((samples, config.ACTION_SIZE), dtype=np.float32)
    tactical_targets = np.zeros((samples,), dtype=np.int64)

    for i in range(samples):
        obs = _sample_observation(rng)
        label = _rule_based_tactical_label(obs)
        act = _build_action_vector(rng, label)
        observations[i] = obs
        actions[i] = act
        tactical_targets[i] = label

    return DatasetSplit(observations=observations, actions=actions, tactical_targets=tactical_targets)


def save_split(path: str, split: DatasetSplit) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        observations=split.observations,
        actions=split.actions,
        tactical_targets=split.tactical_targets,
    )


def load_split(path: str) -> DatasetSplit:
    data = np.load(path)
    return DatasetSplit(
        observations=data["observations"].astype(np.float32),
        actions=data["actions"].astype(np.float32),
        tactical_targets=data["tactical_targets"].astype(np.int64),
    )
