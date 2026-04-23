"""Supervised warm-start and RL fine-tune utilities for TinySLMPolicy."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from slm.slm_model import TinySLMPolicy


@dataclass
class SLMTrainResult:
    artifact_path: str
    train_accuracy: float
    mean_reward_last_20: float


def _supervised_step(model: TinySLMPolicy, obs: np.ndarray, y: int, lr: float) -> float:
    p = model.probs(obs)
    one_hot = np.zeros_like(p)
    one_hot[y] = 1.0
    diff = p - one_hot
    hist_feat = model._history_features(obs)

    model.obs_w -= lr * np.outer(obs, diff).astype(np.float32)
    model.hist_w -= lr * np.outer(hist_feat, diff).astype(np.float32)
    model.b -= lr * diff.astype(np.float32)
    return float(-np.log(max(p[y], 1e-8)))


def supervised_warm_start(
    observations: np.ndarray,
    tactical_targets: np.ndarray,
    epochs: int = 6,
    learning_rate: float = 0.003,
) -> TinySLMPolicy:
    model = TinySLMPolicy(obs_dim=observations.shape[1], n_actions=6, seed=42)
    rng = np.random.default_rng(42)
    n = len(observations)

    for ep in range(1, epochs + 1):
        idx = rng.permutation(n)
        loss_sum = 0.0
        for i in idx:
            loss_sum += _supervised_step(model, observations[i], int(tactical_targets[i]), learning_rate)
        print(f"[slm warmstart {ep:02d}] loss={loss_sum/n:.4f}")
    return model


def rl_finetune(
    model: TinySLMPolicy,
    observations: np.ndarray,
    tactical_targets: np.ndarray,
    episodes: int = 120,
    steps_per_episode: int = 120,
    learning_rate: float = 0.002,
    gamma: float = 0.98,
) -> float:
    rng = np.random.default_rng(123)
    rewards_window: list[float] = []

    for ep in range(1, episodes + 1):
        ep_reward = 0.0
        traj: list[tuple[np.ndarray, int, float]] = []

        idx = int(rng.integers(0, len(observations)))
        for _ in range(steps_per_episode):
            obs = observations[idx]
            target = int(tactical_targets[idx])
            probs = model.probs(obs)
            action = int(rng.choice(6, p=probs))
            reward = 1.0 if action == target else -0.4
            traj.append((obs, action, reward))
            ep_reward += reward
            idx = int(rng.integers(0, len(observations)))

        # REINFORCE-style update with discounted returns.
        returns = []
        running = 0.0
        for _, _, r in reversed(traj):
            running = r + gamma * running
            returns.append(running)
        returns.reverse()
        ret = np.array(returns, dtype=np.float32)
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)

        for (obs, action, _), g_t in zip(traj, ret):
            p = model.probs(obs)
            one_hot = np.zeros_like(p)
            one_hot[action] = 1.0
            diff = (p - one_hot) * float(g_t)
            hist_feat = model._history_features(obs)
            model.obs_w -= learning_rate * np.outer(obs, diff).astype(np.float32)
            model.hist_w -= learning_rate * np.outer(hist_feat, diff).astype(np.float32)
            model.b -= learning_rate * diff.astype(np.float32)

        rewards_window.append(ep_reward)
        if ep % 20 == 0:
            print(f"[slm rl {ep:03d}] mean_reward(last20)={np.mean(rewards_window[-20:]):.3f}")

    return float(np.mean(rewards_window[-20:])) if len(rewards_window) >= 20 else float(np.mean(rewards_window))


def train_slm_end_to_end(
    observations: np.ndarray,
    tactical_targets: np.ndarray,
    episodes: int,
    steps_per_episode: int,
    learning_rate: float,
    gamma: float,
    output_dir: str = "artifacts",
) -> SLMTrainResult:
    model = supervised_warm_start(observations, tactical_targets, epochs=6, learning_rate=learning_rate * 0.5)

    preds = np.array([model.predict(o) for o in observations], dtype=np.int64)
    train_accuracy = float((preds == tactical_targets).mean())

    mean_reward_last_20 = rl_finetune(
        model=model,
        observations=observations,
        tactical_targets=tactical_targets,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        learning_rate=learning_rate * 0.35,
        gamma=gamma,
    )

    os.makedirs(output_dir, exist_ok=True)
    artifact = os.path.join(output_dir, "slm_tactical_policy.npz")
    np.savez_compressed(artifact, obs_w=model.obs_w, hist_w=model.hist_w, b=model.b)

    return SLMTrainResult(
        artifact_path=artifact,
        train_accuracy=train_accuracy,
        mean_reward_last_20=mean_reward_last_20,
    )
